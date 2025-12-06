from typing import List, Set, Dict, Optional, FrozenSet, cast

from dataclasses import dataclass

from config import get_logger
from src.interfaces import (
    ISingleVehicleRoutePlanner,
    VehicleAvailability,
    Request,
    FeasibleRoute,
)
from src.interfaces.graphs import RTVGraph, TripId, RVGraph
from src.interfaces.services import IRTVGraphBuilder, INetworkService

logger = get_logger(__name__)

# Constants
INF = 1e9
OMEGA = 600.0  # Max wait time (should probably come from config)
DELTA = 900.0  # Max detor/delay time (should probably come from config)
MAX_PERMUTATIONS = 20  # Safety limit to prevent combinatorial explosion


@dataclass(frozen=True)
class Stop:
    id: str  # Original ID (Request ID or Vehicle ID)
    node_id: int  # Physical Road Network Node

    # Absolute Time Constraints
    hard_earliest: float
    hard_latest: float

    # Metadata for debugging/sorting
    type: str = "request"  # "pickup" | "dropoff" | "vehicle_start" | "vehicle_drop"

    def __repr__(self) -> str:
        return f"{self.id}({self.type[:1]})"


@dataclass(frozen=True)
class TripPermutation:
    """
    A specific valid permutation of stops.
    Stores cached validation metrics and slack score.
    """

    stops: List[Stop]

    # Cached validation metrics
    total_travel_time: float
    earliest_arrivals: List[float]
    latest_arrivals: List[float]

    # Slack score for pruning (higher is better/looser)
    slack_score: float

    @staticmethod
    def validate(
        sequence: List[Stop], network: INetworkService
    ) -> Optional["TripPermutation"]:
        """
        Validates a fixed sequence of stops and computes tightest time windows.
        Returns TripPermutation if valid, None otherwise.
        """
        n = len(sequence)
        if n == 0:
            return TripPermutation([], 0.0, [], [], 0.0)

        # 1. Forward Pass (Earliest Arrival)
        earliest = [0.0] * n
        # The sequence is relative. For a set of requests, "time 0" is virtual?
        # No, for requests, we use their absolute constraint times.
        # earliest[0] must be at least hard_earliest[0].

        current_t = sequence[0].hard_earliest
        earliest[0] = current_t

        cumulative_dist = 0.0

        for i in range(1, n):
            prev = sequence[i - 1]
            curr = sequence[i]

            # Use get_shortest_path to be consistent with previous code using travel_time
            # Ideally we used get_travel_time but previous code used path object
            # optimizing: use get_travel_time if available or cache
            # The interface INetworkService likely has get_shortest_path
            path = network.get_shortest_path(prev.node_id, curr.node_id)
            tt = path.travel_time if path else INF

            if tt == INF:
                return None

            cumulative_dist += tt

            # Arrival time
            arrival = max(earliest[i - 1] + tt, curr.hard_earliest)
            earliest[i] = arrival

            # Basic feasibility check during forward pass (Fail Early)
            if arrival > curr.hard_latest:
                return None

        # 2. Backward Pass (Latest Arrival)
        latest = [0.0] * n
        latest[n - 1] = sequence[n - 1].hard_latest

        for i in range(n - 2, -1, -1):
            prev = sequence[i]
            curr = sequence[i + 1]
            path = network.get_shortest_path(prev.node_id, curr.node_id)
            tt = path.travel_time if path else INF

            # Latest we can leave prev to reach curr by latest[i+1]
            required_by_next = latest[i + 1] - tt
            latest[i] = min(prev.hard_latest, required_by_next)

            if latest[i] < earliest[i] - 1e-5:
                return None

        # Calculate Slack Score (Sum of (L - E))
        # This represents the "robustness" or "schedulability" of the route
        total_slack = sum(l - e for l, e in zip(latest, earliest))

        return TripPermutation(
            stops=sequence,
            total_travel_time=cumulative_dist,
            earliest_arrivals=earliest,
            latest_arrivals=latest,
            slack_score=total_slack,
        )


@dataclass
class GlobalTrip:
    """
    Represents a SET of requests {r1, r2...}
    and the valid permutations that serve them.
    """

    id: TripId
    request_ids: FrozenSet[str]
    permutations: List[TripPermutation]
    candidate_vehicle_ids: Set[str]  # The "Intersection" set


class PermutationRTVGraphBuilder(IRTVGraphBuilder):
    """
    RTV Builder using "Persistent Permutations" (Alonso-Mora Inductive Step).
    Stores valid topological sorts of requests to avoid re-solving TSP.
    """

    def __init__(self, route_planner: ISingleVehicleRoutePlanner) -> None:
        super().__init__(route_planner)
        self._network = route_planner._network

    def build_graph(
        self,
        rv_graph: RVGraph,
        vehicles: Set[VehicleAvailability],
        requests: Set[Request],
        max_trip_size: int,
    ) -> RTVGraph:
        """
        Builds the RTV graph using the Global Trip + Incremental Permutation approach.
        """
        rtv_graph = RTVGraph()

        # Map vehicles/requests for easy access
        vehicles_by_id = {v.vehicle_id: v for v in vehicles}
        requests_by_id = {r.id: r for r in requests}

        # Global Trips Storage: size -> {TripId -> GlobalTrip}
        global_trips: Dict[int, Dict[TripId, GlobalTrip]] = {}
        for k in range(1, max_trip_size + 1):
            global_trips[k] = {}

        # --- PHASE 1: SIZE 1 (Base Case) ---
        # Optimization: Only consider requests that have at least one reachable vehicle.
        for r_id, connected_vehs in rv_graph.request_vehicle.items():
            r = requests_by_id.get(r_id)
            if not r:
                continue

            t_id: TripId = frozenset([r.id])

            # Create Base Request Stops
            stops = [
                Stop(r.id, r.origin_node, r.request_time, r.request_time + OMEGA, type="pickup"),
                Stop(r.id, r.destination_node, r.request_time, r.earliest_dropoff_time + DELTA, type="dropoff"),
            ]

            # Validate (construct) the single request.
            # Implied feasible by RVGraph existence.
            topo = TripPermutation.validate(stops, self._network)
            
            if topo is not None:
                # Integrate Validation & Assignment Logic
                valid_vehicles = self._assign_and_prune_vehicles(
                    t_id, [topo], connected_vehs, vehicles_by_id, requests_by_id, rtv_graph
                )

                if valid_vehicles:
                    global_trips[1][t_id] = GlobalTrip(
                        id=t_id,
                        request_ids=t_id,
                        permutations=[topo],
                        candidate_vehicle_ids=valid_vehicles,
                    )

        # --- PHASE 2 & 3: EXPANSION (Recursive Step) ---
        for k in range(2, max_trip_size + 1):
            prev_level = global_trips[k - 1]

            for base_trip in prev_level.values():
                if not base_trip.candidate_vehicle_ids:
                    continue

                # 1. CLIQUE OPTIMIZATION & ORDERING
                max_req_in_base = max(base_trip.request_ids)
                
                # Intersect neighbors of ALL requests strings in the trip.
                candidates = set.intersection(
                    *(rv_graph.get_rr_neighbors(r) for r in base_trip.request_ids)
                )

                if not candidates:
                    continue

                for new_r_id in candidates:
                    # Enforce Ordering
                    if new_r_id <= max_req_in_base:
                        continue

                    # Create ID for potential new trip
                    new_ids: TripId = base_trip.request_ids | {new_r_id}

                    if new_ids in global_trips[k]:
                        continue

                    # --- CHECK 1: Vehicle Intersection ---
                    new_r_vehs = rv_graph.get_rv_neighbors(new_r_id)
                    common_vehs = base_trip.candidate_vehicle_ids & new_r_vehs
                    if not common_vehs:
                        continue

                    # --- CHECK 2: Subtrip Lemma (Apriori) ---
                    is_valid_subtrip = True
                    for r_removed in new_ids:
                        sub_trip_id = new_ids - {r_removed}
                        if sub_trip_id == base_trip.request_ids:
                            continue
                        if sub_trip_id not in prev_level:
                            is_valid_subtrip = False
                            break
                    if not is_valid_subtrip:
                        continue

                    # --- CHECK 3: Permutation Extension ---
                    new_req = requests_by_id[new_r_id]
                    new_pickup = Stop(
                        new_req.id,
                        new_req.origin_node,
                        new_req.request_time,
                        new_req.request_time + OMEGA,
                        type="pickup",
                    )
                    new_dropoff = Stop(
                        new_req.id,
                        new_req.destination_node,
                        new_req.request_time,
                        new_req.earliest_dropoff_time + DELTA,
                        type="dropoff",
                    )

                    new_permutations = self._extend_permutations(
                        base_trip.permutations, new_pickup, new_dropoff
                    )

                    if not new_permutations:
                        continue

                    # Pruning
                    if len(new_permutations) > MAX_PERMUTATIONS:
                        new_permutations.sort(
                            key=lambda p: p.slack_score, reverse=True
                        )
                        new_permutations = new_permutations[:MAX_PERMUTATIONS]

                    # Integrate Validation & Assignment Logic
                    valid_vehicles = self._assign_and_prune_vehicles(
                        new_ids,
                        new_permutations,
                        common_vehs,
                        vehicles_by_id,
                        requests_by_id,
                        rtv_graph,
                    )

                    if valid_vehicles:
                        global_trips[k][new_ids] = GlobalTrip(
                            id=new_ids,
                            request_ids=new_ids,
                            permutations=new_permutations,
                            candidate_vehicle_ids=valid_vehicles,
                        )

        return rtv_graph


    def _assign_and_prune_vehicles(
        self,
        trip_id: TripId,
        permutations: List[TripPermutation],
        candidate_vehs: Set[str],
        vehicles_by_id: Dict[str, VehicleAvailability],
        requests_by_id: Dict[str, Request],
        rtv_graph: RTVGraph,
    ) -> Set[str]:
        """
        Checks each candidate vehicle against the permutations.
        Adds valid assignments to rtv_graph.
        Returns the subset of candidate_vehs that are actually feasible.
        """
        valid_vehs: Set[str] = set()
        
        for vid in candidate_vehs:
            vehicle = vehicles_by_id.get(vid)
            if not vehicle:
                continue

            # Construct Vehicle Partial Path
            v_stops = [
                Stop(
                    vid,
                    vehicle.available_at_node,
                    vehicle.available_from_time,
                    INF,
                    type="vehicle_start",
                )
            ]

            for p in vehicle.passengers_onboard:
                v_stops.append(
                    Stop(
                        f"drop_{p.original_request.id}",
                        p.original_request.destination_node,
                        vehicle.available_from_time,
                        p.original_request.earliest_dropoff_time + DELTA,
                        type="vehicle_drop",
                    )
                )

            # Try to match Vehicle to EACH valid permutation
            best_route: Optional[FeasibleRoute] = None
            best_cost = INF

            for perm in permutations:
                full_sequence = v_stops + perm.stops

                # Validate this specific chain
                final_topo = TripPermutation.validate(
                    full_sequence, self._network
                )

                if final_topo:
                    # It's feasible!
                    reqs = {requests_by_id[rid] for rid in trip_id}
                    res = self._route_planner.check_feasibility(reqs, vehicle)

                    if res.is_feasible:
                        f_res = cast(FeasibleRoute, res)
                        if f_res.cost < best_cost:
                            best_route = f_res
                            best_cost = f_res.cost
                            
                        # Optimization: detailed route planner might be slow, so maybe stop at first?
                        # (Or check all permutations for best cost - using all)

            if best_route:
                rtv_graph.add_trip_for_vehicle(trip_id, vid, best_route)
                valid_vehs.add(vid)

        return valid_vehs

    def _extend_permutations(
        self,
        base_permutations: List[TripPermutation],
        new_pickup: Stop,
        new_dropoff: Stop,
    ) -> List[TripPermutation]:
        """
        Inserts new_pickup and new_dropoff into all valid positions of base_permutations.
        Returns flattened list of ALL valid extended permutations.
        """
        valid_results = []

        for base in base_permutations:
            base_stops = base.stops
            n = len(base_stops)

            # Optimization: Pre-check time windows against base bounds?
            # If new_pickup.hard_latest < base.earliest_start, maybe valid at start?

            # Insert Pickup at index i (0 to n)
            for i in range(n + 1):
                # Optimize: If we insert P after a node that arrives AFTER P.latest, impossible.
                # if i > 0 and base.earliest_arrivals[i-1] > new_pickup.hard_latest: break (approx)

                temp_stops_p = base_stops[:i] + [new_pickup] + base_stops[i:]

                # Insert Dropoff at index j (i+1 to n+1) -> strictly after Pickup
                for j in range(i + 1, n + 2):
                    final_stops = temp_stops_p[:j] + [new_dropoff] + temp_stops_p[j:]

                    # Validate
                    topo = TripPermutation.validate(final_stops, self._network)
                    if topo:
                        valid_results.append(topo)

        return valid_results

