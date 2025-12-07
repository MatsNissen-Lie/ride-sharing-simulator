import numpy as np
from typing import List, Set, Dict, Optional, Tuple, FrozenSet, cast

from dataclasses import dataclass

from config import get_logger
from config.config import DELTA, OMEGA, VEHICLE_CAPACITY
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
INF = float("inf")
MAX_PERMUTATIONS = 10000  # Safety limit to prevent combinatorial explosion


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
    Stores cached validation metrics as NumPy arrays for O(1) vectorized checks.
    """

    stops: List[Stop]

    # Cached validation metrics
    arrival_times: np.ndarray  # shape (N,)
    slacks: np.ndarray  # shape (N,) - How much delay each node can absorb
    total_travel_time: float
    slack_score: float

    @staticmethod
    def validate(
        sequence: List[Stop],
        network: INetworkService,
        initial_load: int = 0,  # <--- NEW PARAMETER (For vehicle context)
    ) -> Optional["TripPermutation"]:
        """
        Validates a sequence for Time Windows AND Capacity.
        """
        max_capacity = VEHICLE_CAPACITY
        n = len(sequence)
        if n == 0:
            return TripPermutation([], np.array([]), np.array([]), 0.0, 0.0)

        # --- 0. Capacity & Load Check (New) ---
        current_load = initial_load

        # We check capacity BEFORE doing expensive network calls.
        # This is a very fast O(N) integer check.
        for stop in sequence:
            if stop.type == "pickup":
                current_load += 1
                # If your requests have varying sizes (groups), use: current_load += stop.load
            elif stop.type == "dropoff" or stop.type == "vehicle_drop":
                current_load -= 1

            # The Hard Constraint
            if current_load > max_capacity:
                return None

            # Sanity check for negative load (logic error in sequence generation)
            # if current_load < 0: return None

        # --- 1. Forward Pass (Earliest Arrival) ---
        earliest = np.zeros(n)
        current_t = sequence[0].hard_earliest
        earliest[0] = current_t

        cumulative_dist = 0.0

        for i in range(1, n):
            prev = sequence[i - 1]
            curr = sequence[i]

            path = network.get_shortest_path(prev.node_id, curr.node_id)
            tt = path.travel_time if path else INF

            if tt == INF:
                return None

            cumulative_dist += tt

            # Arrival time logic
            arrival = max(earliest[i - 1] + tt, curr.hard_earliest)
            earliest[i] = arrival

            if arrival > curr.hard_latest:
                return None

        # --- 2. Backward Pass (Latest Arrival) ---
        latest = np.zeros(n)
        latest[n - 1] = sequence[n - 1].hard_latest

        for i in range(n - 2, -1, -1):
            prev = sequence[i]
            curr = sequence[i + 1]
            path = network.get_shortest_path(prev.node_id, curr.node_id)
            tt = path.travel_time if path else INF

            required_by_next = latest[i + 1] - tt
            latest[i] = min(prev.hard_latest, required_by_next)

            if latest[i] < earliest[i] - 1e-5:
                return None

        slacks = latest - earliest
        total_slack = np.sum(slacks)

        return TripPermutation(
            stops=sequence,
            arrival_times=earliest,
            # latest_arrivals=latest,  # Added missing field based on context
            slacks=slacks,
            total_travel_time=cumulative_dist,
            slack_score=float(total_slack),
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


class VectorizedInserter:
    """
    Perform matrix-based feasibility checks for vehicle insertion.
    """

    def __init__(self, matrix: np.ndarray, node_to_idx: Dict[int, int]):
        self.matrix = matrix
        self.node_to_idx = node_to_idx

    def find_best_vehicle_weave(
        self, base_perm: TripPermutation, v_stops: List[Stop]
    ) -> Optional[float]:
        """
        Returns the Best Cost (initial_delay + min_detour) if feasible.
        """
        if not v_stops:
            return None

        try:
            trip_indices = [self.node_to_idx[s.node_id] for s in base_perm.stops]
            v_indices = [self.node_to_idx[s.node_id] for s in v_stops]
        except KeyError:
            return None

        trip_nodes = np.array(trip_indices, dtype=np.int32)
        slacks = base_perm.slacks

        if len(slacks) > 0:
            downstream_slack = np.minimum.accumulate(slacks[::-1])[::-1]
        else:
            downstream_slack = np.array([])

        limit_vector = np.append(downstream_slack, INF)

        # 1. Check Prefix (V -> T0)
        s1 = v_stops[0]
        v_idx = v_indices[0]

        if len(trip_indices) > 0:
            t0_idx = trip_indices[0]
            travel_time = self.matrix[v_idx, t0_idx]

            # Time we arrive at T0 vs Planned T0
            arrival_at_t0 = s1.hard_earliest + travel_time

            # Hard Constraint: T0 Deadline
            # Note: base_perm.stops[0].hard_latest might be different from base limits?
            # Yes, uses absolute time.
            if arrival_at_t0 > base_perm.stops[0].hard_latest:
                return None

            # Delay check
            initial_delay = max(0.0, arrival_at_t0 - base_perm.arrival_times[0])
            if initial_delay > limit_vector[0]:
                return None
        else:
            initial_delay = 0.0

        if len(v_stops) == 1:
            return initial_delay

        # 2. Weave S2
        s2 = v_stops[1]
        s2_idx = v_indices[1]

        return self._solve_weave_s2(
            trip_nodes,
            limit_vector,
            initial_delay,
            s2_idx,
            s2.hard_latest,
            v_idx,
            s1.hard_earliest,
            base_perm.arrival_times,
        )

    def _solve_weave_s2(
        self,
        trip_nodes: np.ndarray,
        limit_vector: np.ndarray,
        initial_delay: float,
        s2_idx: int,
        s2_deadline: float,
        v_idx: int,
        v_start_time: float,
        base_arrivals: np.ndarray,
    ) -> Optional[float]:

        N = len(trip_nodes)

        # Prepare arrays for J=0..N
        # Prev Node Indices: [V, T0, T1... TN-1]
        prevs_idx = np.empty(N + 1, dtype=np.int32)
        prevs_idx[0] = v_idx
        if N > 0:
            prevs_idx[1:] = trip_nodes

        # Prev Node Planned Arrivals (for S2 arrival calc)
        # For J=0, Prev is V, 'Planned' is V_start_time
        # For J>0, Prev is T_{j-1}, Planned is base_arrivals[j-1]
        prev_arrivals = np.empty(N + 1, dtype=np.float64)
        prev_arrivals[0] = v_start_time
        if N > 0:
            prev_arrivals[1:] = base_arrivals

        # --- CALC DETOURS & ARRIVALS (Vectorized) ---

        # 1. Dist(Prev, S2)
        d_prev_s2 = self.matrix[prevs_idx, s2_idx]

        # 2. Arrival at S2 check
        # S2 arrival = Prev_Arr + initial_delay (if J>0 else 0) + Dist(Prev, S2)
        # Shift Vector: [0, initial_delay, initial_delay...]
        shifts = np.full(N + 1, initial_delay)
        shifts[0] = 0.0

        arrival_s2 = prev_arrivals + shifts + d_prev_s2

        # Constraint: S2 Deadline
        valid_s2 = arrival_s2 <= s2_deadline

        if not np.any(valid_s2):
            return None

        # 3. Calculate Detour Costs
        costs = np.full(N + 1, INF)

        # J = 0 to N-1 (Insert in middle)
        if N > 0:
            n_vec = trip_nodes
            d_s2_next = self.matrix[s2_idx, n_vec]
            d_prev_next = self.matrix[prevs_idx[:N], n_vec]
            detours = d_prev_s2[:N] + d_s2_next - d_prev_next

            # Total Delay = initial_delay + detours
            # (Note for J=0, detours captures the diff correctly as derived before)
            total_delay = initial_delay + detours

            # Check Limits
            feasible = valid_s2[:N] & (total_delay <= limit_vector[:N])
            costs[:N] = np.where(feasible, total_delay, INF)

        # J = N (Insert at end)
        if valid_s2[N]:
            # No downstream delay constraint
            costs[N] = initial_delay

        best_idx = np.argmin(costs)
        min_cost = costs[best_idx]

        if min_cost == INF:
            return None

        return float(min_cost)


class PermutationRTVGraphBuilder(IRTVGraphBuilder):
    """
    RTV Builder using "Persistent Permutations" (Alonso-Mora Inductive Step).
    Stores valid topological sorts of requests to avoid re-solving TSP.
    """

    def __init__(self, route_planner: ISingleVehicleRoutePlanner) -> None:
        super().__init__(route_planner)
        self._network = route_planner._network

    def _create_dense_matrix(
        self, requests: Set[Request], vehicles: Set[VehicleAvailability]
    ) -> Tuple[np.ndarray, Dict[int, int]]:
        """
        Collects all unique nodes from Requests and Vehicles (Current location + Destinations).
        Fetches the dense travel time matrix.
        Returns (Matrix, Node_Map).
        """
        nodes = set()
        for r in requests:
            nodes.add(r.origin_node)
            nodes.add(r.destination_node)

        for v in vehicles:
            nodes.add(v.available_at_node)
            # Add dropoff destinations for passengers
            for p in v.passengers_onboard:
                nodes.add(p.original_request.destination_node)

        # Ensure we have a list for consistent ordering
        sorted_nodes = sorted(list(nodes))
        node_to_idx = {node: i for i, node in enumerate(sorted_nodes)}

        # Use existing service method
        matrix = self._network.get_travel_time_matrix(sorted_nodes, sorted_nodes)
        return matrix, node_to_idx

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

        # --- PRE-COMPUTE DENSE MATRIX ---
        matrix, node_map = self._create_dense_matrix(requests, vehicles)
        inserter = VectorizedInserter(matrix, node_map)

        # Save inserter for use in methods if needed, or pass it
        self._inserter = inserter

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
                Stop(
                    r.id,
                    r.origin_node,
                    r.request_time,
                    r.request_time + OMEGA,
                    type="pickup",
                ),
                Stop(
                    r.id,
                    r.destination_node,
                    r.request_time,
                    r.earliest_dropoff_time + DELTA,
                    type="dropoff",
                ),
            ]

            # Validate (construct) the single request.
            # Implied feasible by RVGraph existence.
            topo = TripPermutation.validate(stops, self._network)
            if not topo:
                continue
            global_trips[1][t_id] = GlobalTrip(
                id=t_id,
                request_ids=t_id,
                permutations=[topo],
                candidate_vehicle_ids=set(connected_vehs),
            )
            # add all singles to RTV graph
            for v_id in connected_vehs:
                rv_route = rv_graph.rv_details.get((r_id, v_id))
                if rv_route:
                    rtv_graph.add_trip_for_vehicle(t_id, v_id, rv_route)

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
                        new_permutations.sort(key=lambda p: p.slack_score, reverse=True)
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
        Checks each candidate vehicle against the permutations using Vectorized Weave.
        Adds valid assignments to rtv_graph.
        Returns the subset of candidate_vehs that are actually feasible.
        """
        valid_vehs: Set[str] = set()

        for vid in candidate_vehs:
            vehicle = vehicles_by_id.get(vid)
            if not vehicle:
                continue

            # Construct Vehicle Partial Path (The "Chain")
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
                        p.original_request.id,
                        p.original_request.destination_node,
                        vehicle.available_from_time,
                        p.original_request.earliest_dropoff_time + DELTA,
                        type="vehicle_drop",
                    )
                )

            # TRY VECTORIZED MATCH
            best_cost = INF
            # We don't reconstruct the full route here yet in this optimization phase,
            # we just check feasibility/cost.
            # But the RTV graph needs the 'route'.
            # If cost < INF, we can either re-run the detailed check or (better)
            # have the vectorized inserter return enough info to build it.

            # For now, let's keep the high-level logic:
            # Check ALL permutations with VectorizedInsterter.
            # If ANY fits, we perform the detailed check on the BEST one to actully get the route object.

            potential_best_perm = None

            for perm in permutations:
                # The Vectorized Feasibility Check
                # Does v_stops fit into perm?
                # Returns estimated cost (or None if infeasible)
                cost = self._inserter.find_best_vehicle_weave(perm, v_stops)

                if cost is not None and cost < best_cost:
                    best_cost = cost
                    potential_best_perm = perm

            if potential_best_perm:
                # Found a feasible permutation!
                # Now we must construct the actual FeasibleRoute object for the RTV graph.
                # We reuse the logic from before, but ONLY for this best permutation

                full_sequence = v_stops + potential_best_perm.stops

                # Note: 'validate' here re-checks, which is redundant if we trust Inserter,
                # but necessary to get the final list of stops/times in the format RTV expects.
                final_topo = TripPermutation.validate(
                    full_sequence,
                    self._network,
                    initial_load=len(vehicle.passengers_onboard),
                )

                if final_topo:
                    reqs = {requests_by_id[rid] for rid in trip_id}
                    res = self._route_planner.check_feasibility(
                        reqs, vehicle
                    )  # This constructs the detailed route

                    if res.is_feasible:
                        f_res = cast(FeasibleRoute, res)
                        rtv_graph.add_trip_for_vehicle(trip_id, vid, f_res)
                        valid_vehs.add(vid)

                        # Optimization: We found a valid one. We could continue to find a BETTER one?
                        # Above loop visits ALL permutations, picking the best 'cost'.
                        # So potential_best_perm IS the best one.

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
                    topo = TripPermutation.validate(
                        final_stops, self._network, initial_load=0
                    )
                    if topo:
                        valid_results.append(topo)

        return valid_results
