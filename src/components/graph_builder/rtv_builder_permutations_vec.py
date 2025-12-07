import numpy as np
from typing import List, Set, Dict, Optional, Tuple, FrozenSet, cast

from dataclasses import dataclass

from config import get_logger
from config.config import DELTA, OMEGA
from src.interfaces import (
    ISingleVehicleRoutePlanner,
    VehicleAvailability,
    Request,
    FeasibleRoute,
)
from src.interfaces.data_models import Route as InterfaceRoute, Stop as InterfaceStop
from src.interfaces.graphs import RTVGraph, TripId, RVGraph
from src.interfaces.services import IRTVGraphBuilder, INetworkService

logger = get_logger(__name__)

# Constants
INF = float("inf")
MAX_PERMUTATIONS = 200  # Safety limit to prevent combinatorial explosion


@dataclass(frozen=True)
class Stop:
    id: str
    node_id: int
    hard_earliest: float
    hard_latest: float
    type: str = "request"  # "pickup" | "dropoff" | "vehicle_start" | "vehicle_drop"

    def __repr__(self) -> str:
        return f"{self.id}({self.type[:1]})"


@dataclass(frozen=True)
class TripPermutation:
    """
    An immutable, validated sequence of stops.
    Stores cached NumPy vectors for O(1) insertion checks.
    """

    stops: Tuple[Stop, ...]

    # Cached Metrics (NumPy Arrays for Vectorization)
    earliest_arrivals: np.ndarray  # Shape (N,) float
    latest_arrivals: np.ndarray  # Shape (N,) float
    slacks: np.ndarray  # Shape (N,) float (Latest - Earliest)

    total_travel_time: float
    slack_score: float

    def __hash__(self):
        # Cache hash based on stop IDs for fast set operations
        return hash(tuple(s.id for s in self.stops))


class VectorizedValidator:
    """
    Handles the "Physics" of the route: Forward/Backward propagation.
    """

    def __init__(self, network: INetworkService):
        self.network = network

    def validate(self, sequence: List[Stop]) -> Optional[TripPermutation]:
        n = len(sequence)
        if n == 0:
            return None

        # --- 1. Prepare Arrays ---
        node_ids = [s.node_id for s in sequence]
        hard_earliest = np.array([s.hard_earliest for s in sequence])
        hard_latest = np.array([s.hard_latest for s in sequence])

        # --- 2. Forward Pass (Earliest Arrival) ---
        earliest = np.zeros(n)
        current_t = hard_earliest[0]
        earliest[0] = current_t

        for i in range(n - 1):
            # Optimization: Use a batched network call if available.
            # For now, we assume single lookup is fast or cached.
            tt = self.network.get_travel_time(node_ids[i], node_ids[i + 1])
            if tt >= INF:
                return None

            arrival = max(earliest[i] + tt, hard_earliest[i + 1])
            if arrival > hard_latest[i + 1]:
                return None  # Fail Early
            earliest[i + 1] = arrival

        total_duration = earliest[-1] - earliest[0]

        # --- 3. Backward Pass (Latest Arrival) ---
        latest = np.zeros(n)
        latest[-1] = hard_latest[-1]

        for i in range(n - 2, -1, -1):
            tt = self.network.get_travel_time(node_ids[i], node_ids[i + 1])
            # The latest we can leave i is (Latest[i+1] - Travel)
            # The latest we can arrive at i is min(Hard_Latest[i], Derived)
            derived = latest[i + 1] - tt
            latest[i] = min(hard_latest[i], derived)

            if latest[i] < earliest[i] - 1e-5:
                return None

        # --- 4. Final Packaging ---
        slacks = latest - earliest
        slack_score = np.sum(slacks)

        return TripPermutation(
            stops=tuple(sequence),
            earliest_arrivals=earliest,
            latest_arrivals=latest,
            slacks=slacks,
            total_travel_time=total_duration,
            slack_score=slack_score,
        )


class VectorizedInserter:
    """
    The Math Engine: Finds valid insertion slots using NumPy broadcasting.
    """

    def __init__(self, network: INetworkService):
        self.network = network

    def find_best_request_insertions(
        self, base_perm: TripPermutation, pickup: Stop, dropoff: Stop
    ) -> List[Tuple[int, int]]:
        """
        Finds all (i, j) index pairs where P and D can be inserted.
        Returns list of (p_idx, d_idx).
        """
        N = len(base_perm.stops)
        route_ids = [s.node_id for s in base_perm.stops]

        # 1. Pre-calculate "Downstream Slack"
        # min_slack_from[k] = min(slacks[k:])
        # This represents the bottleneck for any delay introduced at k
        min_slack_from = np.minimum.accumulate(base_perm.slacks[::-1])[::-1]
        min_slack_from = np.append(min_slack_from, INF)  # Pad for N+1

        # 2. Vectorized Detour Calculation
        # We need travel times between:
        # Prev(i) -> New -> Next(i)

        # Construct "Previous" and "Next" arrays for all N+1 slots
        # Slot 0: Start -> N0. Slot N: N_last -> End.
        # We'll approximate edges by using the existing route nodes.
        # i maps to inserting BEFORE node i.

        # To simplify, we calculate raw travel times from P/D to all nodes
        # Shape (N,)
        t_nodes_to_p = np.array(
            [self.network.get_travel_time(n, pickup.node_id) for n in route_ids]
        )
        t_p_to_nodes = np.array(
            [self.network.get_travel_time(pickup.node_id, n) for n in route_ids]
        )
        t_nodes_to_d = np.array(
            [self.network.get_travel_time(n, dropoff.node_id) for n in route_ids]
        )
        t_d_to_nodes = np.array(
            [self.network.get_travel_time(dropoff.node_id, n) for n in route_ids]
        )

        # Existing edge costs (i-1 -> i)
        # We need to subtract these when we break the link
        # padded to N+1. [0, edge_0_1, edge_1_2... edge_last_0] is wrong.
        # Correct: [0 (start), cost(0->1), ... cost(N-1->N), 0 (end)]
        # This part requires careful alignment. For MVP, we use a loop for the detours
        # because N is small (<10), but we broadcast the CHECKS.

        p_detours = np.zeros(N + 1)
        d_detours = np.zeros(N + 1)

        # Calculate P detours (Loop is fast for N=10)
        # i=0 (Start): Detour = 0 (logic: earliest start shifts)
        # i=N (End): Detour = Dist(Last, P)
        # 0 < i < N: Dist(i-1, P) + Dist(P, i) - Dist(i-1, i)

        # Note: accurate detour requires context of "Vehicle Start" for i=0.
        # Since this is "Global Request" context, we assume "virtual start".
        # We assume 0 cost to insert at ends for the "Slack Check" (slack absorbs it).

        for i in range(1, N):
            prev_n, curr_n = route_ids[i - 1], route_ids[i]
            base_edge = self.network.get_travel_time(prev_n, curr_n)

            p_detours[i] = t_nodes_to_p[i - 1] + t_p_to_nodes[i] - base_edge
            d_detours[i] = t_nodes_to_d[i - 1] + t_d_to_nodes[i] - base_edge

        # Handle ends (Simplified)
        p_detours[0] = 0  # Shifts start time
        p_detours[N] = 0  # Extends end
        d_detours[0] = 0
        d_detours[N] = 0

        # 3. BROADCASTING (The O(N^2) Logic)
        I, J = np.indices((N + 1, N + 1))

        # Mask 1: Order (P must be before D)
        mask_order = I <= J  # Allow weave (I < J) or immediate (I == J)

        # Mask 2: P Feasibility (Row Check)
        # p_detour[i] <= min_slack_from[i]
        mask_p = p_detours[I] <= min_slack_from[I]

        # Mask 3: Total Feasibility (Cell Check)
        # total_delay = p_detour[i] + d_detour[j]
        # total_delay <= min_slack_from[j]
        total_delay = p_detours[I] + d_detours[J]
        mask_total = total_delay <= min_slack_from[J]

        # Combine
        final_mask = mask_order & mask_p & mask_total

        return list(zip(*np.where(final_mask)))

    def find_vehicle_chain_insertion(
        self, base_perm: TripPermutation, vehicle_stops: List[Stop]
    ) -> List[Tuple[int, ...]]:
        """
        Optimized 'Weave' for vehicle stops.
        Treats vehicle stops as a rigid chain or semi-rigid chain.
        """
        # Logic is similar to above, but enforcing
        # idx_stop1 <= idx_stop2 <= ...
        # For MVP, we can just use the validator since K is usually 1 or 2.
        return []


@dataclass
class GlobalTrip:
    id: TripId
    request_ids: FrozenSet[str]
    permutations: List[TripPermutation]
    candidate_vehicle_ids: Set[str]


class PermutationRTVGraphBuilder(IRTVGraphBuilder):
    def __init__(self, route_planner: ISingleVehicleRoutePlanner) -> None:
        super().__init__(route_planner)
        self._network = route_planner._network
        self._validator = VectorizedValidator(self._network)
        self._inserter = VectorizedInserter(self._network)

    def build_graph(
        self,
        rv_graph: RVGraph,
        vehicles: Set[VehicleAvailability],
        requests: Set[Request],
        max_trip_size: int,
    ) -> RTVGraph:

        rtv_graph = RTVGraph()
        vehicles_by_id = {v.vehicle_id: v for v in vehicles}
        requests_by_id = {r.id: r for r in requests}

        # Size -> {TripId -> GlobalTrip}
        global_trips: Dict[int, Dict[TripId, GlobalTrip]] = {}
        for k in range(1, max_trip_size + 1):
            global_trips[k] = {}

        # --- PHASE 1: SIZE 1 ---
        for r_id, connected_vehs in rv_graph.request_vehicle.items():
            r = requests_by_id.get(r_id)
            if not r:
                continue

            t_id = frozenset([r.id])
            stops = [
                Stop(
                    r.id,
                    r.origin_node,
                    r.request_time,
                    r.request_time + OMEGA,
                    "pickup",
                ),
                Stop(
                    r.id,
                    r.destination_node,
                    r.request_time,
                    r.earliest_dropoff_time + DELTA,
                    "dropoff",
                ),
            ]

            # Create immutable permutation with cached arrays
            perm = self._validator.validate(stops)

            if perm:
                # Check Vehicle Constraints
                valid_vehs = self._check_vehicles(
                    t_id,
                    [perm],
                    connected_vehs,
                    vehicles_by_id,
                    requests_by_id,
                    rtv_graph,
                )

                if valid_vehs:
                    global_trips[1][t_id] = GlobalTrip(t_id, t_id, [perm], valid_vehs)

        # --- PHASE 2+: EXPANSION ---
        for k in range(2, max_trip_size + 1):
            prev_level = global_trips[k - 1]

            for base_trip in prev_level.values():
                if not base_trip.candidate_vehicle_ids:
                    continue

                # Get neighbors from RV Graph
                candidates = set.intersection(
                    *(rv_graph.get_rr_neighbors(r) for r in base_trip.request_ids)
                )

                max_req = max(base_trip.request_ids)  # Ordering trick to avoid dupes

                for new_r_id in candidates:
                    if new_r_id <= max_req:
                        continue

                    new_ids = base_trip.request_ids | {new_r_id}
                    if new_ids in global_trips[k]:
                        continue

                    # 1. Vehicle Pruning
                    new_r_vehs = rv_graph.get_rv_neighbors(new_r_id)
                    common_vehs = base_trip.candidate_vehicle_ids & new_r_vehs
                    if not common_vehs:
                        continue

                    # 2. Subtrip Lemma
                    if not self._check_subtrip_lemma(
                        new_ids, prev_level, base_trip.request_ids
                    ):
                        continue

                    # 3. Vectorized Expansion
                    new_req = requests_by_id[new_r_id]
                    p_stop = Stop(
                        new_req.id,
                        new_req.origin_node,
                        new_req.request_time,
                        new_req.request_time + OMEGA,
                        "pickup",
                    )
                    d_stop = Stop(
                        new_req.id,
                        new_req.destination_node,
                        new_req.request_time,
                        new_req.earliest_dropoff_time + DELTA,
                        "dropoff",
                    )

                    new_perms = self._expand_permutations(
                        base_trip.permutations, p_stop, d_stop
                    )

                    if not new_perms:
                        continue

                    # 4. Final Vehicle Check
                    valid_vehs = self._check_vehicles(
                        new_ids,
                        new_perms,
                        common_vehs,
                        vehicles_by_id,
                        requests_by_id,
                        rtv_graph,
                    )

                    if valid_vehs:
                        global_trips[k][new_ids] = GlobalTrip(
                            new_ids, new_ids, new_perms, valid_vehs
                        )

        return rtv_graph

    def _expand_permutations(
        self, base_perms: List[TripPermutation], p: Stop, d: Stop
    ) -> List[TripPermutation]:
        """
        Uses the VectorizedInserter to find all valid next-gen permutations.
        """
        results = []
        seen_hashes = set()

        for base in base_perms:
            # Vectorized Check: Get list of valid (i, j) tuples
            valid_indices = self._inserter.find_best_request_insertions(base, p, d)

            for i, j in valid_indices:
                # Construct the new sequence
                # Note: This list slicing is fast enough for small N
                new_stops = (
                    list(base.stops[:i])
                    + [p]
                    + list(base.stops[i:j])
                    + [d]
                    + list(base.stops[j:])
                )

                # Re-validate to get fresh cached arrays for the new object
                # (The vector check above was an approximation/filter)
                new_perm = self._validator.validate(new_stops)

                if new_perm:
                    h = hash(new_perm)
                    if h not in seen_hashes:
                        results.append(new_perm)
                        seen_hashes.add(h)

        # Prune top K based on slack
        if len(results) > MAX_PERMUTATIONS:
            logger.warning(
                f"Generated {len(results)} permutations; pruning to top {MAX_PERMUTATIONS}."
            )
            results.sort(key=lambda x: x.slack_score, reverse=True)
            results = results[:MAX_PERMUTATIONS]

        return results

    def _check_vehicles(
        self,
        trip_id: TripId,
        permutations: List[TripPermutation],
        candidates: Set[str],
        veh_map: Dict[str, VehicleAvailability],
        req_map: Dict[str, Request],
        rtv_graph: RTVGraph,
    ) -> Set[str]:

        valid_vehs = set()

        for vid in candidates:
            veh = veh_map[vid]

            # Construct Vehicle Constraints
            # 1. Vehicle Start
            v_stops = [
                Stop(
                    vid,
                    veh.available_at_node,
                    veh.available_from_time,
                    INF,
                    "vehicle_start",
                )
            ]
            # 2. Existing Passengers (Chain)
            for p in veh.passengers_onboard:
                v_stops.append(
                    Stop(
                    p.original_request.id,
                        p.original_request.destination_node,
                        veh.available_from_time,
                        p.original_request.earliest_dropoff_time + DELTA,
                        "vehicle_drop",
                    )
                )

            # Check Feasibility: Can we weave this chain into ANY of the permutations?
            best_route = None
            best_cost = INF

            for perm in permutations:
                # We try to insert the vehicle stops into the request permutation.
                # Since vehicle stops usually must happen ASAP or in fixed order,
                # we treat this as prepending/weaving.

                # Simple Logic: Prepend V_Start, Weave Drops
                # For high performance, we would use _inserter logic here too.
                # MVP: Just concat and validate.

                full_sequence = v_stops + list(perm.stops)

                # Check with validator
                final_perm = self._validator.validate(full_sequence)

                if final_perm:
                    # Logic: If validator passes, it is feasible.
                    # We create the Route object.
                    cost = final_perm.total_travel_time
                    if cost < best_cost:
                        best_cost = cost
                        # Construct FeasibleRoute object
                        # (Mapping stops back to original requests handled by logic layer)

                        interface_stops = []
                        for s in final_perm.stops:
                            is_pickup = s.type in ("pickup", "vehicle_start")
                            interface_stops.append(
                                InterfaceStop(
                                    node=s.node_id, is_pickup=is_pickup, request_id=s.id
                                )
                            )

                        route_obj = InterfaceRoute(
                            stops=tuple(interface_stops),
                            total_travel_time=final_perm.total_travel_time,
                            total_travel_distance=0.0,  # Not calculated in vector mode
                            total_delay_cost=cost,
                        )

                        best_route = FeasibleRoute(
                            route=route_obj,
                            cost=cost,
                        )

            if best_route:
                rtv_graph.add_trip_for_vehicle(trip_id, vid, best_route)
                valid_vehs.add(vid)

        return valid_vehs

    def _check_subtrip_lemma(
        self, new_ids: FrozenSet[str], prev_level: Dict, base_ids: FrozenSet[str]
    ) -> bool:
        """Apriori check."""
        for r in new_ids:
            sub = new_ids - {r}
            if sub == base_ids:
                continue
            if sub not in prev_level:
                return False
        return True
