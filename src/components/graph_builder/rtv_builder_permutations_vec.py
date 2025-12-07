import numpy as np
import itertools
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
        # Cache the matrix for direct access
        # We assume node_id corresponds directly to matrix index
        try:
             self.matrix = self.network.get_l1_cache()
        except NotImplementedError:
             self.matrix = None

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

    def find_optimal_vehicle_integration(
        self, 
        base_perm: TripPermutation, 
        v_stops: List[Stop]
    ) -> Optional[Tuple[float, List[Stop], List[int]]]:
        """
        1. Fixes v_stops[0] (Start).
        2. Permutes v_stops[1:] (Drops).
        3. Finds the best weave cost across all permutations.
        Returns: (cost, ordered_v_stops, insertion_indices)
        """
        if not v_stops:
            return 0.0, [], []

        # Need the matrix for this
        if self.matrix is None:
             # Fallback: cannot optimize without matrix
             return None

        # Optimization: If only Start + 1 Drop, no need to permute
        if len(v_stops) <= 2:
            res = self._solve_ordered_weave(base_perm, v_stops)
            if res:
                return res[0], v_stops, res[1]
            return None

        best_cost = INF
        best_res = None
        best_perm = None

        start_node = v_stops[0]
        drop_nodes = v_stops[1:]

        # Iterate all valid orderings of the passengers on board
        for p_drops in itertools.permutations(drop_nodes):
            # Construct candidate chain: [Start, Drop_A, Drop_B...]
            candidate_chain = [start_node] + list(p_drops)
            
            # Use the fast matrix weaver for this specific order
            res = self._solve_ordered_weave(base_perm, candidate_chain)
            
            if res:
                cost, indices = res
                if cost < best_cost:
                    best_cost = cost
                    best_res = (cost, candidate_chain, indices)
                    best_perm = candidate_chain

        return best_res

    def _solve_ordered_weave(
        self, 
        base_perm: TripPermutation, 
        ordered_v_stops: List[Stop]
    ) -> Optional[Tuple[float, List[int]]]:
        if self.matrix is None:
             return None

        """
        Weaves a FIXED chain of Vehicle stops into the Base Permutation.
        Returns (final_cost, list_of_insertion_indices).
        Index j means "Before T[j]".
        """
        N = len(base_perm.stops)
        K = len(ordered_v_stops)
        
        # Node indices
        # We assume node_id is the index.
        t_indices = [s.node_id for s in base_perm.stops]
        v_indices = [s.node_id for s in ordered_v_stops]
        
        # --- 0. Pre-computation ---
        if N > 0:
            limit_vector = np.minimum.accumulate(base_perm.slacks[::-1])[::-1]
            limit_vector = np.append(limit_vector, INF) # Pad for end
            base_arrivals = base_perm.earliest_arrivals 
        else:
            limit_vector = np.full(1, INF)
            base_arrivals = np.array([]) 

        # --- 1. DP State Initialization ---
        v0 = ordered_v_stops[0]
        v0_idx = v_indices[0]
        start_time = v0.hard_earliest
        
        current_delay = np.full(N + 1, INF)
        
        # Backtracking pointers: parents[k, j] stores the index i (prev_j)
        # We need to store parents for k=1..K-1. 
        # k=0 is fixed at 0.
        parents = np.zeros((K, N + 1), dtype=int)
        
        # Calculate transition V0 -> T0 (If N > 0)
        if N > 0:
            t0_idx = t_indices[0]
            travel = self.network.get_travel_time(v0_idx, t0_idx)

            assert travel is not None
            # assert self.matrix is not None (Checked in caller or via implicit)
            # Pyright complains optional. We handle via cast or assert.
            # Ideally we asserted at top of _solve_ordered_weave.
 
            
            if np.isnan(travel): return None
                
            arr_t0 = start_time + travel
            
            if arr_t0 > base_perm.stops[0].hard_latest:
                return None
                
            delay = max(0.0, arr_t0 - base_arrivals[0])
            
            if delay <= limit_vector[0]:
                current_delay[0] = delay # V0 inserted at 0
        else:
            current_delay[0] = 0.0

        # --- 2. DP Loop for V1...VK ---
        for k in range(1, K):
            v_curr = ordered_v_stops[k]
            v_idx = v_indices[k]
            v_prev_idx = v_indices[k-1]
            
            # Matrix of valid (i, j) pairs
            I, J = np.indices((N + 1, N + 1))
            mask_order = I <= J
            mask_prev_valid = current_delay[I] != INF
            valid_transitions = mask_order & mask_prev_valid
            
            if not np.any(valid_transitions):
                return None
                
            std_detours = self._calc_standard_detours(t_indices, v_idx, N)
            cost_separated = current_delay[I] + std_detours[J]
            
            chained_detours = self._calc_chained_detours(t_indices, v_prev_idx, v_idx, N)
            cost_chained = current_delay[I] + chained_detours[I]
            
            total_delay_grid = np.where(I < J, cost_separated, INF)
            np.fill_diagonal(total_delay_grid, cost_chained)
            
            total_delay_grid = np.where(valid_transitions, total_delay_grid, INF)
            
            feasible_limit = total_delay_grid <= limit_vector[J]
            valid_grid = np.where(feasible_limit, total_delay_grid, INF)
            
            # Reduce
            next_delay = np.min(valid_grid, axis=0) # min over I
            best_prevs = np.argmin(valid_grid, axis=0) # Index I that gave min
            
            if np.all(next_delay == INF):
                return None
            
            current_delay = next_delay
            parents[k] = best_prevs

        # End of Loop
        min_final_cost = np.min(current_delay)
        if min_final_cost == INF:
            return None
            
        # Backtrack
        final_j = np.argmin(current_delay)
        indices = [int(final_j)]
        curr = final_j
        
        for k in range(K - 1, 0, -1):
            prev = parents[k, curr]
            indices.append(int(prev))
            curr = prev
            
        indices.append(0) # V0 is always at 0
        indices.reverse()
        
        return min_final_cost, indices

    def _calc_standard_detours(self, t_indices: List[int], v_idx: int, N: int) -> np.ndarray:
        """Calc detour of inserting V into T at every pos j (assuming prev is T_{j-1})"""
        assert self.matrix is not None
        # Detour at j: Dist(T_{j-1}, V) + Dist(V, T_j) - Dist(T_{j-1}, T_j)
        # Detour at j: Dist(T_{j-1}, V) + Dist(V, T_j) - Dist(T_{j-1}, T_j)
        # Boundary j=0: Dist(Start?, V) + Dist(V, T0) - ... undefined.
        # But for std_detours we assume i < j, so j must be >= 1 usually if we start at 0.
        # If j=0, it means we insert at start. But we start at Start.
        # Actually i < j implies we visited at least one Trip Node if j > 0.
        
        # We need numpy arrays for T indices.
        # To avoid re-creating arrays every time, we could cache them on TripPermutation?
        # For now, create on fly.
        
        detours = np.full(N + 1, INF)
        
        if N == 0:
            return detours
        
        # Array of T nodes
        T = np.array(t_indices)
        
        # Dist(V, T_j) for j=0..N-1
        # dist_v_t = self.matrix[v_idx, T] 
        # But we need Dist(V, T_j) for j being the insertion index.
        # Insertion at j means before T[j].
        # So Next Node is T[j].
        # Valid for j=0..N-1. For j=N, Next is End (Virtual).
        
        # Dist(T_{j-1}, V)
        # Prev Node is T[j-1].
        # Valid for j=1..N.
        
        # We process range 1..N-1 (Inner)
        # j range [1, N-1] indices of detours
        # Prev: T[0..N-2]
        # Next: T[1..N-1]
        
        if N > 1:
            prev_nodes = T[:-1] # 0..N-2
            next_nodes = T[1:]  # 1..N-1
            
            d_prev_v = np.array([self.network.get_travel_time(n, v_idx) for n in prev_nodes])
            d_v_next = np.array([self.network.get_travel_time(v_idx, n) for n in next_nodes])
            d_prev_next = np.array([self.network.get_travel_time(p, n) for p, n in zip(prev_nodes, next_nodes)])
            
            detours[1:N] = d_prev_v + d_v_next - d_prev_next
            
        # Boundary j=N (End)
        # Prev: T[N-1]
        # Next: None?
        # Detour is just Dist(T_{N-1}, V)? 
        # No, "Detour" implies change in path info.
        # If we append V at end, the "Trip" ends at V instead of T_{N-1}.
        # Cost added to delay? 
        # If we consider the trip finishes at T_{N-1}, adding V adds Dist(T_{N-1}, V).
        # Yes.
        d_last_v = self.network.get_travel_time(T[-1], v_idx)
        detours[N] = d_last_v
        
        # Boundary j=0 (Start)
        # Only valid if i < j is impossible index-wise? 
        # If i=0, j=0 is chain.
        # If i=-1? No.
        # So for standard detour (gap), j must be >= 1 really.
        # We verify this in logic: I < J. If I >= 0, then J >= 1.
        
        return detours

    def _calc_chained_detours(self, t_indices: List[int], v_prev_idx: int, v_curr_idx: int, N: int) -> np.ndarray:
        """Calc detour of inserting V_curr immediately after V_prev at pos i"""
        assert self.matrix is not None
        # Cost = Dist(V_prev, V_curr) + Dist(V_curr, Next) - Dist(V_prev, Next)
        # Insertion at i. Next node is T[i] (if i < N).   
        
        detours = np.full(N + 1, INF)
        
        if N == 0:
            detours[0] = self.network.get_travel_time(v_prev_idx, v_curr_idx)
            return detours
            
        T = np.array(t_indices)
        
        # Constant Term
        d_vv = self.network.get_travel_time(v_prev_idx, v_curr_idx)
        
        # Terms dependent on Next Node (T[i]) for i=0..N-1
        d_curr_next = np.array([self.network.get_travel_time(v_curr_idx, n) for n in T])
        d_prev_next = np.array([self.network.get_travel_time(v_prev_idx, n) for n in T])
        
        detours[:N] = d_vv + d_curr_next - d_prev_next
        
        # Boundary i=N (End)
        # Next is None.
        # We act as if we just append V_curr after V_prev.
        # Cost is Dist(V_prev, V_curr).
        detours[N] = d_vv
        
        return detours

    def find_vehicle_chain_insertion(
        self, base_perm: TripPermutation, vehicle_stops: List[Stop]
    ) -> List[Tuple[int, ...]]:
         return [] # Deprecated/Unused in new logic



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

            # Check Feasibility using Vectorized Weaver
            best_route = None
            best_cost = INF

            for perm in permutations:
                # Returns (cost, ordered_v_stops, insertion_indices) or None
                res = self._inserter.find_optimal_vehicle_integration(perm, v_stops)
                
                if res is not None:
                     cost, best_v_stops, indices = res
                     # The cost returned is MINIMUM ADDITIONAL DELAY.
                     # We might want total travel time as metric?
                     # TripPermutation stores 'total_travel_time'.
                     # New Time ~= Old Time + Delay? 
                     # Not exactly, but close enough for greedy.
                     # Better: use the delay as the criterion.
                     
                     if cost < best_cost:
                        best_cost = cost
                        
                        # Reconstruct the merged sequence
                        merged_stops = []
                        
                        # Pointer to the trip permutation
                        t_ptr = 0
                        
                        # Pointer to the vehicle stops
                        v_ptr = 0
                        K = len(best_v_stops)
                        
                        # N trip stops. Indices 0..N.
                        # Index j means "Insert BEFORE T[j]".
                        
                        for j in range(len(perm.stops) + 1):
                            # Append all vehicle stops destined for this slot
                            while v_ptr < K and indices[v_ptr] == j:
                                merged_stops.append(best_v_stops[v_ptr])
                                v_ptr += 1
                                
                            # Append the trip stop if valid
                            if j < len(perm.stops):
                                merged_stops.append(perm.stops[j])
                                
                        # FINAL CHECK VALIDATOR (To get precise timings/cost)
                        full_perm = self._validator.validate(merged_stops)
                        
                        if full_perm:
                            final_cost = full_perm.total_travel_time
                            
                            interface_stops = []
                            for s in full_perm.stops:
                                is_pickup = s.type in ("pickup", "vehicle_start")
                                interface_stops.append(
                                    InterfaceStop(
                                        node=s.node_id, is_pickup=is_pickup, request_id=s.id
                                    )
                                )

                            route_obj = InterfaceRoute(
                                stops=tuple(interface_stops),
                                total_travel_time=final_cost,
                                total_travel_distance=0.0,
                                total_delay_cost=final_cost, # Or slack?
                            )

                            best_route = FeasibleRoute(
                                route=route_obj,
                                cost=final_cost,
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
