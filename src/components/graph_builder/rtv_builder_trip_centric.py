from collections import defaultdict
from typing import List, Set, Dict, Optional, Tuple, FrozenSet
import numpy as np
from dataclasses import dataclass, field
from enum import IntEnum, auto

from config import get_logger
from config.config import (
    OMEGA,
    DELTA,
)  # Assuming these might be in config based on previous context, or I define them
from src.interfaces import (
    ISingleVehicleRoutePlanner,
    VehicleAvailability,
    Request,
    FeasibleRoute,
    Route,
)
from src.interfaces.graphs import RTVGraph, TripId, RVGraph
from src.interfaces.services import IRTVGraphBuilder, INetworkService

logger = get_logger(__name__)

# Constants (Fallback if not in config)
INF = float("inf")


class StopType(IntEnum):
    PICKUP = auto()
    DROPOFF = auto()
    VEHICLE_START = auto()
    VEHICLE_DROP = auto()


@dataclass(frozen=True)
class Stop:
    id: str  # Original ID (Request ID or Vehicle ID)
    node_id: int  # Physical Road Network Node

    # Absolute Time Constraints
    hard_earliest: float
    hard_latest: float

    # Metadata for debugging/sorting
    type: StopType

    @property
    def is_pickup(self) -> bool:
        return self.type == StopType.PICKUP

    @property
    def is_vehicle_start(self) -> bool:
        return self.type == StopType.VEHICLE_START


@dataclass
class UnifiedTripGraph:
    """
    A graph representing a set of stops and the valid transitions between them.
    Uses 'Branch-and-Bound' to find optimal Hamiltonian path.
    """

    stops: List[Stop]
    adj: np.ndarray  # [N, N] float
    prereqs: np.ndarray  # [N] bitmask
    n: int

    # Pruning Vectors (Calculated during extend)
    E: np.ndarray  # [N] Earliest arrival at node i
    L: np.ndarray  # [N] Latest arrival at node i

    @classmethod
    def empty(cls) -> "UnifiedTripGraph":
        return cls(
            [], np.zeros((0, 0)), np.zeros(0, dtype=int), 0, np.zeros(0), np.zeros(0)
        )

    def extend(
        self,
        new_stops: List[Stop],
        precedence_rules: List[Tuple[int, int]],
        network: INetworkService,
    ) -> Optional["UnifiedTripGraph"]:
        """
        Expands the graph with new stops and strictly defined precedence rules.
        """
        n_old = self.n
        n_new = len(new_stops)
        n_total = n_old + n_new

        # 1. Expand Matrix
        new_adj = np.full((n_total, n_total), INF)
        if n_old > 0:
            new_adj[:n_old, :n_old] = self.adj

        # 2. Expand Prerequisites (Bitmasks)
        new_prereqs = np.zeros(n_total, dtype=int)
        if n_old > 0:
            new_prereqs[:n_old] = self.prereqs

        # 3. Compute Edges (Raw Feasibility)
        all_stops = self.stops + new_stops
        node_ids = [s.node_id for s in all_stops]

        # Precompute strict time windows
        # For new nodes, use their intrinsic constraints.
        # For old nodes, we could inherit old E/L but simple approach is to rebuild from scratch
        # or just use intrinsic. Let's use intrinsic for now, and rely on solve() to handle tightness?
        # Actually, for B&B to be fast, we want tight L.
        # We can do a quick Backward Pass here if we want "Perfect Pruning",
        # or we just rely on intrinsic hard_latest.
        # Let's strictly use intrinsic hard_earliest/latest for E/L for now to match constraints.
        # IMPROVEMENT: We could propagate L backwards to prune earlier.

        earliest_vec = np.array([s.hard_earliest for s in all_stops])
        latest_vec = np.array([s.hard_latest for s in all_stops])

        for i in range(n_total):
            for j in range(n_total):
                if i < n_old and j < n_old:
                    continue  # Skip old internal
                if i == j:
                    continue

                # Get Travel Time
                path = network.get_shortest_path(node_ids[i], node_ids[j])
                tt = path.travel_time if path else INF

                if tt == INF:
                    continue

                # Basic Time Window Feasibility
                if earliest_vec[i] + tt <= latest_vec[j]:
                    new_adj[i, j] = tt

        # 4. Apply Precedence Rules
        for pre_idx, post_idx in precedence_rules:
            u = n_old + pre_idx
            v = n_old + post_idx
            new_prereqs[v] |= 1 << u
            new_adj[v, u] = INF  # Disable backward edge strictly

        # 5. Backward Propagation for L (Optional but recommended for B&B)
        # We want L[u] to be min(u.late, min(L[v] - travel(u,v)))
        # This gives us a much tighter bound.
        # Iterate backwards from deadlines? Or just simple N-pass?
        # Given N is small (<= 10), we can just do a few relaxation passes or strict order if topo sort known.
        # Since we don't have strict topo sort guaranteed (general TSP), let's just use N relaxation passes (Bellman-Ford style).

        L = latest_vec.copy()

        # Simple relaxation for Tight L
        # For k in range(n_total):
        #    changed = False
        #    for u in range(n_total):
        #        for v in range(n_total):
        #             if new_adj[u, v] != INF:
        #                 # To arrive at v by L[v], we must leave u by L[v] - travel
        #                 # So we must arrive at u by ... min(L[v] - travel - service?)
        #                 # We assuming 0 service time for now or included in travel.
        #                 req_arrival_u = L[v] - new_adj[u, v]
        #                 if req_arrival_u < L[u]:
        #                     L[u] = req_arrival_u
        #                     changed = True
        #    if not changed: break

        # But wait, we cannot violate u's intrinsic L.
        # And we can't arrive earlier than u's E.
        # If L[u] < E[u] after propagation, then graph is valid? No, Graph is INVALID.
        # B&B handles invalidity, but catching it here is faster?
        # Let's trust B&B for validity, just use L for pruning.
        # Keep intrinsic L for now to keep implementation simple and correct to plan.
        # The user's B&B code uses 'self.L[u]'.

        return UnifiedTripGraph(
            stops=all_stops,
            adj=new_adj,
            prereqs=new_prereqs,
            n=n_total,
            E=earliest_vec,  # Intrinsic E
            L=L,  # Intrinsic L
        )

    def precompute_suffixes(
        self,
    ) -> Dict[Tuple[str, FrozenSet[str]], Tuple[float, List[str]]]:
        """
        Populates a cache of optimal paths for all subsets of the graph.
        Used for the 'Universal Suffix Cache'.
        Returns: Key -> (Cost, List[StopIDs])
        """
        # We only cache purely internal paths (no time windows? or abstract time windows?)
        # The prompt says: "The answer to this question is universal... the geometry... doesn't change."
        # This implies we solve for Distance/Geometry primarily, or Feasible Time Windows assuming t=Earliest?
        # Actually, for the suffix cache to be valid, it must be valid for *some* time.
        # But since we verify time later, we can just find the Min-Cost Geometric Path
        # that obeys Precedence and Intrinsic Time Windows (assuming broad availability).

        # We use a Memoized approach as requested.
        cache: Dict[Tuple[str, FrozenSet[str]], Tuple[float, List[str]]] = {}

        # Map ID to Index
        id_to_idx = {s.id: i for i, s in enumerate(self.stops)}

        # Recursive Solver
        def solve(u_id: str, remaining: FrozenSet[str]) -> Tuple[float, List[str]]:
            key = (u_id, remaining)
            if key in cache:
                return cache[key]

            if not remaining:
                return (0.0, [])

            best_c, best_seq = float("inf"), []

            u_idx = id_to_idx[u_id]

            for v_id in remaining:
                v_idx = id_to_idx[v_id]

                # A. Connectivity
                dist = self.adj[u_idx, v_idx]
                if dist == INF:
                    continue

                # B. Precedence (Must respect strictly)
                # We need to know if v requires something in 'remaining'
                # if mask(v) & mask(remaining) != 0 -> Invalid
                # Bitmask check:
                v_prereqs = self.prereqs[v_idx]
                # If any bit in v_prereqs corresponds to a node in 'remaining', we cannot visit v yet.
                is_blocked = False
                for r_id in remaining:
                    if r_id == v_id:
                        continue
                    r_idx = id_to_idx[r_id]
                    if v_prereqs & (1 << r_idx):
                        is_blocked = True
                        break
                if is_blocked:
                    continue

                # C. Recurse
                new_rem = remaining - {v_id}
                sub_cost, sub_seq = solve(v_id, new_rem)

                total = dist + sub_cost
                if total < best_c:
                    best_c = total
                    best_seq = [v_id] + sub_seq

            res = (best_c, best_seq)
            cache[key] = res
            return res

        # Run for all logical start nodes?
        # The vehicle will arrive at *any* node 'u' of this trip.
        # So we should populate the cache for common entries.
        # Or, we just populate it lazily?
        # The USER said "Pre-Computer... runs this once."
        # So we iterate all stops as potential 'u', and all subsets as 'remaining'?
        # That's 2^N * N. N<=8. 256*8 is tiny.

        all_ids = frozenset(s.id for s in self.stops)

        # Populate cache for all possible (u_id, remaining_set) pairs
        # This is a full precomputation for the geometric paths
        for u_stop in self.stops:
            # For each stop as a potential starting point 'u'
            # We need to consider all subsets of other stops as 'remaining'
            # This is typically done by iterating through bitmasks for 'remaining'

            # For N <= 8, we can iterate through all subsets
            # The `solve` function will naturally populate the cache for all subproblems
            # when called for the full problem.

            # To ensure full population, we can call `solve` for each possible start node
            # and the set of all other nodes.

            # The `solve` function is designed to be called with a specific `u_id` and `remaining` set.
            # It will memoize results for all subproblems it encounters.
            # So, calling it for each `u_id` with `all_ids - {u_id}` will populate the cache
            # for all relevant suffixes.

            initial_remaining = all_ids - {u_stop.id}
            solve(u_stop.id, initial_remaining)

        # We return the solver wrapper or populator?
        # Let's return the cache Dict that is built on demand or full build.
        # Let's do Full Build for maximum speed later.

        # Trigger for all possible split points?
        # Actually, since it's recursive, triggering it for "All nodes" with "All meaningful subsets" is complex.
        # Maybe better to let the Vehicle Solver call `solve_suffix` and populate the shared dict?
        # The User said "When we create the GlobalTrip... we run this once."
        # OK, let's implement `solve_suffix` using the `suffix_cache` passed in.
        return cache

    def solve_best_path(
        self, 
        suffix_cache: Optional[Dict] = None,
        current_load: int = 0,
        max_capacity: int = 4
    ) -> Optional[FeasibleRoute]:
        """
        Finds the OPTIMAL Hamiltonian path (visiting all nodes).
        Uses 'suffix_cache' (Universal Cache) to short-circuit the tail.
        Enforces strict Capacity Constraints dynamically.
        """
        self.best_cost = float("inf")  # Reset for each solve call
        self.best_path = []  # Reset for each solve call
        self.suffix_cache = suffix_cache if suffix_cache is not None else {}
        self.max_capacity = max_capacity # Store for DFS

        # Identify Start Node (Vehicle Start)
        start_idx = -1
        vehicle_stops_mask = 0

        for i, s in enumerate(self.stops):
            if s.type == StopType.VEHICLE_START:
                start_idx = i
                vehicle_stops_mask |= 1 << i
            elif s.type == StopType.VEHICLE_DROP:
                vehicle_stops_mask |= 1 << i

        self.vehicle_stops_mask = vehicle_stops_mask
        self.id_to_idx = {s.id: i for i, s in enumerate(self.stops)}

        # State
        self.best_cost = float("inf")
        self.best_path = []

        if start_idx == -1:
            return None

        # DFS
        # Start load is passed in (number of existing pax)
        # Note: VEHICLE_START doesn't add load itself, it just represents the state "at start".
        # But we must ensure initial load <= capacity (redundant if vehicle is valid)
        if current_load > max_capacity: return None

        self._bnb_dfs(
            u=start_idx,
            visited=(1 << start_idx),
            time=self.stops[start_idx].hard_earliest,
            cost=0.0,
            load=current_load,
            path=[start_idx],
        )

        if self.best_cost == float("inf"):
            return None

        # Reconstruct
        ordered_stops = [self.stops[i] for i in self.best_path]
        final_stops = [s for s in ordered_stops if s.type != StopType.VEHICLE_START]

        return FeasibleRoute(
            route=Route(tuple(final_stops), self.best_cost, 0.0, 0.0),
            cost=self.best_cost,
        )

    def _bnb_dfs(self, u: int, visited: int, time: float, cost: float, load: int, path: List[int]):
        # 1. Pruning: Cost
        if cost >= self.best_cost:
            return

        # 2. Pruning: Lookahead (Time)
        if time > self.L[u] + 1e-5:
            return

        # --- OPTIMIZATION: CACHE HIT ---
        if (visited & self.vehicle_stops_mask) == self.vehicle_stops_mask:
            current_id = self.stops[u].id
            remaining_ids = []
            for i in range(self.n):
                if not (visited & (1 << i)):
                    if not (self.vehicle_stops_mask & (1 << i)):
                        remaining_ids.append(self.stops[i].id)

            if not remaining_ids:
                if cost < self.best_cost:
                    self.best_cost = cost
                    self.best_path = list(path)
                return

            remaining_set = frozenset(remaining_ids)
            cache_key = (current_id, remaining_set)

            if cache_key in self.suffix_cache:
                cached_cost, cached_seq_ids = self.suffix_cache[cache_key]
                if self._verify_tail_times(time, cached_seq_ids, u):
                    # verify capacity for the suffix?
                    # The Suffix Cache is purely geometric. 
                    # We must verify capacity for the cached path too!
                    # This is tricky. The cache doesn't know our current load.
                    # We have to walk the suffix steps and check violations.
                    if self._verify_tail_capacity(load, cached_seq_ids):
                         full_cost = cost + cached_cost
                         if full_cost < self.best_cost:
                             self.best_cost = full_cost
                             seq_indices = [self.id_to_idx[sid] for sid in cached_seq_ids]
                             self.best_path = path + seq_indices
                return

        # 3. Branching (Standard)
        candidates = []
        for v in range(self.n):
            if not (visited & (1 << v)):
                dist = self.adj[u, v]
                if dist == INF: continue

                # Precedence
                if (visited & self.prereqs[v]) != self.prereqs[v]: continue

                # Time Windows
                arrival = time + dist
                start_v = max(arrival, self.E[v])
                if start_v > self.L[v]: continue
                
                # --- CAPACITY CHECK ---
                # Check impact of VISITING v
                # If v is PICKUP, load increases. Check limit.
                # If v is DROPOFF, load decreases.
                v_stop = self.stops[v]
                new_load = load
                
                if v_stop.type == StopType.PICKUP:
                    new_load += 1
                elif v_stop.type == StopType.DROPOFF or v_stop.type == StopType.VEHICLE_DROP:
                    new_load -= 1
                
                # Sanity: Load can't be negative (logic error if so)
                # But allowing new_load=0 is fine.
                
                # HARD CONSTRAINT: Do not exceed capacity
                if new_load > self.max_capacity:
                    continue
                
                # END CAPACITY CHECK
                
                step_cost = dist + (start_v - arrival)
                candidates.append((v, start_v, cost + step_cost, new_load))

        candidates.sort(key=lambda x: x[2])

        for v, start_v, new_cost, n_load in candidates:
            path.append(v)
            self._bnb_dfs(v, visited | (1 << v), start_v, new_cost, n_load, path)
            path.pop()

    def _verify_tail_times(self, start_time: float, suffix_ids: List[str], current_u_idx: int) -> bool:
        """
        Checks if the cached geometric path is valid given the current arrival time.
        """
        t = start_time
        curr = current_u_idx
        
        for next_id in suffix_ids:
            next_idx = self.id_to_idx[next_id]
            
            # Travel
            dt = self.adj[curr, next_idx]
            if dt == INF: return False 
            
            t += dt
            
            # Windows
            if t < self.stops[next_idx].hard_earliest:
                t = self.stops[next_idx].hard_earliest
                
            if t > self.stops[next_idx].hard_latest:
                return False
                
            curr = next_idx
            
        return True

    def _verify_tail_capacity(self, start_load: int, suffix_ids: List[str]) -> bool:
        """
        Checks if the cached path violates capacity constraints starting with `start_load`.
        """
        current_load = start_load
        
        for next_id in suffix_ids:
            next_idx = self.id_to_idx[next_id]
            stop = self.stops[next_idx]
            
            if stop.type == StopType.PICKUP:
                current_load += 1
            elif stop.type == StopType.DROPOFF or stop.type == StopType.VEHICLE_DROP:
                current_load -= 1
            
            if current_load > self.max_capacity:
                return False
                
        return True


@dataclass
class GlobalTrip:
    id: TripId
    request_ids: FrozenSet[str]
    topology: UnifiedTripGraph
    candidate_vehicle_ids: Set[str]
    # CACHE: Key -> (Geometric Cost, Sequence of IDs)
    suffix_cache: Dict[Tuple[str, FrozenSet[str]], Tuple[float, List[str]]] = field(
        default_factory=dict
    )


class TripCentricRTVGraphBuilder(IRTVGraphBuilder):
    """
    New RTV Builder implementation using Micro-DFS Validator.
    Replaces global propagation with on-the-fly Hamiltonian search.
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
        rtv = RTVGraph()
        req_map = {r.id: r for r in requests}
        veh_map = {v.vehicle_id: v for v in vehicles}

        # Master storage: size -> { frozenset(ids) : GlobalTrip }
        global_trips: Dict[int, Dict[TripId, GlobalTrip]] = defaultdict(dict)

        # --- PHASE 1: SIZE 1 (Base Case) ---
        for r in requests:
            t_id = frozenset([r.id])

            # 1. Get Initial Candidates from RV Graph
            connected_vids = rv_graph.get_rv_neighbors(r.id)
            if not connected_vids:
                continue

            # 2. Build Base Graph (Request Only)
            # Create P -> D graph for later expansion
            stops = [
                Stop(
                    r.id,
                    r.origin_node,
                    r.request_time,
                    r.request_time + OMEGA,
                    type=StopType.PICKUP,
                ),
                Stop(
                    r.id,
                    r.destination_node,
                    r.request_time,
                    r.earliest_dropoff_time + DELTA,
                    type=StopType.DROPOFF,
                ),
            ]
            base_graph = UnifiedTripGraph.empty().extend(stops, [(0, 1)], self._network)

            if not base_graph:
                continue

            # 3. Filter Vehicles & Build RTV
            valid_vids = set()
            for vid in connected_vids:
                # Trust RV Graph for Size 1
                route = rv_graph.get_rv_route(r.id, vid)
                if route:
                    rtv.add_trip_for_vehicle(t_id, vid, route)
                    valid_vids.add(vid)

            if valid_vids:
                global_trips[1][t_id] = GlobalTrip(
                    id=t_id,
                    request_ids=t_id,
                    topology=base_graph,
                    candidate_vehicle_ids=valid_vids,
                )

        # --- PHASE 2 & 3: EXPANSION LOOP ---

        for k in range(2, max_trip_size + 1):
            prev_level = global_trips[k - 1]
            if not prev_level:
                break

            # We iterate to generate candidates
            for base_trip in prev_level.values():
                anchor_id = next(iter(base_trip.request_ids))
                neighbors = rv_graph.get_rr_neighbors(anchor_id)

                for new_r_id in neighbors:
                    if new_r_id in base_trip.request_ids:
                        continue

                    new_ids: TripId = base_trip.request_ids | {new_r_id}

                    # 1. SUBTRIP LEMMA CHECK
                    # If we've already processed this set (via another path), skip
                    if new_ids in global_trips[k]:
                        continue

                    # Verify all subsets of size k-1 exist
                    is_valid_subtrip = True
                    for r_removed in new_ids:
                        sub_trip_id = new_ids - {r_removed}
                        if sub_trip_id not in prev_level:
                            is_valid_subtrip = False
                            break
                    if not is_valid_subtrip:
                        continue

                    # 2. VEHICLE INTERSECTION (Fast Prune)
                    new_r_vehs = rv_graph.get_rv_neighbors(new_r_id)
                    candidates = base_trip.candidate_vehicle_ids & new_r_vehs

                    if not candidates:
                        continue

                    # 3. BUILD BASE GRAPH (Request Only)
                    # Extend the k-1 graph with the 1 new request
                    new_r = req_map[new_r_id]
                    req_stops = [
                        Stop(
                            new_r.id,
                            new_r.origin_node,
                            new_r.request_time,
                            new_r.request_time + OMEGA,
                            type=StopType.PICKUP,
                        ),
                        Stop(
                            new_r.id,
                            new_r.destination_node,
                            new_r.request_time,
                            new_r.earliest_dropoff_time + DELTA,
                            type=StopType.DROPOFF,
                        ),
                    ]
                    # Precedence: P -> D. P is index 0 relative to new list, D is 1.
                    req_graph = base_trip.topology.extend(
                        req_stops, [(0, 1)], self._network
                    )

                    if not req_graph:
                        continue

                    # 4. VEHICLE VALIDATION & ASSIGNMENT
                    surviving_vehicles = set()

                    # Precompute Suffixes for the New Request Set (Geometric)
                    # This is the "Universal Cache" the user requested.
                    # It works for ANY vehicle once they clear their own drops.
                    suffix_cache = req_graph.precompute_suffixes()

                    for vid in candidates:
                        vehicle = veh_map.get(vid)
                        if not vehicle: continue
                        
                        # A. Inject Vehicle into Graph
                        veh_graph = self._inject_vehicle(req_graph, vehicle)
                        if not veh_graph: continue 

                        # B. Solve (DFS B&B) with Suffix Cache & Capacity Constraint
                        # We pass 'current_occupancy' and 'capacity' to the solver.
                        # Note: The solver starts at 'V_Start', effectively t=Available.
                        # The initial load at V_Start is len(vehicle.passengers_onboard).
                        # We must track load dynamically.
                        initial_load = len(vehicle.passengers_onboard)
                        
                        result = veh_graph.solve_best_path(
                            suffix_cache=suffix_cache,
                            current_load=initial_load,
                            max_capacity=vehicle.capacity
                        )
                        
                        if result:
                            # SUCCESS:
                            rtv.add_trip_for_vehicle(new_ids, vid, result)
                            surviving_vehicles.add(vid)

                    # 5. SURVIVAL CHECK
                    if surviving_vehicles:
                        global_trips[k][new_ids] = GlobalTrip(
                            id=new_ids,
                            request_ids=new_ids,
                            topology=req_graph,
                            candidate_vehicle_ids=surviving_vehicles,
                            suffix_cache=suffix_cache,
                        )

        return rtv

    def _inject_vehicle(
        self, base_graph: UnifiedTripGraph, vehicle: VehicleAvailability
    ) -> Optional[UnifiedTripGraph]:
        """
        Merges 'New Requests' (Base Graph) with 'Vehicle State' (Start + Existing Dropoffs).

        Optimization: We know V->P_new is valid (from RV Graph).
        The solver's job is to check the cross-constraints:
        - Can we go Drop_Old -> Pick_New?
        - Can we go Pick_New -> Drop_Old?
        """

        # 1. Define Vehicle Nodes
        v_stops = []

        # A. Vehicle Start (Anchor)
        start = Stop(
            id="V_Start",
            node_id=vehicle.available_at_node,
            hard_earliest=vehicle.available_from_time,
            hard_latest=INF,
            type=StopType.VEHICLE_START,
        )
        v_stops.append(start)

        # B. Existing Passengers (Floating Dropoffs)
        # They have strict deadlines but NO fixed order relative to new stops.
        for p in vehicle.passengers_onboard:
            s = Stop(
                id=p.original_request.id,
                node_id=p.original_request.destination_node,
                hard_earliest=vehicle.available_from_time,  # Ready to drop immediately
                hard_latest=p.original_request.earliest_dropoff_time + DELTA,
                type=StopType.VEHICLE_DROP,
            )
            v_stops.append(s)

        # 2. Precedence Rules
        # Rule: Vehicle Start (Index 0) must precede ALL other vehicle nodes.
        # We DO NOT enforce constraints between existing drops. The solver optimizes them.

        rules = []
        for i in range(1, len(v_stops)):
            rules.append((0, i))

        return base_graph.extend(v_stops, rules, self._network)
