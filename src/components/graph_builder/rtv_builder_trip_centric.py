from collections import defaultdict
from typing import List, Set, Dict, Optional, Tuple, FrozenSet, cast
import numpy as np
from dataclasses import dataclass
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
    Route
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
    E: np.ndarray # [N] Earliest arrival at node i
    L: np.ndarray # [N] Latest arrival at node i

    @classmethod
    def empty(cls) -> "UnifiedTripGraph":
        return cls(
            [], 
            np.zeros((0, 0)), 
            np.zeros(0, dtype=int), 
            0,
            np.zeros(0),
            np.zeros(0)
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
            new_adj[v, u] = INF # Disable backward edge strictly

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
            E=earliest_vec, # Intrinsic E
            L=L             # Intrinsic L
        )

    def solve_best_path(self) -> Optional[FeasibleRoute]:
        """
        Finds the OPTIMAL Hamiltonian path (visiting all nodes) 
        that respects all time/precedence constraints.
        Minimizes total_delay_cost.
        """
        n = len(self.stops)
        # target_mask = (1 << n) - 1 # Unused variable
        
        # 1. Identify Start Node
        # We assume the last added node (Vehicle Start) is the start (Optimization)
        # Or search for generic vehicle_start
        start_node_idx = -1
        for i, s in enumerate(self.stops):
            if s.type == StopType.VEHICLE_START:
                start_node_idx = i
                break
        
        if start_node_idx == -1:
            # Fallback for pure request graphs: 
            # Try to start at the earliest pickup?
            # Or just return None because a trip without a vehicle isn't a route?
            # Actually, in Phase 2/3 we build request-only graphs.
            # We don't call solve() on those! We only call extend().
            # We ONLY call solve() when we inject a vehicle.
            # So start_node_idx SHOULD exist if we are calling solve().
            return None

        # State for optimization
        self.best_cost = float('inf')
        self.best_path = []

        # 2. Run Branch & Bound
        self._bnb_dfs(
            u=start_node_idx, 
            visited=(1 << start_node_idx), 
            time=self.stops[start_node_idx].hard_earliest,
            cost=0.0,
            path=[start_node_idx]
        )

        if self.best_cost == float('inf'):
            return None

        # 3. Reconstruct FeasibleRoute Object
        ordered_stops = [self.stops[i] for i in self.best_path]
        
        # Exclude 'Vehicle Start' phantom from final route object stops
        final_stops = [s for s in ordered_stops if s.type != StopType.VEHICLE_START]
        
        # The cost is sum of delays.
        return FeasibleRoute(
            route=Route(
                stops=tuple(final_stops),
                total_delay_cost=self.best_cost,
                total_travel_time=0.0, # Placeholder
                total_travel_distance=0.0 # Placeholder
            ),
            cost=self.best_cost
        )

    def _bnb_dfs(self, u: int, visited: int, time: float, cost: float, path: List[int]):
        """
        Recursive Branch-and-Bound with Slack Pruning.
        """
        # 1. Pruning: Cost Bound
        if cost >= self.best_cost:
            return

        # 2. Pruning: Slack / Time Windows
        # Lookahead: If we are already past L[u], clearly invalid. 
        # But checking `time > L[u]` is for the CURRENT node u.
        # We verified `start_v <= L[v]` before entering v.
        # However, checking against loose L[u] is still a safeguard.
        if time > self.L[u] + 1e-5:
            return

        # 3. Goal Check
        target_mask = (1 << self.n) - 1
        if visited == target_mask:
            if cost < self.best_cost:
                self.best_cost = cost
                self.best_path = list(path)
            return

        # 4. Branching
        candidates = []
        for v in range(self.n):
            if not (visited & (1 << v)):
                
                # A. Matrix Check
                dist = self.adj[u, v]
                if dist == INF: continue
                
                # B. Precedence Check
                # "Have I visited all prerequisites of v?"
                if (visited & self.prereqs[v]) != self.prereqs[v]:
                    continue

                # C. Arrival Logic
                arrival = time + dist
                
                # Wait if early (Slack absorption)
                # Ensure we don't start before earliest allowed time at v
                start_v = max(arrival, self.E[v]) 
                                                  
                # D. Deadline Check (Instant Pruning)
                if start_v > self.L[v]:
                    continue
                
                # E. Cost Increment
                # Cost = Wait time + Travel time? OR Lateness?
                # User's B&B snippet: step_cost = dist + wait.
                # Total Cost = Sum of (dist + wait) = Total Duration of Trip?
                # Total Delay Cost usually means (Arrival - RequestTime).
                # But prioritizing MINIMUM COMPLEITION TIME (Makespan) is equivalent to minimizing Wait+Travel.
                # Let's use `dist + wait` as the cost metric for B&B minimization.
                wait = start_v - arrival
                step_cost = dist + wait
                
                candidates.append((v, start_v, cost + step_cost))

        # Sort by cost (heuristic)
        candidates.sort(key=lambda x: x[2])

        for v, start_v, new_cost in candidates:
            path.append(v)
            self._bnb_dfs(v, visited | (1 << v), start_v, new_cost, path)
            path.pop()


@dataclass
class GlobalTrip:
    """
    Represents a SET of requests {r1, r2...} 
    and the Vehicles that can theoretically serve them.
    """
    id: TripId
    request_ids: FrozenSet[str]
    topology: UnifiedTripGraph
    candidate_vehicle_ids: Set[str]  # The "Intersection" set


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
            if not connected_vids: continue

            # 2. Build Base Graph (Request Only)
            # Create P -> D graph for later expansion
            stops = [
                Stop(r.id, r.origin_node, r.request_time, r.request_time + OMEGA, type=StopType.PICKUP),
                Stop(r.id, r.destination_node, r.request_time, r.earliest_dropoff_time + DELTA, type=StopType.DROPOFF)
            ]
            base_graph = UnifiedTripGraph.empty().extend(stops, [(0, 1)], self._network)
            
            if not base_graph: continue 

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
                    candidate_vehicle_ids=valid_vids
                )

        # --- PHASE 2 & 3: EXPANSION LOOP ---
        
        for k in range(2, max_trip_size + 1):
            prev_level = global_trips[k - 1]
            if not prev_level: break
            
            # We iterate to generate candidates
            for base_trip in prev_level.values():
                anchor_id = next(iter(base_trip.request_ids))
                neighbors = rv_graph.get_rr_neighbors(anchor_id)
                
                for new_r_id in neighbors:
                    if new_r_id in base_trip.request_ids: continue
                    
                    new_ids: TripId = base_trip.request_ids | {new_r_id}
                    
                    # 1. SUBTRIP LEMMA CHECK
                    # If we've already processed this set (via another path), skip
                    if new_ids in global_trips[k]: continue
                    
                    # Verify all subsets of size k-1 exist
                    is_valid_subtrip = True
                    for r_removed in new_ids:
                        sub_trip_id = new_ids - {r_removed}
                        if sub_trip_id not in prev_level:
                            is_valid_subtrip = False
                            break
                    if not is_valid_subtrip: continue
                    
                    # 2. VEHICLE INTERSECTION (Fast Prune)
                    new_r_vehs = rv_graph.get_rv_neighbors(new_r_id)
                    candidates = base_trip.candidate_vehicle_ids & new_r_vehs
                    
                    if not candidates: continue

                    # 3. BUILD BASE GRAPH (Request Only)
                    # Extend the k-1 graph with the 1 new request
                    new_r = req_map[new_r_id]
                    req_stops = [
                        Stop(new_r.id, new_r.origin_node, new_r.request_time, new_r.request_time + OMEGA, type=StopType.PICKUP),
                        Stop(new_r.id, new_r.destination_node, new_r.request_time, new_r.earliest_dropoff_time + DELTA, type=StopType.DROPOFF)
                    ]
                    # Precedence: P -> D. P is index 0 relative to new list, D is 1.
                    req_graph = base_trip.topology.extend(req_stops, [(0, 1)], self._network)
                    
                    if not req_graph: continue 
                    
                    # 4. VEHICLE VALIDATION & ASSIGNMENT
                    surviving_vehicles = set()
                    
                    for vid in candidates:
                        vehicle = veh_map.get(vid)
                        if not vehicle: continue
                        
                        # A. Inject Vehicle into Graph
                        veh_graph = self._inject_vehicle(req_graph, vehicle)
                        if not veh_graph: continue 

                        # B. Solve (DFS B&B)
                        result = veh_graph.solve_best_path()
                        
                        if result:
                            # SUCCESS:
                            rtv.add_trip_for_vehicle(new_ids, vid, result)
                            surviving_vehicles.add(vid)
                    
                    # 5. SURVIVAL CHECK
                    if surviving_vehicles:
                        global_trips[k][new_ids] = GlobalTrip(
                            id=new_ids,
                            request_ids=new_ids,
                            topology=req_graph, # Save the REQUEST graph
                            candidate_vehicle_ids=surviving_vehicles
                        )

        return rtv

    def _inject_vehicle(self, base_graph: UnifiedTripGraph, vehicle: VehicleAvailability) -> Optional[UnifiedTripGraph]:
        """
        Temporarily adds vehicle start/stops to the graph to check feasibility.
        """
        v_stops = []
        
        # 1. Start Node (Current State)
        start = Stop(
            vehicle.vehicle_id, 
            vehicle.available_at_node, 
            vehicle.available_from_time, 
            INF,
            type=StopType.VEHICLE_START
        )
        v_stops.append(start)
        
        # 2. Existing Passengers
        for p in vehicle.passengers_onboard:
            s = Stop(
                f"drop_{p.original_request.id}",
                p.original_request.destination_node, 
                vehicle.available_from_time, 
                p.original_request.earliest_dropoff_time + DELTA, # Using DELTA as per existing logic, or strict deadline
                type=StopType.VEHICLE_DROP
            )
            v_stops.append(s)
            
        # Precedence: Start -> Drop1 -> Drop2 ... 
        # (Assuming FIFO dropping for onboard passengers simplifies things, 
        # or we let them be free but constrained by time? 
        # Code example used sequential. We'll stick to sequential V->Drop->Drop for simplicity 
        # unless we specifically want to re-optimize onboard drops.)
        
        rules = []
        # Start (0) must precede everything
        for i in range(1, len(v_stops)):
            rules.append((0, i))
            
        # Optional: Enforce existing passenger order?
        # for i in range(1, len(v_stops) - 1):
        #    rules.append((i, i+1))
        
        return base_graph.extend(v_stops, rules, self._network)
