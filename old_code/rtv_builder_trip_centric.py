from collections import defaultdict
from typing import Optional, FrozenSet, List, Dict, Set, Tuple, Union, Literal
import numpy as np
from dataclasses import dataclass, field
from enum import IntEnum, auto

from config import get_logger
from config.config import (
    MAX_REQUESTS_ABOVE_CAPACITY,
    OMEGA,
    DELTA,
    VEHICLE_CAPACITY,
)
from src.interfaces import (
    ISingleVehicleRoutePlanner,
    VehicleAvailability,
    Request,
    FeasibleRoute,
    Route,
)
from src.interfaces.data_models import Stop as DMStop
from src.interfaces.graphs import RTVGraph, TripId, RVGraph
from src.interfaces.services import IRTVGraphBuilder, INetworkService

logger = get_logger(__name__)
INF = float("inf")


class StopType(IntEnum):
    PICKUP = auto()
    DROPOFF = auto()
    VEHICLE_START = auto()
    VEHICLE_DROP = auto()


@dataclass(frozen=True)
class GraphStop:
    """Internal Stop representation for the Graph Builder."""

    id: str
    node_id: int
    hard_earliest: float
    hard_latest: float
    type: StopType

    @property
    def is_pickup(self) -> bool:
        return self.type == StopType.PICKUP


# ID Type for Cache Key
StopID = Tuple[str, bool]


@dataclass
class Solution:
    """The cached solution for a suffix."""

    cost: float  # Travel + Wait
    sequence: List[str]  # IDs of the path
    latest_start: float  # Pruning Bound (Time)
    capacity_needed: int  # Pruning Bound (Cap)


@dataclass
class CacheEntry:
    # 1. The Accelerator (Best known valid path)
    best_sol: Optional[Solution] = None

    # 2. The Pruners (Aggregated failure knowledge)
    fail_time: float = float("inf")
    fail_cap: int = -1


@dataclass
class TripGraph:
    """
    Lightweight immutable container for the sequence constraints.
    """

    stops: List[GraphStop]
    adj: np.ndarray  # [N, N] Travel Times
    prereqs: np.ndarray  # [N] Bitmask
    suffix_cache: Dict[Tuple[StopID, FrozenSet[StopID]], CacheEntry] = field(
        default_factory=dict
    )

    # Pruning state
    best_cost: float = float("inf")
    best_path: List[int] = field(default_factory=list)
    vehicle_stops_mask: int = 0
    id_to_idx: Dict[str, int] = field(default_factory=dict)
    max_capacity: int = 4

    @classmethod
    def from_pair(
        cls, r1: Request, r2: Request, network: INetworkService
    ) -> "TripGraph":
        stops = cls._make_stops(r1) + cls._make_stops(r2)
        return cls._build(stops, network)

    @classmethod
    def from_single(cls, r1: Request, network: INetworkService) -> "TripGraph":
        stops = cls._make_stops(r1)
        return cls._build(stops, network)

    def extend_request(self, req: Request, network: INetworkService) -> "TripGraph":
        new_stops = self.stops + self._make_stops(req)
        # Pass forward the cache reference!
        return self._build(
            new_stops, network, self.adj, self.prereqs, self.suffix_cache
        )

    def extend_vehicle(
        self, vehicle: VehicleAvailability, network: INetworkService
    ) -> "TripGraph":
        # 1. Start Node
        v_start = GraphStop(
            id="V_Start",
            node_id=vehicle.available_at_node,
            hard_earliest=vehicle.available_from_time,
            hard_latest=INF,
            type=StopType.VEHICLE_START,
        )

        # 2. Existing Drops (Floating)
        v_drops = []
        for p in vehicle.passengers_onboard:
            s = GraphStop(
                id=p.original_request.id,
                node_id=p.original_request.destination_node,
                hard_earliest=vehicle.available_from_time,
                hard_latest=p.original_request.earliest_dropoff_time + DELTA,
                type=StopType.VEHICLE_DROP,
            )
            v_drops.append(s)

        vehicle_stops = [v_start] + v_drops
        all_stops = vehicle_stops + self.stops

        # Expand matrix
        g = self._build(all_stops, network)

        # Transfer Cache & Prereqs
        g.suffix_cache = self.suffix_cache
        g.prereqs[1:] |= 1  # V_Start prereq
        return g

    def precompute_suffixes(self):
        """No-op. Lazy caching is used."""
        pass

    def solve(
        self, vehicle_start_time: float, capacity: int, current_load: int
    ) -> Optional[FeasibleRoute]:
        """Finds optimal path using Selfish B&B DFS."""
        self.capacity = capacity
        # We don't use self.best_cost global tracking anymore,
        # _dfs_selfish returns the best result for the call.

        self.max_capacity = capacity  # Used in selfish solver
        self.vehicle_stops_mask = 0
        self.id_to_idx = {s.id: i for i, s in enumerate(self.stops)}

        for i, s in enumerate(self.stops):
            if s.type == StopType.VEHICLE_START or s.type == StopType.VEHICLE_DROP:
                self.vehicle_stops_mask |= 1 << i

        start_idx = 0
        if current_load > capacity:
            return None

        # Start Selfish DFS
        res = self._dfs_selfish(
            u_idx=start_idx,
            visited=(1 << start_idx),
            time=vehicle_start_time,
            load=current_load,
        )

        if res[0] != "Success":
            return None

        # Explicit unpacking for type safety
        _, cost, path_indices, _, _ = res

        ordered_stops = [self.stops[i] for i in path_indices]
        # Filter out V_Start from result route
        final_stops_local = [
            s for s in ordered_stops if s.type != StopType.VEHICLE_START
        ]

        # Convert to Interface Stop
        final_stops_dm = []
        for s in final_stops_local:
            # VEHICLE_DROP is also a drop-off (is_pickup=False)
            is_pick = s.type == StopType.PICKUP
            final_stops_dm.append(
                DMStop(node=s.node_id, is_pickup=is_pick, request_id=s.id)
            )

        return FeasibleRoute(
            route=Route(tuple(final_stops_dm), cost, 0.0, 0.0), cost=cost
        )

    def _dfs_selfish(self, u_idx: int, visited: int, time: float, load: int) -> Union[
        Tuple[Literal["Success"], float, List[int], float, int],
        Tuple[Literal["FailTime"], float],
        Tuple[Literal["FailCap"], int],
    ]:
        """
        Selfish & Collaborative Solver.
        """

        # 1. HARD PRUNING (Static Graph Constraints)
        if time > self.stops[u_idx].hard_latest + 1e-4:
            return ("FailTime", self.stops[u_idx].hard_latest)

        # 2. INDEPENDENCE CHECK (Are we in the Suffix?)
        is_independent = (visited & self.vehicle_stops_mask) == self.vehicle_stops_mask

        cache_key = None
        entry = None

        if is_independent and self.suffix_cache is not None:
            # Create Key: (Current_Node_ID, Remaining_Node_IDs)
            u_node = self.stops[u_idx]
            curr_id = (u_node.id, u_node.is_pickup)

            rem_ids = []
            for i in range(len(self.stops)):
                if not (visited & (1 << i)):
                    s = self.stops[i]
                    rem_ids.append((s.id, s.is_pickup))

            cache_key = (curr_id, frozenset(rem_ids))

            # Init Entry if missing
            if cache_key not in self.suffix_cache:
                self.suffix_cache[cache_key] = CacheEntry()
            entry = self.suffix_cache[cache_key]

            # --- A. PRUNING (The Walls) ---
            if time >= entry.fail_time:
                return ("FailTime", entry.fail_time)

            rest_cap = self.max_capacity - load
            if rest_cap <= entry.fail_cap:
                return ("FailCap", entry.fail_cap)

            # --- B. ACCELERATION (The Solution) ---
            if entry.best_sol:
                sol = entry.best_sol

                # Check 1: Time Validity
                if time <= sol.latest_start:
                    # Check 2: Capacity Validity
                    if load + sol.capacity_needed <= self.max_capacity:
                        # HIT! We use the cached path.
                        # Translate IDs back to indices
                        # Warning: Sequence in Solution is StopIDs (str), need indices
                        # We stored IDs in Solution to be generic.
                        # Wait, Solution stores IDs? Yes.
                        # We need to map back.
                        seq_indices = [self.id_to_idx[sid] for sid in sol.sequence]
                        return (
                            "Success",
                            sol.cost,
                            seq_indices,
                            sol.latest_start,
                            sol.capacity_needed,
                        )

                    # CAPACITY MISS: Fall through
                else:
                    # TIME MISS: Prune
                    return ("FailTime", sol.latest_start)

        # 3. BASE CASE
        if visited == (1 << len(self.stops)) - 1:
            return ("Success", 0.0, [], self.stops[u_idx].hard_latest, 0)

        # 4. RECURSIVE SEARCH
        best_res = None
        best_cost = float("inf")

        min_fail_time = float("inf")
        max_fail_cap: int = -1

        # Heuristic Sort
        candidates = []
        for v in range(len(self.stops)):
            if not (visited & (1 << v)):
                dist = self.adj[u_idx, v]
                if dist == INF:
                    continue
                if (visited & self.prereqs[v]) != self.prereqs[v]:
                    continue

                arrival = time + dist
                if arrival > self.stops[v].hard_latest:
                    continue

                v_stop = self.stops[v]
                delta = 0
                if v_stop.type == StopType.PICKUP:
                    delta = 1
                elif (
                    v_stop.type == StopType.DROPOFF
                    or v_stop.type == StopType.VEHICLE_DROP
                ):
                    delta = -1

                new_load = load + delta

                if new_load > self.max_capacity:
                    continue  # Vehicle Hard Cap

                candidates.append((v, arrival, dist, new_load, delta))

        candidates.sort(key=lambda x: self.stops[x[0]].hard_latest)

        if not candidates:
            # Dead end (topology block or time/cap block locally)
            min_fail_time = time

        for v, arr_time, dist, n_load, delta in candidates:

            res = self._dfs_selfish(v, visited | (1 << v), arr_time, n_load)

            # Explicit checking to help Pyright narrow the Union type
            if res[0] == "Success":
                _, s_cost, s_seq, s_L, s_cap = res
                total_cost = dist + s_cost

                if total_cost < best_cost:
                    best_cost = total_cost
                    # Back-propagate constraints
                    my_L = min(self.stops[u_idx].hard_latest, s_L - dist)
                    my_cap = max(0, delta, delta + s_cap)
                    best_res = ("Success", total_cost, [v] + s_seq, my_L, my_cap)

            elif res[0] == "FailTime":
                min_fail_time = min(min_fail_time, res[1])
            elif res[0] == "FailCap":
                max_fail_cap = max(max_fail_cap, res[1])

        # 5. AGGREGATE & UPDATE CACHE
        final_result = None

        if best_res:
            final_result = best_res
            # Update Best Solution if this is a new global optimum
            if is_independent and entry:
                # Store IDs in Solution
                sol_seq_ids = [self.stops[ix].id for ix in best_res[2]]

                new_sol = Solution(best_res[1], sol_seq_ids, best_res[3], best_res[4])

                if not entry.best_sol or best_res[1] < entry.best_sol.cost:
                    entry.best_sol = new_sol
                # Also, we might improve L or Cap even if cost is same?
                # For now strict cost dominance.

        elif min_fail_time < float("inf"):
            final_result = ("FailTime", min_fail_time)
            if is_independent and entry:
                entry.fail_time = min(entry.fail_time, min_fail_time)

        elif max_fail_cap > -1:
            final_result = ("FailCap", max_fail_cap)
            if is_independent and entry:
                entry.fail_cap = max(entry.fail_cap, max_fail_cap)

        return final_result or ("FailTime", time)

    # --- HELPERS ---
    @staticmethod
    def _make_stops(r: Request) -> List[GraphStop]:
        return [
            GraphStop(
                r.id,
                r.origin_node,
                r.request_time,
                r.request_time + OMEGA,
                StopType.PICKUP,
            ),
            GraphStop(
                r.id,
                r.destination_node,
                r.request_time,
                r.earliest_dropoff_time + DELTA,
                StopType.DROPOFF,
            ),
        ]

    @classmethod
    def _build(
        cls,
        stops: List[GraphStop],
        network: INetworkService,
        old_adj=None,
        old_prereqs=None,
        cache=None,
    ) -> "TripGraph":
        n = len(stops)
        locs = [s.node_id for s in stops]
        adj = network.get_travel_time_matrix(locs, locs)
        prereqs = np.zeros(n, dtype=int)

        # P->D Constraints
        id_map = {}
        for i, s in enumerate(stops):
            if s.type == StopType.PICKUP:
                id_map[s.id] = i
            elif s.type == StopType.DROPOFF and s.id in id_map:
                p_idx = id_map[s.id]
                prereqs[i] |= 1 << p_idx

        g = cls(stops, adj, prereqs)
        if cache is not None:
            g.suffix_cache = cache
        return g


@dataclass
class GlobalTrip:
    id: TripId
    request_ids: FrozenSet[str]
    topology: TripGraph
    candidate_vehicle_ids: Set[str]


class TripCentricRTVGraphBuilder(IRTVGraphBuilder):
    """
    RTV-Graph Builder using the Trip-Centric (Combinatorial) approach.
    """

    def __init__(self, route_planner: ISingleVehicleRoutePlanner) -> None:
        super().__init__(route_planner)
        self._network = route_planner._network

    def build_graph(
        self,
        rv_graph: RVGraph,
        vehicles: Set[VehicleAvailability],
        requests: Set[Request],
    ) -> RTVGraph:
        max_trip_size = VEHICLE_CAPACITY + MAX_REQUESTS_ABOVE_CAPACITY
        rtv = RTVGraph()
        req_map = {r.id: r for r in requests}
        veh_map = {v.vehicle_id: v for v in vehicles}
        global_trips: dict[int, dict[TripId, GlobalTrip]] = defaultdict(dict)

        # --- PHASE 1: SIZE 1 ---
        for r in requests:
            t_id = frozenset([r.id])
            connected_vids = rv_graph.get_rv_neighbors(r.id)
            if not connected_vids:
                continue

            # Build Base Graph
            base_graph = TripGraph.from_single(r, self._network)

            # Filter & Build RTV
            valid_vids = set()
            for vid in connected_vids:
                # Naive check: does size 1 fit?
                # Optimization: Trust RV Graph route? or Re-solve?
                # User says: "Trust RV Graph" for Phase 1.
                route = rv_graph.get_rv_route(r.id, vid)
                if route:
                    # Capacity check for safety
                    v = veh_map.get(vid)
                    if v and (len(v.passengers_onboard) + 1 <= v.capacity):
                        rtv.add_trip_for_vehicle(t_id, vid, route)
                        valid_vids.add(vid)

            if valid_vids:
                global_trips[1][t_id] = GlobalTrip(t_id, t_id, base_graph, valid_vids)

        # --- EXPANSION LOOP ---
        for k in range(2, max_trip_size + 1):
            prev_level = global_trips[k - 1]
            if not prev_level:
                break

            for base_trip in prev_level.values():
                anchor_id = next(iter(base_trip.request_ids))
                neighbors = rv_graph.get_rr_neighbors(anchor_id)

                for new_r_id in neighbors:
                    if new_r_id in base_trip.request_ids:
                        continue
                    new_ids = base_trip.request_ids | {new_r_id}

                    if new_ids in global_trips[k]:
                        continue

                    # Subtrip Check
                    is_valid_subtrip = True
                    for r_rem in new_ids:
                        if (new_ids - {r_rem}) not in prev_level:
                            is_valid_subtrip = False
                            break
                    if not is_valid_subtrip:
                        continue

                    # Vehicle Intersection
                    new_r_vehs = rv_graph.get_rv_neighbors(new_r_id)
                    candidates = base_trip.candidate_vehicle_ids & new_r_vehs
                    if not candidates:
                        continue

                    # Construct New Global Trip
                    new_r = req_map[new_r_id]
                    req_graph = base_trip.topology.extend_request(new_r, self._network)

                    # Precompute Suffixes
                    # Removed eager call: req_graph.precompute_suffixes()
                    # Lazy caching happens in veh_graph.solve() sharing suffix_cache.

                    surviving_vehicles = set()
                    for vid in candidates:
                        vehicle = veh_map.get(vid)
                        if not vehicle:
                            continue

                        # 0. Capacity Prune (Pre-Check)
                        if len(vehicle.passengers_onboard) + k > vehicle.capacity:
                            continue

                        # A. Extend with Vehicle
                        veh_graph = req_graph.extend_vehicle(vehicle, self._network)

                        # B. Solve
                        result = veh_graph.solve(
                            vehicle_start_time=vehicle.available_from_time,
                            capacity=vehicle.capacity,
                            current_load=len(vehicle.passengers_onboard),
                        )

                        if result:
                            rtv.add_trip_for_vehicle(new_ids, vid, result)
                            surviving_vehicles.add(vid)

                    if surviving_vehicles:
                        global_trips[k][new_ids] = GlobalTrip(
                            new_ids, new_ids, req_graph, surviving_vehicles
                        )

        return rtv
