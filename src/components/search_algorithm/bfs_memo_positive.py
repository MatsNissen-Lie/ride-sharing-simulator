from collections.abc import Iterable
import heapq
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from config import get_logger
from config.config import (
    OMEGA,
    DELTA,
)
from src.interfaces import (
    VehicleAvailability,
    Request,
    FeasibleRoute,
    Route,
    Stop as DMStop,
)
from src.interfaces.data_models import FeasibilityResult, Infeasible

logger = get_logger(__name__)
INF = float("inf")


class TripSolverCached:
    """
    A* BFS Solver using Priority Queue with Global Memoization.
    Memoizes optimal path suffixes for (node, remaining_requests) states.
    NOTE: optimizing for 'batch-only' segments where vehicle-specific onboard passengers
    have already been served.
    """

    # Global Memoization Cache
    # Key: (current_node_idx, frozenset(remaining_stop_ids))
    #   - current_node_idx: GLOBAL Map Node ID (Location).
    #   - remaining_stop_ids: Set of Unique Logical IDs (ReqID, 'p'/'d').
    #     This ensures that even if two stops are at the same location, they are distinct if the logical action differs.
    # Value: (path_suffix_ids, cost_constant, max_start_time, peak_load)
    # path_suffix_ids is List[Tuple[str, str]] e.g. [('req1', 'p'), ('req1', 'd')]
    _memo: Dict[
        Tuple[int, frozenset], Tuple[List[Tuple[str, str]], float, float, int]
    ] = {}

    @classmethod
    def clear_cache(cls):
        cls._memo.clear()

    @classmethod
    def get_cache_size(cls):
        return len(cls._memo)

    @staticmethod
    def solve(
        requests: Iterable[Request],
        vehicle: VehicleAvailability,
        time_matrix: np.ndarray,
    ) -> FeasibilityResult:

        # --- 1. SETUP LOCAL CONTEXT ---
        v_node = vehicle.available_at_node
        v_start = vehicle.available_from_time
        v_cap = vehicle.capacity
        initial_load = len(vehicle.passengers_onboard)

        node_indices = []
        is_pickup = []
        hard_latest = []
        req_times = []
        req_ids_ordered = []

        # We need store stop_types to build cache keys: (id, 'p') or (id, 'd')
        stop_types = []

        # 1. Onboard Passengers
        for p in vehicle.passengers_onboard:
            r = p.original_request
            node_indices.append(r.destination_node)
            is_pickup.append(False)
            hard_latest.append(r.latest_dropoff_time)
            req_times.append(r.request_time)
            req_ids_ordered.append(r.id)
            stop_types.append((r.id, "d"))

        n_onboard = len(node_indices)

        # 2. New Requests
        current_static_idx = 0
        prereqs = [0] * n_onboard

        for r in requests:
            node_indices.append(r.origin_node)
            is_pickup.append(True)
            hard_latest.append(r.latest_pickup_time)
            req_times.append(r.request_time)
            prereqs.append(0)
            req_ids_ordered.append(r.id)
            stop_types.append((r.id, "p"))

            pk_local_idx = n_onboard + current_static_idx
            current_static_idx += 1

            node_indices.append(r.destination_node)
            is_pickup.append(False)
            hard_latest.append(r.latest_dropoff_time)
            req_times.append(r.request_time)
            prereqs.append(1 << pk_local_idx)
            req_ids_ordered.append(r.id)
            stop_types.append((r.id, "d"))
            current_static_idx += 1

        total_stops = len(node_indices)
        full_mask = (1 << total_stops) - 1

        onboard_mask_all = (1 << n_onboard) - 1

        # Prepare lookup map for current context
        # stop_types[i] = (req_id, type_str) e.g. ('req_123', 'p')
        id_type_to_index = {st: i for i, st in enumerate(stop_types)}

        nodes_arr = np.array(node_indices, dtype=np.int32)
        load_deltas = [1 if p else -1 for p in is_pickup]
        cost_mask = [not p for p in is_pickup]

        # --- HEURISTIC PRE-CALCULATION ---
        onboard_data = []
        for i in range(n_onboard):
            onboard_data.append((i, node_indices[i], req_times[i], hard_latest[i]))

        waiting_data = []
        for i in range(n_onboard, total_stops, 2):
            origin = node_indices[i]
            dest = node_indices[i + 1]
            static_dist = time_matrix[origin, dest]
            waiting_data.append(
                (
                    i,
                    i + 1,
                    origin,
                    dest,
                    req_times[i],
                    static_dist,
                    hard_latest[i],
                    hard_latest[i + 1],
                )
            )

        # --- A* BFS STATE ---
        def calc_heuristic(u_idx, c_time, mask):
            h = 0.0
            # 1. Onboard
            for i, dest_idx, r_time, limit in onboard_data:
                if not ((mask >> i) & 1):
                    arr = c_time + time_matrix[u_idx, dest_idx]
                    if arr > limit:
                        return INF
                    if arr > r_time:
                        h += arr - r_time

            # 2. Waiting
            for (
                pk_i,
                dp_i,
                origin,
                dest,
                r_time,
                static,
                pk_lim,
                dp_lim,
            ) in waiting_data:
                if not ((mask >> pk_i) & 1):
                    dist_u_origin = time_matrix[u_idx, origin]
                    arr_pk = c_time + dist_u_origin
                    arr_dp = arr_pk + static
                    if arr_pk > pk_lim or arr_dp > dp_lim:
                        return INF
                    if arr_dp > r_time:
                        h += arr_dp - r_time
                elif not ((mask >> dp_i) & 1):
                    arr = c_time + time_matrix[u_idx, dest]
                    if arr > dp_lim:
                        return INF
                    if arr > r_time:
                        h += arr - r_time
            return h

        h_start = calc_heuristic(v_node, v_start, 0)
        if h_start == INF:
            return Infeasible(reason="Initial state infeasible")

        # PQ State: (f, g, c_time, u_idx, load, mask, path)
        pq = [(h_start, 0.0, v_start, v_node, initial_load, 0, [])]
        closed = {}

        while pq:
            f, g, c_time, u_idx, load, mask, path = heapq.heappop(pq)

            # --- MEMOIZATION CHECK ---
            if (mask & onboard_mask_all) == onboard_mask_all:
                remaining_indices = []
                for i in range(total_stops):
                    if not ((mask >> i) & 1):
                        remaining_indices.append(i)

                if not remaining_indices:
                    pass  # Goal Reached, handled below
                else:
                    rem_set = frozenset([stop_types[i] for i in remaining_indices])
                    cache_key = (u_idx, rem_set)

                    if cache_key in TripSolverCached._memo:
                        suffix_ids, cost_const, t_max, peak_load = (
                            TripSolverCached._memo[cache_key]
                        )

                        if c_time <= t_max and peak_load <= v_cap:
                            try:
                                suffix_indices = [
                                    id_type_to_index[sid] for sid in suffix_ids
                                ]
                            except KeyError:
                                # Fallback if IDs don't match (should not happen if logic is correct)
                                suffix_indices = None

                            if suffix_indices is not None:
                                full_path_indices = path + suffix_indices

                                # Suffix Cost Calculation
                                n_suffix_drops = sum(
                                    1 for idx in suffix_indices if not is_pickup[idx]
                                )
                                added_suffix_cost = n_suffix_drops * c_time + cost_const

                                final_g = g + added_suffix_cost

                                final_stops = []
                                for idx in full_path_indices:
                                    final_stops.append(
                                        DMStop(
                                            node=node_indices[idx],
                                            is_pickup=is_pickup[idx],
                                            request_id=req_ids_ordered[idx],
                                        )
                                    )

                                return FeasibleRoute(
                                    route=Route(tuple(final_stops), final_g, 0.0, 0.0),
                                    cost=final_g,
                                )

            # Goal Check
            if mask == full_mask:
                final_stops = []
                for idx in path:
                    final_stops.append(
                        DMStop(
                            node=node_indices[idx],
                            is_pickup=is_pickup[idx],
                            request_id=req_ids_ordered[idx],
                        )
                    )

                # --- CACHE POPULATION ---
                path_len = len(path)

                # Re-calculate load state flow
                # load at i=0 corresponds to AFTER path[0] if we update inside loop?
                # Let's track load BEFORE processing path[i].

                curr_load = initial_load
                temp_mask = 0
                curr_u = v_node

                for i in range(path_len):
                    # State (curr_u, temp_mask, curr_load)

                    if (temp_mask & onboard_mask_all) == onboard_mask_all:
                        # Eligible for caching
                        local_suffix = path[i:]
                        suffix_ids = [stop_types[idx] for idx in local_suffix]

                        rem_indices = [
                            idx
                            for idx in range(total_stops)
                            if not ((temp_mask >> idx) & 1)
                        ]
                        rem_set = frozenset([stop_types[idx] for idx in rem_indices])

                        key = (int(curr_u), rem_set)

                        c_dist = 0.0
                        c_max = INF
                        c_cost_const = 0.0

                        # Track peak load encountered during suffix
                        sim_load = curr_load
                        peak_load_suffix = sim_load

                        prev_sim = curr_u

                        for s_idx in local_suffix:
                            d = time_matrix[prev_sim, nodes_arr[s_idx]]
                            c_dist += d
                            limit = hard_latest[s_idx]

                            possible_start = limit - c_dist
                            if possible_start < c_max:
                                c_max = possible_start

                            if not is_pickup[s_idx]:
                                c_cost_const += c_dist - req_times[s_idx]

                            # Load update
                            sim_load += load_deltas[s_idx]
                            if sim_load > peak_load_suffix:
                                peak_load_suffix = sim_load

                            prev_sim = nodes_arr[s_idx]

                        TripSolverCached._memo[key] = (
                            suffix_ids,
                            c_cost_const,
                            c_max,
                            peak_load_suffix,
                        )

                    # Advance
                    stop_idx = path[i]
                    temp_mask |= 1 << stop_idx
                    curr_u = nodes_arr[stop_idx]
                    curr_load += load_deltas[stop_idx]

                return FeasibleRoute(
                    route=Route(tuple(final_stops), g, 0.0, 0.0), cost=g
                )

            # Pruning
            state_key = (u_idx, mask)
            if state_key in closed and closed[state_key] <= c_time:
                continue
            closed[state_key] = c_time

            # Expand
            for v in range(total_stops):
                if (mask >> v) & 1:
                    continue
                if (prereqs[v] & mask) != prereqs[v]:
                    continue

                new_load = load + load_deltas[v]
                if new_load > v_cap:
                    continue

                v_global_idx = nodes_arr[v]
                dist = time_matrix[u_idx, v_global_idx]
                arrival = c_time + dist

                if arrival > hard_latest[v]:
                    continue

                added_cost = 0.0
                if cost_mask[v]:
                    added_cost = arrival - req_times[v]

                new_g = g + added_cost
                new_mask = mask | (1 << v)

                h = calc_heuristic(v_global_idx, arrival, new_mask)
                if h == INF:
                    continue

                new_f = new_g + h

                heapq.heappush(
                    pq,
                    (
                        new_f,
                        new_g,
                        arrival,
                        v_global_idx,
                        new_load,
                        new_mask,
                        path + [v],
                    ),
                )

        return Infeasible(reason="No feasible route found")
