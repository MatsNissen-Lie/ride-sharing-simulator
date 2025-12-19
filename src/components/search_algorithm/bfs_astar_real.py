from collections.abc import Iterable
import heapq
import numpy as np

from config import get_logger
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


class TripSolver:
    """
    A* BFS Solver using Priority Queue.
    """

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

        # 1. Onboard Passengers
        for p in vehicle.passengers_onboard:
            r = p.original_request
            node_indices.append(r.destination_node)
            is_pickup.append(False)
            hard_latest.append(r.latest_dropoff_time)
            req_times.append(r.request_time)
            req_ids_ordered.append(r.id)

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

            pk_local_idx = n_onboard + current_static_idx
            current_static_idx += 1

            node_indices.append(r.destination_node)
            is_pickup.append(False)
            hard_latest.append(r.latest_dropoff_time)
            req_times.append(r.request_time)
            prereqs.append(1 << pk_local_idx)
            req_ids_ordered.append(r.id)
            current_static_idx += 1

        total_stops = len(node_indices)
        full_mask = (1 << total_stops) - 1

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
        # Priority Queue Item: (f_score, g_score, current_time, u_global_idx, current_load, visited_mask, path)
        # We use tuples for path to make it hashable if needed, but list is fine inside tuple if we don't hash the item.
        # Actually, heapq compares items. We need f_score first.

        # Initial State
        # f = g + h
        # g = 0
        # h = calc_heuristic(...)

        # (We could calculate initial heuristic, but let's just push 0 and let the loop handle it)
        # Actually, for A* to work best, we should calc h for start node.

        # Let's define the heuristic function to reuse
        def calc_heuristic(u_idx, c_time, mask):
            # h(s) = |R_unserved| * t_now + sum(MinRemainingTime(r)) - sum(t_req^r)

            unserved_count = 0
            sum_min_remaining = 0.0
            sum_req_times = 0.0

            # 1. Onboard Passengers (only dropoff needed)
            for i, dest_idx, r_time, limit in onboard_data:
                if not ((mask >> i) & 1):
                    # Unserved
                    dist = time_matrix[u_idx, dest_idx]
                    arrival_est = c_time + dist

                    # Pruning check
                    if arrival_est > limit:
                        return INF

                    unserved_count += 1
                    sum_min_remaining += dist
                    sum_req_times += r_time

            # 2. Waiting Requests (pickup and dropoff needed)
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
                # Check if served (dropoff visited)
                if (mask >> dp_i) & 1:
                    continue

                # Not served (fully)
                unserved_count += 1
                sum_req_times += r_time

                if not ((mask >> pk_i) & 1):
                    # Not picked up yet: Current -> Pickup -> Drop
                    dist_u_pk = time_matrix[u_idx, origin]
                    # MinRemaining = dist(u, pk) + dist(pk, dp)
                    # Note: static = dist(pk, dp)

                    # Pruning checks
                    arr_pk = c_time + dist_u_pk
                    arr_dp = arr_pk + static
                    if arr_pk > pk_lim or arr_dp > dp_lim:
                        return INF

                    sum_min_remaining += dist_u_pk + static

                else:
                    # Picked up, not dropped: Current -> Drop
                    dist_u_dp = time_matrix[u_idx, dest]

                    # Pruning checks
                    arr_dp = c_time + dist_u_dp
                    if arr_dp > dp_lim:
                        return INF

                    sum_min_remaining += dist_u_dp

            h = (unserved_count * c_time) + sum_min_remaining - sum_req_times
            return h

        h_start = calc_heuristic(v_node, v_start, 0)
        if h_start == INF:
            return Infeasible(reason="Initial state infeasible")

        # PQ: (f, g, time, u_idx, load, mask, path)
        pq = [(h_start, 0.0, v_start, v_node, initial_load, 0, [])]

        # Closed Set: (u_idx, mask) -> min_time
        # We want to keep the state if we arrive with LOWER time.
        closed = {}

        while pq:
            f, g, c_time, u_idx, load, mask, path = heapq.heappop(pq)

            # Goal Check
            if mask == full_mask:
                # Reconstruct
                final_stops = []
                for idx in path:
                    final_stops.append(
                        DMStop(
                            node=node_indices[idx],
                            is_pickup=is_pickup[idx],
                            request_id=req_ids_ordered[idx],
                        )
                    )
                return FeasibleRoute(
                    route=Route(tuple(final_stops), g, 0.0, 0.0), cost=g
                )

            state_key = (u_idx, mask)

            # if state_key in closed and closed[state_key] <= c_time:
            #     continue
            # closed[state_key] = c_time

            # # --- RESTORED & REFINED PRUNING ---
            state_key = (u_idx, mask)

            if state_key in closed:
                # Retrieve the best previous state for this node/mask combo
                prev_g, prev_time = closed[state_key]

                # If the previous path to get here was "better" in every way:
                # 1. It cost less (or equal)
                # 2. AND it arrived earlier (or same time)
                # Then the current path is strictly worse. Discard it.
                if prev_g <= g and prev_time <= c_time:
                    continue

                # If the current path is NOT strictly dominated (e.g. it's slower but cheaper),
                # we keep it alive. We also update the closed set to reflect this new frontier.
                # (Ideally, we store a list of Pareto-optimal states, but simply updating
                # if we are strictly better is usually sufficient for VRP speeds).
                if g < prev_g and c_time < prev_time:
                    closed[state_key] = (g, c_time)
            else:
                closed[state_key] = (g, c_time)

            # Expand
            # Iterate over all possible next stops v
            for v in range(total_stops):
                # 1. Check if already visited
                if (mask >> v) & 1:
                    continue

                # 2. Check Prereqs (Pickup before Drop)
                if (prereqs[v] & mask) != prereqs[v]:
                    continue

                # 3. Check Capacity
                new_load = load + load_deltas[v]
                if new_load > v_cap:
                    continue

                # 4. Calculate Time & Cost
                v_global_idx = nodes_arr[v]
                dist = time_matrix[u_idx, v_global_idx]
                arrival = c_time + dist

                # 5. Check Hard Time Window
                if arrival > hard_latest[v]:
                    continue

                added_cost = 0.0
                if cost_mask[v]:
                    added_cost = arrival - req_times[v]

                new_g = g + added_cost
                new_mask = mask | (1 << v)

                # 6. Calculate Heuristic
                h = calc_heuristic(v_global_idx, arrival, new_mask)
                if h == INF:
                    continue

                new_f = new_g + h

                # Push to PQ
                # Note: path + [v] creates a new list
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
