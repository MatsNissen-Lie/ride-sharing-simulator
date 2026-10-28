"""Exact single-vehicle routing: the travel(v, T) function of Alonso-Mora et al.

Given a vehicle snapshot and a set of new requests, find the stop order that
drops off every onboard passenger and serves every new request within their
pickup and drop-off deadlines, minimising total delay. A* over (location,
visited-stop mask) with an admissible bound from direct travel times.

Cost (SI Appendix, Eq. 7 and footnote 1): the delay of a rider is
t_d - t*, where t* = request_time + tau(o, d) is the earliest possible
drop-off. A trip's cost is the total delay of onboard passengers and new
requests, minus the onboard passengers' delay if the vehicle took no new
requests. It is therefore the marginal delay the trip adds.
"""

from collections.abc import Iterable
import heapq
import numpy as np

from config import get_logger
from src.interfaces import (
    ITripSolver,
    VehicleAvailability,
    Request,
    FeasibleRoute,
    Route,
    Stop as DMStop,
)
from src.interfaces.data_models import FeasibilityResult, Infeasible

logger = get_logger(__name__)
INF = float("inf")

_BASELINE_CACHE_SIZE = 50_000


class AStarTripSolver(ITripSolver):
    """Exact travel(v, T) minimising delay, by A* search over stop orders."""

    # The RV builder's vectorised edges for empty and one-passenger vehicles
    # reproduce this solver exactly (tests/test_graph_builders.py).
    matches_rv_closed_forms = True

    def __init__(self, time_matrix: np.ndarray):
        self._matrix = time_matrix
        self._baseline_cache: dict[tuple, float] = {}

    def solve(
        self, requests: Iterable[Request], vehicle: VehicleAvailability
    ) -> FeasibilityResult:
        """Optimal route for `vehicle` serving `requests`; cost is the marginal delay."""
        result = self.solve_absolute(requests, vehicle)
        if not result.is_feasible:
            return result
        cost = result.cost - self.onboard_baseline_delay(vehicle)  # type: ignore[attr-defined]
        route = result.route  # type: ignore[attr-defined]
        return FeasibleRoute(
            route=Route(route.stops, route.total_travel_time, 0.0, cost), cost=cost
        )

    def onboard_baseline_delay(self, vehicle: VehicleAvailability) -> float:
        """Minimum total delay of the vehicle's current passengers with no new requests."""
        if not vehicle.passengers_onboard:
            return 0.0
        # Request equality is by id only, so key on the fields the search reads.
        key = (
            vehicle.available_at_node,
            vehicle.available_from_time,
            frozenset(
                (r.id, r.origin_node, r.destination_node, r.request_time, r.latest_dropoff_time)
                for r in (p.original_request for p in vehicle.passengers_onboard)
            ),
        )
        cached = self._baseline_cache.get(key)
        if cached is not None:
            return cached
        result = self.solve_absolute((), vehicle)
        if result.is_feasible:
            value = result.cost  # type: ignore[attr-defined]
        else:
            logger.warning(
                "Vehicle %s cannot drop off its passengers in time; baseline set to 0.",
                vehicle.vehicle_id,
            )
            value = 0.0
        if len(self._baseline_cache) >= _BASELINE_CACHE_SIZE:
            self._baseline_cache.clear()
        self._baseline_cache[key] = value
        return value

    def solve_absolute(
        self, requests: Iterable[Request], vehicle: VehicleAvailability
    ) -> FeasibilityResult:
        """Optimal route; cost is the total delay of onboard and new riders."""
        time_matrix = self._matrix

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
            req_times.append(float(r.request_time + time_matrix[r.origin_node, r.destination_node]))
            req_ids_ordered.append(r.id)

        n_onboard = len(node_indices)

        # 2. New Requests
        current_static_idx = 0
        prereqs = [0] * n_onboard

        for r in requests:
            t_star = float(r.request_time + time_matrix[r.origin_node, r.destination_node])
            node_indices.append(r.origin_node)
            is_pickup.append(True)
            hard_latest.append(r.latest_pickup_time)
            req_times.append(t_star)
            prereqs.append(0)
            req_ids_ordered.append(r.id)

            pk_local_idx = n_onboard + current_static_idx
            current_static_idx += 1

            node_indices.append(r.destination_node)
            is_pickup.append(False)
            hard_latest.append(r.latest_dropoff_time)
            req_times.append(t_star)
            prereqs.append(1 << pk_local_idx)
            req_ids_ordered.append(r.id)
            current_static_idx += 1

        total_stops = len(node_indices)
        full_mask = (1 << total_stops) - 1

        nodes_arr = np.array(node_indices, dtype=np.int32)
        # Travel times between the start (local index 0) and the stops (stop i at
        # i + 1) as Python floats: one lookup per element of the NumPy matrix
        # dominated the search.
        local_nodes = np.array([v_node, *node_indices])
        tt = time_matrix[local_nodes[:, None], local_nodes].tolist()
        load_deltas = [1 if p else -1 for p in is_pickup]
        cost_mask = [not p for p in is_pickup]

        # --- HEURISTIC PRE-CALCULATION ---
        onboard_data = []
        for i in range(n_onboard):
            onboard_data.append((i, i + 1, req_times[i], hard_latest[i]))

        waiting_data = []
        for i in range(n_onboard, total_stops, 2):
            static_dist = tt[i + 1][i + 2]
            waiting_data.append(
                (
                    i,
                    i + 1,
                    i + 1,  # local index of the origin
                    i + 2,  # and of the destination
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
        def calc_heuristic(u_loc, c_time, mask):
            # h(s) = |R_unserved| * t_now + sum(MinRemainingTime(r)) - sum(t_req^r)

            unserved_count = 0
            sum_min_remaining = 0.0
            sum_req_times = 0.0

            # 1. Onboard Passengers (only dropoff needed)
            for i, dest_idx, r_time, limit in onboard_data:
                if not ((mask >> i) & 1):
                    # Unserved
                    dist = tt[u_loc][dest_idx]
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
                    dist_u_pk = tt[u_loc][origin]
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
                    dist_u_dp = tt[u_loc][dest]

                    # Pruning checks
                    arr_dp = c_time + dist_u_dp
                    if arr_dp > dp_lim:
                        return INF

                    sum_min_remaining += dist_u_dp

            h = (unserved_count * c_time) + sum_min_remaining - sum_req_times
            return h

        h_start = calc_heuristic(0, v_start, 0)
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
                    route=Route(tuple(final_stops), c_time - v_start, 0.0, g), cost=g
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
            u_times = tt[path[-1] + 1 if path else 0]
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
                dist = u_times[v + 1]
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
                h = calc_heuristic(v + 1, arrival, new_mask)
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
