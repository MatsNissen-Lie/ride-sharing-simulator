from collections.abc import Iterable
import numpy as np

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


class TripSolver:
    """
    Zero-Copy Solver.
    Uses the global network matrix directly via indirect addressing.
    """

    @staticmethod
    def solve(
        requests: Iterable[Request],
        vehicle: VehicleAvailability,
        time_matrix: np.ndarray,
    ) -> FeasibilityResult:

        # --- 1. SETUP LOCAL CONTEXT ---
        # We map local DFS steps (0, 1, 2...) to Global Matrix Indices

        # A. Vehicle State
        v_node = vehicle.available_at_node
        v_start = vehicle.available_from_time
        v_cap = vehicle.capacity
        initial_load = len(vehicle.passengers_onboard)

        # B. Nodes & Constraints Arrays
        # Structure: [Onboard_Drops] + [New_Pickups, New_Drops...]

        node_indices = []  # The Map: local_idx -> global_matrix_idx
        is_pickup = []  # Bool
        hard_latest = []  # Float
        req_times = []  # Float (for cost calc)
        req_ids_ordered = []  # Str (for result reconstruction)

        # 1. Onboard Passengers (Drop-only)
        for p in vehicle.passengers_onboard:
            r = p.original_request
            node_indices.append(r.destination_node)
            is_pickup.append(False)

            limit = r.earliest_dropoff_time
            hard_latest.append((limit + DELTA) if limit else INF)
            req_times.append(r.request_time)
            req_ids_ordered.append(r.id)

        n_onboard = len(node_indices)

        # 2. New Requests (Pickup + Drop)
        # We assign indices sequentially: 0, 1 for R1; 2, 3 for R2... relative to start of static part
        current_static_idx = 0
        prereqs = [0] * n_onboard  # Onboard has no prereqs

        for r in requests:
            # Pickup
            node_indices.append(r.origin_node)
            is_pickup.append(True)
            hard_latest.append(r.request_time + OMEGA)
            req_times.append(r.request_time)
            prereqs.append(0)
            req_ids_ordered.append(r.id)

            pk_local_idx = n_onboard + current_static_idx
            current_static_idx += 1

            # Dropoff
            node_indices.append(r.destination_node)
            is_pickup.append(False)

            limit = r.earliest_dropoff_time
            hard_latest.append((limit + DELTA) if limit else INF)
            req_times.append(r.request_time)

            # Prereq: The Pickup we just added
            # Bitshift must reference the index in the FULL list
            prereqs.append(1 << pk_local_idx)
            req_ids_ordered.append(r.id)

            current_static_idx += 1

        total_stops = len(node_indices)
        full_mask = (1 << total_stops) - 1

        # Pre-convert lists to fast arrays/vars for the loop
        nodes_arr = np.array(node_indices, dtype=np.int32)

        # Load Deltas: -1 for drop, +1 for pickup
        load_deltas = [1 if p else -1 for p in is_pickup]

        # Cost masking (only Dropoffs incur delay cost)
        # Onboard are all drops (True), New reqs mix.
        cost_mask = [not p for p in is_pickup]

        # --- HEURISTIC PRE-CALCULATION ---
        # Optimize for Python loop speed (avoid dict lookups, use tuples/locals)

        # 1. Onboard Data: (bit_idx, dest_node_idx, req_time, hard_limit)
        onboard_data = []
        for i in range(n_onboard):
            onboard_data.append((i, node_indices[i], req_times[i], hard_latest[i]))

        # 2. Waiting Data: (pk_bit_idx, dp_bit_idx, origin_node, dest_node, req_time, static_dist, pk_limit, dp_limit)
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

        # --- DFS STATE ---
        best_cost = INF
        best_path = []

        # Inner Recursive Function
        def _dfs(
            u_global_idx, current_time, current_load, visited_mask, current_cost, path
        ):
            nonlocal best_cost, best_path

            if current_cost >= best_cost:
                return

            # --- HEURISTIC PRUNING (Optimized Python Loops) ---
            h_cost = 0.0

            # 1. Onboard Passengers
            for i, dest_idx, r_time, limit in onboard_data:
                # Check if unvisited (bit is 0)
                if not ((visited_mask >> i) & 1):
                    arr = current_time + time_matrix[u_global_idx, dest_idx]
                    if arr > limit:
                        return  # Dead end
                    if arr > r_time:
                        h_cost += arr - r_time

            # 2. Waiting Requests
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
                # Check Pickup
                if not ((visited_mask >> pk_i) & 1):
                    # Pickup Unvisited -> Go to Origin -> Dest
                    dist_u_origin = time_matrix[u_global_idx, origin]
                    arr_pk = current_time + dist_u_origin
                    arr_dp = arr_pk + static

                    if arr_pk > pk_lim or arr_dp > dp_lim:
                        return  # Dead end

                    if arr_dp > r_time:
                        h_cost += arr_dp - r_time

                # Check Dropoff (only if Pickup visited)
                elif not ((visited_mask >> dp_i) & 1):
                    # Pickup Visited, Dropoff Unvisited -> Go to Dest
                    arr = current_time + time_matrix[u_global_idx, dest]

                    if arr > dp_lim:
                        return  # Dead end

                    if arr > r_time:
                        h_cost += arr - r_time

            if current_cost + h_cost >= best_cost:
                return
            # -------------------------

            if visited_mask == full_mask:
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_path = list(path)
                return

            # OPTIMIZATION:
            # We iterate 0..total_stops.

            # Fetch row for u from global matrix ONCE (if numpy)
            # This represents dist(u, all_nodes)

            for v in range(total_stops):
                if (visited_mask >> v) & 1:
                    continue

                if (prereqs[v] & visited_mask) != prereqs[v]:
                    continue

                new_load = current_load + load_deltas[v]
                if new_load > v_cap:
                    continue

                # --- DIRECT GLOBAL LOOKUP ---
                v_global_idx = nodes_arr[v]
                dist = time_matrix[u_global_idx, v_global_idx]

                arrival = current_time + dist

                # Check constraints...
                if arrival > hard_latest[v]:
                    continue

                added_cost = 0.0
                if cost_mask[v]:
                    added_cost = arrival - req_times[v]

                path.append(v)
                _dfs(
                    v_global_idx,
                    arrival,
                    new_load,
                    visited_mask | (1 << v),
                    current_cost + added_cost,
                    path,
                )
                path.pop()

        # --- EXECUTE ---
        # Start from Vehicle Node
        _dfs(v_node, v_start, initial_load, 0, 0.0, [])

        if best_cost == INF:
            return Infeasible(reason="No feasible route found")

        # --- RECONSTRUCT ---
        final_stops = []
        for idx in best_path:
            final_stops.append(
                DMStop(
                    node=node_indices[idx],
                    is_pickup=is_pickup[idx],
                    request_id=req_ids_ordered[idx],
                )
            )

        return FeasibleRoute(
            route=Route(tuple(final_stops), best_cost, 0.0, 0.0), cost=best_cost
        )
