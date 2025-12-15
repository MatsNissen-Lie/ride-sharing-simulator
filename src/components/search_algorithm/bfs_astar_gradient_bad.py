from collections.abc import Iterable
import heapq
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
        def compute_mst(nodes, u_idx):
            """
            Compute MST weight for a set of nodes including current node u_idx.
            Using Prim's algorithm.
            """
            if not nodes:
                return 0.0

            # Nodes to visit (excluding u_idx which is in tree)
            pending = set(nodes)
            if u_idx in pending:
                pending.remove(u_idx)

            if not pending:
                return 0.0

            mst_cost = 0.0

            # Distance from tree to node
            # Initialize with dist from u_idx
            min_dists = {v: time_matrix[u_idx, v] for v in pending}

            while pending:
                # Find closest node to current tree
                next_node = min(pending, key=lambda n: min_dists[n])

                # Add check for INF connectivity if needed, but time_matrix usually solid
                cost = min_dists[next_node]
                if cost == INF:
                    return INF

                mst_cost += cost
                pending.remove(next_node)

                # Update distances
                for v in pending:
                    d = time_matrix[next_node, v]
                    if d < min_dists[v]:
                        min_dists[v] = d

            return mst_cost

        def calc_heuristic(u_idx, c_time, mask):
            h_sunk = 0.0
            active_nodes = set()
            active_drop_nodes = set()

            # 1. Onboard Passengers
            for i, dest_idx, r_time, limit in onboard_data:
                # Check if dropped off (bit i is SET if visited)
                # Wait, existing Logic: 'if not ((mask >> i) & 1)' means NOT visited.
                if not ((mask >> i) & 1):
                    # Not yet dropped off
                    # Active Request
                    h_sunk += c_time - r_time

                    # Target node is destination
                    active_nodes.add(dest_idx)
                    active_drop_nodes.add(dest_idx)

                    # Hard constraint check (min time to dest)
                    direct_arr = c_time + time_matrix[u_idx, dest_idx]
                    if direct_arr > limit:
                        return INF

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
                pk_done = (mask >> pk_i) & 1
                dp_done = (mask >> dp_i) & 1

                if not pk_done:
                    # Still waiting at origin
                    # Active Request
                    h_sunk += c_time - r_time

                    active_nodes.add(origin)
                    active_nodes.add(dest)
                    active_drop_nodes.add(dest)

                    # Hard constraint (min time to pickup)
                    direct_pk = c_time + time_matrix[u_idx, origin]
                    if direct_pk > pk_lim:
                        return INF
                    # Min time to drop (pk -> dp)
                    if direct_pk + static > dp_lim:
                        return INF

                elif not dp_done:
                    # Picked up, not dropped (Onboard)
                    # Active Request
                    h_sunk += c_time - r_time

                    active_nodes.add(dest)
                    active_drop_nodes.add(dest)

                    # Hard constraint
                    direct_dp = c_time + time_matrix[u_idx, dest]
                    if direct_dp > dp_lim:
                        return INF

            # 3. Compute MST of all active points
            # Active set = {u_idx} U {active_nodes}
            # Note: u_idx is start of MST
            h_structure = compute_mst(active_nodes, u_idx)
            if h_structure == INF:
                return INF

            # 4. Bottleneck: Dist to nearest Dropoff
            # We only care about drop locations of ACTIVE requests.
            # active_drop_nodes populated above.
            h_bottleneck = 0.0
            if active_drop_nodes:
                h_bottleneck = min(time_matrix[u_idx, d] for d in active_drop_nodes)

            # Final Heuristic
            return h_sunk + h_structure + h_bottleneck

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

            # --- RESTORED & REFINED PRUNING ---
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
