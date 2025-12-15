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

        # We also need a map to look up pickup time for onboard passengers if available
        # But 'vehicle.passengers_onboard' stores 'Passenger' objects which have 'pickup_time'.
        
        # Initial Cost Calculation (Sunk Cost)
        # To ensure the final returned cost equals "Total Delay" (DropTime - RequestTime),
        # we must include ALL time that has already passed for onboard passengers.
        # Sunk Cost = v_start - request_time.
        # This implicitly includes:
        #   1. Sunk Wait Time (Pickup - Request)
        #   2. Sunk Ride Time (v_start - Pickup)
        
        initial_g = 0.0
        
        # 1. Onboard Passengers
        for p in vehicle.passengers_onboard:
            r = p.original_request
            node_indices.append(r.destination_node)
            is_pickup.append(False)
            hard_latest.append(r.latest_dropoff_time)
            req_times.append(r.request_time)
            req_ids_ordered.append(r.id)
            
            # Sunk Delay
            delay = v_start - r.request_time
            if delay > 0:
                initial_g += delay

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

        # --- HYBRID HEURISTIC ---
        # g(s) includes:
        #   - Wait Time (Paid at Pickup)
        #   - Ride Time So Far (Paid continuously by onboard)
        #
        # h(s) must estimate:
        #   - Remaining Wait Time (For waiting requests)
        #   - Remaining Ride Time (For everyone)
        
        def calc_heuristic(u_idx, c_time, mask):
            h = 0.0
            
            # 1. Onboard Passengers
            # Wait time is paid. Ride time partially paid.
            # We need Future Ride Time.
            for i, dest_idx, r_time, limit in onboard_data:
                if not ((mask >> i) & 1):
                    # Future Ride
                    dist = time_matrix[u_idx, dest_idx]
                    arr = c_time + dist
                    if arr > limit:
                        return INF
                    h += dist

            # 2. Waiting Requests
            # Nothing paid into g yet.
            # We need Total Delay = (Wait) + (Ride).
            # Wait = (Arr_Pickup - Req_Time)
            # Ride = (Arr_Drop - Arr_Pickup) = Dist(O, D) roughly
            # Total = (Arr_Pickup - Req_Time) + Dist(O, D)
            # Arr_Pickup >= c_time + Dist(u, Origin)
            
            # So H = (c_time + Dist(u, Origin) - Req_Time) + Dist(Origin, Dest)
            #      = (c_time - Req_Time) + Dist(u, Origin) + Dist(Origin, Dest)
            
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
                    # Not yet picked up
                    dist_u_origin = time_matrix[u_idx, origin]
                    arr_pk = c_time + dist_u_origin
                    arr_dp = arr_pk + static
                    if arr_pk > pk_lim or arr_dp > dp_lim:
                        return INF
                    
                    # Cost = Total Delay Estimate
                    # We use arr_pk for wait calculation because it's the earliest we can get there
                    if arr_dp > r_time:
                         h += (arr_dp - r_time)
                         
                elif not ((mask >> dp_i) & 1):
                    # Picked up, not dropped (currently onboard in this hypothetical state context)
                    # For a waiting request that is NOW onboard (bit flipped in mask):
                    # The Wait Time Cost was effectively added to 'g' when we traversed the stored pickup edge.
                    # RIDE TIME cost is accrued in 'g' for every step since then.
                    # We only need FUTURE RIDE TIME.
                    
                    dist = time_matrix[u_idx, dest]
                    arr = c_time + dist
                    if arr > dp_lim:
                        return INF
                    h += dist

            return h

        h_start = calc_heuristic(v_node, v_start, 0)
        if h_start == INF:
            return Infeasible(reason="Initial state infeasible")

        # PQ: (f, g, c_time, u_idx, load, mask, path)
        pq = [(h_start + initial_g, initial_g, v_start, v_node, initial_load, 0, [])]

        # Closed Set: (u_idx, mask) -> (min_g, min_time)
        closed = {}

        while pq:
            f, g, c_time, u_idx, load, mask, path = heapq.heappop(pq)

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
                return FeasibleRoute(
                    route=Route(tuple(final_stops), g, 0.0, 0.0), cost=g
                )

            state_key = (u_idx, mask)
            if state_key in closed:
                prev_g, prev_time = closed[state_key]
                if prev_g <= g and prev_time <= c_time:
                    continue
                if g < prev_g and c_time < prev_time:
                    closed[state_key] = (g, c_time)
            else:
                closed[state_key] = (g, c_time)

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

                # --- HYBRID COST UPDATE ---
                # 1. Ride Time for Onboard Passengers (Continuous)
                #    All 'load' passengers suffer 'dist' delay.
                edge_cost = load * dist

                # 2. Wait Time for New Passenger (Lump Sum at Pickup)
                event_cost = 0.0
                if is_pickup[v]:
                    # Wait Time = Arrival - Request Time
                    delay = arrival - req_times[v]
                    if delay > 0:
                        event_cost = delay
                
                # 3. Dropoff: No Cost (End of Ride Time accumulation)
                
                added_cost = edge_cost + event_cost
                new_g = g + added_cost
                new_mask = mask | (1 << v)

                # Heuristic
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
