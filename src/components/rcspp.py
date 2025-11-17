import itertools
from typing import Set, List, Tuple, Dict, Optional

# --- Configuration Imports ---
# Assumed available from your config.py
from config import OMEGA, DELTA

# --- Interface Imports ---
# Assumed available from your project structure
from src.interfaces import (
    FeasibilityResult,
    FeasibleRoute,
    Infeasible,
    Request,
    Route,
    Stop,
    VehicleAvailability,
    INetworkService,
)


class RCSPPSolver:
    """
    Exact solver for the Vehicle Routing Problem with Time Windows (VRPTW)
    for small instance sizes (N <= 12 stops).

    Uses dynamic programming with bitmask state and dominance pruning
    (conceptually similar to Label Setting algorithms for RCSPP).
    """

    def __init__(self, network_service: INetworkService):
        self._network = network_service

    def find_optimal_route(
        self,
        availability: VehicleAvailability,
        requests_to_add: Set[Request],
    ) -> FeasibilityResult:
        """
        Finds the cost-optimal sequence for a vehicle and a set of requests
        by exploring ALL valid reorderings.
        """

        # --- 1. Setup Data Structures ---

        onboard_psg = list(availability.passengers_onboard)
        new_reqs_list = list(requests_to_add)

        all_stops: List[Stop] = []

        # Map request_id -> (pickup_index, dropoff_index)
        # For existing passengers, pickup_index is -1 (already done)
        req_indices: Dict[str, Tuple[int, int]] = {}

        # A) Add Drop-offs for existing passengers
        for p in onboard_psg:
            stop = Stop(
                node=p.original_request.destination_node,
                is_pickup=False,
                request_id=p.original_request.id,
            )
            all_stops.append(stop)
            req_indices[p.original_request.id] = (-1, len(all_stops) - 1)

        # B) Add Pickup/Dropoff pairs for new requests
        for r in new_reqs_list:
            # Pickup
            p_stop = Stop(node=r.origin_node, is_pickup=True, request_id=r.id)
            all_stops.append(p_stop)
            p_idx = len(all_stops) - 1

            # Dropoff
            d_stop = Stop(node=r.destination_node, is_pickup=False, request_id=r.id)
            all_stops.append(d_stop)
            d_idx = len(all_stops) - 1

            req_indices[r.id] = (p_idx, d_idx)

        # Constants for the bitmask DP
        N = len(all_stops)
        FULL_MASK = (1 << N) - 1

        # Safety check for recursion depth/complexity
        if N > 14:
            return Infeasible(reason=f"RCSPP input too large: {N} stops")

        # --- 2. Build Travel Time Cache ---
        # We need times between: StartNode <-> Stops, and Stops <-> Stops
        nodes_to_cache = {s.node for s in all_stops}
        nodes_to_cache.add(availability.available_at_node)
        tt_cache = self._build_travel_time_cache(nodes_to_cache)

        # --- 3. Build Request Map for Constraints ---
        all_req_map: Dict[str, Request] = {
            p.original_request.id: p.original_request for p in onboard_psg
        }
        all_req_map.update({r.id: r for r in new_reqs_list})

        # --- 4. DP State Initialization ---

        # Memoization table for dominance pruning.
        # Key: (visited_bitmask, current_last_stop_index)
        # Value: (min_accumulated_cost, min_current_time)
        memo: Dict[Tuple[int, int], Tuple[float, float]] = {}

        best_route_indices: Optional[List[int]] = None
        min_total_cost = float("inf")

        # --- 5. The Recursive Solver (DFS) ---

        def solve_recursive(
            current_stop_index: int,  # -1 indicates virtual start at vehicle location
            current_time: float,
            current_load: int,
            visited_mask: int,
            accumulated_cost: float,
            path_indices: List[int],
        ):
            nonlocal best_route_indices, min_total_cost

            # --- Dominance Pruning ---
            # If we have reached this exact state (same set of stops visited,
            # ending at the same node) before, check if the previous path was better.
            state_key = (visited_mask, current_stop_index)

            if state_key in memo:
                prev_cost, prev_time = memo[state_key]
                # Dominance Rule: If previous path had lower/equal cost AND
                # lower/equal time, this current path cannot lead to a better optima.
                if prev_cost <= accumulated_cost and prev_time <= current_time:
                    return

            # Update state with this new Pareto-optimal (or non-dominated) path
            memo[state_key] = (accumulated_cost, current_time)

            # --- Base Case: All stops visited ---
            if visited_mask == FULL_MASK:
                if accumulated_cost < min_total_cost:
                    min_total_cost = accumulated_cost
                    best_route_indices = list(path_indices)
                return

            # --- Bound Pruning ---
            # If we are already worse than the best complete solution found so far
            if accumulated_cost >= min_total_cost:
                return

            # --- Expansion: Try visiting every unvisited node ---
            for i in range(N):
                # Check if bit 'i' is set in mask
                if (visited_mask >> i) & 1:
                    continue

                next_stop = all_stops[i]

                # A. Precedence Constraint
                # If next_stop is a Dropoff, has its Pickup been visited?
                p_idx, _ = req_indices[next_stop.request_id]

                # p_idx == -1 means it's an existing passenger (pickup done)
                if p_idx != -1:
                    # Check if pickup bit is set in mask
                    if not ((visited_mask >> p_idx) & 1):
                        continue  # Invalid: Trying to drop off before pickup

                # B. Capacity Constraint
                # If picking up, do we have space?
                load_change = 1 if next_stop.is_pickup else -1
                next_load = current_load + load_change

                if next_load > availability.capacity:
                    continue

                # C. Travel Time Calculation
                # Determine previous node ID
                if current_stop_index == -1:
                    prev_node_id = availability.available_at_node
                else:
                    prev_node_id = all_stops[current_stop_index].node

                # Get cached time
                try:
                    travel_time = tt_cache[(prev_node_id, next_stop.node)]
                except KeyError:
                    # Fallback / Safety (should be covered by cache builder)
                    # [Image of safety net]
                    raise ValueError(
                        f"Missing travel time cache for ({prev_node_id}, {next_stop.node})"
                    )
                    travel_time = self._network.get_shortest_path(
                        prev_node_id, next_stop.node
                    ).travel_time

                arrival_time = current_time + travel_time
                req = all_req_map[next_stop.request_id]

                # D. Time Window Constraints

                next_accumulated_cost = accumulated_cost

                if next_stop.is_pickup:
                    # Constraint: Max Wait Time (Omega)
                    wait_time = arrival_time - req.request_time

                    # 1. Hard System Limit
                    if wait_time > OMEGA:
                        continue

                    # 2. Promise Limit (if applicable)
                    if req.promised_pickup_time:
                        # Allow a tiny float tolerance (e.g., 1e-6)
                        if wait_time > (
                            req.promised_pickup_time - req.request_time + 1e-6
                        ):
                            continue

                    # Pickups do not add to the cost function (delay),
                    # though they consume time.

                else:
                    # Constraint: Max Total Delay (Delta)
                    # Note: earliest_dropoff_time = request_time + direct_travel_time
                    total_delay = arrival_time - req.earliest_dropoff_time

                    if total_delay > DELTA:
                        continue

                    # Dropoffs contribute to the objective function
                    next_accumulated_cost += total_delay

                # --- Recurse ---
                path_indices.append(i)
                solve_recursive(
                    current_stop_index=i,
                    current_time=arrival_time,
                    current_load=next_load,
                    visited_mask=visited_mask | (1 << i),
                    accumulated_cost=next_accumulated_cost,
                    path_indices=path_indices,
                )
                path_indices.pop()

        # --- 6. Start the Search ---
        # Start at "virtual index -1", representing the vehicle's availability state
        solve_recursive(
            current_stop_index=-1,
            current_time=availability.available_from_time,
            current_load=len(onboard_psg),
            visited_mask=0,
            accumulated_cost=0.0,
            path_indices=[],
        )

        # --- 7. Return Result ---
        if best_route_indices is not None:
            final_stops = [all_stops[i] for i in best_route_indices]

            # Create Route object
            # Note: total_travel_distance is not calculated here to save time,
            # as ILP usually optimizes cost (delay).
            return FeasibleRoute(
                route=Route(
                    stops=final_stops,
                    total_travel_time=0.0,  # Can be populated if needed
                    total_travel_distance=0.0,
                    total_delay_cost=min_total_cost,
                ),
                cost=min_total_cost,
            )

        return Infeasible(reason="No feasible permutation found via RCSPP")

    def _build_travel_time_cache(self, nodes: Set[int]) -> Dict[Tuple[int, int], float]:
        """
        Helper to pre-calculate all-pairs travel times for the involved nodes.
        """
        cache = {}
        # itertools.product is efficient for generating pairs
        for u, v in itertools.product(nodes, repeat=2):
            if u == v:
                cache[(u, v)] = 0.0
            else:
                path = self._network.get_shortest_path(u, v)
                cache[(u, v)] = path.travel_time if path else float("inf")
        return cache
