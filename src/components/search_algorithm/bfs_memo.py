from collections.abc import Iterable
import heapq
import numpy as np
from typing import Dict, Tuple, FrozenSet, List

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

# Global Cache: (sorted_remaining_stops, current_stop_id) -> List[(latest_arrival_time, min_capacity_surge)]
# StopID is Tuple[str, bool] -> (request_id, is_pickup)
_UNIVERSAL_CACHE: Dict[
    Tuple[Tuple[Tuple[str, bool], ...], Tuple[str, bool]], List[Tuple[float, int]]
] = {}
EPSILON = 1e-5


class TripSolver:
    """
    A* BFS Solver with Universal Memoization (Lexicographic State).
    """

    @staticmethod
    def _get_constraints(
        current_stop: Tuple[str, bool],
        remaining_stops: Tuple[Tuple[str, bool], ...],
        requests_map: Dict[str, Request],
        time_matrix: np.ndarray,
    ) -> List[Tuple[float, int]]:
        """
        Recursive function to compute constraints for a state.
        State = (current_stop, remaining_stops_to_visit)
        Returns a Pareto Frontier of (limit, kappa) constraints.
        """
        state_key = (remaining_stops, current_stop)
        if state_key in _UNIVERSAL_CACHE:
            return _UNIVERSAL_CACHE[state_key]

        # 1. Get properties of current stop
        r_id, is_pk = current_stop
        req = requests_map[r_id]

        if is_pk:
            # Pickup

            current_deadline = req.latest_pickup_time
            current_node = req.origin_node
        else:
            # Dropoff
            current_deadline = req.latest_dropoff_time
            current_node = req.destination_node

        # Base Case: No more stops to visit
        if not remaining_stops:
            # If we are at the end, the constraint is just our own deadline.
            # Kappa is 0 because there is no future path.
            res = [(current_deadline, 0)]
            _UNIVERSAL_CACHE[state_key] = res
            return res

        # Recursive Step
        all_constraints = []
        valid_next_exists = False

        # We need to try visiting every possible next stop from remaining_stops
        for i, next_stop in enumerate(remaining_stops):
            next_r_id, next_is_pk = next_stop

            # Prereq Check:
            if not next_is_pk:
                pickup_still_pending = False
                for check_stop in remaining_stops:
                    if check_stop == (next_r_id, True):
                        pickup_still_pending = True
                        break
                if pickup_still_pending:
                    continue

            valid_next_exists = True

            # Construct new remaining tuple (remove next_stop)
            new_remaining = remaining_stops[:i] + remaining_stops[i + 1 :]

            # Recurse - Get List of Constraints
            child_constraints = TripSolver._get_constraints(
                next_stop, new_remaining, requests_map, time_matrix
            )

            if not child_constraints:
                continue

            # Get distance from current to next
            next_req = requests_map[next_r_id]
            next_node = (
                next_req.origin_node if next_is_pk else next_req.destination_node
            )
            dist = time_matrix[current_node, next_node]

            # Delta at next stop
            delta_v = 1 if next_is_pk else -1

            # Propagate constraints back to current
            for c_limit, c_kappa in child_constraints:
                if c_limit == -INF:
                    continue

                # Time: Must arrive at current such that (arrival + dist) <= c_limit
                val_time = c_limit - dist

                # Capacity:
                # We need 'load_at_next + c_kappa <= capacity'
                # load_at_next = load_at_current + delta_v
                # So: load_at_current + delta_v + c_kappa <= capacity
                # kappa_current = max(0, delta_v + c_kappa)

                val_kappa = max(0, delta_v + c_kappa)

                # Also bound by current deadline
                final_time = min(current_deadline, val_time)

                all_constraints.append((final_time, val_kappa))

        if not valid_next_exists or not all_constraints:
            # Dead end or invalid path
            res = []
            _UNIVERSAL_CACHE[state_key] = res
            return res

        # Pareto Filtering
        # We want to keep (Limit, Kappa) pairs that are "easiest".
        # Easier = Higher Limit (Later), Lower Kappa (Less Surge).
        # Sort by Limit Descending (Best Time First).
        all_constraints.sort(key=lambda x: x[0], reverse=True)

        pareto_front = []
        min_kappa_seen = INF

        for limit, kappa in all_constraints:
            if limit == -INF:
                continue

            # If this kappa is better (lower) than anything we've seen for "better or equal" times, keep it.
            if kappa < min_kappa_seen:
                pareto_front.append((limit, kappa))
                min_kappa_seen = kappa

        _UNIVERSAL_CACHE[state_key] = pareto_front
        return pareto_front

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

        # Map Request ID -> Request Object for easy lookup
        requests_map = {r.id: r for r in requests}
        for p in vehicle.passengers_onboard:
            requests_map[p.original_request.id] = p.original_request

        # Build the initial "To-Do" list (Remaining Stops)
        # 1. Onboard Passengers -> Only Dropoffs remaining
        # 2. New Requests -> Pickup AND Dropoff remaining

        initial_stops = []

        # Onboard
        for p in vehicle.passengers_onboard:
            initial_stops.append((p.original_request.id, False))  # Dropoff only

        # New Requests
        for r in requests:
            initial_stops.append((r.id, True))  # Pickup
            initial_stops.append((r.id, False))  # Dropoff

        # Sort to create Canonical State
        # Sort by Request ID, then by Type (Pickup=True first? No, False=0, True=1.
        # We want consistent ordering. (ID, is_pickup) is fine.)
        initial_stops.sort()
        initial_stops_tuple = tuple(initial_stops)

        # Pre-compute constraints for all reachable states in the "Universal Cache"
        # We don't need to pre-compute everything. We can do it lazily inside the search.
        # But to match the previous logic, we can just let the search drive it.

        # --- A* BFS STATE ---
        # State: (current_node_idx, current_load, visited_mask_tuple?)
        # Actually, we need to track "Remaining Stops" in the BFS state to query the cache.
        # Using a Tuple in the BFS state is slightly heavier than a bitmask, but necessary for the Universal Cache.
        # Optimization: We can map the initial_stops to integers 0..N-1 for the BFS *local* state,
        # and reconstruct the Tuple only when querying the cache.
        # OR: Just use the Tuple in the BFS state. Python tuples are hashable and reasonably fast.

        # Let's use the Tuple directly in the BFS state for correctness first.

        # Heuristic Cache for this specific solve (optional, but good for A*)
        # We can use the Universal Cache for the heuristic too!
        # H(state) = max(0, earliest_possible_arrival - latest_feasible_arrival) ?
        # No, H is usually "estimated cost to go".
        # Here we are just checking feasibility. A* is for minimizing cost (delay).

        # Initial State
        # (f_score, g_score, current_time, current_node_idx, current_load, remaining_stops_tuple, path)

        start_node = v_node
        start_time = v_start

        # Check initial feasibility with cache?
        # The cache tells us constraints for the *next* node.
        # We are at v_node (which is NOT a request stop, usually).
        # We need to check if we can reach ANY next stop from v_node.

        pq = [(0.0, 0.0, start_time, start_node, initial_load, initial_stops_tuple, [])]
        closed = {}  # (current_node_idx, remaining_stops) -> min_arrival_time

        while pq:
            f, g, c_time, u_idx, load, remaining, path = heapq.heappop(pq)

            # Goal Check
            if not remaining:
                # Reconstruct Route
                final_stops = []
                for stop_info in path:
                    r_id, is_pk = stop_info
                    req = requests_map[r_id]
                    node_idx = req.origin_node if is_pk else req.destination_node
                    final_stops.append(
                        DMStop(
                            node=node_idx,
                            is_pickup=is_pk,
                            request_id=r_id,
                        )
                    )
                return FeasibleRoute(
                    route=Route(tuple(final_stops), g, 0.0, 0.0), cost=g
                )

            state_key = (u_idx, remaining)
            if state_key in closed and closed[state_key] <= c_time:
                continue
            closed[state_key] = c_time

            # Explore Next Stops
            for i, next_stop in enumerate(remaining):
                next_r_id, next_is_pk = next_stop

                # Prereq Check (Local)
                if not next_is_pk:
                    # Check if pickup is still in remaining
                    # (next_r_id, True)
                    pickup_pending = False
                    for s in remaining:
                        if s == (next_r_id, True):
                            pickup_pending = True
                            break
                    if pickup_pending:
                        continue

                # Load Check
                delta = 1 if next_is_pk else -1
                new_load = load + delta
                if new_load > v_cap:
                    continue

                # Get Node Info
                req = requests_map[next_r_id]
                v_global_idx = req.origin_node if next_is_pk else req.destination_node

                # Travel Time
                dist = time_matrix[u_idx, v_global_idx]
                arrival = c_time + dist
                # --- UNIVERSAL MEMOIZATION CHECK ---
                # We are moving TO 'next_stop'.
                # The state AFTER visiting 'next_stop' will have 'new_remaining'.
                # We want to know if 'next_stop' is feasible given 'new_remaining'.
                # Query: _get_constraints(next_stop, new_remaining)

                new_remaining = remaining[:i] + remaining[i + 1 :]

                # This returns a LIST of Pareto-optimal constraints for 'next_stop'
                constraints_list = TripSolver._get_constraints(
                    next_stop, new_remaining, requests_map, time_matrix
                )

                if not constraints_list:
                    continue

                # Check if (arrival, new_load) satisfies ANY of the constraints
                # A constraint (limit, kappa) is satisfied if:
                # 1. arrival <= limit
                # 2. new_load + kappa <= v_cap

                is_feasible = False
                for constraint_time, min_kappa in constraints_list:
                    if arrival <= constraint_time + EPSILON:
                        if new_load + min_kappa <= v_cap:
                            is_feasible = True
                            break

                if not is_feasible:
                    continue

                # Cost Calculation (Delay)
                added_cost = 0.0
                if not next_is_pk:
                    # Delay cost is only for dropoffs (arrival - request_time)
                    # Wait, cost function might vary. Assuming standard delay cost.
                    added_cost = arrival - req.request_time

                new_g = g + added_cost

                # Heuristic
                # We can use the difference between 'arrival' and 'constraint_time' as a heuristic?
                # Or just 0 for now.
                h = 0.0

                new_f = new_g + h

                heapq.heappush(
                    pq,
                    (
                        new_f,
                        new_g,
                        arrival,
                        v_global_idx,
                        new_load,
                        new_remaining,
                        path + [next_stop],
                    ),
                )

        return Infeasible(reason="No feasible route found")
