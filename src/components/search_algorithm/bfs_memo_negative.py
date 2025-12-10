from collections.abc import Iterable
import heapq
import numpy as np
from typing import Dict, Tuple, List

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

# Global Cache: (remaining_stops, current_stop_node_idx) -> List[(arrival_time, residual_capacity)]
# We store "Witnesses" (states that have been visited/processed).
# Key: (remaining_stops_tuple, current_node_idx)
# Value: Pareto Frontier of (Time, Residual).
# If a new state has (Time >= W.Time) and (Residual <= W.Residual), it is Dominated (Pruned).
_UNIVERSAL_CACHE: Dict[
    Tuple[int, Tuple[Tuple[str, bool], ...]], List[Tuple[float, int]]
] = {}
EPSILON = 0


class TripSolver:
    """
    A* BFS Solver with Universal Memoization (Lexicographic State).
    """

    @staticmethod
    def _is_dominated(
        state_key: Tuple[int, Tuple[Tuple[str, bool], ...]],
        arrival: float,
        residual: int,
    ) -> bool:
        """
        Check if the current state is dominated by any witness in the cache.
        Dominated means: Exists W such that (Arrival >= W.Time) AND (Residual <= W.Residual).
        (Meaning: W arrived earlier/same-time AND W had more/same space).
        """
        if state_key not in _UNIVERSAL_CACHE:
            return False

        witnesses = _UNIVERSAL_CACHE[state_key]
        for w_time, w_residual in witnesses:
            if arrival >= w_time - EPSILON and residual <= w_residual:
                return True
        return False

    @staticmethod
    def _update_cache(
        state_key: Tuple[int, Tuple[Tuple[str, bool], ...]],
        arrival: float,
        residual: int,
    ):
        """
        Add (arrival, residual) to cache and maintain Pareto frontier.
        Remove any witnesses that are dominated by this new state.
        Dominance: New state dominates W if (New.Time <= W.Time) AND (New.Residual >= W.Residual).
        """
        if state_key not in _UNIVERSAL_CACHE:
            _UNIVERSAL_CACHE[state_key] = [(arrival, residual)]
            return

        witnesses = _UNIVERSAL_CACHE[state_key]
        new_witnesses = []
        is_dominated = False

        for w_time, w_residual in witnesses:
            # If new state dominates existing witness
            if arrival <= w_time + EPSILON and residual >= w_residual:
                # We dominate or equal W. W is removed.
                pass

            # Check if WE are dominated
            elif arrival >= w_time - EPSILON and residual <= w_residual:
                is_dominated = True
                new_witnesses.append((w_time, w_residual))
            else:
                new_witnesses.append((w_time, w_residual))

        if not is_dominated:
            new_witnesses.append((arrival, residual))
            _UNIVERSAL_CACHE[state_key] = new_witnesses

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

        # Clear Universal Cache for this run (Local Negative Pruning)
        # We cannot prune based on other vehicles' performance (different origins).
        _UNIVERSAL_CACHE.clear()

        # Check initial feasibility with cache?
        # The cache tells us constraints for the *next* node.
        # We are at v_node (which is NOT a request stop, usually).
        # We need to check if we can reach ANY next stop from v_node.

        pq = [(0.0, 0.0, start_time, start_node, initial_load, initial_stops_tuple, [])]
        # closed = {}  <-- Replaced by _UNIVERSAL_CACHE Logic

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

            # Residual Capacity
            residual = v_cap - load

            # --- NEGATIVE PRUNING CHECK ---
            if TripSolver._is_dominated(state_key, c_time, residual):
                continue

            # --- UPDATE NEGATIVE CACHE ---
            TripSolver._update_cache(state_key, c_time, residual)

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

                # Hard Deadline Check
                if next_is_pk:
                    deadline = req.latest_pickup_time
                else:
                    deadline = req.latest_dropoff_time

                if arrival > deadline:
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

                new_remaining = remaining[:i] + remaining[i + 1 :]

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
