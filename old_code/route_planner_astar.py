"""Minimal single-vehicle route planner used during prototyping."""

from dataclasses import dataclass, field
from typing import List, Optional
from config import DELTA, OMEGA, get_logger

from src.interfaces import (
    FeasibilityResult,
    FeasibleRoute,
    Infeasible,
    Request,
    Stop,
    Route,
    Passenger,
    INetworkService,
    VehicleAvailability,
    ISingleVehicleRoutePlanner,
)
import heapq

logger = get_logger(__name__)


@dataclass(order=True)
class SearchNode:
    """Represents a state in the A* search tree."""

    f_score: float  # Priority (g + h)
    g_score: float = field(compare=False)  # Actual cost so far (cumulative delay)
    time: float = field(compare=False)  # Current time at this node
    load: int = field(compare=False)  # Current passenger count
    mask: int = field(compare=False)  # Bitmask of visited stops
    curr_node_idx: int = field(compare=False)  # Index of the stop we are currently at
    parent: Optional["SearchNode"] = field(compare=False, default=None)

    def get_path(self, stops: list[Stop]) -> list[Stop]:
        """Reconstructs the path by backtracking."""
        path = []
        curr = self
        while curr.parent is not None:
            # We skip the root/dummy start node in the path list
            # providing the stop indices align with the stops list
            if curr.curr_node_idx >= 0:
                path.append(stops[curr.curr_node_idx])
            curr = curr.parent
        return path[::-1]


class RoutePlanner(ISingleVehicleRoutePlanner):
    """A brute-force feasibility checker for two-request R-R cases."""

    def __init__(self, network_service: INetworkService) -> None:
        super().__init__(network_service)

    def check_feasibility(
        self,
        requests_to_add: set[Request],
        vehicle: VehicleAvailability | None = None,
        find_best_route: bool = True,
    ) -> FeasibilityResult:

        if not requests_to_add:
            raise ValueError("Requires at least one request.")

        # --- CASE 1: R-R Check (No Vehicle) ---
        if vehicle is None:
            if len(requests_to_add) != 2:
                # If you ever need 3+ request pooling without a vehicle,
                # you'd implement a generic RR-solver here.
                raise NotImplementedError(
                    "Only pairwise R-R supported without vehicle."
                )
            raise NotImplementedError("R-R checks are not implemented in this planner.")

        # --- CASE 2: General Vehicle + N Requests (RV & RTV) ---
        # Whether it is 1 request (RV) or 3 requests (RTV), the logic is identical.
        _ = find_best_route  # Currently unused
        return self._handle_rtv_check(vehicle, requests_to_add)

    def _handle_rtv_check(
        self,
        availability: VehicleAvailability,
        requests_to_add: set[Request],
        find_best_route: bool = True,
    ) -> FeasibilityResult:
        """
        A* Solver for the exact Single-Vehicle Routing Problem.
        """

        # --- 1. SETUP (Same as before) ---
        start_node = availability.available_at_node
        start_time = availability.available_from_time
        initial_load = len(availability.passengers_onboard)
        capacity = availability.capacity

        stops: list[Stop] = []
        # Add existing drop-offs
        for p in availability.passengers_onboard:
            stops.append(
                Stop(p.original_request.destination_node, False, p.original_request.id)
            )

        # Add new requests (Pickup & Dropoff)
        new_req_ids = set()
        for r in requests_to_add:
            new_req_ids.add(r.id)
            stops.append(Stop(r.origin_node, True, r.id))
            stops.append(Stop(r.destination_node, False, r.id))

        N = len(stops)
        if N == 0:
            # Trivial case: nothing to do
            return FeasibleRoute(Route((), 0.0, 0.0, 0.0), 0.0)

        # Lookup Maps & Constraints
        all_requests = {
            p.original_request.id: p.original_request
            for p in availability.passengers_onboard
        }
        all_requests.update({r.id: r for r in requests_to_add})

        pickup_indices = {s.request_id: i for i, s in enumerate(stops) if s.is_pickup}

        # Precompute Constraints
        stop_req_ids = [s.request_id for s in stops]
        is_pickup = [s.is_pickup for s in stops]
        stop_nodes = [s.node for s in stops]

        # Precompute Time Windows & Target Dropoff Times
        earliest_arrival = []
        latest_arrival = []
        target_times = []  # For cost calculation (dropoff delay)

        for i, s in enumerate(stops):
            req = all_requests[s.request_id]
            if s.is_pickup:
                start = req.request_time
                end = (
                    req.promised_pickup_time
                    if req.promised_pickup_time
                    else (req.request_time + OMEGA)
                )
                earliest_arrival.append(start)
                latest_arrival.append(end)
                target_times.append(
                    0.0
                )  # Pickups don't incur "delay cost" in this objective
            else:
                earliest_arrival.append(req.request_time)
                # Max delay constraint
                latest_arrival.append(
                    req.earliest_dropoff_time + DELTA
                    if req.earliest_dropoff_time
                    else float("inf")
                )
                # For cost: we want to minimize (arrival - earliest_dropoff)
                target_times.append(req.earliest_dropoff_time)

        # Travel Time Cache
        unique_nodes = set(stop_nodes) | {start_node}
        tt_cache = self._build_travel_time_cache(unique_nodes)  # type: ignore

        # --- 2. HEURISTIC FUNCTION ---
        def calculate_heuristic(curr_time: float, curr_node: int, mask: int) -> float:
            """
            Admissible Heuristic: Max unavoidable delay for unvisited drop-offs.
            If we drove straight to any remaining drop-off, what is the minimum delay we'd face?
            We take the MAX of these minimums as a safe lower bound.
            """
            max_lower_bound_delay = 0.0

            for i in range(N):
                # If unvisited and is a dropoff
                if not ((mask >> i) & 1) and not is_pickup[i]:
                    # We MUST visit this dropoff eventually.
                    # Fastest way: Direct line from current node
                    dist = tt_cache.get((curr_node, stop_nodes[i]), 0.0)
                    min_arrival = curr_time + dist

                    # Cost = max(0, arrival - target)
                    potential_delay = max(0.0, min_arrival - target_times[i])

                    if potential_delay > max_lower_bound_delay:
                        max_lower_bound_delay = potential_delay

            return max_lower_bound_delay

        # --- 3. A* SEARCH ---

        # Priority Queue: Stores SearchNode
        pq = []

        # Initial State (Dummy Root)
        # Mask 0, Time start_time, Load initial, Node start_node
        root = SearchNode(0.0, 0.0, start_time, initial_load, 0, -1, None)
        heapq.heappush(pq, root)

        # Closed Set for Dominance Pruning
        # Maps (mask, last_node_idx) -> (min_cost, min_time)
        # If we find a path to the same state that is WORSE in both cost and time, we skip.
        visited: dict[tuple[int, int], tuple[float, float]] = {}

        best_solution_node = None
        min_total_cost = float("inf")

        target_mask = (1 << N) - 1

        while pq:
            node = heapq.heappop(pq)

            # 1. Pruning: Dominance Check
            state_key = (node.mask, node.curr_node_idx)
            if state_key in visited:
                prev_cost, prev_time = visited[state_key]
                # If we have been here before with lower cost AND earlier/equal time,
                # this path is dominated. Drop it.
                if prev_cost <= node.g_score and prev_time <= node.time:
                    continue

            # Record state
            visited[state_key] = (node.g_score, node.time)

            # 2. Goal Check
            if node.mask == target_mask:
                if node.g_score < min_total_cost:
                    min_total_cost = node.g_score
                    best_solution_node = node

                    # Optimization: If we only care about feasibility (any valid route)
                    # A* guarantees the first goal reached is optimal IF heuristic is consistent.
                    # Even if h is just admissible, it's usually very good.
                    if not find_best_route:
                        break
                continue

            # 3. Expand Children
            # Current physical location
            u_node = (
                start_node
                if node.curr_node_idx == -1
                else stop_nodes[node.curr_node_idx]
            )

            for i in range(N):
                # If already visited, skip
                if (node.mask >> i) & 1:
                    continue

                # --- VALIDATION (Identical to DFS) ---

                # A. Precedence: If dropoff, is pickup visited?
                if not is_pickup[i] and stop_req_ids[i] in new_req_ids:
                    p_idx = pickup_indices[stop_req_ids[i]]
                    if not ((node.mask >> p_idx) & 1):
                        continue

                # B. Capacity
                delta_load = 1 if is_pickup[i] else -1
                if node.load + delta_load > capacity:
                    continue

                # C. Time Constraints
                v_node = stop_nodes[i]
                travel_time = tt_cache.get((u_node, v_node), float("inf"))
                arrival_time = node.time + travel_time

                # Wait if early
                actual_arrival = max(arrival_time, earliest_arrival[i])

                # Late check
                if actual_arrival > latest_arrival[i]:
                    continue

                # --- SCORE CALCULATION ---

                added_delay = 0.0
                if not is_pickup[i]:
                    # Cost = Delay relative to earliest dropoff
                    delay = actual_arrival - target_times[i]
                    if delay > 0:
                        added_delay = delay

                new_g = node.g_score + added_delay

                # Optimization: Pruning by upper bound
                # If we already have a full solution, and this partial path is already worse, stop.
                if new_g >= min_total_cost:
                    continue

                # Heuristic
                h = calculate_heuristic(actual_arrival, v_node, node.mask | (1 << i))
                new_f = new_g + h

                # Add to Queue
                new_node = SearchNode(
                    f_score=new_f,
                    g_score=new_g,
                    time=actual_arrival,
                    load=node.load + delta_load,
                    mask=node.mask | (1 << i),
                    curr_node_idx=i,
                    parent=node,
                )
                heapq.heappush(pq, new_node)

        # --- 4. RESULT ---
        if best_solution_node:
            final_stops = tuple(best_solution_node.get_path(stops))
            # Construct Route object (simplified)
            route = Route(
                stops=final_stops,
                total_travel_time=best_solution_node.time - start_time,
                total_travel_distance=0.0,  # Not calculated
                total_delay_cost=best_solution_node.g_score,
            )
            return FeasibleRoute(route=route, cost=best_solution_node.g_score)

        return Infeasible(reason="No feasible route found via A*")

    def _simulate_route(
        self,
        start_node: int,
        start_time: float,
        passengers_onboard: frozenset[Passenger],
        all_requests_map: dict[str, Request],
        stops_to_visit: list[Stop],
        capacity: int,
        travel_time_cache: dict[tuple[int, int], float],
        is_rr_check: bool = False,
    ) -> FeasibilityResult:
        """
        A general-purpose route simulator that checks all constraints.
        """
        current_node = start_node
        current_time = start_time
        # Start with passengers already in the car
        current_occupancy = len(passengers_onboard)

        # Keep track of who has been picked up
        picked_up_req_ids = {p.original_request.id for p in passengers_onboard}

        pickup_times = {
            p.original_request.id: p.pickup_time for p in passengers_onboard
        }
        dropoff_times = {}
        total_travel_time = 0.0
        stop_times = []

        for stop in stops_to_visit:
            # 1. Travel to the next stop
            travel_time_key = (current_node, stop.node)
            travel_time = travel_time_cache.get(travel_time_key)
            if travel_time is None:
                # Cache miss: This is expected if find_best_route=False
                if current_node == stop.node:
                    travel_time = 0.0
                else:
                    # Do the "expensive" network call on-demand
                    path = self._network.get_shortest_path(current_node, stop.node)
                    if path is None:
                        return Infeasible(
                            reason=f"No path between nodes {current_node} and {stop.node}"
                        )
                    travel_time = path.travel_time

                # Store it in the cache in case this route uses
                # the same edge again (unlikely, but good practice)
                travel_time_cache[travel_time_key] = travel_time

            stop_times.append(current_time)

            current_time += travel_time
            current_node = stop.node
            total_travel_time += travel_time

            req = all_requests_map[stop.request_id]

            if stop.is_pickup:
                # Enforce causality: cannot pick up before request is made
                if current_time < req.request_time and not is_rr_check:
                    raise ValueError(
                        f"Attempting to pick up request {req.id} before it was made."
                    )
                elif is_rr_check:
                    # In R-R checks, we allow waiting at the pickup node
                    current_time = max(current_time, req.request_time)

                # 2. Check Wait Time (Omega)
                wait_time = current_time - req.request_time
                max_wait_allowed = OMEGA
                if req.promised_pickup_time is not None:
                    max_wait_allowed = req.promised_pickup_time - req.request_time
                    if wait_time > max_wait_allowed + 1e-6:
                        return Infeasible(
                            reason=f"Rematch would worsen wait for {req.id}"
                        )
                if wait_time > OMEGA:
                    return Infeasible(reason=f"Violates max wait for {req.id}")

                pickup_times[req.id] = current_time
                picked_up_req_ids.add(req.id)
                current_occupancy += 1

                # 3. Check Capacity
                if current_occupancy > capacity:
                    return Infeasible(reason="Violates capacity")

            else:  # Is dropoff
                # Check if this passenger was ever picked up (sanity check)
                if req.id not in picked_up_req_ids:
                    return Infeasible(reason=f"Dropoff before pickup for {req.id}")

                dropoff_times[req.id] = current_time
                current_occupancy -= 1

                # 4. Check Total Delay (Delta)
                # This check applies to EVERYONE (new and old passengers)
                if req.earliest_dropoff_time is None:
                    raise ValueError(
                        f"Earliest dropoff time not set for request {req.id}"
                    )
                total_delay = current_time - req.earliest_dropoff_time
                if total_delay > DELTA:
                    return Infeasible(reason=f"Violates max delay for {req.id}")

        # If we got here, the route is feasible
        # Cost is the sum of delays for *all* passengers dropped off
        total_cost = 0.0
        for r_id, drop_time in dropoff_times.items():
            req = all_requests_map[r_id]
            total_cost += drop_time - req.earliest_dropoff_time

        final_route = Route(
            stops=tuple(stops_to_visit),
            total_travel_time=total_travel_time,
            total_travel_distance=0,  # Not calculated
            total_delay_cost=total_cost,
        )

        return FeasibleRoute(route=final_route, cost=total_cost)

    def _build_travel_time_cache(self, nodes: set[int]) -> dict[tuple[int, int], float]:
        """Pre-calculates all-pairs travel times for a small set of nodes."""
        # assume all nodes pares are primed in the cache
        return self._network.get_travel_times_l1_cache_subset(nodes)

    @staticmethod
    def _build_stops(request_a: Request, request_b: Request) -> tuple[Stop, ...]:
        return (
            Stop(node=request_a.origin_node, is_pickup=True, request_id=request_a.id),
            Stop(
                node=request_a.destination_node,
                is_pickup=False,
                request_id=request_a.id,
            ),
            Stop(node=request_b.origin_node, is_pickup=True, request_id=request_b.id),
            Stop(
                node=request_b.destination_node,
                is_pickup=False,
                request_id=request_b.id,
            ),
        )


if __name__ == "__main__":
    from src.demos.route_planner_demo import run_demo

    run_demo()
