"""Minimal single-vehicle route planner used during prototyping."""

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


logger = get_logger(__name__)


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
        return self._handle_rtv_dfs(vehicle, requests_to_add, find_best_route)

    def _handle_rtv_dfs(
        self,
        availability: VehicleAvailability,
        requests_to_add: set[Request],
        find_best_route: bool,
    ) -> FeasibilityResult:
        """
        Unified DFS Branch-and-Bound solver for RV and RTV.
        """
        # --- 1. SETUP ---
        start_node = availability.available_at_node
        start_time = availability.available_from_time
        initial_load = len(availability.passengers_onboard)
        capacity = availability.capacity

        # Collect all stops (Existing Drop-offs + New Pickups/Drop-offs)
        stops: list[Stop] = []

        # Existing passengers (Drop-off only)
        for p in availability.passengers_onboard:
            stops.append(
                Stop(p.original_request.destination_node, False, p.original_request.id)
            )

        # New requests (Pickup + Drop-off)
        new_req_ids = set()
        for r in requests_to_add:
            new_req_ids.add(r.id)
            stops.append(Stop(r.origin_node, True, r.id))
            stops.append(Stop(r.destination_node, False, r.id))

        N = len(stops)

        # Mappings & Constraints
        all_reqs = {
            p.original_request.id: p.original_request
            for p in availability.passengers_onboard
        }
        all_reqs.update({r.id: r for r in requests_to_add})

        # Arrays for fast indexing (avoid dict lookups in loop)
        s_node = [s.node for s in stops]
        s_is_pick = [s.is_pickup for s in stops]
        s_req_id = [s.request_id for s in stops]

        # Time Windows
        t_earliest = []
        t_latest = []

        for i in range(N):
            r = all_reqs[s_req_id[i]]
            if s_is_pick[i]:
                t_earliest.append(r.request_time)
                # Strict wait time constraint
                t_latest.append(r.request_time + OMEGA)
            else:
                t_earliest.append(r.request_time)  # Can't drop off before requested
                # Max detour constraint
                limit = getattr(
                    r, "earliest_dropoff_time", None
                )  # pre-calculated direct arrival
                t_latest.append((limit + DELTA) if limit else float("inf"))

        # Precedence: Pickup index for every dropoff index (if applicable)
        # We map RequestID -> Pickup Index
        pick_map = {}
        for i in range(N):
            if s_is_pick[i]:
                pick_map[s_req_id[i]] = i

        # Cache Travel Times
        unique = set(s_node) | {start_node}
        tt = self._build_travel_time_cache(unique)

        # --- 2. DFS STATE ---
        best_cost = float("inf")
        best_path: list[int] = []
        full_mask = (1 << N) - 1

        def solve(u_node, u_time, load, mask, cost, path):
            nonlocal best_cost, best_path

            # Pruning
            if cost >= best_cost:
                return

            # Goal
            if mask == full_mask:
                if cost < best_cost:
                    best_cost = cost
                    best_path = list(path)
                return

            # Branching
            for i in range(N):
                # If not visited
                if not ((mask >> i) & 1):

                    # A. Precedence (Dropoff requires visited Pickup)
                    # Existing passengers don't have pickups in 'stops', so check 'in new_req_ids'
                    if not s_is_pick[i] and s_req_id[i] in new_req_ids:
                        p_idx = pick_map[s_req_id[i]]
                        if not ((mask >> p_idx) & 1):
                            continue

                    # B. Capacity
                    # Pickup +1, Dropoff -1
                    new_load = load + (1 if s_is_pick[i] else -1)
                    if new_load > capacity:
                        continue

                    # C. Time
                    travel = tt.get((u_node, s_node[i]), float("inf"))
                    arr = u_time + travel
                    # Wait if early (not applicable for dropoffs usually, but logic holds)
                    act_arr = max(arr, t_earliest[i])

                    if act_arr > t_latest[i]:
                        continue

                    # D. Cost Function (Objective: Minimize Total Delay)
                    # Cost = Cumulative Travel Time + Wait Time?
                    # Standard: Sum of (Arrival_Dropoff - Request_Time)
                    # Your generic metric: Sum of Travel Times + Lateness Penalty

                    # Simple accumulation:
                    new_cost = cost + travel
                    # Add wait time at pickup (if driver arrives early)
                    if act_arr > arr:
                        new_cost += act_arr - arr

                    path.append(i)
                    solve(s_node[i], act_arr, new_load, mask | (1 << i), new_cost, path)
                    path.pop()  # Backtrack

        # --- 3. RUN ---
        solve(start_node, start_time, initial_load, 0, 0.0, [])

        if best_cost == float("inf"):
            return Infeasible(reason="DFS found no solution")

        return FeasibleRoute(
            route=Route(
                stops=tuple(stops[i] for i in best_path),
                total_travel_time=0.0,  # Not tracking strictly, but best_cost approx
                total_travel_distance=0.0,
                total_delay_cost=best_cost,
            ),
            cost=best_cost,
        )

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
