"""Minimal single-vehicle route planner used during prototyping."""

import itertools
from typing import cast

from config import BATCH_DURATION, DELTA, OMEGA, get_logger
from src.components.batch_controller import BatchController
from src.components.network_service import NetworkXService
from src.interfaces import (
    FeasibilityResult,
    FeasibleRoute,
    Infeasible,
    Request,
    Route,
    Stop,
    Vehicle,
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
        vehicle: VehicleAvailability,
        requests_to_add: set[Request],
        find_best_route: bool = False,  # <-- ADDED PARAM
    ) -> FeasibilityResult:

        if not requests_to_add:
            raise ValueError("RoutePlanner requires at least one request to evaluate")

        self._validate_batch_window(vehicle, requests_to_add)

        passengers_onboard = vehicle.passengers_onboard

        # --- CASE 1: R-R Check ---
        # (Two new requests, empty vehicle)
        if len(requests_to_add) == 2 and not passengers_onboard:
            return self._handle_rr_check(
                vehicle,
                requests_to_add,
                find_best_route,
            )

        # --- CASE 2: R-V Check ---
        # (One new request, vehicle state is 0..N passengers)
        if len(requests_to_add) == 1:
            return self._handle_rv_check(
                vehicle,
                list(requests_to_add)[0],
                find_best_route,
            )

        return Infeasible(reason="Planner only supports R-R and R-V checks")

    # --- R-R Handler ---
    def _handle_rr_check(
        self,
        vehicle: VehicleAvailability,
        requests: set[Request],
        find_best_route: bool,
    ) -> FeasibilityResult:

        if vehicle.capacity < 2:
            return Infeasible(reason="Vehicle capacity too low for sharing")

        r1, r2 = tuple(requests)

        # 1. Lower-Bound Check
        path_1_to_2 = self._network.get_shortest_path(r1.origin_node, r2.origin_node)
        path_2_to_1 = self._network.get_shortest_path(r2.origin_node, r1.origin_node)
        tt_1_to_2 = path_1_to_2.travel_time if path_1_to_2 else float("inf")
        arrival_at_r2 = r1.request_time + tt_1_to_2
        if arrival_at_r2 - r2.request_time > OMEGA:
            tt_2_to_1 = path_2_to_1.travel_time if path_2_to_1 else float("inf")
            arrival_at_r1 = r2.request_time + tt_2_to_1
            if arrival_at_r1 - r1.request_time > OMEGA:
                return Infeasible(reason="Fails R-R lower-bound check")
        # END Lower-Bound Check

        p1 = Stop(node=r1.origin_node, is_pickup=True, request_id=r1.id)
        d1 = Stop(node=r1.destination_node, is_pickup=False, request_id=r1.id)
        p2 = Stop(node=r2.origin_node, is_pickup=True, request_id=r2.id)
        d2 = Stop(node=r2.destination_node, is_pickup=False, request_id=r2.id)

        valid_sequences = [
            (p1, p2, d1, d2),
            (p1, p2, d2, d1),
            (p2, p1, d1, d2),
            (p2, p1, d2, d1),
            (p1, d1, p2, d2),
            (p2, d2, p1, d1),
        ]

        # 2. Build Travel Time Cache
        travel_time_cache = {}  # Start with an empty cache
        if find_best_route:
            nodes_to_cache = {p1.node, d1.node, p2.node, d2.node}
            travel_time_cache = self._build_travel_time_cache(nodes_to_cache)

        # This is a general request map for the simulation
        requests_map = {r.id: r for r in requests}

        best_route = None
        min_cost = float("inf")

        for route_stops in valid_sequences:
            first_pickup = route_stops[0]  # guaranteed pickup in our list
            first_req = requests_map[first_pickup.request_id]

            sim_result = self._simulate_route(
                start_node=first_pickup.node,
                start_time=first_req.request_time,
                passengers_onboard=frozenset(),  # Empty for R-R
                all_requests_map=requests_map,
                stops_to_visit=list(route_stops),
                capacity=vehicle.capacity,
                travel_time_cache=travel_time_cache,
            )

            if sim_result.is_feasible and isinstance(sim_result, FeasibleRoute):
                feasible_route = sim_result
                if not find_best_route:
                    return feasible_route  # Early exit for RV graph

                if feasible_route.cost < min_cost:
                    if feasible_route.cost < 0:
                        # NOTE: remove after validating the system end-to-end
                        logger.error(
                            "Negative cost in feasible route: %.2f", feasible_route.cost
                        )
                        # look at the vehicle and requests to debug
                        logger.error("Vehicle: %s", vehicle)
                        logger.error("Requests: %s", requests)
                        logger.error("Feasible Route: %s", feasible_route)
                        raise ValueError("Negative cost encountered in feasible route")
                    min_cost = feasible_route.cost
                    best_route = feasible_route.route
        # yes
        return (
            FeasibleRoute(route=best_route, cost=min_cost)
            if best_route
            else Infeasible(reason="No R-R route permutation satisfied constraints")
        )

    # --- R-V Handler ---
    def _handle_rv_check(
        self,
        availability: VehicleAvailability,
        request_to_add: Request,
        find_best_route: bool,  # <-- NEW PARAM
    ) -> FeasibilityResult:

        passengers_onboard = availability.passengers_onboard
        m = len(passengers_onboard)

        # Throw error if m > 4
        if m > 4:
            # We only implement exhaustive search, which is O(m!)
            raise NotImplementedError(
                f"RoutePlanner exhaustive search does not support m > 4 (got {m})"
            )

        # 1. Define all stops
        p_new = Stop(
            node=request_to_add.origin_node,
            is_pickup=True,
            request_id=request_to_add.id,
        )
        d_new = Stop(
            node=request_to_add.destination_node,
            is_pickup=False,
            request_id=request_to_add.id,
        )

        existing_stops: list[Stop] = list(availability.remaining_stops_to_serve)

        # YOUR CHECK: Assert no existing pickups
        if any(stop.is_pickup for stop in existing_stops):
            # This logic assumes the controller correctly populates the snapshot
            # with only *future* stops. This check is a good assertion.
            raise ValueError("VehicleAvailability snapshot has existing pickup stops.")

        # 2. Build Cache & Request Map
        travel_time_cache = {}
        if find_best_route:
            # Pre-populate ONLY if we're finding the best route (for RTV graph)
            nodes_to_cache = {s.node for s in existing_stops}
            nodes_to_cache.update(
                {p_new.node, d_new.node, availability.available_at_node}
            )
            travel_time_cache = self._build_travel_time_cache(nodes_to_cache)
        else:
            network_path = self._network.get_shortest_path(
                availability.available_at_node, p_new.node
            )
            travel_time_cache[(availability.available_at_node, p_new.node)] = (
                network_path.travel_time if network_path else float("inf")
            )

        all_requests_map = {
            p.original_request.id: p.original_request for p in passengers_onboard
        }
        all_requests_map[request_to_add.id] = request_to_add

        # 3. YOUR CHECK: Lower-bound pickup time
        LB_pickup_time = (
            availability.available_from_time
            + travel_time_cache[(availability.available_at_node, p_new.node)]
        )
        if LB_pickup_time - request_to_add.request_time > OMEGA:
            return Infeasible(reason="Fails lower-bound wait time")

        # 4. Iterate all valid PERMUTATIONS of existing stops
        #    and all valid INSERTIONS of new stops
        best_route = None
        min_cost = float("inf")

        # Outer loop permutes existing stops (O(m!))
        for base_route_tuple in itertools.permutations(existing_stops):

            base_route = list(base_route_tuple)
            n = len(base_route)  # n == m

            # Inner loops do O(n^2) insertions
            for i in range(n + 1):
                route_with_p = base_route[:i] + [p_new] + base_route[i:]

                for j in range(i + 1, len(route_with_p) + 1):
                    route_stops = route_with_p[:j] + [d_new] + route_with_p[j:]

                    # 5. Simulate this route
                    sim_result = self._simulate_route(
                        start_node=availability.available_at_node,
                        start_time=availability.available_from_time,
                        passengers_onboard=passengers_onboard,
                        all_requests_map=all_requests_map,
                        stops_to_visit=route_stops,
                        capacity=availability.capacity,
                        travel_time_cache=travel_time_cache,
                    )

                    if sim_result.is_feasible:
                        # YOUR PARAM: Handle early exit
                        if not find_best_route:
                            return sim_result  # Found one, we're done (for R-V graph)

                        # We are finding the best (for RTV graph)
                        feasible_route = cast(FeasibleRoute, sim_result)
                        if feasible_route.cost < min_cost:
                            min_cost = feasible_route.cost
                            best_route = feasible_route.route

        return (
            FeasibleRoute(route=best_route, cost=min_cost)
            if best_route
            else Infeasible(reason="No R-V route satisfied constraints")
        )

    def _validate_batch_window(
        self, vehicle: VehicleAvailability, requests: set[Request]
    ) -> None:
        times = [req.request_time for req in requests]
        earliest = min(times)
        latest = max(times)
        if latest - earliest > BATCH_DURATION:
            raise ValueError("RoutePlanner received requests outside the batch window")

        if vehicle.available_from_time < earliest:
            raise ValueError(
                "Vehicle availability precedes batch time; snapshot is stale"
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

            current_time += travel_time
            current_node = stop.node
            total_travel_time += travel_time

            req = all_requests_map[stop.request_id]

            if stop.is_pickup:
                # 2. Check Wait Time (Omega)
                wait_time = current_time - req.request_time
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
            stops=stops_to_visit,
            total_travel_time=total_travel_time,
            total_travel_distance=0,  # Not calculated
            total_delay_cost=total_cost,
        )

        return FeasibleRoute(route=final_route, cost=total_cost)

    def _build_travel_time_cache(self, nodes: set[int]) -> dict[tuple[int, int], float]:
        """Pre-calculates all-pairs travel times for a small set of nodes."""
        cache = {}
        for u, v in itertools.product(nodes, repeat=2):
            if (u, v) not in cache:
                if u == v:
                    cache[(u, v)] = 0.0
                else:
                    path = self._network.get_shortest_path(u, v)
                    cache[(u, v)] = path.travel_time if path else float("inf")
        return cache

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
    from itertools import combinations, islice

    from src.components.simulation_data_provider import SimulationDataProvider

    provider = SimulationDataProvider()
    network_service = NetworkXService(provider.graph)
    planner = RoutePlanner(network_service=network_service)
    batch_controller = BatchController(network=network_service)

    def _log_result(label: str, result: FeasibilityResult) -> None:
        if result.is_feasible and isinstance(result, FeasibleRoute):
            logger.info("[%s] cost %.2f", label, result.cost)
            for idx, stop in enumerate(result.route.stops, start=1):
                action = "Pickup" if stop.is_pickup else "Dropoff"
                logger.info(
                    "[%s] %d. %s request %s at node %d",
                    label,
                    idx,
                    action,
                    stop.request_id,
                    stop.node,
                )
        else:
            logger.warning(
                "[%s] infeasible: %s",
                label,
                getattr(result, "reason", "unknown"),
            )

    meta_requests = list(islice(provider.iter_requests(), 100))
    if len(meta_requests) < 2:
        raise SystemExit("Need at least two rider requests for demos")

    request_pool: list[Request] = list(
        batch_controller.convert_meta_requests(meta_requests)
    )

    request_pool.sort(key=lambda r: r.id)
    if len(request_pool) < 2:
        raise SystemExit("Need at least two convertible requests for demos")

    vehicles = list(provider.iter_vehicles())
    if not vehicles:
        raise SystemExit("No vehicles available from provider")
    base_vehicle = vehicles[0]

    def _vehicle_snapshot(start_time: float) -> VehicleAvailability:
        return VehicleAvailability.from_vehicle(base_vehicle, start_time)

    def _windowed_requests(pool: list[Request]):
        for seed in pool:
            start = seed.request_time
            window = [
                req for req in pool if 0 <= req.request_time - start <= BATCH_DURATION
            ]
            if window:
                yield start, window

    def _run_rr_demo() -> None:
        feasible_found = False
        infeasible_found = False

        for start_time, window in _windowed_requests(request_pool):
            if len(window) < 2:
                continue
            vehicle = _vehicle_snapshot(start_time)
            for r1, r2 in combinations(window, 2):
                try:
                    result = planner.check_feasibility(
                        vehicle,
                        {r1, r2},
                        find_best_route=True,
                    )
                except ValueError as exc:
                    logger.warning("[RR] skipped pair %s/%s: %s", r1.id, r2.id, exc)
                    continue
                _log_result("RR", result)
                if result.is_feasible:
                    feasible_found = True
                else:
                    infeasible_found = True
                if feasible_found and infeasible_found:
                    return

        logger.warning(
            "RR demo: unable to find both feasible and infeasible combinations"
        )

    def _run_rv_demo() -> None:
        feasible_found = False
        infeasible_found = False

        for existing_req in request_pool:
            passenger = Passenger(
                original_request=existing_req,
                pickup_time=existing_req.request_time,
            )
            for start_time, window in _windowed_requests(request_pool):
                if not window:
                    continue
                availability = VehicleAvailability(
                    vehicle_id="demo-rv",
                    capacity=2,
                    available_at_node=existing_req.origin_node,
                    available_from_time=start_time,
                    passengers_onboard=frozenset({passenger}),
                    remaining_stops_to_serve=(
                        Stop(
                            node=existing_req.destination_node,
                            is_pickup=False,
                            request_id=existing_req.id,
                        ),
                    ),
                )
                for new_req in window:
                    if new_req.id == existing_req.id:
                        continue
                    try:
                        result = planner.check_feasibility(
                            availability,
                            {new_req},
                            find_best_route=True,
                        )
                    except ValueError as exc:
                        logger.warning(
                            "[RV] skipped pair %s/%s: %s",
                            existing_req.id,
                            new_req.id,
                            exc,
                        )
                        continue
                    _log_result("RV", result)
                    if result.is_feasible:
                        feasible_found = True
                    else:
                        infeasible_found = True
                    if feasible_found and infeasible_found:
                        return

        logger.warning(
            "RV demo: unable to find both feasible and infeasible combinations"
        )

    _run_rr_demo()
    _run_rv_demo()
