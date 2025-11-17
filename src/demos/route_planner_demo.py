"""Standalone demo routines for RoutePlanner."""

from itertools import combinations, islice
from typing import Iterable, cast

from config import BATCH_DURATION, get_logger
from src.components.batch_controller import BatchController
from src.components.network_service import NetworkXService
from src.components.route_planner import RoutePlanner
from src.components.simulation_data_provider import SimulationDataProvider
from src.interfaces import (
    FeasibilityResult,
    FeasibleRoute,
    Passenger,
    Request,
    Stop,
    VehicleAvailability,
    Route,
)

logger = get_logger(__name__)


def _windowed_requests(pool: list[Request]):
    for seed in pool:
        seed_time = seed.request_time
        window = [
            req for req in pool if 0 <= req.request_time - seed_time <= BATCH_DURATION
        ]
        if window:
            start_time = max(req.request_time for req in window)
            yield start_time, window


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


def _chk(
    planner: RoutePlanner,
    availability: VehicleAvailability,
    reqs: set[Request],
    *,
    find_best_route: bool,
) -> FeasibilityResult:
    return planner.check_feasibility(
        reqs,
        availability,
        find_best_route=find_best_route,
    )


def _rr_feasible(
    planner: RoutePlanner, requests: set[Request]
) -> tuple[bool, Route | None]:

    latest_req = max(requests, key=lambda r: r.request_time)

    virtual_vehicle = VehicleAvailability(
        vehicle_id="virt",
        capacity=2,
        available_at_node=latest_req.origin_node,
        available_from_time=latest_req.request_time,
        passengers_onboard=frozenset(),
        remaining_stops_to_serve=tuple(),
    )
    result = _chk(
        planner,
        virtual_vehicle,
        requests,
        find_best_route=False,
    )
    if not result.is_feasible:
        return False, None
    result = cast(FeasibleRoute, result)
    return bool(result.is_feasible), result.route


def _mk_availability_from_pair(
    base_vehicle: VehicleAvailability,
    start_time: float,
    pair: tuple[Request, Request],  # last picked up is the back one
) -> VehicleAvailability:
    first, second = pair
    passengers = frozenset(
        {
            Passenger(original_request=first, pickup_time=first.request_time),
            Passenger(original_request=second, pickup_time=second.request_time),
        }
    )
    remaining_stops = (
        Stop(node=first.destination_node, is_pickup=False, request_id=first.id),
        Stop(node=second.destination_node, is_pickup=False, request_id=second.id),
    )
    return VehicleAvailability(
        vehicle_id=f"demo-{first.id}-{second.id}",
        capacity=base_vehicle.capacity,
        available_at_node=second.origin_node,
        available_from_time=start_time,
        passengers_onboard=passengers,
        remaining_stops_to_serve=remaining_stops,
    )


def _run_rr_demo(
    planner: RoutePlanner,
    request_pool: list[Request],
) -> None:
    feasible_found = False
    infeasible_found = False

    for _, window in _windowed_requests(request_pool):
        if len(window) < 2:
            continue
        for r1, r2 in combinations(window, 2):
            try:
                result = planner.check_feasibility(
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

    logger.warning("RR demo: unable to find both feasible and infeasible combinations")


def _run_rv_demo(
    planner: RoutePlanner,
    request_pool: list[Request],
) -> None:
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
                        {new_req},
                        availability,
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

    logger.warning("RV demo: unable to find both feasible and infeasible combinations")


def _run_rtv_demo(
    planner: RoutePlanner,
    base_vehicle: VehicleAvailability,
    request_pool: list[Request],
) -> None:
    feasible_one_found = False
    feasible_two_found = False
    infeasible_found = False

    previous_window: list[Request] | None = None

    for start_time, window in _windowed_requests(request_pool):
        ordered_window = sorted(window, key=lambda r: r.request_time)

        if previous_window is None:
            previous_window = ordered_window
            continue

        prior = sorted(previous_window, key=lambda r: r.request_time)
        if len(prior) < 2:
            previous_window = ordered_window
            continue

        feasible_prior_pairs: list[tuple[Request, Request]] = []
        for r1, r2 in combinations(prior, 2):
            check_with_order, route = _rr_feasible(planner, {r1, r2})
            if route is not None:
                if len(route.stops) == 4:
                    # who is picked up last?
                    last_pickup = [stop for stop in route.stops if stop.is_pickup][-1]
                    if last_pickup.request_id == r1.id:
                        feasible_prior_pairs.append((r2, r1))
                    else:
                        feasible_prior_pairs.append((r1, r2))
                else:
                    raise RuntimeError(
                        "Unexpected route length for two-request RR feasibility"
                    )

        if not feasible_prior_pairs:
            previous_window = ordered_window
            continue

        for pair in feasible_prior_pairs:
            availability = _mk_availability_from_pair(base_vehicle, start_time, pair)

            singly_feasible: list[Request] = []
            for new_req in ordered_window:
                try:
                    res = _chk(
                        planner,
                        availability,
                        {new_req},
                        find_best_route=True,
                    )
                except Exception as exc:
                    logger.warning("[RTV1] skip %s: %s", new_req.id, exc)
                    continue
                _log_result("RTV1", res)
                if res.is_feasible:
                    feasible_one_found = True
                    singly_feasible.append(new_req)
                else:
                    infeasible_found = True

                if feasible_one_found and feasible_two_found and infeasible_found:
                    return

            for r_a, r_b in combinations(singly_feasible, 2):
                try:
                    res2 = _chk(
                        planner,
                        availability,
                        {r_a, r_b},
                        find_best_route=True,
                    )
                except Exception as exc:
                    logger.warning("[RTV2] skip %s/%s: %s", r_a.id, r_b.id, exc)
                    continue
                _log_result("RTV2", res2)
                if res2.is_feasible:
                    feasible_two_found = True
                else:
                    infeasible_found = True

                if feasible_one_found and feasible_two_found and infeasible_found:
                    return

        previous_window = ordered_window

    logger.warning("RTV demo: unable to find both feasible and infeasible combinations")


def run_demo() -> None:
    provider = SimulationDataProvider()
    network_service = NetworkXService(provider.graph)
    planner = RoutePlanner(network_service=network_service)
    batch_controller = BatchController(network=network_service)

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
    base_availability = VehicleAvailability.from_vehicle(base_vehicle, 0.0)

    _run_rr_demo(planner, request_pool)
    _run_rv_demo(planner, request_pool)
    _run_rtv_demo(planner, base_availability, request_pool)


if __name__ == "__main__":
    run_demo()
