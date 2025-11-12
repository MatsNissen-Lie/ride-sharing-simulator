"""Focused test coverage for src.components.route_planner.RoutePlanner."""

from dataclasses import dataclass

import pytest

from config import DELTA, OMEGA
from src.components.route_planner import RoutePlanner
from src.interfaces import (
    FeasibleRoute,
    NetworkPath,
    Passenger,
    Request,
    Stop,
    VehicleAvailability,
)
from src.interfaces.services import INetworkService


class RecordingNetworkService(INetworkService):
    """Simple INetworkService stub that records how often each pair is requested."""

    def __init__(
        self,
        *,
        travel_times: dict[tuple[int, int], float] | None = None,
        default_time: float = 100.0,
    ) -> None:
        self.call_counts: dict[tuple[int, int], int] = {}
        self._travel_times = travel_times or {}
        self._default_time = default_time

    def get_shortest_path(self, origin_node: int, destination_node: int) -> NetworkPath:  # type: ignore[override]
        key = (origin_node, destination_node)
        self.call_counts[key] = self.call_counts.get(key, 0) + 1
        travel = self._travel_times.get(key)
        if travel is None:
            travel = (
                0.0 if origin_node == destination_node else float(self._default_time)
            )
        return NetworkPath(
            nodes=[origin_node, destination_node],
            travel_time=travel,
            travel_distance=travel,
        )


def make_request(
    rid: str,
    *,
    origin: int,
    destination: int,
    request_time: float = 0.0,
    earliest_dropoff: float = 0.0,
    promised_pickup: float | None = None,
) -> Request:
    return Request(
        id=rid,
        origin_node=origin,
        destination_node=destination,
        request_time=request_time,
        earliest_dropoff_time=earliest_dropoff,
        promised_pickup_time=promised_pickup,
    )


def make_vehicle_availability(
    *,
    vehicle_id: str = "veh-1",
    capacity: int = 2,
    available_node: int = 0,
    available_time: float = 0.0,
    passengers: frozenset[Passenger] | None = None,
    remaining_stops: tuple[Stop, ...] = (),
) -> VehicleAvailability:
    return VehicleAvailability(
        vehicle_id=vehicle_id,
        capacity=capacity,
        available_at_node=available_node,
        available_from_time=available_time,
        passengers_onboard=passengers or frozenset(),
        remaining_stops_to_serve=remaining_stops,
    )


def stop_for(request: Request, *, pickup: bool, node: int | None = None) -> Stop:
    return Stop(
        node=(
            node
            if node is not None
            else (request.origin_node if pickup else request.destination_node)
        ),
        is_pickup=pickup,
        request_id=request.id,
    )


def make_passenger(request: Request, pickup_time: float = 0.0) -> Passenger:
    return Passenger(original_request=request, pickup_time=pickup_time)


def test_rr_prefers_low_delay_route_and_uses_cache() -> None:
    nodes = [0, 1, 2, 3, 4]
    rr_times = {(u, v): (0.0 if u == v else 100.0) for u in nodes for v in nodes}
    rr_times.update(
        {
            (1, 2): 5.0,
            (2, 3): 10.0,
            (3, 4): 5.0,
            (4, 1): 10.0,
            (1, 3): 40.0,
            (3, 1): 40.0,
        }
    )
    network = RecordingNetworkService(travel_times=rr_times)
    planner = RoutePlanner(network)

    r1 = make_request("r1", origin=1, destination=2, earliest_dropoff=5.0)
    r2 = make_request("r2", origin=3, destination=4, earliest_dropoff=5.0)
    vehicle = make_vehicle_availability(capacity=2)

    result = planner.check_feasibility(vehicle, {r1, r2})

    assert result.is_feasible
    assert isinstance(result, FeasibleRoute)
    assert result.route.total_delay_cost == pytest.approx(15.0)
    assert result.route.total_travel_time == pytest.approx(20.0)

    stops = result.route.stops
    assert len(stops) == 4
    assert stops[0].is_pickup and not stops[1].is_pickup
    assert stops[2].is_pickup and not stops[3].is_pickup
    assert stops[0].request_id == stops[1].request_id
    assert stops[2].request_id == stops[3].request_id
    assert {stops[0].request_id, stops[2].request_id} == {r1.id, r2.id}

    total_calls = sum(network.call_counts.values())
    assert total_calls == 20  # 5 nodes => 5^2 minus self loops


def test_rr_rejects_capacity_lt_two() -> None:
    network = RecordingNetworkService()
    planner = RoutePlanner(network)

    r1 = make_request("r1", origin=1, destination=2, earliest_dropoff=5.0)
    r2 = make_request("r2", origin=3, destination=4, earliest_dropoff=5.0)
    vehicle = make_vehicle_availability(capacity=1)

    result = planner.check_feasibility(vehicle, {r1, r2})

    assert not result.is_feasible
    assert "capacity" in getattr(result, "reason", "")
    assert network.call_counts == {}


def test_rv_insertion_respects_existing_drop_order() -> None:
    rv_times = {
        (0, 1): 1.0,
        (1, 10): 1.0,
        (10, 2): 1.0,
        (2, 20): 1.0,
        (1, 2): 8.0,
        (0, 10): 8.0,
        (10, 20): 8.0,
        (20, 2): 8.0,
        (2, 10): 8.0,
    }
    network = RecordingNetworkService(travel_times=rv_times)
    planner = RoutePlanner(network)

    old_a = make_request(
        "old-a", origin=50, destination=10, earliest_dropoff=1.0, request_time=-100.0
    )
    old_b = make_request(
        "old-b", origin=60, destination=20, earliest_dropoff=3.0, request_time=-100.0
    )
    passenger_a = make_passenger(old_a, pickup_time=-90.0)
    passenger_b = make_passenger(old_b, pickup_time=-80.0)
    remaining = (
        stop_for(old_a, pickup=False, node=10),
        stop_for(old_b, pickup=False, node=20),
    )
    vehicle = make_vehicle_availability(
        capacity=3,
        passengers=frozenset({passenger_a, passenger_b}),
        remaining_stops=remaining,
    )

    new_request = make_request("new", origin=1, destination=2, earliest_dropoff=2.0)

    result = planner.check_feasibility(vehicle, {new_request})

    assert result.is_feasible
    assert isinstance(result, FeasibleRoute)

    stop_signature = [(s.request_id, s.is_pickup, s.node) for s in result.route.stops]
    assert stop_signature == [
        (new_request.id, True, new_request.origin_node),
        (old_a.id, False, 10),
        (new_request.id, False, new_request.destination_node),
        (old_b.id, False, 20),
    ]
    assert result.route.total_delay_cost == pytest.approx(3.0)


def test_rv_rejects_pickup_too_far() -> None:
    travel_times = {(0, 5): OMEGA + 50.0}
    network = RecordingNetworkService(travel_times=travel_times)
    planner = RoutePlanner(network)

    vehicle = make_vehicle_availability(capacity=2)
    request = make_request("req", origin=5, destination=6, earliest_dropoff=10.0)

    result = planner.check_feasibility(vehicle, {request})

    assert not result.is_feasible
    assert getattr(result, "reason", "") == "Too long to reach new pickup"


@pytest.mark.parametrize(
    "promised_pickup, travel_time, expected_reason",
    [
        (10.0, 15.0, "Rematch would worsen wait"),
        (None, float(OMEGA + 10.0), "Violates max wait"),
    ],
)
def test_simulate_route_detects_wait_and_promised_pickup_violations(
    promised_pickup: float | None, travel_time: float, expected_reason: str
) -> None:
    planner = RoutePlanner(RecordingNetworkService())
    request = make_request(
        "wait",
        origin=1,
        destination=2,
        earliest_dropoff=50.0,
        promised_pickup=(
            (0.0 + promised_pickup) if promised_pickup is not None else None
        ),
    )
    stops = [
        stop_for(request, pickup=True),
        stop_for(request, pickup=False),
    ]
    travel_time_cache = {
        (0, request.origin_node): travel_time,
        (request.origin_node, request.destination_node): 1.0,
    }

    result = planner._simulate_route(
        start_node=0,
        start_time=0.0,
        passengers_onboard=frozenset(),
        all_requests_map={request.id: request},
        stops_to_visit=stops,
        capacity=2,
        travel_time_cache=travel_time_cache,
    )

    assert not result.is_feasible
    assert expected_reason in getattr(result, "reason", "")


@dataclass
class SimulationScenario:
    start_node: int
    start_time: float
    passengers: frozenset[Passenger]
    all_requests: dict[str, Request]
    stops: list[Stop]
    capacity: int
    travel_time_cache: dict[tuple[int, int], float]
    expected_reason: str


def _capacity_violation_scenario() -> SimulationScenario:
    req_a = make_request("ca", origin=1, destination=2, earliest_dropoff=5.0)
    req_b = make_request("cb", origin=3, destination=4, earliest_dropoff=5.0)
    stops = [
        stop_for(req_a, pickup=True),
        stop_for(req_b, pickup=True),
        stop_for(req_a, pickup=False),
        stop_for(req_b, pickup=False),
    ]
    travel_time_cache = {
        (0, req_a.origin_node): 1.0,
        (req_a.origin_node, req_b.origin_node): 1.0,
        (req_b.origin_node, req_a.destination_node): 1.0,
        (req_a.destination_node, req_b.destination_node): 1.0,
    }
    return SimulationScenario(
        start_node=0,
        start_time=0.0,
        passengers=frozenset(),
        all_requests={req_a.id: req_a, req_b.id: req_b},
        stops=stops,
        capacity=1,
        travel_time_cache=travel_time_cache,
        expected_reason="Violates capacity",
    )


def _drop_before_pickup_scenario() -> SimulationScenario:
    req = make_request("drop-first", origin=10, destination=11, earliest_dropoff=5.0)
    stops = [stop_for(req, pickup=False)]
    travel_time_cache = {(0, req.destination_node): 1.0}
    return SimulationScenario(
        start_node=0,
        start_time=0.0,
        passengers=frozenset(),
        all_requests={req.id: req},
        stops=stops,
        capacity=2,
        travel_time_cache=travel_time_cache,
        expected_reason="Dropoff before pickup",
    )


def _delay_violation_scenario() -> SimulationScenario:
    req = make_request("delay", origin=5, destination=9, earliest_dropoff=100.0)
    stops = [
        stop_for(req, pickup=True),
        stop_for(req, pickup=False),
    ]
    travel_time_cache = {
        (req.origin_node, req.origin_node): 0.0,
        (req.origin_node, req.destination_node): req.earliest_dropoff_time
        + DELTA
        + 1.0,
    }
    return SimulationScenario(
        start_node=req.origin_node,
        start_time=0.0,
        passengers=frozenset(),
        all_requests={req.id: req},
        stops=stops,
        capacity=2,
        travel_time_cache=travel_time_cache,
        expected_reason="Violates max delay",
    )


@pytest.mark.parametrize(
    "scenario",
    [
        _capacity_violation_scenario(),
        _drop_before_pickup_scenario(),
        _delay_violation_scenario(),
    ],
    ids=["capacity", "drop-before-pickup", "delay"],
)
def test_simulate_route_detects_additional_constraints(
    scenario: SimulationScenario,
) -> None:
    planner = RoutePlanner(RecordingNetworkService())

    result = planner._simulate_route(
        start_node=scenario.start_node,
        start_time=scenario.start_time,
        passengers_onboard=scenario.passengers,
        all_requests_map=scenario.all_requests,
        stops_to_visit=scenario.stops,
        capacity=scenario.capacity,
        travel_time_cache=scenario.travel_time_cache,
    )

    assert not result.is_feasible
    assert scenario.expected_reason in getattr(result, "reason", "")
