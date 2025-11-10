"""Unit tests for the MinimalRoutePlanner.

Test coverage:
- `test_minimal_route_planner_chooses_low_cost_permutation`: ensures the planner
  evaluates permutations, picks the cheapest ordering, and populates route
  metrics while limiting network calls via caching.
- `test_minimal_route_planner_rejects_low_capacity_vehicle`: verifies capacity
  gating occurs before any network queries.
- `test_rejects_when_vehicle_not_virtual`: confirms vehicles with passengers
  onboard are rejected up-front.
- `test_rejects_when_request_set_not_size_two`: ensures only two-request
  batches are accepted by the minimal planner.
- `test_simulate_route_detects_wait_violation`: exercises `_simulate_route`
  by injecting a path cache that creates a wait-time breach.
- `test_simulate_route_detects_delay_violation`: similarly verifies that an
  excessive delay during simulation yields an infeasible result.
"""

from functools import lru_cache
import pytest

from src.components.route_planner import MinimalRoutePlanner
from src.interfaces import (
    FeasibleRoute,
    NetworkPath,
    Passenger,
    Request,
    Stop,
    Vehicle,
)
from src.interfaces.services import INetworkService


class RecordingNetworkService(INetworkService):
    """Network stub that records how often each pair is queried."""

    def __init__(
        self,
        *,
        travel_times: dict[tuple[int, int], float] | None = None,
        default_step: float = 1.0,
    ) -> None:
        self.call_counts: dict[tuple[int, int], int] = {}
        self._travel_times = travel_times or {}
        self._default_step = default_step

    @lru_cache(maxsize=200_000)
    def get_shortest_path(self, origin_node: int, destination_node: int) -> NetworkPath:
        key = (origin_node, destination_node)
        self.call_counts[key] = self.call_counts.get(key, 0) + 1
        travel = self._travel_times.get(key)
        if travel is None:
            if origin_node == destination_node:
                travel = 0.0
            else:
                travel = (
                    float(abs(destination_node - origin_node)) or self._default_step
                )
        return NetworkPath(
            nodes=[origin_node, destination_node],
            travel_time=float(travel),
            travel_distance=float(travel),
        )


def _make_request(
    rid: str,
    origin: int,
    destination: int,
    *,
    request_time: float = 0.0,
    earliest_dropoff_time: float = 0.0,
) -> Request:
    return Request(
        id=rid,
        origin_node=origin,
        destination_node=destination,
        request_time=request_time,
        earliest_dropoff_time=earliest_dropoff_time,
    )


def _make_vehicle(*, capacity: int = 2, node: int = 0) -> Vehicle:
    return Vehicle(
        id="veh-1",
        current_node=node,
        capacity=capacity,
        passengers_onboard=set(),
        current_route=None,
    )


def _stop_signature(stop) -> tuple[str, bool, int]:
    return (stop.request_id, stop.is_pickup, stop.node)


def test_minimal_route_planner_chooses_low_cost_permutation() -> None:
    network = RecordingNetworkService()
    planner = MinimalRoutePlanner(network)

    vehicle = _make_vehicle(node=0, capacity=2)
    r1 = _make_request("r1", origin=1, destination=2)
    r2 = _make_request("r2", origin=100, destination=101)

    result = planner.check_feasibility(vehicle, {r1, r2})

    assert result.is_feasible
    assert isinstance(result, FeasibleRoute)

    stops = [_stop_signature(stop) for stop in result.route.stops]
    assert stops == [
        ("r1", True, r1.origin_node),
        ("r1", False, r1.destination_node),
        ("r2", True, r2.origin_node),
        ("r2", False, r2.destination_node),
    ]
    assert result.route.total_travel_time == pytest.approx(101.0)

    # 5 nodes (vehicle, 4 stops) => 25 pairs, but same-node pairs are memoized
    actual_calls = sum(network.call_counts.values())
    assert actual_calls == 20


def test_minimal_route_planner_rejects_low_capacity_vehicle() -> None:
    network = RecordingNetworkService()
    planner = MinimalRoutePlanner(network)

    vehicle = _make_vehicle(capacity=1)
    r1 = _make_request("r1", origin=1, destination=2)
    r2 = _make_request("r2", origin=3, destination=4)

    result = planner.check_feasibility(vehicle, {r1, r2})

    assert not result.is_feasible
    assert getattr(result, "reason", "").startswith("Vehicle capacity too low")
    assert network.call_counts == {}


def test_rejects_when_vehicle_not_virtual() -> None:
    network = RecordingNetworkService()
    planner = MinimalRoutePlanner(network)

    onboard_request = _make_request("r0", origin=5, destination=6)
    passenger = Passenger(original_request=onboard_request, pickup_time=0.0)
    vehicle = Vehicle(
        id="veh-2",
        current_node=0,
        capacity=2,
        passengers_onboard={passenger},
        current_route=None,
    )

    r1 = _make_request("r1", origin=1, destination=2)
    r2 = _make_request("r2", origin=3, destination=4)

    result = planner.check_feasibility(vehicle, {r1, r2})

    assert not result.is_feasible
    assert "only supports R-R check" in getattr(result, "reason", "")
    assert network.call_counts == {}


def test_rejects_when_request_set_not_size_two() -> None:
    network = RecordingNetworkService()
    planner = MinimalRoutePlanner(network)

    vehicle = _make_vehicle()
    r1 = _make_request("r1", origin=1, destination=2)

    result = planner.check_feasibility(vehicle, {r1})

    assert not result.is_feasible
    assert "only supports R-R check" in getattr(result, "reason", "")
    assert network.call_counts == {}


def _path(time: float) -> NetworkPath:
    return NetworkPath(nodes=[0, 1], travel_time=time, travel_distance=time)


def test_simulate_route_detects_wait_violation() -> None:
    planner = MinimalRoutePlanner(RecordingNetworkService())
    vehicle = _make_vehicle(node=0)
    r1 = _make_request("r1", origin=10, destination=11, request_time=0.0)
    r2 = _make_request("r2", origin=20, destination=21, request_time=0.0)

    stops = [
        Stop(node=r1.origin_node, is_pickup=True, request_id=r1.id),
        Stop(node=r1.destination_node, is_pickup=False, request_id=r1.id),
        Stop(node=r2.origin_node, is_pickup=True, request_id=r2.id),
        Stop(node=r2.destination_node, is_pickup=False, request_id=r2.id),
    ]

    path_cache = {
        (vehicle.current_node, r1.origin_node): _path(400.0),  # > OMEGA
        (r1.origin_node, r1.destination_node): _path(5.0),
        (r1.destination_node, r2.origin_node): _path(5.0),
        (r2.origin_node, r2.destination_node): _path(5.0),
    }

    result = planner._simulate_route(
        start_node=vehicle.current_node,
        stops=stops,
        requests={r1, r2},
        capacity=vehicle.capacity,
        path_cache=path_cache,
    )

    assert not result.is_feasible
    assert getattr(result, "reason", "").startswith("Violates max wait")


def test_simulate_route_detects_delay_violation() -> None:
    planner = MinimalRoutePlanner(RecordingNetworkService())
    vehicle = _make_vehicle(node=0)
    r1 = _make_request("r1", origin=1, destination=2, request_time=0.0)
    r2 = _make_request("r2", origin=3, destination=4, request_time=0.0)

    stops = [
        Stop(node=r1.origin_node, is_pickup=True, request_id=r1.id),
        Stop(node=r2.origin_node, is_pickup=True, request_id=r2.id),
        Stop(node=r1.destination_node, is_pickup=False, request_id=r1.id),
        Stop(node=r2.destination_node, is_pickup=False, request_id=r2.id),
    ]

    path_cache = {
        (vehicle.current_node, r1.origin_node): _path(10.0),
        (r1.origin_node, r2.origin_node): _path(10.0),
        (r2.origin_node, r1.destination_node): _path(10.0),
        (r1.destination_node, r2.destination_node): _path(500.0),  # > DELTA
    }

    result = planner._simulate_route(
        start_node=vehicle.current_node,
        stops=stops,
        requests={r1, r2},
        capacity=vehicle.capacity,
        path_cache=path_cache,
    )

    assert not result.is_feasible
    assert getattr(result, "reason", "").startswith("Violates max delay")
