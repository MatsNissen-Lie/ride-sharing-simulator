"""Batch controller: fleet projection, commitments and statistics."""

import pytest

from config.config import OMEGA
from src.components.batch_controller import BatchController
from src.interfaces import Route, Stop, Vehicle
from src.interfaces.data_models import Assignment, AssignmentResult, MetaRequest


def make_controller(network, nodes):
    fleet = [
        Vehicle(
            id=f"v{k}",
            capacity=4,
            passengers_onboard=set(),
            last_visited_node=node,
            last_visited_time=0.0,
        )
        for k, node in enumerate(nodes)
    ]
    return BatchController(network, fleet)


def assign(controller, t_now, vehicle_id, request):
    route = Route(
        (Stop(request.origin_node, True, request.id), Stop(request.destination_node, False, request.id)),
        0.0, 0.0, 0.0,
    )
    controller.commit_assignment(
        t_now,
        AssignmentResult(
            {Assignment(vehicle_id, frozenset({request.id}), route, 0.0)}, set(), 0.0, True
        ),
    )


def test_ride_is_replayed_on_the_network_and_counted(network):
    controller = make_controller(network, [0])
    (req,) = controller.prepare_batch(10.0, [MetaRequest("r", 14, 35, 5.0)])[0]
    assign(controller, 10.0, "v0", req)

    pickup = 10.0 + network.get_travel_time(0, 14)
    dropoff = pickup + network.get_travel_time(14, 35)
    controller.prepare_batch(dropoff + 1, [])
    stats = controller.get_and_reset_metrics()

    assert stats.delivered_count == 1
    assert stats.avg_wait_time == pytest.approx(pickup - 5.0)
    assert stats.avg_total_delay == pytest.approx(dropoff - 5.0 - network.get_travel_time(14, 35))
    vehicle = controller._fleet[0]
    assert vehicle.is_idle and vehicle.last_visited_node == 35


def test_vehicle_does_not_gain_time_when_reassigned_mid_leg(network):
    """A reassigned vehicle starts its new route from its snapshot, not from t_now."""
    controller = make_controller(network, [0])
    (far,) = controller.prepare_batch(10.0, [MetaRequest("far", 35, 0, 5.0)])[0]
    assign(controller, 10.0, "v0", far)

    (snapshot,) = controller.prepare_batch(20.0, [])[1]
    assert snapshot.available_from_time >= 20.0
    vehicle = controller._fleet[0]
    assert vehicle.last_visited_time == snapshot.available_from_time

    assign(controller, 20.0, "v0", controller._request_index["far"])
    assert vehicle.route_assignment_time == snapshot.available_from_time


def test_idle_vehicle_snapshot_uses_its_arrival_time(network):
    controller = make_controller(network, [0])
    vehicle = controller._fleet[0]
    vehicle.last_visited_time = 500.0  # still arriving at node 0
    (snapshot,) = controller.prepare_batch(100.0, [])[1]
    assert snapshot.available_from_time == 500.0


def test_vehicle_that_has_not_departed_stays_put(network):
    controller = make_controller(network, [0])
    vehicle = controller._fleet[0]
    vehicle.current_route = Route((Stop(35, False, "rebalance_x"),), 0.0, 0.0, 0.0)
    vehicle.route_assignment_time = 300.0  # starts in the future
    (snapshot,) = controller.prepare_batch(100.0, [])[1]
    assert snapshot.available_at_node == 0
    assert snapshot.available_from_time == 300.0


def test_unserved_request_expires_after_omega(network):
    controller = make_controller(network, [0])
    controller.prepare_batch(10.0, [MetaRequest("r", 14, 35, 5.0)])
    requests, _ = controller.prepare_batch(5.0 + OMEGA + 1, [])
    assert not requests
    assert controller.get_and_reset_metrics().expired_count == 1


def test_assigned_request_gets_promised_pickup_time(network):
    controller = make_controller(network, [0])
    (req,) = controller.prepare_batch(10.0, [MetaRequest("r", 35, 0, 5.0)])[0]
    assign(controller, 10.0, "v0", req)
    requests, _ = controller.prepare_batch(20.0, [])
    (promised,) = requests
    expected = 10.0 + network.get_travel_time(0, 35)
    assert promised.promised_pickup_time == pytest.approx(expected)
    assert promised.latest_pickup_time == pytest.approx(expected)
