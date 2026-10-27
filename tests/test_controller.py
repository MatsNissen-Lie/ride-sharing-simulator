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
    assert promised.latest_pickup_time == pytest.approx(expected, abs=1e-2)


def test_promised_rider_stays_feasible_for_the_same_vehicle(network):
    """Re-planning recomputes the promised arrival from the travel-time matrix,
    which can differ from the replayed path in the last digits."""
    from src.components.trip_solver import AStarTripSolver

    controller = make_controller(network, [0])
    (req,) = controller.prepare_batch(10.0, [MetaRequest("r", 35, 0, 5.0)])[0]
    assign(controller, 10.0, "v0", req)
    (promised,), (snapshot,) = controller.prepare_batch(20.0, [])

    matrix = network.get_l1_cache().copy()
    matrix[snapshot.available_at_node, promised.origin_node] += 1e-7  # float noise
    assert AStarTripSolver(matrix).solve([promised], snapshot).is_feasible


def test_rebalancing_vehicle_keeps_its_destination_until_arrival(network):
    from src.interfaces.data_models import AssignmentResult

    controller = make_controller(network, [0])
    controller.prepare_batch(10.0, [])
    controller.apply_rebalancing({"v0": Route((Stop.rebalancing(35, "x"),), 0, 0, 0)}, 10.0)
    arrival = 10.0 + network.get_travel_time(0, 35)
    empty = AssignmentResult(set(), set(), 0.0, True)

    t, batches_en_route = 10.0, 0
    while True:
        t += 10.0
        _, (snapshot,) = controller.prepare_batch(t, [])
        idle, _, en_route = controller.get_rebalancing_candidates(empty, {snapshot})
        if snapshot.available_at_node == 35:
            break
        assert [s.node for s in snapshot.remaining_stops_to_serve] == [35]
        assert idle == [] and en_route == [snapshot]
        batches_en_route += 1
    assert batches_en_route >= 1
    # At (or arriving at) the destination: idle there from the arrival time.
    _, (snapshot,) = controller.prepare_batch(t + 10.0, [])
    idle, _, en_route = controller.get_rebalancing_candidates(empty, {snapshot})
    assert snapshot.available_at_node == 35 and snapshot.remaining_stops_to_serve == ()
    assert snapshot.available_from_time == pytest.approx(max(arrival, t + 10.0))
    assert idle == [snapshot] and en_route == []


def test_rebalancing_vehicle_can_be_assigned_a_rider_on_the_way(network):
    controller = make_controller(network, [0])
    controller.prepare_batch(10.0, [])
    controller.apply_rebalancing({"v0": Route((Stop.rebalancing(35, "x"),), 0, 0, 0)}, 10.0)
    (req,), (snapshot,) = controller.prepare_batch(20.0, [MetaRequest("r", 14, 7, 15.0)])
    assert snapshot.remaining_stops_to_serve  # still heading to 35
    assign(controller, 20.0, "v0", req)
    route = controller._fleet[0].current_route
    assert [s.request_id for s in route.stops] == ["r", "r"]  # rebalancing replaced by the ride


def test_rebalancing_vehicle_can_be_redirected_or_stopped(network):
    controller = make_controller(network, [0])
    controller.prepare_batch(10.0, [])
    controller.apply_rebalancing({"v0": Route((Stop.rebalancing(35, "x"),), 0, 0, 0)}, 10.0)
    _, (snapshot,) = controller.prepare_batch(20.0, [])
    controller.apply_rebalancing({"v0": Route((Stop.rebalancing(5, "y"),), 0, 0, 0)}, 20.0)
    assert [s.node for s in controller._fleet[0].current_route.stops] == [5]

    _, (snapshot,) = controller.prepare_batch(30.0, [])
    controller.apply_rebalancing({"v0": Route((), 0, 0, 0)}, 30.0)
    _, (stopped,) = controller.prepare_batch(40.0, [])
    assert stopped.remaining_stops_to_serve == ()
    assert stopped.available_at_node == snapshot.available_at_node
    assert stopped.available_from_time == max(snapshot.available_from_time, 40.0)


def test_rebalancing_never_overwrites_a_vehicle_with_riders(network):
    controller = make_controller(network, [0])
    (req,), _ = controller.prepare_batch(10.0, [MetaRequest("r", 14, 7, 5.0)])
    assign(controller, 10.0, "v0", req)
    controller.apply_rebalancing({"v0": Route((), 0, 0, 0)}, 10.0)
    assert [s.request_id for s in controller._fleet[0].current_route.stops] == ["r", "r"]
