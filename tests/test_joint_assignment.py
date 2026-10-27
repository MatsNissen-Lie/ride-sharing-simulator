"""Joint assignment and rebalancing ILP."""

import random

import numpy as np
import pytest

from src.components.assignment_solver import HighsAssignmentSolver
from src.components.graph_builder.rtv_trip_centric import TripCentricRTVGraphBuilder
from src.components.graph_builder.rv_builder import RVGraphBuilder
from src.components.joint_assignment_solver import JointAssignmentSolver, ZeroValue
from src.components.objectives import DelayObjective
from src.components.rebalance_solver import NoRebalancingSolver
from src.components.trip_solver import AStarTripSolver
from src.interfaces import IVehicleValue, VehicleAvailability
from tests.conftest import make_request, random_requests, random_vehicle
from tests.test_simulation import run

T_NOW = 10_000.0


class Zones:
    """Two zones on the synthetic grid: left half (1) and right half (2)."""

    zone_ids = np.array([1, 2])
    center_node = {1: 14, 2: 21}


class NodeValue(IVehicleValue):
    """A fixed value per node, the same at every time."""

    def __init__(self, per_node):
        self.per_node = per_node

    def values(self, nodes, times, t_now):
        return np.array([self.per_node.get(int(n), 0.0) for n in nodes])


def idle(vid, node, t=T_NOW):
    return VehicleAvailability(vid, 4, node, t, frozenset(), ())


def instance(network, vehicles, requests):
    trip_solver = AStarTripSolver(network.get_l1_cache())
    rv = RVGraphBuilder(network, trip_solver).build_graph(set(requests), set(vehicles))
    return TripCentricRTVGraphBuilder(trip_solver).build_graph(rv, set(vehicles), set(requests))


def joint(network, value, **kwargs):
    return JointAssignmentSolver(DelayObjective(), network, Zones(), value, **kwargs)


@pytest.mark.parametrize("seed", range(5))
def test_zero_value_is_the_papers_ilp(network, seed):
    rng = random.Random(seed)
    vehicles = {random_vehicle(network, rng, f"v{k}", T_NOW, rng.choice([0, 0, 1, 2])) for k in range(4)}
    requests = set(random_requests(network, rng, 6, T_NOW))
    rtv = instance(network, vehicles, requests)

    plain = HighsAssignmentSolver().solve(rtv, requests, vehicles, 30, T_NOW)
    combined = joint(network, ZeroValue()).solve(rtv, requests, vehicles, 30, T_NOW)

    assert combined.total_cost == pytest.approx(plain.total_cost)
    assert combined.rebalancing == {}


def test_value_moves_idle_vehicles_towards_it(network):
    vehicles = {idle("a", 0), idle("b", 1)}  # both in the left zone
    solver = joint(network, NodeValue({21: 1e4}), max_move=1e6)
    result = solver.solve(instance(network, vehicles, []), set(), vehicles, 30, T_NOW)
    assert set(result.rebalancing) == {"a", "b"}
    assert all(r.stops[0].node == 21 and r.stops[0].is_rebalancing for r in result.rebalancing.values())


def test_move_needs_a_gain_above_its_cost(network):
    vehicles = {idle("a", 0)}
    travel = network.get_travel_time(0, 21)
    small = joint(network, NodeValue({21: 0.5 * travel}), max_move=1e6)
    large = joint(network, NodeValue({21: 2.0 * travel}), max_move=1e6)
    rtv = instance(network, vehicles, [])
    assert small.solve(rtv, set(), vehicles, 30, T_NOW).rebalancing == {}
    assert set(large.solve(rtv, set(), vehicles, 30, T_NOW).rebalancing) == {"a"}


def test_value_of_staying_changes_who_serves(network):
    """The nearer vehicle sits somewhere valuable, so the farther one takes the ride."""
    near, far = idle("near", 8), idle("far", 2)
    rider = make_request(network, "r", 7, 35, T_NOW)
    vehicles, requests = {near, far}, {rider}
    rtv = instance(network, vehicles, requests)
    assert network.get_travel_time(8, 7) < network.get_travel_time(2, 7)

    (plain,) = HighsAssignmentSolver().solve(rtv, requests, vehicles, 30, T_NOW).assignments
    (valued,) = joint(network, NodeValue({8: 1e4}), max_move=0).solve(
        rtv, requests, vehicles, 30, T_NOW
    ).assignments
    assert plain.vehicle_id == "near" and valued.vehicle_id == "far"


def test_riders_come_before_moves(network):
    """A waiting rider (penalty 1e5) outweighs a move worth less than that."""
    vehicles = {idle("a", 0)}
    rider = make_request(network, "r", 1, 2, T_NOW)
    rtv = instance(network, vehicles, [rider])
    result = joint(network, NodeValue({21: 5e4}), max_move=1e6).solve(
        rtv, {rider}, vehicles, 30, T_NOW
    )
    assert [a.vehicle_id for a in result.assignments] == ["a"] and result.rebalancing == {}


def zero_joint(network):
    return lambda: joint(network, ZeroValue())


@pytest.mark.parametrize("seed", range(2))
def test_zero_value_simulation_matches_the_paper(network, seed):
    plain, _ = run(network, TripCentricRTVGraphBuilder, HighsAssignmentSolver, seed=seed,
                   rebalancer=NoRebalancingSolver())
    combined, _ = run(network, TripCentricRTVGraphBuilder, zero_joint(network), seed=seed,
                      rebalancer=NoRebalancingSolver())
    assert [b.requests_delivered for b in combined] == [b.requests_delivered for b in plain]
    assert [b.avg_total_delay for b in combined] == pytest.approx([b.avg_total_delay for b in plain])


def test_simulation_with_moves_keeps_every_deadline(network):
    """CheckedController asserts every committed route; moves must not break promises."""
    value = NodeValue({14: 2e3, 21: 3e3})  # the zone centres
    batches, _ = run(network, TripCentricRTVGraphBuilder,
                     lambda: joint(network, value, max_move=1e6), rebalancer=NoRebalancingSolver())
    assert sum(b.vehicles_rebalanced for b in batches) > 0
    assert sum(b.requests_delivered for b in batches) > 50
