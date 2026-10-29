"""Two-stage stochastic joint ILP over sampled futures."""

import random
from dataclasses import replace

import numpy as np
import pytest

from config.config import OMEGA
from src.components.assignment_solver import HighsAssignmentSolver
from src.components.graph_builder.rtv_trip_centric import TripCentricRTVGraphBuilder
from src.components.joint_assignment_solver import ZeroValue
from src.components.objectives import DelayObjective
from src.components.rebalance_solver import NoRebalancingSolver
from src.components.two_stage_assignment_solver import TwoStageAssignmentSolver
from src.interfaces import IRequestSampler
from tests.conftest import make_request, random_requests, random_vehicle
from tests.test_joint_assignment import idle, instance
from tests.test_sampled_requests import NoVirtualPickups, RandomSampler
from tests.test_simulation import run

T_NOW = 10_000.0


class Zones:
    """Nodes 0-17 are zone 1 (centre 14), nodes 18-35 zone 2 (centre 21)."""

    zone_ids = np.array([1, 2])
    center_node = {1: 14, 2: 21}

    def zones_of(self, nodes):
        return np.where(np.asarray(nodes) < 18, 1, 2)


class SameFuture(IRequestSampler):
    """Every sampled future holds the same given riders (shifted to the batch time)."""

    def __init__(self, riders=()):
        self.riders = list(riders)

    def sample(self, t_start, t_end):
        return list(self.riders)


def two_stage(network, riders=(), **kwargs):
    kwargs = {"n_scenarios": 3, "horizon": 600.0, "weight": 0.5} | kwargs
    return TwoStageAssignmentSolver(DelayObjective(), network, Zones(), ZeroValue(), SameFuture(riders), **kwargs)


def far_rider(network, origin=0, rid="virtual_1", dt=300.0):
    """A future rider that a vehicle at `origin` cannot reach in time but one at zone 2's centre can."""
    times = network.get_l1_cache()
    node = max(range(18, 36), key=lambda n: times[origin, n] - times[21, n])
    assert times[origin, node] > times[21, node]
    patience = (times[origin, node] + times[21, node]) / 2
    return make_request(network, rid, node, 0, T_NOW + dt, latest_pickup=T_NOW + dt + patience)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("futures", [0, 3])
def test_without_future_riders_is_the_papers_ilp(network, seed, futures):
    rng = random.Random(seed)
    vehicles = {random_vehicle(network, rng, f"v{k}", T_NOW, rng.choice([0, 0, 1, 2])) for k in range(4)}
    requests = set(random_requests(network, rng, 6, T_NOW))
    rtv = instance(network, vehicles, requests)
    plain = HighsAssignmentSolver().solve(rtv, requests, vehicles, 30, T_NOW)
    result = two_stage(network, n_scenarios=futures).solve(rtv, requests, vehicles, 30, T_NOW)
    assert result.total_cost == pytest.approx(plain.total_cost)
    assert result.assignments == plain.assignments and result.rebalancing == {}


def test_vehicle_moves_to_be_near_future_riders(network):
    vehicles = {idle("a", 0)}
    result = two_stage(network, [far_rider(network)]).solve(instance(network, vehicles, []), set(), vehicles, 30, T_NOW)
    (stop,) = result.rebalancing["a"].stops
    assert stop.node == 21 and stop.is_zone_move


def test_no_move_when_the_rider_is_reachable_from_where_the_vehicle_waits(network):
    times = network.get_l1_cache()
    node = next(n for n in range(18, 36) if times[0, n] < OMEGA / 2)
    vehicles = {idle("a", 0)}
    near = make_request(network, "virtual_1", node, 0, T_NOW + 300)
    result = two_stage(network, [near]).solve(instance(network, vehicles, []), set(), vehicles, 30, T_NOW)
    assert result.rebalancing == {}


def test_one_future_rider_draws_one_vehicle(network):
    vehicles = {idle("a", 0), idle("b", 0)}
    result = two_stage(network, [far_rider(network)]).solve(instance(network, vehicles, []), set(), vehicles, 30, T_NOW)
    assert len(result.rebalancing) == 1


def test_real_rider_beats_a_future_one(network):
    vehicles = {idle("a", 0)}
    rider = make_request(network, "r", 1, 2, T_NOW)
    solver = two_stage(network, [far_rider(network)])
    result = solver.solve(instance(network, vehicles, [rider]), {rider}, vehicles, 30, T_NOW)
    assert [a.trip_id for a in result.assignments] == [frozenset({"r"})]
    assert result.rebalancing == {}


def test_future_riders_count_by_their_share_of_futures(network):
    """A rider in every future is worth w * chi; in one future of three, a third of that."""

    class OneInThree(IRequestSampler):
        def __init__(self, rider):
            self.rider, self.calls = rider, 0

        def sample(self, t_start, t_end):
            self.calls += 1
            return [self.rider] if self.calls % 3 == 1 else []

    vehicles = {idle("a", 0)}
    rider = far_rider(network)
    move = network.get_travel_time(0, 21)
    w = 2 * move / 1e5  # a sure rider pays twice the move; a third of one does not
    sure = two_stage(network, [rider], weight=w).solve(instance(network, vehicles, []), set(), vehicles, 30, T_NOW)
    assert "a" in sure.rebalancing
    solver = TwoStageAssignmentSolver(
        DelayObjective(), network, Zones(), ZeroValue(), OneInThree(rider), n_scenarios=3, weight=w
    )
    rare = solver.solve(instance(network, vehicles, []), set(), vehicles, 30, T_NOW)
    assert rare.rebalancing == {}


def test_future_riders_are_never_committed(network, monkeypatch):
    import tests.test_simulation as sim

    monkeypatch.setattr(sim, "CheckedController", NoVirtualPickups)

    class Hotspot(RandomSampler):
        """Future riders all at zone 2's centre, waiting 30 s: only a vehicle nearby can serve them."""

        def sample(self, t_start, t_end):
            return [
                replace(q, origin_node=21, latest_pickup_time=q.request_time + 30)
                for q in super().sample(t_start, t_end) if q.destination_node != 21
            ]

    solver = lambda: TwoStageAssignmentSolver(
        DelayObjective(), network, Zones(), ZeroValue(), Hotspot(network)
    )
    batches, controller = run(network, TripCentricRTVGraphBuilder, solver, rebalancer=NoRebalancingSolver())
    assert sum(b.vehicles_rebalanced for b in batches) > 0
    assert not any(r.startswith("virtual_") for r in controller._request_index)


def test_no_futures_simulation_matches_the_paper(network):
    plain, _ = run(network, TripCentricRTVGraphBuilder, HighsAssignmentSolver, rebalancer=NoRebalancingSolver())
    result, _ = run(network, TripCentricRTVGraphBuilder, lambda: two_stage(network, n_scenarios=0),
                    rebalancer=NoRebalancingSolver())
    assert [b.requests_delivered for b in result] == [b.requests_delivered for b in plain]
    assert [b.avg_total_delay for b in result] == [b.avg_total_delay for b in plain]
