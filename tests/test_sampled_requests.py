"""Virtual future requests: the sampler and the sampled joint ILP."""

import random
from datetime import date, datetime

import numpy as np
import polars as pl
import pytest

from config.config import DELTA, OMEGA
from src.components.assignment.assignment_solver import HighsAssignmentSolver
from src.components.demand.demand_model import HistoricalDemandModel
from src.components.graph_builder.rtv_trip_centric import TripCentricRTVGraphBuilder
from src.components.assignment.joint_assignment_solver import ZeroValue
from src.components.economics.objectives import DelayObjective
from src.components.rebalancing.rebalance_solver import NoRebalancingSolver
from src.components.demand.request_sampler import HistoricalRequestSampler
from src.components.assignment.sampled_assignment_solver import (
    BidirectionalChainSolver,
    BidirectionalSampledSolver,
    FullMatchingSampledSolver,
    SampledChainAssignmentSolver,
    SampledRequestsAssignmentSolver,
    SupplyMatchedSampledSolver,
)
from src.interfaces import IRequestSampler, Passenger, Stop, VehicleAvailability
from tests.conftest import make_request, random_requests, random_vehicle
from tests.test_joint_assignment import Zones, idle, instance
from tests.test_simulation import CheckedController, run

T0 = datetime(2025, 1, 16)  # Thursday 00:00 = simulation time 0
SIX_PM = 18 * 3600
T_NOW = 10_000.0


class HalfZones:
    """Nodes 0-17 are zone 1, nodes 18-35 zone 2."""

    zone_ids = np.array([1, 2])

    def zones_of(self, nodes):
        return np.where(np.asarray(nodes) < 18, 1, 2)


def trip_table():
    rows = []
    for day in (14, 15):  # Tuesday and Wednesday: training days
        rows += [(datetime(2025, 1, day, 18, m), o, 30) for m, o in [(1, 3), (4, 5), (9, 3)]]
        rows += [(datetime(2025, 1, day, 18, 7), 20, 2)]
        rows += [(datetime(2025, 1, day, 18, 20), 25, 1)]  # next slot: zone 2 only
    rows += [(datetime(2025, 1, 16, 18, m), 4, 35) for m in range(15)]  # the simulated day
    return pl.DataFrame(rows, schema=["pickup_datetime", "orig_node", "dest_node"], orient="row")


def sampler(network, seed=0):
    trips = trip_table()
    demand = HistoricalDemandModel.fit(trips, HalfZones(), T0, exclude={date(2025, 1, 16)})
    return HistoricalRequestSampler(
        demand, trips, HalfZones(), T0, network, exclude={date(2025, 1, 1), date(2025, 1, 16)}, seed=seed
    )


def test_sampler_matches_the_model_and_never_uses_the_simulated_day(network):
    s, n = sampler(network), 4000
    batches = [s.sample(SIX_PM, SIX_PM + 900) for _ in range(n)]
    zones = HalfZones()
    for zone, expected in [(1, 3.0), (2, 1.0)]:  # trips per training day in 18:00-18:15
        counts = [sum(zones.zones_of(q.origin_node) == zone for q in b) for b in batches]
        assert np.mean(counts) == pytest.approx(expected, abs=4 * np.sqrt(expected / n))
        assert np.var(counts) == pytest.approx(expected, rel=0.1)  # Poisson
    requests = [q for b in batches for q in b]
    assert {(q.origin_node, q.destination_node) for q in requests} == {(3, 30), (5, 30), (20, 2)}
    share_3 = np.mean([q.origin_node == 3 for q in requests if q.origin_node in (3, 5)])
    assert share_3 == pytest.approx(2 / 3, abs=0.02)  # empirical origin distribution
    for q in requests[:50]:
        assert SIX_PM <= q.request_time < SIX_PM + 900
        assert q.latest_pickup_time == q.request_time + OMEGA
        direct = network.get_travel_time(q.origin_node, q.destination_node)
        assert q.latest_dropoff_time == pytest.approx(q.request_time + direct + DELTA)


def test_sampler_follows_slots_and_is_seeded(network):
    s = sampler(network)
    requests = [q for _ in range(500) for q in s.sample(SIX_PM + 600, SIX_PM + 1200)]
    early = {q.origin_node for q in requests if q.request_time < SIX_PM + 900}
    late = {q.origin_node for q in requests if q.request_time >= SIX_PM + 900}
    assert early == {3, 5, 20} and late == {25}
    first = [(q.origin_node, q.request_time) for q in sampler(network, seed=3).sample(SIX_PM, SIX_PM + 900)]
    again = [(q.origin_node, q.request_time) for q in sampler(network, seed=3).sample(SIX_PM, SIX_PM + 900)]
    assert first == again and len({q.id for q in requests}) == len(requests)


class FixedSampler(IRequestSampler):
    """Hands out given virtual requests when their time enters the horizon."""

    def __init__(self, requests=()):
        self.requests = list(requests)
        self.calls = []

    def sample(self, t_start, t_end, t_now=None):
        self.calls.append((t_start, t_end))
        return [q for q in self.requests if t_start <= q.request_time < t_end]


def virtual(network, rid, origin, dt=60.0, dest=35):
    return make_request(network, f"virtual_{rid}", origin, dest, T_NOW + dt)


def sampled(network, requests=(), **kwargs):
    kwargs = {"horizon": 300.0, "weight": 0.05} | kwargs
    return SampledRequestsAssignmentSolver(
        DelayObjective(), network, Zones(), ZeroValue(), FixedSampler(requests), **kwargs
    )


def test_free_vehicle_drives_to_the_virtual_pickup(network):
    vehicles = {idle("a", 0)}
    result = sampled(network, [virtual(network, 1, 21)]).solve(
        instance(network, vehicles, []), set(), vehicles, 30, T_NOW
    )
    (stop,) = result.rebalancing["a"].stops
    assert stop.node == 21 and stop.is_rebalancing and not stop.is_pickup
    assert result.assignments == set()


def test_real_rider_beats_virtual_rider(network):
    vehicles = {idle("a", 0)}
    rider = make_request(network, "r", 1, 2, T_NOW)
    solver = sampled(network, [virtual(network, k, 21) for k in range(3)], weight=0.1)
    result = solver.solve(instance(network, vehicles, [rider]), {rider}, vehicles, 30, T_NOW)
    assert [a.trip_id for a in result.assignments] == [frozenset({"r"})]
    assert result.rebalancing == {}


def test_each_virtual_rider_is_taken_at_most_once(network):
    vehicles = {idle("a", 0), idle("b", 1)}
    result = sampled(network, [virtual(network, 1, 21)]).solve(
        instance(network, vehicles, []), set(), vehicles, 30, T_NOW
    )
    assert len(result.rebalancing) == 1


def test_unreachable_virtual_rider_is_ignored(network):
    vehicles = {idle("a", 0)}
    far = make_request(network, "virtual_far", 35, 0, T_NOW + 10, latest_pickup=T_NOW + 11)
    result = sampled(network, [far]).solve(instance(network, vehicles, []), set(), vehicles, 30, T_NOW)
    assert result.rebalancing == {}


def test_scenario_rolls_and_keeps_virtual_riders_until_their_latest_pickup(network):
    source = FixedSampler([virtual(network, 1, 21, dt=20), virtual(network, 2, 22, dt=305)])
    solver = SampledRequestsAssignmentSolver(DelayObjective(), network, Zones(), ZeroValue(), source)
    vehicles = {idle("a", 0)}
    rtv = instance(network, vehicles, [])
    solver.solve(rtv, set(), vehicles, 30, T_NOW)
    assert [q.id for q in solver.virtual] == ["virtual_1"]
    solver.solve(rtv, set(), vehicles, 30, T_NOW + 10)
    assert [q.id for q in solver.virtual] == ["virtual_1", "virtual_2"]
    solver.solve(rtv, set(), vehicles, 30, T_NOW + 20 + OMEGA)
    assert [q.id for q in solver.virtual] == ["virtual_2"]
    assert source.calls == [(T_NOW, T_NOW + 300), (T_NOW + 300, T_NOW + 310),
                                     (T_NOW + 310, T_NOW + 500)]


@pytest.mark.parametrize("seed", range(5))
def test_no_virtual_requests_is_the_papers_ilp(network, seed):
    rng = random.Random(seed)
    vehicles = {random_vehicle(network, rng, f"v{k}", T_NOW, rng.choice([0, 0, 1, 2])) for k in range(4)}
    requests = set(random_requests(network, rng, 6, T_NOW))
    rtv = instance(network, vehicles, requests)
    plain = HighsAssignmentSolver().solve(rtv, requests, vehicles, 30, T_NOW)
    combined = sampled(network).solve(rtv, requests, vehicles, 30, T_NOW)
    assert combined.total_cost == pytest.approx(plain.total_cost)
    assert combined.rebalancing == {}


class RandomSampler(IRequestSampler):
    def __init__(self, network, rate=0.05):
        self.rng, self.network, self.rate, self.k = random.Random(1), network, rate, 0

    def sample(self, t_start, t_end, t_now=None):
        out = []
        for _ in range(np.random.default_rng(int(t_start)).poisson(self.rate * (t_end - t_start))):
            o, d = self.rng.sample(range(self.network.N), 2)
            out.append(make_request(self.network, f"virtual_{self.k}", o, d, self.rng.uniform(t_start, t_end)))
            self.k += 1
        return out


class NoVirtualPickups(CheckedController):
    def apply_rebalancing(self, routes, t_now):
        assert all(s.is_rebalancing for r in routes.values() for s in r.stops)
        super().apply_rebalancing(routes, t_now)


def test_virtual_riders_are_never_committed(network, monkeypatch):
    import tests.test_simulation as sim

    monkeypatch.setattr(sim, "CheckedController", NoVirtualPickups)
    solver = lambda: SampledRequestsAssignmentSolver(
        DelayObjective(), network, Zones(), ZeroValue(), RandomSampler(network)
    )
    batches, controller = run(network, TripCentricRTVGraphBuilder, solver, rebalancer=NoRebalancingSolver())
    assert sum(b.vehicles_rebalanced for b in batches) > 0
    for v in controller._fleet:
        assert not any(p.original_request.id.startswith("virtual_") for p in v.passengers_onboard)
    assert not any(r.startswith("virtual_") for r in controller._request_index)


@pytest.mark.parametrize("seed", range(2))
def test_no_virtual_requests_simulation_matches_the_paper(network, seed):
    plain, _ = run(network, TripCentricRTVGraphBuilder, HighsAssignmentSolver, seed=seed,
                   rebalancer=NoRebalancingSolver())
    combined, _ = run(network, TripCentricRTVGraphBuilder, lambda: sampled(network), seed=seed,
                      rebalancer=NoRebalancingSolver())
    assert [b.requests_delivered for b in combined] == [b.requests_delivered for b in plain]
    assert [b.avg_total_delay for b in combined] == [b.avg_total_delay for b in plain]


def chain(network, requests=(), **kwargs):
    kwargs = {"horizon": 300.0, "weight": 0.05} | kwargs
    return SampledChainAssignmentSolver(
        DelayObjective(), network, Zones(), ZeroValue(), FixedSampler(requests), **kwargs
    )


def test_vehicle_ending_its_plan_near_a_virtual_rider_spares_a_move(network):
    """A busy vehicle that drops off next to the virtual rider claims it; the free one stays."""
    rider = make_request(network, "p", 14, 21, T_NOW - 100)
    busy = VehicleAvailability("busy", 4, 20, T_NOW, frozenset({Passenger(rider, T_NOW - 60)}),
                               (Stop(21, False, "p"),))
    vehicles = {busy, idle("free", 0)}
    q = virtual(network, 1, 21, dt=120)
    rtv = instance(network, vehicles, [])
    assert set(sampled(network, [q]).solve(rtv, set(), vehicles, 30, T_NOW).rebalancing) == {"free"}
    assert chain(network, [q]).solve(rtv, set(), vehicles, 30, T_NOW).rebalancing == {}


def test_chain_commits_only_the_real_trip(network):
    vehicles = {idle("a", 0), idle("c", 27)}
    rider = make_request(network, "r", 1, 21, T_NOW)
    q = virtual(network, 1, 21, dt=200)
    rtv = instance(network, vehicles, [rider])
    result = chain(network, [q]).solve(rtv, {rider}, vehicles, 30, T_NOW)
    (a,) = result.assignments
    assert a.trip_id == frozenset({"r"}) and a.route == rtv.get_route(frozenset({"r"}), a.vehicle_id).route
    assert result.rebalancing == {} and result.unassigned_requests == set()
    assert set(sampled(network, [q]).solve(rtv, {rider}, vehicles, 30, T_NOW).rebalancing) == {"c"}


@pytest.mark.parametrize("seed", range(3))
def test_chain_without_virtual_requests_is_the_papers_ilp(network, seed):
    rng = random.Random(seed)
    vehicles = {random_vehicle(network, rng, f"v{k}", T_NOW, rng.choice([0, 0, 1, 2])) for k in range(4)}
    requests = set(random_requests(network, rng, 6, T_NOW))
    rtv = instance(network, vehicles, requests)
    plain = HighsAssignmentSolver().solve(rtv, requests, vehicles, 30, T_NOW)
    assert chain(network).solve(rtv, requests, vehicles, 30, T_NOW).total_cost == pytest.approx(plain.total_cost)


def test_supply_matched_offers_about_one_virtual_rider_per_free_vehicle(network):
    riders = [virtual(network, k, k % network.N, dt=1 + k % 200) for k in range(400)]
    vehicles = {idle(f"v{k}", k) for k in range(20)}
    solver = SupplyMatchedSampledSolver(
        DelayObjective(), network, Zones(), ZeroValue(), FixedSampler(riders), virtual_per_free=2.0
    )
    rtv = instance(network, vehicles, [])
    solver.solve(rtv, set(), vehicles, 30, T_NOW)
    first = {q.id for q in solver.active}
    assert 20 <= len(first) <= 60  # about 2 * 20 of 400
    solver.solve(rtv, set(), vehicles, 30, T_NOW + 10)
    assert {q.id for q in solver.active} == first  # same marks, same share: stable


def test_supply_matched_with_room_for_all_is_the_sampled_solver(network):
    vehicles = {idle("a", 0), idle("b", 1)}
    riders = [virtual(network, 1, 21)]
    rtv = instance(network, vehicles, [])
    matched = SupplyMatchedSampledSolver(
        DelayObjective(), network, Zones(), ZeroValue(), FixedSampler(riders), virtual_per_free=1.0
    ).solve(rtv, set(), vehicles, 30, T_NOW)
    plain = sampled(network, riders).solve(rtv, set(), vehicles, 30, T_NOW)
    assert matched.rebalancing == plain.rebalancing and len(plain.rebalancing) == 1


def test_bidirectional_offers_each_rider_to_its_nearest_vehicles(network):
    vehicles = {idle("a", 0), idle("b", 1)}
    riders = [virtual(network, 1, 2, dt=200), virtual(network, 2, 35, dt=200)]
    rtv = instance(network, vehicles, [])
    args = (DelayObjective(), network, Zones(), ZeroValue(), FixedSampler(riders))
    one_side = SupplyMatchedSampledSolver(*args, n_virtual=1).solve(rtv, set(), vehicles, 30, T_NOW)
    both = BidirectionalSampledSolver(*args, n_virtual=1).solve(rtv, set(), vehicles, 30, T_NOW)
    assert len(one_side.rebalancing) == 1 and len(both.rebalancing) == 2


def test_chain_supply_counts_vehicles_free_within_the_horizon(network):
    rider = make_request(network, "p", 14, 21, T_NOW - 100)
    busy = VehicleAvailability("busy", 4, 20, T_NOW, frozenset({Passenger(rider, T_NOW - 60)}),
                               (Stop(21, False, "p"),))
    solver = BidirectionalChainSolver(DelayObjective(), network, Zones(), ZeroValue(), FixedSampler())
    assert solver._supply({busy, idle("a", 0)}, T_NOW) == 2
    assert solver._supply({busy, idle("a", 0)}, T_NOW - 1000) == 1


def test_full_matching_moves_every_vehicle_of_a_cluster(network):
    """Three vehicles on one node, one option each: top-k alone moves one, the matching all three."""
    vehicles = {idle(f"v{k}", 0, T_NOW) for k in range(3)}
    riders = [virtual(network, k, node, dt=200) for k, node in enumerate([1, 6, 35])]
    rtv = instance(network, vehicles, [])
    args = (DelayObjective(), network, Zones(), ZeroValue(), FixedSampler(riders))
    kw = {"n_virtual": 1, "virtual_per_free": 1.0}
    assert len(SupplyMatchedSampledSolver(*args, **kw).solve(rtv, set(), vehicles, 30, T_NOW).rebalancing) == 1
    assert len(FullMatchingSampledSolver(*args, **kw).solve(rtv, set(), vehicles, 30, T_NOW).rebalancing) == 3
