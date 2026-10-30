"""RV and RTV graphs: compared with exhaustive travel(v, T) checks."""

import itertools
import random

import numpy as np
import pytest

from config.config import MAX_REQUESTS_ABOVE_CAPACITY
from src.components.graph_builder.rtv_trip_centric import TripCentricRTVGraphBuilder
from src.components.graph_builder.rtv_vehicle_centric import (
    VehicleCentricRTVGraphBuilder,
)
from src.components.graph_builder.rv_builder import RVGraphBuilder
from src.components.network.trip_solver import AStarTripSolver
from src.interfaces import ITripSolver
from tests.conftest import random_requests, random_vehicle

T_NOW = 10_000.0


def make_batch(network, seed, n_requests=10, n_vehicles=6):
    rng = random.Random(seed)
    requests = random_requests(network, rng, n_requests, T_NOW)
    vehicles = [
        random_vehicle(network, rng, f"v{k}", T_NOW, rng.choice([0, 0, 1, 1, 2, 3]))
        for k in range(n_vehicles)
    ]
    return requests, vehicles


@pytest.mark.parametrize("seed", range(8))
def test_rv_edges_match_trip_solver(network, seed):
    requests, vehicles = make_batch(network, seed)
    matrix = network.get_l1_cache()
    rv = RVGraphBuilder(network, AStarTripSolver(network.get_l1_cache())).build_graph(set(requests), set(vehicles))

    for r in requests:
        for v in vehicles:
            expected = AStarTripSolver(matrix).solve([r], v)
            edge = rv.rv_details.get((r.id, v.vehicle_id))
            assert (edge is not None) == expected.is_feasible, (r.id, v.vehicle_id)
            if edge is not None:
                assert edge.cost == pytest.approx(expected.cost, abs=1e-6)


@pytest.mark.parametrize("seed", range(8))
def test_rr_edges_cover_every_shareable_pair(network, seed):
    """If any vehicle can serve two requests together, they need an RR edge."""
    requests, vehicles = make_batch(network, seed, n_vehicles=10)
    matrix = network.get_l1_cache()
    rv = RVGraphBuilder(network, AStarTripSolver(network.get_l1_cache())).build_graph(set(requests), set(vehicles))
    for r1, r2 in itertools.combinations(requests, 2):
        if any(AStarTripSolver(matrix).solve([r1, r2], v).is_feasible for v in vehicles):
            assert r2.id in rv.request_request.get(r1.id, set()), (r1.id, r2.id)


def all_feasible_trips(requests, vehicles, matrix):
    """Every (trip, vehicle) pair with travel(v, T) feasible, by enumeration."""
    out = {}
    for v in vehicles:
        limit = v.capacity - len(v.passengers_onboard) + MAX_REQUESTS_ABOVE_CAPACITY
        for k in range(1, min(limit, 4) + 1):
            for combo in itertools.combinations(requests, k):
                result = AStarTripSolver(matrix).solve(list(combo), v)
                if result.is_feasible:
                    out[(frozenset(r.id for r in combo), v.vehicle_id)] = result.cost
    return out


@pytest.mark.parametrize("seed", range(8))
def test_rtv_builders_find_every_feasible_trip(network, seed):
    requests, vehicles = make_batch(network, seed)
    matrix = network.get_l1_cache()
    rv = RVGraphBuilder(network, AStarTripSolver(network.get_l1_cache())).build_graph(set(requests), set(vehicles))
    expected = all_feasible_trips(requests, vehicles, matrix)
    assert any(len(t) >= 2 for t, _ in expected), "instance has no shared trips"

    for builder in (TripCentricRTVGraphBuilder, VehicleCentricRTVGraphBuilder):
        rtv = builder(AStarTripSolver(matrix)).build_graph(rv, set(vehicles), set(requests))
        got = {
            key: route.cost
            for key, route in rtv.trip_vehicle_details.items()
            if len(key[0]) <= 4
        }
        assert got.keys() == expected.keys(), builder.__name__
        for key, cost in expected.items():
            assert got[key] == pytest.approx(cost, abs=1e-6), (builder.__name__, key)


@pytest.mark.parametrize("seed", range(3))
def test_trip_and_vehicle_centric_agree_on_large_batch(network, seed):
    requests, vehicles = make_batch(network, 100 + seed, n_requests=16, n_vehicles=10)
    rv = RVGraphBuilder(network, AStarTripSolver(network.get_l1_cache())).build_graph(set(requests), set(vehicles))
    trip = TripCentricRTVGraphBuilder(AStarTripSolver(network.get_l1_cache())).build_graph(rv, set(vehicles), set(requests))
    veh = VehicleCentricRTVGraphBuilder(AStarTripSolver(network.get_l1_cache())).build_graph(rv, set(vehicles), set(requests))
    assert trip.trip_vehicle_details.keys() == veh.trip_vehicle_details.keys()
    for key, route in trip.trip_vehicle_details.items():
        assert route.cost == pytest.approx(veh.trip_vehicle_details[key].cost)


class PlainTripSolver(ITripSolver):
    """A swapped-in trip solver: same answers, but no closed-form RV shortcut."""

    def __init__(self, matrix):
        self._inner = AStarTripSolver(matrix)

    def solve(self, requests, vehicle):
        return self._inner.solve(requests, vehicle)


def test_rv_builder_uses_any_trip_solver_for_every_edge(network):
    requests, vehicles = make_batch(network, 3)
    matrix = network.get_l1_cache()
    plain = RVGraphBuilder(network, PlainTripSolver(matrix)).build_graph(set(requests), set(vehicles))
    fast = RVGraphBuilder(network, AStarTripSolver(matrix)).build_graph(set(requests), set(vehicles))
    assert plain.rv_details.keys() == fast.rv_details.keys()
    for key, route in plain.rv_details.items():
        assert route.cost == pytest.approx(fast.rv_details[key].cost)


@pytest.mark.parametrize("seed", range(4))
def test_occupied_filter_keeps_every_pair_the_solver_can_serve(network, seed):
    rng = random.Random(seed)
    solver = AStarTripSolver(network.get_l1_cache())
    builder = RVGraphBuilder(network, solver)
    requests = random_requests(network, rng, 40, T_NOW)
    vehicles = [
        random_vehicle(network, rng, f"v{k}", T_NOW, rng.choice([2, 3, 4]), capacity=rng.choice([4, 5]))
        for k in range(20)
    ]
    v_idx, r_idx = (a.ravel() for a in np.meshgrid(range(len(vehicles)), range(len(requests))))
    keep = builder._may_be_feasible(v_idx, r_idx, vehicles, requests)
    feasible = np.array(
        [solver.solve([requests[r]], vehicles[v]).is_feasible for v, r in zip(v_idx, r_idx)]
    )
    assert not (feasible & ~keep).any()
    assert (~keep).any() and feasible.any()


@pytest.mark.parametrize("seed", range(4))
def test_reachable_pairs_match_every_pair(network, seed):
    rng = random.Random(seed)
    matrix = network.get_l1_cache()
    builder = RVGraphBuilder(network, AStarTripSolver(matrix), omega=rng.choice([60.0, 120.0]))
    locs = [rng.randrange(network.N) for _ in range(60)]
    times = np.array([T_NOW + rng.choice([0.0, rng.uniform(0, 30)]) for _ in locs])
    origins = np.array([rng.randrange(network.N) for _ in range(25)])
    for spread in (builder._omega, 3 * builder._omega):  # near-node index, then every pair
        latest = T_NOW + np.array([rng.uniform(0, spread) for _ in origins])
        arrival = times[:, None] + matrix[np.ix_(locs, origins)]
        expected = np.where(arrival <= latest[None, :])
        v_idx, r_idx, arr = builder._reachable_pairs(locs, times, origins, latest)
        assert (v_idx == expected[0]).all() and (r_idx == expected[1]).all()
        assert (arr == arrival[expected]).all()


def test_reused_builder_gives_the_graph_of_a_fresh_one(network):
    rng = random.Random(5)
    trip_solver = AStarTripSolver(network.get_l1_cache())
    requests = random_requests(network, rng, 40, T_NOW)
    vehicles = [random_vehicle(network, rng, f"v{k}", T_NOW, rng.choice([0, 1, 2])) for k in range(8)]
    builder = RVGraphBuilder(network, trip_solver)
    for window in (requests[:30], requests[10:], requests[5:35], list(reversed(requests[20:]))):
        reused = builder.build_graph(set(window), set(vehicles))
        fresh = RVGraphBuilder(network, trip_solver).build_graph(set(window), set(vehicles))
        assert reused == fresh
        assert list(reused.rr_details) == list(fresh.rr_details)  # same insertion order
