"""RV and RTV graphs: compared with exhaustive travel(v, T) checks."""

import itertools
import random

import pytest

from config.config import MAX_REQUESTS_ABOVE_CAPACITY
from src.components.graph_builder.rtv_trip_centric import TripCentricRTVGraphBuilder
from src.components.graph_builder.rtv_vehicle_centric import (
    VehicleCentricRTVGraphBuilder,
)
from src.components.graph_builder.rv_builder import RVGraphBuilder
from src.components.trip_solver import TripSolver
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
    rv = RVGraphBuilder(network).build_graph(set(requests), set(vehicles))

    for r in requests:
        for v in vehicles:
            expected = TripSolver.solve([r], v, matrix)
            edge = rv.rv_details.get((r.id, v.vehicle_id))
            assert (edge is not None) == expected.is_feasible, (r.id, v.vehicle_id)
            if edge is not None:
                assert edge.cost == pytest.approx(expected.cost, abs=1e-6)


@pytest.mark.parametrize("seed", range(8))
def test_rr_edges_cover_every_shareable_pair(network, seed):
    """If any vehicle can serve two requests together, they need an RR edge."""
    requests, vehicles = make_batch(network, seed, n_vehicles=10)
    matrix = network.get_l1_cache()
    rv = RVGraphBuilder(network).build_graph(set(requests), set(vehicles))
    for r1, r2 in itertools.combinations(requests, 2):
        if any(TripSolver.solve([r1, r2], v, matrix).is_feasible for v in vehicles):
            assert r2.id in rv.request_request.get(r1.id, set()), (r1.id, r2.id)


def all_feasible_trips(requests, vehicles, matrix):
    """Every (trip, vehicle) pair with travel(v, T) feasible, by enumeration."""
    out = {}
    for v in vehicles:
        limit = v.capacity - len(v.passengers_onboard) + MAX_REQUESTS_ABOVE_CAPACITY
        for k in range(1, min(limit, 4) + 1):
            for combo in itertools.combinations(requests, k):
                result = TripSolver.solve(list(combo), v, matrix)
                if result.is_feasible:
                    out[(frozenset(r.id for r in combo), v.vehicle_id)] = result.cost
    return out


@pytest.mark.parametrize("seed", range(8))
def test_rtv_builders_find_every_feasible_trip(network, seed):
    requests, vehicles = make_batch(network, seed)
    matrix = network.get_l1_cache()
    rv = RVGraphBuilder(network).build_graph(set(requests), set(vehicles))
    expected = all_feasible_trips(requests, vehicles, matrix)
    assert any(len(t) >= 2 for t, _ in expected), "instance has no shared trips"

    for builder in (TripCentricRTVGraphBuilder, VehicleCentricRTVGraphBuilder):
        rtv = builder(network).build_graph(rv, set(vehicles), set(requests))
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
    rv = RVGraphBuilder(network).build_graph(set(requests), set(vehicles))
    trip = TripCentricRTVGraphBuilder(network).build_graph(rv, set(vehicles), set(requests))
    veh = VehicleCentricRTVGraphBuilder(network).build_graph(rv, set(vehicles), set(requests))
    assert trip.trip_vehicle_details.keys() == veh.trip_vehicle_details.keys()
    for key, route in trip.trip_vehicle_details.items():
        assert route.cost == pytest.approx(veh.trip_vehicle_details[key].cost)
