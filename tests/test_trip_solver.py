"""travel(v, T): the A* trip solver must match exhaustive search exactly."""

import random

import pytest

from src.components.trip_solver import AStarTripSolver
from tests.conftest import brute_force_route, random_requests, random_vehicle


def test_trip_solver_matches_brute_force(network):
    rng = random.Random(0)
    matrix = network.get_l1_cache()
    t_now = 10_000.0
    outcomes = {True: 0, False: 0}
    for _ in range(400):
        n_onboard = rng.choice([0, 0, 1, 2])
        n_new = rng.choice([1, 2, 3]) if n_onboard < 2 else rng.choice([1, 2])
        vehicle = random_vehicle(network, rng, "v", t_now, n_onboard)
        requests = random_requests(network, rng, n_new, t_now)
        feasible, cost, _ = brute_force_route(requests, vehicle, matrix)

        result = AStarTripSolver(matrix).solve(requests, vehicle)

        assert result.is_feasible == feasible
        if feasible:
            assert result.cost == pytest.approx(cost, abs=1e-6)
        outcomes[feasible] += 1
    assert outcomes[True] > 80 and outcomes[False] > 80, outcomes


def test_route_respects_constraints(network):
    rng = random.Random(1)
    matrix = network.get_l1_cache()
    checked = 0
    for _ in range(200):
        vehicle = random_vehicle(network, rng, "v", 10_000.0, rng.choice([0, 1, 2]))
        requests = random_requests(network, rng, 2, 10_000.0)
        result = AStarTripSolver(matrix).solve(requests, vehicle)
        if not result.is_feasible:
            continue
        checked += 1
        by_id = {r.id: r for r in requests}
        by_id |= {p.original_request.id: p.original_request for p in vehicle.passengers_onboard}
        node, t = vehicle.available_at_node, vehicle.available_from_time
        load, picked = len(vehicle.passengers_onboard), set()
        for stop in result.route.stops:
            r = by_id[stop.request_id]
            t += matrix[node, stop.node]
            node = stop.node
            if stop.is_pickup:
                picked.add(r.id)
                load += 1
                assert t <= r.latest_pickup_time + 1e-9
            else:
                assert r.id in picked or r.id not in {q.id for q in requests}
                load -= 1
                assert t <= r.latest_dropoff_time + 1e-9
            assert load <= vehicle.capacity
        assert len(result.route.stops) == 2 * len(requests) + len(vehicle.passengers_onboard)
    assert checked > 20


def test_promised_pickup_time_is_a_hard_deadline(network):
    rng = random.Random(3)
    matrix = network.get_l1_cache()
    vehicle = random_vehicle(network, rng, "v", 10_000.0, 0)
    (req,) = random_requests(network, rng, 1, 10_000.0)
    t_pickup = vehicle.available_from_time + matrix[vehicle.available_at_node, req.origin_node]
    tight = type(req)(**{**req.__dict__, "latest_pickup_time": t_pickup - 1})
    assert not AStarTripSolver(matrix).solve([tight], vehicle).is_feasible
