"""Assignment ILP and rebalancing: compared with brute force on small instances."""

import itertools
import random

import numpy as np
import pytest

from src.components.assignment.assignment_solver import GurobiAssignmentSolver, HighsAssignmentSolver
from src.components.economics.objectives import DelayObjective
from src.components.rebalancing.rebalance_solver import ReactiveRebalancingSolver
from src.interfaces import FeasibleRoute, Route, Stop
from src.interfaces.graphs import RTVGraph
from tests.conftest import make_request, random_vehicle

T_NOW = 10_000.0


def random_rtv(rng, n_requests=5, n_vehicles=3):
    graph = RTVGraph()
    req_ids = [f"r{k}" for k in range(n_requests)]
    for v in range(n_vehicles):
        for k in (1, 2, 3):
            for trip in rng.sample(list(itertools.combinations(req_ids, k)), 3):
                cost = rng.uniform(0, 300)
                graph.add_trip_for_vehicle(
                    frozenset(trip), f"v{v}", FeasibleRoute(Route((), 0, 0, cost), cost)
                )
    return graph, req_ids


def brute_force_objective(graph, requests, solver, t_now):
    edges = list(graph.trip_vehicle_details.items())
    best = np.inf
    for chosen in itertools.product([None] + edges, repeat=len({v for (_, v), _ in edges})):
        used_v, used_r, cost = set(), set(), 0.0
        ok = True
        for e in chosen:
            if e is None:
                continue
            (trip, v), route = e
            if v in used_v or used_r & trip:
                ok = False
                break
            used_v.add(v)
            used_r |= trip
            cost += route.cost
        if ok:
            cost += sum(DelayObjective().unserved_penalty(r, t_now) for r in requests if r.id not in used_r)
            best = min(best, cost)
    return best


@pytest.mark.parametrize("solver_cls", [GurobiAssignmentSolver, HighsAssignmentSolver])
@pytest.mark.parametrize("seed", range(5))
def test_assignment_is_optimal(network, solver_cls, seed):
    rng = random.Random(seed)
    graph, req_ids = random_rtv(rng)
    requests = {make_request(network, rid, 0, 1, T_NOW - rng.uniform(0, 120)) for rid in req_ids}
    vehicles = {random_vehicle(network, rng, f"v{v}", T_NOW, 0) for v in range(3)}
    solver = solver_cls()

    result = solver.solve(graph, requests, vehicles, solver_timeout=30, current_time=T_NOW)

    assert result.is_optimal
    assert result.total_cost == pytest.approx(
        brute_force_objective(graph, requests, solver, T_NOW), rel=1e-9
    )
    served = [r for a in result.assignments for r in a.trip_id]
    assert len(served) == len(set(served))
    assert len({a.vehicle_id for a in result.assignments}) == len(result.assignments)
    assert {r.id for r in result.unassigned_requests} == set(req_ids) - set(served)


@pytest.mark.parametrize("seed", range(5))
def test_rebalancing_minimises_total_travel_time(network, seed):
    rng = random.Random(seed)
    vehicles = [random_vehicle(network, rng, f"v{k}", T_NOW, 0) for k in range(4)]
    requests = [
        make_request(network, f"r{k}", *rng.sample(range(network.N), 2), T_NOW)
        for k in range(rng.choice([2, 4, 6]))
    ]
    routes = ReactiveRebalancingSolver(network).solve(vehicles, requests, t_now=T_NOW)

    assert len(routes) == min(len(vehicles), len(requests))
    by_id = {v.vehicle_id: v for v in vehicles}
    total = sum(
        network.get_travel_time(by_id[v].available_at_node, route.stops[0].node)
        for v, route in routes.items()
    )
    m = network.get_travel_time_matrix(
        [v.available_at_node for v in vehicles], [r.origin_node for r in requests]
    )
    k = min(len(vehicles), len(requests))
    best = min(
        sum(m[i, j] for i, j in zip(rows, cols))
        for rows in itertools.permutations(range(len(vehicles)), k)
        for cols in itertools.combinations(range(len(requests)), k)
    )
    assert total == pytest.approx(best)


def test_rebalancing_replans_vehicles_en_route(network):
    """En-route vehicles join the matching; those left without a rider stop."""
    rng = random.Random(0)
    idle = [random_vehicle(network, rng, "idle", T_NOW, 0)]
    en_route = [
        type(v)(**{**v.__dict__, "remaining_stops_to_serve": (Stop.rebalancing(0, "old"),)})
        for v in (random_vehicle(network, rng, f"e{k}", T_NOW, 0) for k in range(2))
    ]
    rider = make_request(network, "r", en_route[0].available_at_node, 35, T_NOW)
    routes = ReactiveRebalancingSolver(network).solve(idle, [rider], T_NOW, en_route)

    assert routes["e0"].stops[0].node == rider.origin_node  # closest vehicle, zero travel
    assert routes["e1"].stops == ()  # stops instead of chasing the old target
    assert "idle" not in routes
