"""Economics: fares, profit trip costs, revenue and vehicle-km accounting."""

import itertools
import random

import numpy as np
import pytest

from src.components.assignment.assignment_solver import GurobiAssignmentSolver, HighsAssignmentSolver
from src.components.network.network_service import NetworkXService
from src.components.economics.objectives import DelayObjective, ProfitObjective
from src.components.economics.pricing import LinearFareModel
from src.interfaces import FeasibleRoute, Route, Stop
from src.interfaces.data_models import MetaRequest
from src.interfaces.graphs import RTVGraph
from tests.conftest import InMemoryPathCache, make_request, make_street_graph, random_vehicle
from tests.test_controller import assign, make_controller

T_NOW = 10_000.0


def test_fare_follows_the_formula_on_the_simulated_path(network):
    fares = LinearFareModel(network, base=5, per_km=1, per_min=0.5, pooling_discount=0.2)
    km = network.get_distance_matrix()[0, 35] / 1000
    minutes = network.get_travel_time(0, 35) / 60
    assert fares.price(0, 35) == pytest.approx(0.8 * (5 + km + 0.5 * minutes))


def test_fare_time_component_uses_the_travel_time_factor():
    graph = make_street_graph(seed=7)
    db = InMemoryPathCache(graph)
    free = LinearFareModel(NetworkXService(graph, db=db), base=0, per_km=0, per_min=1, pooling_discount=0)
    slow = LinearFareModel(
        NetworkXService(graph, db=db, travel_time_factor=2.0), base=0, per_km=0, per_min=1, pooling_discount=0
    )
    assert slow.price(0, 35) == pytest.approx(2 * free.price(0, 35))


def test_profit_trip_cost_is_extra_km_cost_minus_fares(network):
    rng = random.Random(0)
    dist = network.get_distance_matrix()
    vehicle = random_vehicle(network, rng, "v", T_NOW, 0)
    req = make_request(network, "r", 14, 35, T_NOW)
    req = type(req)(**{**req.__dict__, "fare": 20.0})
    route = FeasibleRoute(Route((Stop(14, True, "r"), Stop(35, False, "r")), 0, 0, 0), 0)
    cost = ProfitObjective(network, cost_per_km=0.5).trip_cost(route, vehicle, [req])
    km = (dist[vehicle.available_at_node, 14] + dist[14, 35]) / 1000
    assert cost == pytest.approx(0.5 * km - 20.0)


def test_profit_counts_only_km_beyond_the_current_drop_offs(network):
    rng = random.Random(2)
    vehicle = random_vehicle(network, rng, "v", T_NOW, 1)
    (p,) = vehicle.passengers_onboard
    drop = Stop(p.original_request.destination_node, False, p.original_request.id)
    vehicle = type(vehicle)(**{**vehicle.__dict__, "remaining_stops_to_serve": (drop,)})
    # Serving nothing new along the same drop-off costs nothing.
    same = FeasibleRoute(Route((drop,), 0, 0, 0), 0)
    assert ProfitObjective(network, cost_per_km=1.0).trip_cost(same, vehicle, []) == pytest.approx(0)


def test_unserved_costs_nothing_unless_promised(network):
    objective = ProfitObjective(network, cost_per_km=0.25, promise_penalty=1e4)
    req = make_request(network, "r", 14, 35, T_NOW)
    assert objective.unserved_penalty(req, T_NOW) == 0
    promised = type(req)(**{**req.__dict__, "promised_pickup_time": T_NOW + 30})
    assert objective.unserved_penalty(promised, T_NOW) == 1e4


@pytest.mark.parametrize("solver_cls", [GurobiAssignmentSolver, HighsAssignmentSolver])
@pytest.mark.parametrize("seed", range(4))
def test_profit_assignment_is_optimal(network, solver_cls, seed):
    rng = random.Random(seed)
    requests = []
    for k in range(5):
        r = make_request(network, f"r{k}", *rng.sample(range(network.N), 2), T_NOW - rng.uniform(0, 60))
        promised = T_NOW + 30 if rng.random() < 0.3 else None
        requests.append(type(r)(**{**r.__dict__, "fare": rng.uniform(5, 30), "promised_pickup_time": promised}))
    by_id = {r.id: r for r in requests}
    vehicles = {random_vehicle(network, rng, f"v{v}", T_NOW, 0) for v in range(3)}
    graph = RTVGraph()
    for v in vehicles:
        for k in (1, 2):
            for trip in rng.sample(list(itertools.combinations(sorted(by_id), k)), 3):
                stops = tuple(
                    Stop(by_id[r].origin_node if pick else by_id[r].destination_node, pick, r)
                    for r in trip for pick in (True, False)
                )
                graph.add_trip_for_vehicle(frozenset(trip), v.vehicle_id, FeasibleRoute(Route(stops, 0, 0, 0), 0))
    objective = ProfitObjective(network, cost_per_km=rng.choice([0.1, 2.0, 8.0]))
    veh = {v.vehicle_id: v for v in vehicles}
    cost = {
        key: objective.trip_cost(route, veh[key[1]], [by_id[r] for r in sorted(key[0])])
        for key, route in graph.trip_vehicle_details.items()
    }
    best = float("inf")
    edges = list(cost)
    for chosen in itertools.product([None, *edges], repeat=len(vehicles)):
        picked = [e for e in chosen if e is not None]
        served = set().union(*(trip for trip, _ in picked))
        one_trip_per_vehicle = len({v for _, v in picked}) == len(picked)
        disjoint = len(served) == sum(len(trip) for trip, _ in picked)
        if not (one_trip_per_vehicle and disjoint):
            continue
        total = sum(cost[e] for e in picked) + sum(
            objective.unserved_penalty(r, T_NOW) for r in requests if r.id not in served
        )
        best = min(best, total)

    result = solver_cls(objective).solve(graph, set(requests), vehicles, 30, T_NOW)
    assert result.total_cost == pytest.approx(best, abs=1e-6)


def test_controller_counts_revenue_and_empty_and_loaded_km(network):
    fares = LinearFareModel(network)
    controller = make_controller(network, [0])
    controller._fare_model = fares
    (req,) = controller.prepare_batch(10.0, [MetaRequest("r", 14, 35, 5.0)])[0]
    assert req.fare == pytest.approx(fares.price(14, 35))
    assign(controller, 10.0, "v0", req)

    # Replay in several batches so legs are split mid-way; totals must still add up.
    t = 10.0
    km_empty = km_loaded = revenue = 0.0
    while not controller._fleet[0].is_idle or t == 10.0:
        t += 10.0
        controller.prepare_batch(t, [])
        if "r" in controller._request_index:  # not picked up yet: re-assigned, as every batch
            assign(controller, t, "v0", controller._request_index["r"])
        stats = controller.get_and_reset_metrics()
        km_empty += stats.km_empty
        km_loaded += stats.km_loaded
        revenue += stats.revenue
    dist = network.get_distance_matrix()
    assert km_empty == pytest.approx(dist[0, 14] / 1000)
    assert km_loaded == pytest.approx(dist[14, 35] / 1000)
    assert revenue == pytest.approx(req.fare)


@pytest.mark.parametrize(
    "make", [lambda net: DelayObjective(move_weight=1.3), lambda net: ProfitObjective(net, cost_per_km=0.37)]
)
def test_rebalancing_cost_is_element_wise(network, make):
    """IObjective contract: solvers score many moves at once, with the same cost per element."""
    objective = make(network)
    rng = np.random.default_rng(0)
    times, dists = rng.uniform(0, 900, 50), rng.uniform(0, 9000, 50)
    costs = objective.rebalancing_cost(times, dists)
    assert costs.tolist() == [objective.rebalancing_cost(float(t), float(d)) for t, d in zip(times, dists)]
