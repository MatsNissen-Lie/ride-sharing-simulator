"""Forecast-based rebalancing."""

import random

import numpy as np
import pytest

from src.components.demand_rebalance_solver import DemandRebalancingSolver
from src.interfaces import IDemandModel
from tests.conftest import make_request, random_vehicle

T_NOW = 10_000.0


class Zones:
    """Two zones on the synthetic grid: left half (1) and right half (2)."""

    zone_ids = np.array([1, 2])
    center_node = {1: 14, 2: 21}

    def zone(self, node):
        return 1 if node % 6 < 3 else 2

    def zones_of(self, nodes):
        return np.array([self.zone(n) for n in nodes])


class FixedDemand(IDemandModel):
    def __init__(self, per_zone):
        self.per_zone = per_zone

    def expected_requests(self, t_start, t_end):
        return dict(self.per_zone)


def vehicles_in(network, zone, n, start=0):
    rng = random.Random(zone)
    nodes = [k for k in range(network.N) if Zones().zone(k) == zone]
    out = []
    for k in range(n):
        v = random_vehicle(network, rng, f"v{zone}_{start + k}", T_NOW, 0)
        out.append(type(v)(**{**v.__dict__, "available_at_node": rng.choice(nodes), "available_from_time": T_NOW}))
    return out


def test_moves_surplus_to_where_demand_is(network):
    idle = vehicles_in(network, 1, 8)  # all vehicles on the left
    solver = DemandRebalancingSolver(network, Zones(), FixedDemand({1: 1.0, 2: 3.0}), max_travel=1e6)
    routes = solver.solve(idle, [], T_NOW)
    # target: 2 left, 6 right -> 6 vehicles move right
    assert len(routes) == 6
    assert all(Zones().zone(r.stops[0].node) == 2 for r in routes.values())
    assert all(r.stops[0].is_rebalancing for r in routes.values())


def test_balanced_fleet_stays_put(network):
    idle = vehicles_in(network, 1, 3) + vehicles_in(network, 2, 3)
    solver = DemandRebalancingSolver(network, Zones(), FixedDemand({1: 5.0, 2: 5.0}))
    assert solver.solve(idle, [], T_NOW) == {}


def test_vehicles_en_route_count_at_their_destination(network):
    idle = vehicles_in(network, 1, 4)
    heading_right = vehicles_in(network, 2, 4, start=10)
    en_route = [
        type(v)(**{**v.__dict__, "remaining_stops_to_serve": (__import__("src.interfaces", fromlist=["Stop"]).Stop.rebalancing(21, "z"),)})
        for v in heading_right
    ]
    solver = DemandRebalancingSolver(network, Zones(), FixedDemand({1: 1.0, 2: 1.0}))
    # 4 idle left + 4 heading right = balanced: nothing to do.
    assert solver.solve(idle, [], T_NOW, en_route=en_route) == {}


def test_unassigned_riders_get_a_vehicle_first(network):
    idle = vehicles_in(network, 1, 2)
    rider = make_request(network, "r", 33, 0, T_NOW - 30)  # origin on the right
    solver = DemandRebalancingSolver(network, Zones(), FixedDemand({1: 1.0, 2: 0.0}), max_travel=1e6)
    routes = solver.solve(idle, [rider], T_NOW)
    assert any(r.stops[0].node == 33 for r in routes.values())


def test_integer_targets_sum_to_the_fleet():
    t = DemandRebalancingSolver._integer_targets(np.array([2.4, 3.3, 1.3]))
    assert t.sum() == 7 and list(t) == [3, 3, 1]
