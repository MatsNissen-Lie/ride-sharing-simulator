"""Forecast rebalancing as a transportation problem."""

import itertools
import random

import numpy as np
import pytest

from src.components.rebalancing.transport_rebalance_solver import TransportRebalancingSolver
from src.interfaces import IDemandModel, Stop, VehicleAvailability
from tests.conftest import make_request

T_NOW = 10_000.0


class Zones:
    """Three zones on the 6x6 grid by column: 0-1, 2-3, 4-5."""

    zone_ids = np.array([1, 2, 3])
    center_node = {1: 12, 2: 14, 3: 17}

    def zone(self, node):
        return 1 + (node % 6) // 2

    def zones_of(self, nodes):
        return np.array([self.zone(n) for n in nodes])


class FixedDemand(IDemandModel):
    def __init__(self, per_zone):
        self.per_zone = per_zone

    def expected_requests(self, t_start, t_end):
        return dict(self.per_zone)


def idle(vid, node, t=T_NOW):
    return VehicleAvailability(vid, 4, node, t, frozenset(), ())


def solver(network, demand, **kwargs):
    return TransportRebalancingSolver(network, Zones(), FixedDemand(demand), **kwargs)


def outcome(network, vehicles, slots, routes, max_travel):
    """(places filled, total travel) of a rebalancing decision."""
    zones = Zones()
    count = dict.fromkeys(zones.zone_ids.tolist(), 0)
    travel = 0.0
    for v in vehicles:
        route = routes.get(v.vehicle_id)
        if route and route.stops:
            dest = route.stops[0].node
            travel += network.get_travel_time(v.available_at_node, dest)
            count[zones.zone(dest)] += 1
        else:
            count[zones.zone(v.available_at_node)] += 1
    filled = sum(min(count[z], slots[k]) for k, z in enumerate(zones.zone_ids.tolist()))
    return filled, travel


def brute_force(network, vehicles, slots, max_travel):
    zones = Zones()
    best = (-1, 0.0)
    for choice in itertools.product([None, 1, 2, 3], repeat=len(vehicles)):
        routes, ok = {}, True
        for v, z in zip(vehicles, choice):
            if z is None or z == zones.zone(v.available_at_node):
                continue
            dest = zones.center_node[z]
            if network.get_travel_time(v.available_at_node, dest) > max_travel:
                ok = False
                break
            routes[v.vehicle_id] = type("R", (), {"stops": (Stop.rebalancing(dest, "z"),)})()
        if ok:
            filled, travel = outcome(network, vehicles, slots, routes, max_travel)
            if (filled, -travel) > (best[0], -best[1]):
                best = (filled, travel)
    return best


@pytest.mark.parametrize("seed", range(6))
def test_transport_fills_most_places_with_least_travel(network, seed):
    rng = random.Random(seed)
    vehicles = [idle(f"v{k}", rng.randrange(network.N)) for k in range(5)]
    slots = np.array(rng.choice([[3, 1, 1], [0, 2, 3], [1, 1, 3], [0, 0, 5], [2, 2, 1]]))
    max_travel = rng.choice([60.0, 120.0, 1e6])
    routes = solver(network, {}, max_travel=max_travel)._transport(vehicles, slots)
    got = outcome(network, vehicles, slots, routes, max_travel)
    best = brute_force(network, vehicles, slots, max_travel)
    assert got[0] == best[0]
    assert got[1] == pytest.approx(best[1])


def test_balanced_fleet_stays_put(network):
    vehicles = [idle("a", 0), idle("b", 2), idle("c", 4)]  # one per zone
    assert solver(network, {1: 1.0, 2: 1.0, 3: 1.0}).solve(vehicles, [], T_NOW) == {}


def test_surplus_moves_to_the_nearest_deficit(network):
    vehicles = [idle("near", 1), idle("far", 0), idle("stay", 6)]  # all in zone 1
    routes = solver(network, {1: 1.0, 2: 1.0, 3: 1.0}, max_travel=1e6).solve(vehicles, [], T_NOW)
    assert {r.stops[0].node for r in routes.values()} == {14, 17}
    assert len(routes) == 2 and all(r.stops[0].is_rebalancing for r in routes.values())


def test_only_riders_reachable_in_time_are_chased(network):
    near, far = idle("near", 8), idle("far", 35)
    reachable = make_request(network, "ok", 7, 0, T_NOW)
    hopeless = make_request(network, "late", 30, 0, T_NOW - 170)  # 10 s left
    routes = solver(network, {}).solve([near, far], [reachable, hopeless], T_NOW)
    assert routes["near"].stops[0].request_id.endswith("ok")
    assert "far" not in routes or routes["far"].stops[0].node != 30


def test_vehicles_sent_to_riders_and_zone_moves_count_for_targets(network):
    rider = make_request(network, "r", 5, 0, T_NOW)  # origin in zone 3
    mover = VehicleAvailability(
        "m", 4, 0, T_NOW, frozenset(), (Stop.rebalancing(14, "zone2"),)
    )
    vehicles = [idle("a", 4), idle("b", 0)]  # a reaches the rider, b sits in zone 1
    routes = solver(network, {1: 1.0, 2: 1.0, 3: 1.0}, max_travel=1e6).solve(
        vehicles, [rider], T_NOW, en_route=[mover]
    )
    # a -> rider (zone 3), mover -> zone 2, b stays: one vehicle per zone, no zone moves.
    assert set(routes) == {"a"} and routes["a"].stops[0].node == 5


def test_rider_chasers_stop_when_nothing_is_left_for_them(network):
    chaser = VehicleAvailability("c", 4, 0, T_NOW, frozenset(), (Stop.rebalancing(30, "gone"),))
    routes = solver(network, {1: 1.0}).solve([], [], T_NOW, en_route=[chaser])
    assert routes == {"c": routes["c"]} and routes["c"].stops == ()
