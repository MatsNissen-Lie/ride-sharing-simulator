"""Perfect-information rebalancing benchmark."""

import random

import numpy as np

from src.components.rebalancing.oracle_rebalance_solver import OracleRebalancingSolver
from tests.conftest import random_vehicle

T_NOW = 10_000.0


def test_oracle_sends_idle_vehicles_only_where_they_arrive_in_time(network):
    rng = random.Random(0)
    vehicles = [random_vehicle(network, rng, f"v{k}", T_NOW, 0) for k in range(6)]
    times = np.sort([T_NOW + rng.uniform(1, 600) for _ in range(20)])
    origins = np.array([rng.randrange(network.N) for _ in times])
    solver = OracleRebalancingSolver(network, times, origins, horizon=300, omega=60)

    routes = solver.solve(vehicles, [], t_now=T_NOW)

    assert routes
    by_id = {v.vehicle_id: v for v in vehicles}
    in_horizon = [(t, o) for t, o in zip(times, origins) if T_NOW < t <= T_NOW + 300]
    for vid, route in routes.items():
        v = by_id[vid]
        arrival = v.available_from_time + network.get_travel_time(
            v.available_at_node, route.stops[0].node
        )
        assert any(o == route.stops[0].node and arrival <= t + 60 for t, o in in_horizon)


def test_oracle_ignores_requests_beyond_the_horizon(network):
    rng = random.Random(1)
    vehicles = [random_vehicle(network, rng, "v0", T_NOW, 0)]
    solver = OracleRebalancingSolver(
        network, np.array([T_NOW + 900.0]), np.array([5]), horizon=300
    )
    assert solver.solve(vehicles, [], t_now=T_NOW) == {}
