"""C3's fleet plan over sampled demand futures."""

import random

import numpy as np
import pytest

import src.components.sampled_fleet_mpc as sampled_mpc
from src.components.objectives import DelayObjective
from src.components.sampled_fleet_mpc import SampledFleetMpcValue
from tests.conftest import random_requests, random_vehicle
from tests.test_fleet_mpc import T_NOW, FixedDemand, Zones, mpc, trips_to_same_zone


def sampled(network, demand, trip_time=600.0, cls=SampledFleetMpcValue, **kwargs):
    kwargs = {"step": 60.0, "n_steps": 10, "smoothing": 1.0, "workers": 1, **kwargs}
    return cls(network, Zones(), FixedDemand(demand), trips_to_same_zone(trip_time), DelayObjective(), 1e5, **kwargs)


class ForecastFutures(SampledFleetMpcValue):
    """Every future is the forecast itself."""

    def _sample(self, forecast):
        return forecast.copy()


def fleet(network, seed=0):
    rng = random.Random(seed)
    vehicles = {random_vehicle(network, rng, f"v{k}", T_NOW, rng.choice([0, 0, 1])) for k in range(5)}
    return vehicles, set(random_requests(network, rng, 3, T_NOW))


def test_futures_equal_to_the_forecast_give_the_forecast_plan(network):
    vehicles, riders = fleet(network)
    plain = mpc(network, {1: 0.02, 2: 0.05})
    futures = sampled(network, {1: 0.02, 2: 0.05}, cls=ForecastFutures, n_futures=3)
    for value in (plain, futures):
        value.prepare(vehicles, riders, T_NOW)
    nodes = np.arange(36)
    assert futures.values(nodes, np.full(36, T_NOW + 90), T_NOW) == pytest.approx(
        plain.values(nodes, np.full(36, T_NOW + 90), T_NOW), rel=1e-6, abs=1e-3
    )
    assert futures.moves == pytest.approx(plain.moves, abs=1e-6)


def test_every_future_keeps_the_waiting_riders_and_samples_the_forecast(network, monkeypatch):
    seen = []
    monkeypatch.setattr(sampled_mpc, "_solve", lambda job: (seen.append(job[1]), sampled_mpc.fleet_lp_values(*job))[1])
    vehicles, riders = fleet(network)
    value = sampled(network, {1: 0.02, 2: 0.05}, n_futures=4)
    value.prepare(vehicles, riders, T_NOW)
    waiting = value.demand_grid(riders, T_NOW) - value.demand_grid(set(), T_NOW)
    assert len(seen) == 4 and waiting.sum() == len(riders)
    for demand in seen:
        assert (demand - waiting >= 0).all() and np.allclose(demand - waiting, np.round(demand - waiting))
    assert not all(np.array_equal(seen[0], d) for d in seen[1:])  # futures differ


def test_values_are_the_mean_over_futures(network, monkeypatch):
    results = iter([(np.full((2, 10), 1.0), np.zeros((2, 2))), (np.full((2, 10), 3.0), np.ones((2, 2)))])
    monkeypatch.setattr(sampled_mpc, "_solve", lambda job: next(results))
    value = sampled(network, {1: 0.02}, n_futures=2)
    value.prepare(*fleet(network), T_NOW)
    assert value.values(np.array([0]), np.array([T_NOW]), T_NOW) == pytest.approx([2.0])
    assert value.moves == pytest.approx(np.full((2, 2), 0.5))


def test_process_pool_gives_the_same_plan(network):
    vehicles, riders = fleet(network, seed=3)
    one, pool = (sampled(network, {1: 0.02, 2: 0.05}, n_futures=3, workers=w) for w in (1, 2))
    for value in (one, pool):
        value.prepare(vehicles, riders, T_NOW)
    nodes = np.arange(36)
    assert pool.values(nodes, np.full(36, T_NOW), T_NOW) == pytest.approx(one.values(nodes, np.full(36, T_NOW), T_NOW))
