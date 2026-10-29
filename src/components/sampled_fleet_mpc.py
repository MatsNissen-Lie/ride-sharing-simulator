"""C3's fleet plan over sampled demand futures, solved in parallel."""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from src.components.fleet_mpc import FleetMpcValue, fleet_lp_values
from src.utils.timing import timed


def _solve(args) -> tuple[np.ndarray, np.ndarray]:
    return fleet_lp_values(*args)


class SampledFleetMpcValue(FleetMpcValue):
    """
    FleetMpcValue whose plan averages `n_futures` sampled futures instead of
    planning for the forecast: in each future the demand per zone and step is
    Poisson around the forecast (riders waiting now are known and the same in
    every future). Each future's fleet LP is solved on its own, on a pool of
    `workers` processes, and the duals (values) and step-0 moves are the means
    over futures.

    This approximates the two-stage programme in which the step-0 moves are
    shared and later steps adapt to each future: here each future is planned
    knowing itself, so a move that pays in some futures only is rewarded by its
    share of futures. The exact programme (one LP with all futures) takes 46 s
    with ten futures at the evening peak; ten separate LPs on four processes
    take about 4 s (scripts/analysis/fleet_lp_timing.py).
    """

    def __init__(self, *args, n_futures: int = 10, workers: int = 4, seed: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self._n_futures = n_futures
        self._workers = workers
        self._rng = np.random.default_rng(seed)
        self._forecast = np.zeros(0)
        self._pool: ProcessPoolExecutor | None = None

    def demand_grid(self, requests, t_now):
        self._forecast = super().demand_grid(set(), t_now)
        return super().demand_grid(requests, t_now)

    def _plan(self, supply, demand, shares):
        waiting = demand - self._forecast
        fixed = (shares, *self._lp_constants())
        jobs = [(supply, self._sample(self._forecast) + waiting, *fixed) for _ in range(self._n_futures)]
        with timed("fleet_lp_futures"):
            if self._workers > 1:
                if self._pool is None:
                    # spawn, not fork: the simulator's threads must not be copied into the workers.
                    self._pool = ProcessPoolExecutor(self._workers, mp_context=multiprocessing.get_context("spawn"))
                results = list(self._pool.map(_solve, jobs))
            else:
                results = [_solve(job) for job in jobs]
        return np.mean([r[0] for r in results], axis=0), np.mean([r[1] for r in results], axis=0)

    def _sample(self, forecast: np.ndarray) -> np.ndarray:
        """One future's demand per zone and step."""
        return self._rng.poisson(forecast).astype(float)
