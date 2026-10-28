"""Ablation of a vehicle value: keep its time profile, drop where the vehicle is."""

import numpy as np

from src.interfaces import IVehicleValue


class ZoneAveragedValue(IVehicleValue):
    """
    The mean of `inner` over a set of reference nodes (e.g. zone centres) at
    the same time, so every node is worth the same: a value of vehicle time
    only. Isolates how much of a value's effect comes from timing (short and
    pooled trips) and how much from location.
    """

    def __init__(self, inner: IVehicleValue, reference_nodes: np.ndarray):
        self._inner = inner
        self._reference = np.asarray(reference_nodes, dtype=np.int64)

    def prepare(self, vehicles, requests, t_now):
        self._inner.prepare(vehicles, requests, t_now)

    def values(self, nodes, times, t_now):
        times = np.asarray(times, dtype=float)
        n = self._reference.size
        grid = self._inner.values(np.tile(self._reference, times.size), np.repeat(times, n), t_now)
        return grid.reshape(times.size, n).mean(axis=1)
