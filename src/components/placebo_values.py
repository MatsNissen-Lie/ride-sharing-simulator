"""Placebo controls for a fleet plan's values: same plan and moves, values stripped of meaning.

Used to test whether the joint ILP gains from what the values say (where vehicles
will be needed) or merely from perturbing near-tied choices.
"""

import numpy as np

from src.components.flat_value import ZoneAveragedValue
from src.interfaces import IFleetPlan, IZoneIndex


class ZoneShuffledPlan(IFleetPlan):
    """
    The inner plan's values with zones randomly permuted: a node in zone z is
    valued as the representative node of zone perm(z). Same numbers, same time
    profile, wrong places. The plan's moves are passed through unchanged.
    """

    def __init__(self, inner: IFleetPlan, zones: IZoneIndex, seed: int = 0):
        self._inner = inner
        self._zones = zones
        ids = [int(z) for z in zones.zone_ids]
        perm = np.random.default_rng(seed).permutation(len(ids))
        self._stand_in = {z: zones.center_node[ids[perm[k]]] for k, z in enumerate(ids)}

    def prepare(self, vehicles, requests, t_now):
        self._inner.prepare(vehicles, requests, t_now)

    def values(self, nodes, times, t_now):
        stand_in = [self._stand_in[int(z)] for z in self._zones.zones_of(nodes)]
        return self._inner.values(np.array(stand_in, dtype=np.int64), times, t_now)

    @property
    def t_solved(self):
        return self._inner.t_solved

    @property
    def moves(self):
        return self._inner.moves


class ZoneAveragedPlan(ZoneAveragedValue, IFleetPlan):
    """ZoneAveragedValue (time only, no location) that passes the inner plan's moves through."""

    def __init__(self, inner: IFleetPlan, reference_nodes: np.ndarray):
        super().__init__(inner, reference_nodes)
        self._plan = inner

    @property
    def t_solved(self):
        return self._plan.t_solved

    @property
    def moves(self):
        return self._plan.moves
