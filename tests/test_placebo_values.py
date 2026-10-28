"""Placebo values: same plan and moves, values without location meaning."""

import numpy as np

from src.components.placebo_values import ZoneAveragedPlan, ZoneShuffledPlan
from src.interfaces import IFleetPlan


class Zones:
    """Three zones on the 6x6 grid by column: 0-1, 2-3, 4-5."""

    zone_ids = np.array([1, 2, 3])
    center_node = {1: 12, 2: 14, 3: 17}

    def zone(self, node):
        return 1 + (node % 6) // 2

    def zones_of(self, nodes):
        return np.array([self.zone(n) for n in nodes])


class Plan(IFleetPlan):
    """Value 10 * zone + time; a fixed move matrix."""

    prepared = 0

    def prepare(self, vehicles, requests, t_now):
        self.prepared += 1

    def values(self, nodes, times, t_now):
        return np.array([10.0 * Zones().zone(n) for n in nodes]) + np.asarray(times, dtype=float)

    @property
    def t_solved(self):
        return 5.0

    @property
    def moves(self):
        return np.eye(3)


def test_shuffled_values_are_a_permutation_of_zones():
    plan = Plan()
    shuffled = ZoneShuffledPlan(plan, Zones(), seed=3)
    centres = np.array([12, 14, 17])
    original = plan.values(centres, np.zeros(3), 0.0)
    permuted = shuffled.values(centres, np.zeros(3), 0.0)
    assert sorted(permuted) == sorted(original)
    # Nodes of one zone share the stand-in's value; time still counts.
    assert shuffled.values(np.array([0, 1]), np.array([0.0, 2.0]), 0.0)[1] == shuffled.values(
        np.array([0]), np.array([0.0]), 0.0
    )[0] + 2.0


def test_placebos_pass_the_plan_through():
    plan = Plan()
    for placebo in (ZoneShuffledPlan(plan, Zones()), ZoneAveragedPlan(plan, np.array([12, 14, 17]))):
        placebo.prepare([], [], 0.0)
        assert placebo.t_solved == 5.0 and (placebo.moves == np.eye(3)).all()
    assert plan.prepared == 2


def test_averaged_plan_values_every_node_alike():
    flat = ZoneAveragedPlan(Plan(), np.array([12, 14, 17]))
    v = flat.values(np.array([0, 3, 5]), np.array([1.0, 1.0, 1.0]), 0.0)
    assert np.allclose(v, 21.0)  # mean of 10, 20, 30 plus the time
