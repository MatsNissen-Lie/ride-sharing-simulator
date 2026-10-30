"""Historical demand model and zone index."""

from datetime import date, datetime

import networkx as nx
import polars as pl
import pytest

from src.components.demand.demand_model import HistoricalDemandModel


class FakeZones:
    """Node id = zone id."""

    zone_ids = [1, 2]

    def zones_of(self, nodes):
        import numpy as np

        return np.asarray(nodes)


def trips(rows):
    return pl.DataFrame(rows, schema=["pickup_datetime", "orig_node"], orient="row")


T0 = datetime(2025, 1, 16)  # Thursday 00:00 = simulation time 0


def test_rates_average_over_training_days_and_separate_day_types():
    data = trips([
        (datetime(2025, 1, 14, 18, 5), 1),   # Tuesday
        (datetime(2025, 1, 15, 18, 10), 1),  # Wednesday
        (datetime(2025, 1, 15, 18, 12), 2),
        (datetime(2025, 1, 18, 18, 5), 1),   # Saturday: weekend, not used for Thursday
    ])
    model = HistoricalDemandModel.fit(data, FakeZones(), T0)
    t = 18 * 3600
    assert model.expected_requests(t, t + 900) == pytest.approx({1: 1.0, 2: 0.5})


def test_late_hours_land_in_the_right_slot():
    """Regression: hour * 60 overflowed an 8-bit integer after 02:00."""
    data = trips([(datetime(2025, 1, 15, 23, 50), 1), (datetime(2025, 1, 15, 2, 20), 2)])
    model = HistoricalDemandModel.fit(data, FakeZones(), T0)
    assert model.expected_requests(23 * 3600 + 45 * 60, 24 * 3600) == pytest.approx({1: 1.0})
    assert model.expected_requests(2 * 3600 + 15 * 60, 2 * 3600 + 30 * 60) == pytest.approx({2: 1.0})


def test_partial_slots_are_prorated_and_test_day_excluded():
    data = trips([
        (datetime(2025, 1, 15, 8, 1), 1),
        (datetime(2025, 1, 16, 8, 1), 1),  # the test day itself
    ])
    model = HistoricalDemandModel.fit(data, FakeZones(), T0, exclude={date(2025, 1, 16)})
    t = 8 * 3600
    assert model.expected_requests(t, t + 300) == pytest.approx({1: 1 / 3})
    assert model.expected_requests(t - 450, t + 450) == pytest.approx({1: 0.5})


def test_zone_centres_lie_in_their_zone():
    from src.components.network.zones import ZoneIndex
    from src.state_representation.manhatten_fast import load_graph

    zones = ZoneIndex(load_graph())
    assert len(zones.zone_ids) > 50
    assert all(zones.zone(node) == zone for zone, node in zones.center_node.items())
