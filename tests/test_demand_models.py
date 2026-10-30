"""Swappable demand models: day of week, nowcast, oracle."""

from datetime import date, datetime

import numpy as np
import pytest

from src.components.demand.demand_model import DayOfWeekDemandModel, HistoricalDemandModel, OracleDemandModel
from src.components.demand.nowcast_model import NowcastDemandModel
from src.interfaces import IDemandModel
from tests.test_demand import T0, FakeZones, trips

SIX_PM = 18 * 3600


def test_day_of_week_uses_only_the_same_weekday():
    data = trips([
        (datetime(2025, 1, 9, 18, 5), 1),   # Thursday
        (datetime(2025, 1, 9, 18, 6), 1),
        (datetime(2025, 1, 15, 18, 5), 2),  # Wednesday: a weekday, but not a Thursday
    ])
    thursday = DayOfWeekDemandModel.fit(data, FakeZones(), T0)  # T0 is a Thursday
    weekday = HistoricalDemandModel.fit(data, FakeZones(), T0)
    assert thursday.expected_requests(SIX_PM, SIX_PM + 900) == pytest.approx({1: 2.0})
    assert weekday.expected_requests(SIX_PM, SIX_PM + 900) == pytest.approx({1: 1.0, 2: 0.5})


def test_oracle_is_the_simulated_days_own_requests():
    data = trips([
        (datetime(2025, 1, 15, 18, 5), 1),  # training day: ignored
        (datetime(2025, 1, 16, 18, 1), 2),
        (datetime(2025, 1, 16, 18, 14), 2),
        (datetime(2025, 1, 16, 18, 20), 1),
    ])
    oracle = OracleDemandModel.fit(data, FakeZones(), T0, only={date(2025, 1, 16)})
    assert oracle.expected_requests(SIX_PM, SIX_PM + 900) == pytest.approx({2: 2.0})
    assert oracle.expected_requests(SIX_PM + 900, SIX_PM + 1800) == pytest.approx({1: 1.0})


class Flat(IDemandModel):
    """`rate` requests per second in zone 1."""

    def __init__(self, rate):
        self.rate = rate

    def expected_requests(self, t_start, t_end, t_now=None):
        return {1: self.rate * (t_end - t_start)}


def test_nowcast_scales_by_observed_over_forecast():
    # Forecast 0.1/s, so 360 in the last hour; 720 observed: ratio (720 + 50) / (360 + 50).
    times = np.linspace(SIX_PM - 3600, SIX_PM - 1, 720)
    model = NowcastDemandModel(Flat(0.1), times, window=3600, prior=50, hi=3.0)
    ratio = (720 + 50) / (360 + 50)
    assert model.expected_requests(SIX_PM, SIX_PM + 600, SIX_PM) == pytest.approx({1: 60 * ratio})


def test_nowcast_never_looks_past_now():
    early = np.linspace(SIX_PM - 3600, SIX_PM - 1, 360)
    later = np.concatenate([early, np.linspace(SIX_PM, SIX_PM + 600, 1000)])  # riders still to come
    a, b = (NowcastDemandModel(Flat(0.1), t, window=3600) for t in (early, later))
    assert a.expected_requests(SIX_PM, SIX_PM + 600, SIX_PM) == b.expected_requests(SIX_PM, SIX_PM + 600, SIX_PM)
    assert a.ratio(SIX_PM) == pytest.approx(1.0)


def test_nowcast_without_now_is_the_base_and_the_ratio_is_clipped():
    model = NowcastDemandModel(Flat(0.1), np.zeros(0), window=3600, prior=0.0, lo=0.5, hi=2.0)
    assert model.expected_requests(SIX_PM, SIX_PM + 600) == pytest.approx({1: 60.0})
    assert model.ratio(SIX_PM) == pytest.approx(0.5)  # nothing observed: clipped at lo
