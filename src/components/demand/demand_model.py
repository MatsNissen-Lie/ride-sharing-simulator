"""Demand models: expected requests per zone and time."""

from datetime import date, datetime, timedelta

import numpy as np
import polars as pl

from src.interfaces import IDemandModel, IZoneIndex

# Days left out of the profile: holidays with atypical demand.
HOLIDAYS = {date(2025, 1, 1), date(2025, 1, 20)}


def day_type(d: date) -> str:
    return "weekend" if d.weekday() >= 5 else "weekday"


class HistoricalDemandModel(IDemandModel):
    """
    Average requests per (day type, time slot, zone) over training days.

    Day type is weekday or weekend; slots are `slot_minutes` long; a request
    belongs to the zone of its origin node. The expectation for an interval
    sums the slot averages it covers, prorating partial slots. Subclasses
    group days differently by overriding `day_key`.
    """

    def __init__(
        self,
        rates: dict[tuple, dict[int, float]],
        t0: datetime,
        slot_minutes: int = 15,
    ):
        self._rates = rates
        self._t0 = t0
        self._slot = slot_minutes * 60

    @staticmethod
    def day_key(d: date):
        """Days with the same key share a profile."""
        return day_type(d)

    @classmethod
    def fit(
        cls,
        trips: pl.DataFrame,
        zones: IZoneIndex,
        t0: datetime,
        exclude: set[date] = frozenset(),
        slot_minutes: int = 15,
        only: set[date] | None = None,
    ) -> "HistoricalDemandModel":
        """
        `trips` needs columns pickup_datetime and orig_node. Days in `exclude`
        and holidays are left out; with `only`, just those days are used.
        """
        df = (
            trips.select(
                pl.col("pickup_datetime"),
                pl.Series("zone", zones.zones_of(trips["orig_node"].to_numpy())),
            )
            .with_columns(
                pl.col("pickup_datetime").dt.date().alias("day"),
                (
                    # dt.hour() is an 8-bit integer: cast before multiplying.
                    (pl.col("pickup_datetime").dt.hour().cast(pl.Int32) * 60 + pl.col("pickup_datetime").dt.minute().cast(pl.Int32))
                    // slot_minutes
                ).alias("slot"),
            )
            .filter(~pl.col("day").is_in(sorted(exclude | HOLIDAYS)))
        )
        if only is not None:
            df = df.filter(pl.col("day").is_in(sorted(only)))
        days = df.select("day").unique()["day"].to_list()
        keys = {d: cls.day_key(d) for d in days}
        n_days: dict = {}
        for k in keys.values():
            n_days[k] = n_days.get(k, 0) + 1
        counts = df.group_by("day", "slot", "zone").len()
        rates: dict[tuple, dict[int, float]] = {}
        for day, slot, zone, n in counts.iter_rows():
            k = keys[day]
            zone_rates = rates.setdefault((k, slot), {})
            zone_rates[zone] = zone_rates.get(zone, 0.0) + n / n_days[k]
        return cls(rates, t0, slot_minutes)

    def expected_requests(self, t_start: float, t_end: float, t_now: float | None = None) -> dict[int, float]:
        out: dict[int, float] = {}
        t = t_start
        while t < t_end:
            moment = self._t0 + timedelta(seconds=t)
            seconds_of_day = moment.hour * 3600 + moment.minute * 60 + moment.second
            slot = seconds_of_day // self._slot
            slot_end = t + (self._slot - seconds_of_day % self._slot)
            share = (min(slot_end, t_end) - t) / self._slot
            for zone, rate in self._rates.get((self.day_key(moment.date()), slot), {}).items():
                out[zone] = out.get(zone, 0.0) + share * rate
            t = slot_end
        return out


class DayOfWeekDemandModel(HistoricalDemandModel):
    """Average requests per (weekday, slot, zone): Thursdays are forecast from past Thursdays only."""

    @staticmethod
    def day_key(d: date):
        return d.weekday()


class OracleDemandModel(HistoricalDemandModel):
    """
    The simulated day's own requests per slot and zone (fit with `only` = the
    simulated days): perfect knowledge of demand at slot resolution. Not a
    forecast; the upper bound of what forecasting can give.
    """

    @staticmethod
    def day_key(d: date):
        return d


def zone_vector(expected: dict[int, float], zone_ids: np.ndarray) -> np.ndarray:
    return np.array([expected.get(int(z), 0.0) for z in zone_ids])


def integer_targets(fractional: np.ndarray) -> np.ndarray:
    """Largest-remainder rounding: integers that sum to round(sum(fractional))."""
    base = np.floor(fractional)
    missing = int(round(fractional.sum() - base.sum()))
    order = np.argsort(-(fractional - base))
    base[order[:missing]] += 1
    return base
