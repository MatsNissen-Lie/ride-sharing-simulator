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
    sums the slot averages it covers, prorating partial slots.
    """

    def __init__(
        self,
        rates: dict[tuple[str, int], dict[int, float]],
        t0: datetime,
        slot_minutes: int = 15,
    ):
        self._rates = rates
        self._t0 = t0
        self._slot = slot_minutes * 60

    @classmethod
    def fit(
        cls,
        trips: pl.DataFrame,
        zones: IZoneIndex,
        t0: datetime,
        exclude: set[date] = frozenset(),
        slot_minutes: int = 15,
    ) -> "HistoricalDemandModel":
        """`trips` needs columns pickup_datetime and orig_node."""
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
        days = df.select("day").unique()["day"].to_list()
        n_days = {kind: sum(day_type(d) == kind for d in days) for kind in ("weekday", "weekend")}
        counts = (
            # day_type per row; polars counts Monday = 1 .. Sunday = 7
            df.with_columns((pl.col("day").dt.weekday() >= 6).alias("weekend"))
            .group_by("weekend", "slot", "zone")
            .len()
        )
        rates: dict[tuple[str, int], dict[int, float]] = {}
        for weekend, slot, zone, n in counts.iter_rows():
            kind = "weekend" if weekend else "weekday"
            rates.setdefault((kind, slot), {})[zone] = n / n_days[kind]
        return cls(rates, t0, slot_minutes)

    def expected_requests(self, t_start: float, t_end: float) -> dict[int, float]:
        out: dict[int, float] = {}
        t = t_start
        while t < t_end:
            moment = self._t0 + timedelta(seconds=t)
            seconds_of_day = moment.hour * 3600 + moment.minute * 60 + moment.second
            slot = seconds_of_day // self._slot
            slot_end = t + (self._slot - seconds_of_day % self._slot)
            share = (min(slot_end, t_end) - t) / self._slot
            for zone, rate in self._rates.get((day_type(moment.date()), slot), {}).items():
                out[zone] = out.get(zone, 0.0) + share * rate
            t = slot_end
        return out


def zone_vector(expected: dict[int, float], zone_ids: np.ndarray) -> np.ndarray:
    return np.array([expected.get(int(z), 0.0) for z in zone_ids])


def integer_targets(fractional: np.ndarray) -> np.ndarray:
    """Largest-remainder rounding: integers that sum to round(sum(fractional))."""
    base = np.floor(fractional)
    missing = int(round(fractional.sum() - base.sum()))
    order = np.argsort(-(fractional - base))
    base[order[:missing]] += 1
    return base
