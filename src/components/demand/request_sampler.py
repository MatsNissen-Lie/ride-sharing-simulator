"""Virtual future requests sampled from the demand forecast and historical trips."""

from collections.abc import Iterable
from datetime import date, datetime, timedelta
from itertools import count
from pathlib import Path

import numpy as np
import polars as pl

from config.config import DELTA, OMEGA
from src.interfaces import IZoneIndex
from src.interfaces import IDemandModel, INetworkService, IRequestSampler, Request


def is_weekend(d: date) -> int:
    return int(d.weekday() >= 5)


class HistoricalRequestSampler(IRequestSampler):
    """
    One realisation of future demand, in the spirit of Alonso-Mora, Wallar &
    Rus (2017, IROS):

    - the number of requests per zone is Poisson with the demand model's
      expectation for the interval;
    - each request copies the origin and destination nodes of a random
      historical trip that started in the same zone, day type and time slot,
      on a training day (days in `exclude`, e.g. the simulated day and
      holidays, are never used); day types are weekday and weekend;
    - request times are uniform within each slot piece of the interval;
    - deadlines are those of real requests: pickup within omega, drop-off
      within the direct travel time plus delta.
    """

    def __init__(
        self,
        demand: IDemandModel,
        trips: pl.DataFrame,
        zones: IZoneIndex,
        t0: datetime,
        network: INetworkService,
        exclude: set[date] = frozenset(),
        slot_minutes: int = 15,
        omega: float = OMEGA,
        delta: float = DELTA,
        seed: int = 0,
    ):
        """`trips` needs columns pickup_datetime, orig_node and dest_node."""
        self._demand = demand
        self._t0 = t0
        self._slot = slot_minutes * 60
        self._times = network.get_l1_cache()
        self._omega, self._delta = omega, delta
        self._rng = np.random.default_rng(seed)
        self._ids = count()

        df = (
            trips.select(
                pl.col("pickup_datetime").dt.date().alias("day"),
                (
                    (pl.col("pickup_datetime").dt.hour().cast(pl.Int64) * 60
                     + pl.col("pickup_datetime").dt.minute().cast(pl.Int64)) // slot_minutes
                ).alias("slot"),
                pl.Series("zone", zones.zones_of(trips["orig_node"].to_numpy())),
                pl.col("orig_node").cast(pl.Int32),
                pl.col("dest_node").cast(pl.Int32),
            )
            .filter(~pl.col("day").is_in(sorted(exclude)))
            # Polars weekdays run 1 (Monday) to 7: 6 and 7 are the weekend.
            .with_columns((pl.col("day").dt.weekday() >= 6).cast(pl.Int64).alias("kind"))
            .with_columns(self._key(pl.col("kind"), pl.col("slot"), pl.col("zone")).alias("key"))
            .sort("key")
        )
        self._keys = df["key"].to_numpy()
        self._orig = df["orig_node"].to_numpy()
        self._dest = df["dest_node"].to_numpy()

    @staticmethod
    def read_trips(paths: Iterable[Path]) -> pl.DataFrame:
        """Pickup time, origin and destination node of every trip in the rider files."""
        return pl.concat(
            pl.read_parquet(path, columns=["pickup_datetime", "orig_node", "dest_node"]) for path in paths
        )

    @staticmethod
    def _key(kind, slot, zone):
        return (kind * 1000 + slot) * 1000 + zone

    def sample(self, t_start: float, t_end: float, t_now: float | None = None) -> list[Request]:
        out: list[Request] = []
        t = t_start
        while t < t_end:
            moment = self._t0 + timedelta(seconds=t)
            seconds_of_day = moment.hour * 3600 + moment.minute * 60 + moment.second
            piece_end = min(t + (self._slot - seconds_of_day % self._slot), t_end)
            kind = is_weekend(moment.date())
            slot = seconds_of_day // self._slot
            for zone, expected in sorted(self._demand.expected_requests(t, piece_end, t_now).items()):
                key = self._key(kind, slot, zone)
                lo, hi = np.searchsorted(self._keys, key, "left"), np.searchsorted(self._keys, key, "right")
                n = self._rng.poisson(expected)
                if n == 0 or hi == lo:
                    continue
                rows = self._rng.integers(lo, hi, n)
                for when, o, d in zip(self._rng.uniform(t, piece_end, n), self._orig[rows], self._dest[rows]):
                    out.append(self._request(float(when), int(o), int(d)))
            t = piece_end
        return out

    def _request(self, t: float, origin: int, destination: int) -> Request:
        return Request(
            id=f"virtual_{next(self._ids)}",
            origin_node=origin,
            destination_node=destination,
            request_time=t,
            latest_pickup_time=t + self._omega,
            latest_dropoff_time=t + float(self._times[origin, destination]) + self._delta,
        )
