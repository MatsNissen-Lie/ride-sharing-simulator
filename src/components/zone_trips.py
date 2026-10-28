"""Where riders go and how long their trips take, per pair of zones."""

from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
import polars as pl

from src.components.demand import HOLIDAYS, day_type
from src.interfaces import IZoneIndex


@dataclass
class ZoneTrips:
    """
    Where riders go and how long it takes, per pair of zones (in IZoneIndex.zone_ids order).

    shares[(day type, hour)]: (Z, Z) destination shares of riders by origin zone.
    trip_time: (Z, Z) mean shortest-path travel time of riders' trips plus
    `pickup_time` (the vehicle's drive to the rider), in seconds.
    """

    shares: dict[tuple[str, int], np.ndarray]
    trip_time: np.ndarray
    t0: datetime

    @classmethod
    def fit(
        cls,
        trips: pl.DataFrame,
        zones: IZoneIndex,
        travel_times: np.ndarray,
        t0: datetime,
        exclude: set[date] = frozenset(),
        pickup_time: float = 90.0,
    ) -> "ZoneTrips":
        """`trips` needs pickup_datetime, orig_node, dest_node; `travel_times` is node x node."""
        zone_ids = np.asarray(zones.zone_ids)
        n = zone_ids.size
        o_node, d_node = trips["orig_node"].to_numpy(), trips["dest_node"].to_numpy()
        df = pl.DataFrame({
            "day": trips["pickup_datetime"].dt.date(),
            "hour": trips["pickup_datetime"].dt.hour().cast(pl.Int32),
            "o": np.searchsorted(zone_ids, zones.zones_of(o_node)),
            "d": np.searchsorted(zone_ids, zones.zones_of(d_node)),
            "t": travel_times[o_node, d_node],
        }).filter(~pl.col("day").is_in(sorted(exclude | HOLIDAYS)))

        shares: dict[tuple[str, int], np.ndarray] = {}
        counts = (
            df.with_columns(pl.col("day").map_elements(day_type, return_dtype=pl.String).alias("kind"))
            .group_by("kind", "hour", "o", "d").len()
        )
        for (kind, hour), g in counts.group_by("kind", "hour"):
            m = np.zeros((n, n))
            m[g["o"].to_numpy(), g["d"].to_numpy()] = g["len"].to_numpy()
            shares[(kind, int(hour))] = m / np.maximum(m.sum(axis=1, keepdims=True), 1)

        mean = df.filter(pl.col("t").is_finite() & (pl.col("t") >= 0)).group_by("o", "d").agg(pl.col("t").mean())
        trip_time = np.full((n, n), np.nan)
        trip_time[mean["o"].to_numpy(), mean["d"].to_numpy()] = mean["t"].to_numpy()
        centres = np.array([zones.center_node[int(z)] for z in zone_ids])
        fallback = travel_times[np.ix_(centres, centres)]
        trip_time = np.where(np.isnan(trip_time), fallback, trip_time) + pickup_time
        return cls(shares, trip_time, t0)

    def shares_at(self, t: float) -> np.ndarray:
        moment = self.t0 + timedelta(seconds=t)
        n = self.trip_time.shape[0]
        return self.shares.get((day_type(moment.date()), moment.hour), np.zeros((n, n)))
