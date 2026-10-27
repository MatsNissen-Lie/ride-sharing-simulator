"""Normalized fares: a solo price model and the empirical pooling discount.

1. Solo price model: least squares fare = a + b*km + c*min on solo HVFHS trips
   within Manhattan (base_passenger_fare; trip_miles, trip_time as recorded).
2. Pooling discount: each Manhattan shared request is compared with the median
   solo fare for the same pickup zone, drop-off zone and hour (cells with at
   least MIN_SOLO_TRIPS solo trips). Comparing within the cell avoids using the
   shared ride's own distance and time, which include pooling detours.

    uv run python scripts/analysis/fare_model.py
"""

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.paths import RIDER_DIR, TAXI_ZONES_PATH  # noqa: E402

FHVHV = RIDER_DIR / "fhvhv_tripdata_2025-01.parquet"
FIT_SAMPLE = 500_000
MIN_SOLO_TRIPS = 20
KM_PER_MILE = 1.609344


def manhattan_trips() -> pl.DataFrame:
    zones = gpd.read_file(TAXI_ZONES_PATH)
    manhattan = zones.loc[zones["borough"] == "Manhattan", "LocationID"].astype(int).tolist()
    return (
        pl.scan_parquet(FHVHV)
        .filter(
            (pl.col("trip_miles") > 0.2)
            & (pl.col("trip_time") > 60)
            & (pl.col("base_passenger_fare") > 0)
            & pl.col("PULocationID").is_in(manhattan)
            & pl.col("DOLocationID").is_in(manhattan)
        )
        .select(
            "PULocationID", "DOLocationID", "base_passenger_fare", "trip_miles", "trip_time",
            "shared_request_flag", "shared_match_flag", pl.col("pickup_datetime").dt.hour().alias("hour"),
        )
        .collect()
    )


def solo_price_model(trips: pl.DataFrame) -> None:
    solo = trips.filter(pl.col("shared_request_flag") == "N").sample(FIT_SAMPLE, seed=0)
    x = np.column_stack([
        np.ones(solo.height),
        solo["trip_miles"].to_numpy() * KM_PER_MILE,
        solo["trip_time"].to_numpy() / 60,
    ])
    y = solo["base_passenger_fare"].to_numpy()
    (a, b, c), *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ np.array([a, b, c])
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"solo fare = {a:.2f} + {b:.2f}/km + {c:.2f}/min   "
          f"(R2 {r2:.3f}, median abs error {np.median(np.abs(y - pred)):.2f}, n={solo.height})")


def pooling_discount(trips: pl.DataFrame) -> None:
    cell = ["PULocationID", "DOLocationID", "hour"]
    solo = (
        trips.filter(pl.col("shared_request_flag") == "N")
        .group_by(cell)
        .agg(
            pl.col("base_passenger_fare").median().alias("solo_fare"),
            pl.col("trip_miles").median().alias("solo_miles"),
            pl.col("trip_time").median().alias("solo_time"),
            pl.len().alias("n_solo"),
        )
        .filter(pl.col("n_solo") >= MIN_SOLO_TRIPS)
    )
    shared = trips.filter(pl.col("shared_request_flag") == "Y")
    matched = shared.join(solo, on=cell).with_columns(
        (pl.col("base_passenger_fare") / pl.col("solo_fare")).alias("fare_ratio"),
        (pl.col("trip_miles") / pl.col("solo_miles")).alias("distance_ratio"),
        (pl.col("trip_time") / pl.col("solo_time")).alias("time_ratio"),
    )
    print(f"shared requests compared: {matched.height} of {shared.height}")
    print(matched.group_by("shared_match_flag").agg(
        pl.len(),
        pl.col("fare_ratio").median().round(3),
        pl.col("fare_ratio").quantile(0.25).round(3).alias("q25"),
        pl.col("fare_ratio").quantile(0.75).round(3).alias("q75"),
        pl.col("distance_ratio").median().round(2),
        pl.col("time_ratio").median().round(2),
    ))


if __name__ == "__main__":
    trips = manhattan_trips()
    solo_price_model(trips)
    pooling_discount(trips)
