"""Calibrate config.TRAVEL_TIME_FACTOR: actual trip durations vs free-flow times.

For a sample of enriched TLC trips, compares the recorded duration
(dropoff - pickup) with the free-flow shortest-path time between the same
snapped origin and destination nodes (route database, OSM speed limits).
Reports the median ratio overall and per pickup hour.

    uv run python scripts/analysis/travel_time_factor.py
"""

import sys
from pathlib import Path

import duckdb
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.paths import DATA_DIR, RIDER_DIR  # noqa: E402

SAMPLE = 50_000
MIN_DURATION_S = 60


def main() -> None:
    files = sorted(RIDER_DIR.glob("trips_*_enriched*.parquet"))
    trips = (
        pl.scan_parquet(files)
        .select("orig_node", "dest_node", "pickup_datetime", "dropoff_datetime")
        .filter(pl.col("orig_node") != pl.col("dest_node"))
        .collect()
        .sample(SAMPLE, seed=1)
        .with_columns(
            (pl.col("dropoff_datetime") - pl.col("pickup_datetime")).dt.total_seconds().alias("actual_s"),
            pl.col("pickup_datetime").dt.hour().alias("hour"),
        )
        .filter(pl.col("actual_s") > MIN_DURATION_S)
    )
    con = duckdb.connect(str(DATA_DIR / "duckDB" / "route_db.duckdb"), read_only=True)
    con.register("pairs", trips.select(pl.col("orig_node").alias("o"), pl.col("dest_node").alias("d")).unique().to_arrow())
    free_flow = pl.from_arrow(con.sql("select o, d, time_s, dist_m from time_simple join pairs using (o, d)").arrow())
    j = trips.join(free_flow, left_on=["orig_node", "dest_node"], right_on=["o", "d"]).with_columns(
        (pl.col("actual_s") / pl.col("time_s")).alias("ratio")
    )
    print(f"trips: {j.height}")
    print(f"median actual / free-flow time: {j['ratio'].median():.2f} "
          f"(IQR {j['ratio'].quantile(0.25):.2f}-{j['ratio'].quantile(0.75):.2f})")
    print(f"median speed: free-flow {(j['dist_m'] / j['time_s'] * 3.6).median():.1f} km/h, "
          f"actual {(j['dist_m'] / j['actual_s'] * 3.6).median():.1f} km/h")
    print(j.group_by("hour").agg(pl.col("ratio").median().round(2), pl.len()).sort("hour"))


if __name__ == "__main__":
    main()
