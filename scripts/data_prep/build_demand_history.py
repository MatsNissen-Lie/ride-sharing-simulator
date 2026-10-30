"""Requests per day, 15-min slot and Manhattan taxi zone, from raw TLC trip records.

    uv run python scripts/data_prep/build_demand_history.py

Reads data/history/raw/{fhvhv,yellow}_tripdata_YYYY-MM.parquet (see
download_tlc.sh) and keeps trips with pickup and drop-off in Manhattan taxi
zones, as the rider preparation does (generate_rider_locations_multi.py;
there the trips are then snapped to street nodes). Writes
data/history/zone_counts_15min.parquet with columns day, slot (0-95), zone, n.
Also prints, for the months also in data/riders/, the totals against the
prepared rider files, to show how comparable the two are.
"""

import sys
from pathlib import Path

import geopandas as gpd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.paths import DATA_DIR, TAXI_ZONES_PATH  # noqa: E402

RAW = DATA_DIR / "history" / "raw"
OUT = DATA_DIR / "history" / "zone_counts_15min.parquet"
SLOT_MIN = 15


def manhattan_zones() -> list[int]:
    zones = gpd.read_file(TAXI_ZONES_PATH)
    return sorted(int(z) for z in zones.loc[zones["borough"] == "Manhattan", "LocationID"])


def trips(path: Path) -> pl.LazyFrame:
    pickup = "tpep_pickup_datetime" if path.name.startswith("yellow") else "pickup_datetime"
    return pl.scan_parquet(path).select(
        pl.col(pickup).cast(pl.Datetime("us")).alias("pickup_datetime"),
        pl.col("PULocationID").cast(pl.Int32).alias("zone"),
        pl.col("DOLocationID").cast(pl.Int32).alias("dropoff_zone"),
    )


def counts(path: Path, zones: list[int]) -> pl.DataFrame:
    month = path.stem.split("_")[-1]  # YYYY-MM: records outside the month are data errors
    return (
        trips(path)
        .filter(pl.col("zone").is_in(zones) & pl.col("dropoff_zone").is_in(zones))
        .with_columns(pl.col("pickup_datetime").dt.date().alias("day"))
        .filter(pl.col("day").dt.strftime("%Y-%m") == month)
        .with_columns(
            ((pl.col("pickup_datetime").dt.hour().cast(pl.Int32) * 60
              + pl.col("pickup_datetime").dt.minute().cast(pl.Int32)) // SLOT_MIN).alias("slot")
        )
        .group_by("day", "slot", "zone")
        .len(name="n")
        .collect()
    )


def main() -> None:
    zones = manhattan_zones()
    parts = [counts(p, zones) for p in sorted(RAW.glob("*_tripdata_*.parquet"))]
    table = (
        pl.concat(parts).group_by("day", "slot", "zone").agg(pl.col("n").sum())
        .sort("day", "slot", "zone")
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(OUT)
    days = table["day"].n_unique()
    print(f"{OUT}: {table.height} rows, {days} days ({table['day'].min()} to {table['day'].max()}), "
          f"{table['n'].sum()} requests")

    prepared = sorted((DATA_DIR / "riders").glob("trips_*_enriched*.parquet"))
    if prepared:
        riders = pl.concat(pl.read_parquet(p, columns=["pickup_datetime"]) for p in prepared)
        per_day = riders.group_by(pl.col("pickup_datetime").dt.date().alias("day")).len(name="prepared")
        both = table.group_by("day").agg(pl.col("n").sum().alias("raw")).join(per_day, on="day")
        ratio = both["prepared"].sum() / both["raw"].sum()
        print(f"Days in both: {both.height}; prepared riders / raw-filtered trips = {ratio:.3f}")


if __name__ == "__main__":
    main()
