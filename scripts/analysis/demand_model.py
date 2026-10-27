"""Demand model: fit on other days, evaluate on the test day.

Fits HistoricalDemandModel (weekday/weekend x 15-min slot x taxi zone of the
origin node) on January 2025 excluding the test day and holidays, then
compares the forecast with the test day's actual requests per zone and slot.

    uv run python scripts/analysis/demand_model.py [YYYY-MM-DD]
"""

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import geopandas as gpd
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.paths import RIDER_DIR, TAXI_ZONES_PATH  # noqa: E402
from src.components.demand import HistoricalDemandModel  # noqa: E402
from src.components.zones import ZoneIndex  # noqa: E402
from src.state_representation.manhatten_fast import load_graph  # noqa: E402

SLOT_MIN = 15


def main(test_day: date) -> None:
    trips = (
        pl.scan_parquet(sorted(RIDER_DIR.glob("trips_*_enriched*.parquet")))
        .select("pickup_datetime", "orig_node")
        .filter(pl.col("pickup_datetime").dt.year() == 2025)
        .collect()
    )
    zones = ZoneIndex(load_graph())
    t0 = datetime.combine(test_day, datetime.min.time())
    model = HistoricalDemandModel.fit(trips, zones, t0, exclude={test_day}, slot_minutes=SLOT_MIN)

    actual = (
        trips.filter(pl.col("pickup_datetime").dt.date() == test_day)
        .with_columns(
            pl.Series("zone", zones.zones_of(trips.filter(pl.col("pickup_datetime").dt.date() == test_day)["orig_node"].to_numpy())),
            ((pl.col("pickup_datetime").dt.hour().cast(pl.Int32) * 60 + pl.col("pickup_datetime").dt.minute().cast(pl.Int32)) // SLOT_MIN).alias("slot"),
        )
        .group_by("slot", "zone")
        .len()
    )
    act = {(s, z): n for s, z, n in actual.iter_rows()}
    rows = []
    for slot in range(24 * 60 // SLOT_MIN):
        pred = model.expected_requests(slot * SLOT_MIN * 60, (slot + 1) * SLOT_MIN * 60)
        for z in zones.zone_ids:
            rows.append((slot, int(z), pred.get(int(z), 0.0), act.get((slot, int(z)), 0)))
    df = pl.DataFrame(rows, schema=["slot", "zone", "pred", "actual"], orient="row")

    err = (df["pred"] - df["actual"]).abs().sum()
    print(f"test day {test_day}: {df['actual'].sum()} requests, forecast {df['pred'].sum():.0f}")
    print(f"zone x 15-min: WAPE {err / df['actual'].sum():.1%}, correlation {np.corrcoef(df['pred'], df['actual'])[0, 1]:.3f}")

    # Zone shares per hour: how well the forecast gets the spatial distribution.
    hourly = df.with_columns((pl.col("slot") // (60 // SLOT_MIN)).alias("hour")).group_by("hour", "zone").agg(
        pl.col("pred").sum(), pl.col("actual").sum()
    )
    shares = hourly.with_columns(
        (pl.col("pred") / pl.col("pred").sum().over("hour")).alias("pred_share"),
        (pl.col("actual") / pl.col("actual").sum().over("hour")).alias("actual_share"),
    )
    tv = shares.group_by("hour").agg(
        (0.5 * (pl.col("pred_share") - pl.col("actual_share")).abs().sum()).round(3).alias("share_distance"),
        pl.col("actual").sum().alias("requests"),
    ).sort("hour")
    print("hourly total-variation distance between forecast and actual zone shares:")
    with pl.Config(tbl_rows=24):
        print(tv)

    # Where demand is at night vs in the evening (forecast shares).
    names = gpd.read_file(TAXI_ZONES_PATH).set_index("LocationID")["zone"].to_dict()
    for hour in (3, 18):
        top = shares.filter(pl.col("hour") == hour).sort("pred_share", descending=True).head(5)
        listed = ", ".join(f"{names[z]} {s:.1%}" for z, s in zip(top["zone"], top["pred_share"]))
        print(f"{hour:02d}:00 top zones: {listed}")


if __name__ == "__main__":
    main(date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date(2025, 1, 16))
