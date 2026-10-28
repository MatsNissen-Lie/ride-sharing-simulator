"""Served share, wait, empty km and vehicles idle per clock hour of simulation runs.

    uv run python scripts/analysis/hourly_service.py reports/<run> ...
"""

import sys
from pathlib import Path

import polars as pl

if __name__ == "__main__":
    frames = []
    for run in (Path(p) for p in sys.argv[1:]):
        if not (run / "simulation_stats.csv").exists():
            continue
        df = pl.read_csv(run / "simulation_stats.csv")
        frames.append(
            df.with_columns(pl.col("timestamp").str.slice(11, 2).alias("hour"))
            .group_by("hour")
            .agg(
                delivered=pl.col("requests_delivered").sum(),
                expired=pl.col("requests_expired").sum(),
                wait_s=(pl.col("avg_wait_time") * pl.col("requests_delivered")).sum()
                / pl.col("requests_delivered").sum(),
                km_empty=pl.col("vehicle_km_empty").sum(),
                idle=pl.col("vehicles_idle").mean(),
            )
            .with_columns(
                pl.lit(run.name[20:]).alias("run"),
                (pl.col("delivered") / (pl.col("delivered") + pl.col("expired"))).round(4).alias("served"),
                pl.col("wait_s").round(1),
                pl.col("km_empty").round(0),
                pl.col("idle").round(0),
            )
        )
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_fmt_str_lengths(60)
    print(pl.concat(frames).select("run", "hour", "delivered", "expired", "served", "wait_s", "km_empty", "idle")
          .sort("run", "hour"))
