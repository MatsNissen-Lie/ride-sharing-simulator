"""Summarize simulation runs: service, wait, delay, sharing and economics.

    uv run python scripts/analysis/summarize_runs.py reports/*peak29*

Profit = revenue - cost_per_km * vehicle-km, shown for each cost scenario so
that runs can be compared under the same cost assumption.
"""

import json
import sys
from pathlib import Path

import polars as pl

COST_SCENARIOS = (0.10, 0.25, 0.50)


def summarize(run_dir: Path) -> dict:
    df = pl.read_csv(run_dir / "simulation_stats.csv")
    meta = json.loads((run_dir / "metadata.json").read_text())
    exp = meta.get("experiment", {})
    delivered, expired = df["requests_delivered"].sum(), df["requests_expired"].sum()
    weighted = lambda col: (df[col] * df["requests_delivered"]).sum() / max(delivered, 1)
    row = {
        "run": run_dir.name,
        "fleet": exp.get("fleet", meta.get("NUMBER_OF_VEHICLES")),
        "objective": exp.get("objective", "delay"),
        "rebalancing": exp.get("rebalancing", meta.get("rebalancing", "reactive")),
        "ttf": exp.get("travel_time_factor", 1.0),
        "minutes": df.height * meta.get("BATCH_DURATION", 10) / 60,
        "new": df["requests_new"].sum(),
        "delivered": delivered,
        "expired": expired,
        "served": round(delivered / max(delivered + expired, 1), 4),
        "wait_s": round(weighted("avg_wait_time"), 1),
        "delay_s": round(weighted("avg_total_delay"), 1),
        "shared_trips": sum(df[f"assigned_trips_size_{k}"].sum() for k in range(2, 7) if f"assigned_trips_size_{k}" in df.columns),
        "batch_s": round(df["time_total_batch"].mean(), 2),
    }
    if "revenue" in df.columns:
        km = df["vehicle_km_loaded"].sum() + df["vehicle_km_empty"].sum()
        row |= {
            "revenue": round(df["revenue"].sum()),
            "km_loaded": round(df["vehicle_km_loaded"].sum()),
            "km_empty": round(df["vehicle_km_empty"].sum()),
        }
        for c in COST_SCENARIOS:
            row[f"profit@{c}"] = round(row["revenue"] - c * km)
    return row


if __name__ == "__main__":
    runs = [Path(p) for p in sys.argv[1:] if (Path(p) / "simulation_stats.csv").exists()]
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(300)
    print(pl.DataFrame([summarize(r) for r in runs]).sort(["objective", "fleet"]))
