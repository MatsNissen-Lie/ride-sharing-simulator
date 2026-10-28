"""Served share, idle vehicles and empty km per half hour of measured time.

    uv run python scripts/analysis/half_hours.py reports/*pattern*
"""

import sys
from pathlib import Path

import polars as pl

BATCHES = 180  # 30 min of 10-s batches

if __name__ == "__main__":
    rows = []
    for run in sys.argv[1:]:
        path = Path(run) / "simulation_stats.csv"
        if not path.exists():
            continue
        df = pl.read_csv(path)
        for start in range(0, df.height, BATCHES):
            part = df.slice(start, BATCHES)
            delivered, expired = part["requests_delivered"].sum(), part["requests_expired"].sum()
            rows.append({
                "run": Path(run).name,
                "minutes": f"{start // 6}-{(start + part.height) // 6}",
                "served": round(delivered / max(delivered + expired, 1), 4),
                "idle": round(part["vehicles_idle"].mean(), 1),
                "km_empty": round(part["vehicle_km_empty"].sum()),
            })
    pl.Config.set_tbl_rows(100)
    pl.Config.set_fmt_str_lengths(70)
    print(pl.DataFrame(rows))
