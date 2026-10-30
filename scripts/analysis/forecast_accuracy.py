"""How good is each demand model? Forecast vs actual requests per zone and 15-min slot.

    uv run python scripts/analysis/forecast_accuracy.py

For each test day (evening, 18:00-19:00) and each registered demand model,
the forecast for the four 15-min slots is made at 18:00 (t_now = 18:00, as
the fleet plan sees it) and compared with the requests that arrived:
- bias: (forecast total - actual total) / actual total;
- WAPE: sum over zones and slots of |forecast - actual| / actual total.
Models are fitted exactly as in a simulation of that day (simulated days left
out of the training data).
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.components.simulation.simulation_data_provider import SimulationDataProvider  # noqa: E402
from src.experiment import DEMAND_MODELS, Context, ExperimentConfig, simulation_window  # noqa: E402

DAYS = [16, 23, 28, 29, 30]
SLOT = 900


def main() -> None:
    provider = SimulationDataProvider()
    t0 = provider.first_request_time()
    table = provider.request_table()
    rows = []
    for day in DAYS:
        cfg = ExperimentConfig(start=datetime(2025, 1, day, 17, 45), hours=1)
        start, end = simulation_window(cfg, provider)
        six = start + 900  # 18:00
        day_rows = table.filter(pl.col("pickup_datetime").dt.date() == datetime(2025, 1, day).date())
        times = np.array([(t - t0).total_seconds() for t in day_rows["pickup_datetime"]])
        for name in DEMAND_MODELS:
            ctx = Context(cfg, provider, None, start, end)  # network unused by demand models
            zones = ctx.zones.zones_of(day_rows["orig_node"].to_numpy())
            model = DEMAND_MODELS[name](ctx)
            f_total = a_total = abs_err = 0.0
            for k in range(4):
                lo, hi = six + k * SLOT, six + (k + 1) * SLOT
                forecast = model.expected_requests(lo, hi, six)
                inside = (times >= lo) & (times < hi)
                actual = dict(zip(*np.unique(zones[inside], return_counts=True)))
                keys = set(forecast) | set(actual)
                f_total += sum(forecast.values())
                a_total += sum(actual.values())
                abs_err += sum(abs(forecast.get(z, 0.0) - actual.get(z, 0)) for z in keys)
            rows.append({"day": f"2025-01-{day}", "model": name, "actual": int(a_total),
                         "forecast": round(f_total), "bias_pct": round(100 * (f_total - a_total) / a_total, 1),
                         "wape_pct": round(100 * abs_err / a_total, 1)})
    df = pl.DataFrame(rows)
    pl.Config.set_tbl_rows(-1)
    print(df)
    print(df.group_by("model", maintain_order=True).agg(pl.col("bias_pct").mean().round(1),
                                                        pl.col("wape_pct").mean().round(1)))


if __name__ == "__main__":
    main()
