"""Rolling backtest of the past-only demand models: every day and hour, forecast from the past only.

    uv run python scripts/analysis/forecast_backtest.py [first-day last-day]   (default 2024-12-01 2025-01-31)

For each day D and each hour h, every model forecasts the requests per zone
in the four 15-min slots of [h, h+1), knowing only what happened before
(profiles and gbm are fitted on days before D; nowcast variants also see the
requests of D before h). Truth and training data are the raw TLC counts
(data/history/zone_counts_15min.parquet). Per model:
- bias = sum(forecast) / sum(actual) - 1;
- WAPE = sum |forecast - actual| (over zones and slots) / sum(actual);
overall, by time of day, and the spread of the daily WAPE over days.
February 2025 (data/history/holdout_raw/) is the untouched test month and is
not read here.
"""

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.paths import DATA_DIR  # noqa: E402
from src.components.demand.demand_model import DayOfWeekDemandModel, HistoricalDemandModel  # noqa: E402
from src.components.demand.gbm_model import fit_gbm_forecast  # noqa: E402
from src.components.demand.nowcast_model import NowcastDemandModel  # noqa: E402

COUNTS = DATA_DIR / "history" / "zone_counts_15min.parquet"
SLOT = 900
PERIODS = {"all hours": range(24), "morning 07-10": range(7, 10), "evening 17-20": range(17, 20),
           "night 00-06": range(0, 6)}


_CACHE: dict = {}


def _counts() -> pl.DataFrame:
    if "counts" not in _CACHE:
        _CACHE["counts"] = pl.read_parquet(COUNTS)
    return _CACHE["counts"]


def models_for(day: date, counts: pl.DataFrame, zones: list[int]) -> dict:
    t0 = datetime.combine(day, datetime.min.time())  # simulation time 0 = midnight of D
    base = {
        "last-week": DayOfWeekDemandModel.from_counts(counts, t0, before=day, lookback_days=7),
        "past-weekday-8w": HistoricalDemandModel.from_counts(counts, t0, before=day, lookback_days=56),
        "past-day-of-week-8w": DayOfWeekDemandModel.from_counts(counts, t0, before=day, lookback_days=56),
        "gbm": fit_gbm_forecast(counts, zones, t0, {day}),
    }
    # Request times of D and the day before (negative), at slot resolution (enough: nowcast
    # windows start on whole hours); the day before lets a nowcast after midnight see the last hour.
    seen = counts.filter(pl.col("day").is_in([day - timedelta(days=1), day]))
    offset = np.where(seen["day"].to_numpy() == np.datetime64(day), 0, -86400)
    times = np.repeat(seen["slot"].to_numpy() * SLOT + offset, seen["n"].to_numpy())
    # gbm forecasts day D only: its nowcast does not look before midnight.
    nowcast = {f"nowcast-{k}": NowcastDemandModel(m, times, covered_from=0.0 if k == "gbm" else None)
               for k, m in base.items() if k != "last-week"}
    return base | nowcast


def backtest_day(day: date) -> list[dict]:
    counts = _counts()
    zones = sorted(counts["zone"].unique().to_list())
    today = counts.filter(pl.col("day") == day)
    rows = []
    for name, model in models_for(day, counts, zones).items():
        for h in range(24):
            forecast_total = actual_total = abs_err = 0.0
            for k in range(4):
                slot = 4 * h + k
                forecast = model.expected_requests(slot * SLOT, (slot + 1) * SLOT, h * 3600.0)
                actual = dict(today.filter(pl.col("slot") == slot).select("zone", "n").iter_rows())
                forecast_total += sum(forecast.values())
                actual_total += sum(actual.values())
                abs_err += sum(abs(forecast.get(z, 0.0) - actual.get(z, 0)) for z in set(forecast) | set(actual))
            rows.append({"day": day, "hour": h, "model": name, "forecast": forecast_total,
                         "actual": actual_total, "abs_err": abs_err})
    return rows


def main() -> None:
    first, last = (date.fromisoformat(a) for a in (sys.argv[1:3] if len(sys.argv) > 2 else ("2024-12-01", "2025-01-31")))
    days = [first + timedelta(days=k) for k in range((last - first).days + 1)]
    # One day at a time: the gbm fit already uses every core (a process pool deadlocked on forked threads).
    df = pl.DataFrame([r for d in days for r in backtest_day(d)])
    out = DATA_DIR / "history" / f"backtest_{first}_{last}.parquet"
    df.write_parquet(out)
    pl.Config.set_tbl_rows(-1)
    pl.Config.set_tbl_width_chars(140)
    for period, hours in PERIODS.items():
        part = df.filter(pl.col("hour").is_in(list(hours)))
        daily = part.group_by("model", "day").agg((pl.col("abs_err").sum() / pl.col("actual").sum()).alias("wape"))
        summary = (
            part.group_by("model")
            .agg((100 * (pl.col("forecast").sum() / pl.col("actual").sum() - 1)).round(1).alias("bias_pct"),
                 (100 * pl.col("abs_err").sum() / pl.col("actual").sum()).round(1).alias("wape_pct"))
            .join(daily.group_by("model").agg((100 * pl.col("wape").std()).round(1).alias("daily_wape_sd"),
                                              (100 * pl.col("wape").max()).round(1).alias("worst_day_wape")),
                  on="model")
            .sort("wape_pct")
        )
        print(f"\n{period} ({len(days)} days, {days[0]} to {days[-1]}):")
        print(summary)
    print(f"\nper day and hour: {out}")


if __name__ == "__main__":
    main()
