"""A learned demand forecast: gradient boosting on past requests per zone and slot."""

from datetime import date, datetime, timedelta

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor

from src.components.demand.demand_model import HOLIDAYS, DatedDemandModel

LAGS = (1, 7, 14, 21, 28)  # days back, same slot and zone


def _grid(counts: pl.DataFrame, days: list[date], zones: list[int]) -> pl.DataFrame:
    """Every (day, slot, zone) with its count (0 where no request)."""
    full = (
        pl.DataFrame({"day": days}).join(pl.DataFrame({"slot": np.arange(96, dtype=np.int32)}), how="cross")
        .join(pl.DataFrame({"zone": np.array(zones, dtype=np.int32)}), how="cross")
    )
    return full.join(counts.select("day", "slot", "zone", pl.col("n").cast(pl.Float64)),
                     on=["day", "slot", "zone"], how="left").with_columns(pl.col("n").fill_null(0.0))


def _features(grid: pl.DataFrame) -> pl.DataFrame:
    """Calendar features and the counts of the same slot and zone LAGS days before."""
    out = grid.with_columns(
        pl.col("zone").rank("dense").cast(pl.Int32).sub(1).alias("zone_index"),  # categories 0..Z-1
        pl.col("day").dt.weekday().alias("weekday"),
        pl.col("day").dt.month().alias("month"),
        pl.col("day").is_in(sorted(HOLIDAYS)).cast(pl.Int8).alias("holiday"),
    )
    for lag in LAGS:
        shifted = grid.select((pl.col("day") + timedelta(days=lag)).alias("day"), "slot", "zone",
                              pl.col("n").alias(f"lag{lag}"))
        out = out.join(shifted, on=["day", "slot", "zone"], how="left")
    return out.with_columns(pl.mean_horizontal([f"lag{k}" for k in (7, 14, 21, 28)]).alias("same_weekday_mean"))


FEATURES = ["zone_index", "slot", "weekday", "month", "holiday", *[f"lag{k}" for k in LAGS], "same_weekday_mean"]


def fit_gbm_forecast(
    counts: pl.DataFrame,
    zones: list[int],
    t0: datetime,
    days: set[date],
    train_days: int = 365,
    seed: int = 0,
) -> DatedDemandModel:
    """
    Forecast for `days` (the simulated days) by gradient-boosted trees with a
    Poisson loss, trained on the `train_days` before the first of them only
    (requests per day, slot and zone; `counts` as from
    SimulationDataProvider.demand_history). Features: zone (categorical), slot,
    weekday, month, holiday, and the counts of the same slot and zone 1, 7, 14,
    21 and 28 days earlier. Lags that would fall on a simulated day are
    unknown (missing), so nothing of the simulated days is used.
    """
    first = min(days)
    past = counts.filter(pl.col("day") < first)
    hist_days = sorted(d for d in past["day"].unique().to_list() if d >= first - timedelta(days=train_days + 28))
    grid = _grid(past, hist_days + sorted(days), zones)
    data = _features(grid)
    train = data.filter((pl.col("day") < first) & (pl.col("day") >= first - timedelta(days=train_days))
                        & pl.col("lag28").is_not_null())
    model = HistGradientBoostingRegressor(loss="poisson", max_iter=300, learning_rate=0.1, max_leaf_nodes=63,
                                          categorical_features=[0], random_state=seed)
    model.fit(train.select(FEATURES).to_numpy(), train["n"].to_numpy())
    target = data.filter(pl.col("day").is_in(sorted(days)))
    pred = model.predict(target.select(FEATURES).to_numpy())
    rates: dict[tuple, dict[int, float]] = {}
    for (day, slot, zone), p in zip(target.select("day", "slot", "zone").iter_rows(), pred):
        rates.setdefault((day, slot), {})[zone] = float(p)
    return DatedDemandModel(rates, t0)
