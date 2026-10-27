"""Effective pooling factor of simulation runs: riders' trip time served per vehicle-second busy.

    uv run python scripts/analysis/pooling_factor.py reports/*stagec_tune_highs*

phi = delivered * mean(tau) / (fleet * measured seconds * busy share), where
tau = shortest-path time of a request (times the run's travel-time factor)
plus the fleet MPC's pickup allowance (90 s), averaged over all requests of
the measured window, and busy share = 1 - mean idle vehicles / fleet. This is
the `pooling` argument of FleetMpcValue: riders a vehicle serves per average
trip time. Requests are not tracked individually, so the mean tau is over
all requests, not only the delivered ones.
"""

import json
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.components.simulation_data_provider import SimulationDataProvider  # noqa: E402
from src.database.simple_db import DuckDBPathCache  # noqa: E402

PICKUP_TIME = 90.0


def pooling_factor(run_dir: Path, provider: SimulationDataProvider, db: DuckDBPathCache) -> dict:
    meta = json.loads((run_dir / "metadata.json").read_text())
    exp = meta["experiment"]
    stats = pl.read_csv(run_dir / "simulation_stats.csv")
    t0 = provider.first_request_time()
    lo = t0 + timedelta(seconds=meta["start_time"] + meta["WARMUP_DURATION"])
    hi = lo + timedelta(seconds=stats.height * meta["BATCH_DURATION"])
    trips = provider.request_trips().filter(pl.col("pickup_datetime").is_between(lo, hi, closed="left"))
    nodes = np.unique(np.concatenate([trips["orig_node"].to_numpy(), trips["dest_node"].to_numpy()]))
    index = {int(n): k for k, n in enumerate(nodes)}
    matrix = db.get_travel_time_matrix(nodes.tolist(), nodes.tolist())
    o = np.array([index[int(n)] for n in trips["orig_node"]])
    d = np.array([index[int(n)] for n in trips["dest_node"]])
    tau = matrix[o, d] * exp["travel_time_factor"]
    tau = tau[np.isfinite(tau)].mean() + PICKUP_TIME
    seconds = stats.height * meta["BATCH_DURATION"]
    busy = 1 - stats["vehicles_idle"].mean() / exp["fleet"]
    delivered = stats["requests_delivered"].sum()
    return {
        "run": run_dir.name,
        "fleet": exp["fleet"],
        "mean_tau_s": round(float(tau), 1),
        "busy_share": round(float(busy), 3),
        "delivered": int(delivered),
        "phi": round(float(delivered * tau / (exp["fleet"] * seconds * busy)), 3),
    }


if __name__ == "__main__":
    runs = [Path(p) for p in sys.argv[1:] if (Path(p) / "simulation_stats.csv").exists()]
    provider, db = SimulationDataProvider(), DuckDBPathCache(read_only=True)
    print(pl.DataFrame([pooling_factor(r, provider, db) for r in runs]))
