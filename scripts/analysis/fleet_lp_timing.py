"""How long does C3's fleet LP take with sampled demand futures?

    uv run python scripts/analysis/fleet_lp_timing.py capture   # run to the 18:00 peak and save the LP inputs
    uv run python scripts/analysis/fleet_lp_timing.py time      # time the variants on them

Capture: Thu 2025-01-16 from 17:45, 3000 vehicles, C3 (joint + mpc); the
inputs of the fleet LP solve at or after 18:00 are saved. Variants, each with
S futures whose demand per zone and step is Poisson around the forecast:
- one LP on the forecast (today's C3);
- the extensive form: one LP with S copies that share the step-0 moves
  (two-stage stochastic programme, solved by HiGHS interior point);
- S separate LPs, one per future, one after another and on a process pool
  (their duals and moves averaged afterwards; not a two-stage programme).
"""

import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import block_diag, coo_matrix, vstack

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CAPTURE = ROOT / "data" / "fleet_lp" / "peak_inputs_2025-01-16_1800_f3000.pkl"
PEAK = 18 * 3600  # seconds of day


def capture() -> None:
    import src.components.fleet_mpc as mpc
    import src.experiment as ex

    original = mpc.fleet_lp_values
    t0 = datetime(2025, 1, 16, 17, 45)

    def spy(*args, **kwargs):
        spy.calls += 1
        if spy.calls == 16 and not CAPTURE.exists():
            CAPTURE.parent.mkdir(parents=True, exist_ok=True)  # one solve per minute from 17:45: the 18:00 solve
            CAPTURE.write_bytes(pickle.dumps((args, kwargs)))
            print(f"saved {CAPTURE}")
        return original(*args, **kwargs)

    spy.calls = 0
    mpc.fleet_lp_values = spy
    cfg = ex.ExperimentConfig(name="fleet_lp_capture", fleet=3000, start=t0, hours=1, max_batches=100,
                              assignment_solver="joint", vehicle_value="mpc", rebalancing="mpc", read_only=True)
    ex.run_experiment(cfg)


def solve(model) -> tuple[float, np.ndarray, np.ndarray]:
    cost, a_eq, b_eq, bounds = model
    start = time.perf_counter()
    res = linprog(cost, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs-ipm")
    assert res.status == 0, res.message
    return time.perf_counter() - start, res.x, res.eqlin.marginals


def extensive(models, m0) -> tuple:
    """S copies of the LP, objective averaged, step-0 moves equal across copies."""
    s = len(models)
    cost = np.concatenate([m[0] / s for m in models])
    blocks = block_diag([m[1] for m in models], format="csr")
    n = models[0][1].shape[1]
    rows, cols, vals = [], [], []
    for k in range(1, s):
        for r, c in enumerate(m0):
            rows += [(k - 1) * len(m0) + r] * 2
            cols += [c, k * n + c]
            vals += [1.0, -1.0]
    link = coo_matrix((vals, (rows, cols)), shape=((s - 1) * len(m0), s * n))
    a_eq = vstack([blocks, link]).tocsr()
    b_eq = np.concatenate([m[2] for m in models] + [np.zeros((s - 1) * len(m0))])
    bounds = np.vstack([m[3] for m in models])
    return cost, a_eq, b_eq, bounds


def time_variants() -> None:
    from src.components.fleet_mpc import fleet_lp_model

    args, kwargs = pickle.loads(CAPTURE.read_bytes())
    supply, demand, *rest = args
    rng = np.random.default_rng(0)

    def model(d):
        cost, a_eq, b_eq, bounds, m0, _, _ = fleet_lp_model(supply, d, *rest, **kwargs)
        return (cost, a_eq, b_eq, bounds), m0

    base, m0 = model(demand)
    t, _, _ = solve(base)
    print(f"forecast LP: {base[1].shape[0]} rows x {base[1].shape[1]} columns, {t:.2f} s")
    for s in (3, 5, 10):
        futures = [model(rng.poisson(demand).astype(float))[0] for _ in range(s)]
        start = time.perf_counter()
        for f in futures:
            solve(f)
        sequential = time.perf_counter() - start
        start = time.perf_counter()
        with ProcessPoolExecutor(4) as pool:
            list(pool.map(solve, futures))
        parallel = time.perf_counter() - start
        ext = extensive(futures, m0)
        t_ext, _, _ = solve(ext)
        print(f"S={s:2d}: separate LPs {sequential:.2f} s in sequence, {parallel:.2f} s on 4 processes; "
              f"extensive form ({ext[1].shape[0]} x {ext[1].shape[1]}) {t_ext:.2f} s")


if __name__ == "__main__":
    {"capture": capture, "time": time_variants}[sys.argv[1]]()
