"""Trade-off between riders served, rider delay and profit, per day.

    uv run python scripts/analysis/pareto.py reports/*rebal_evening_f3000* reports/*pareto16_* ...

Each run becomes one point, grouped into families (paper baselines, C2, C3)
and labelled by its setting. A run is on the frontier if no other run of the
same day and fleet serves at least as many riders with at most as much delay (or at
least as much profit). Writes one figure per trade-off to thesis/figures/ and
prints the table.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from summarize_runs import summarize

OUT = Path(__file__).resolve().parents[2] / "thesis" / "figures"
# Categorical slots 1-3 of the reference palette plus a neutral for baselines (validated, light surface);
# shapes repeat the family so identity is never colour alone.
FAMILIES = {
    "baseline": ("#52514e", "s", "Baselines"),
    "C2": ("#eb6834", "^", "C2: sampled riders"),
    "C3": ("#2a78d6", "o", "C3: fleet plan"),
    "C2-SAA": ("#1baf7a", "D", "C2 two-stage: S sampled futures"),
}


def describe(run_dir: Path) -> dict:
    exp = json.loads((run_dir / "metadata.json").read_text()).get("experiment", {})
    solver, rebal = exp.get("assignment_solver"), exp.get("rebalancing")
    if solver == "joint" and exp.get("vehicle_value") == "mpc":
        family, label = "C3", f"α={exp.get('vehicle_value_weight', 1.0):g}"
    elif solver == "two-stage":
        family = "C2-SAA"
        label = f"S={exp.get('scenarios'):g}, w={exp.get('scenario_weight'):g}"
    elif str(solver).startswith("sampled"):
        family = "C2"
        label = f"w={exp.get('virtual_weight'):g}, ρ={exp.get('virtual_per_free'):g}"
    else:
        family, label = "baseline", {"demand": "forecast-greedy", "transport": "forecast-optimal"}.get(rebal, rebal)
    return {"day": str(exp.get("start", ""))[:10], "family": family, "label": label}


def frontier(df: pl.DataFrame, y: str, better_low: bool) -> pl.Series:
    """True for runs no other run of the same panel (day, fleet) beats on served and on y."""
    out = []
    for r in df.iter_rows(named=True):
        same = df.filter(pl.col("panel") == r["panel"])
        y_ok = (pl.col(y) <= r[y]) if better_low else (pl.col(y) >= r[y])
        y_strict = (pl.col(y) < r[y]) if better_low else (pl.col(y) > r[y])
        beaten = same.filter(
            (pl.col("served") >= r["served"]) & y_ok
            & ((pl.col("served") > r["served"]) | y_strict)
        )
        out.append(beaten.height == 0)
    return pl.Series(out)


def plot(df: pl.DataFrame, y: str, ylabel: str, better_low: bool, name: str) -> Path:
    days, fleets = sorted(df["day"].unique()), sorted(df["fleet"].unique())
    fig, axes = plt.subplots(
        len(fleets), len(days), figsize=(6 * len(days), 4.6 * len(fleets)), squeeze=False
    )
    on_front = frontier(df, y, better_low)
    df = df.with_columns(front=on_front)
    for (row, fleet), (col, day) in ((r, c) for r in enumerate(fleets) for c in enumerate(days)):
        ax, panel = axes[row][col], f"{day}, {fleet} vehicles"
        d = df.filter(pl.col("panel") == panel).sort("served")
        if d.height == 0:
            ax.set_visible(False)
            continue
        front = d.filter(pl.col("front"))
        ax.plot(front["served"] * 100, front[y], color="#c3c2b7", lw=2, zorder=1)
        for family, (color, marker, legend) in FAMILIES.items():
            f = d.filter(pl.col("family") == family)
            if f.height == 0:
                continue
            ax.scatter(f["served"] * 100, f[y], s=64, color=color, marker=marker,
                       edgecolor="white", linewidth=2, zorder=3, label=legend)
            # Label baselines and the frontier; the printed table names the rest.
            for r in f.filter(pl.col("front") | (pl.col("family") == "baseline")).iter_rows(named=True):
                ax.annotate(r["label"], (r["served"] * 100, r[y]), textcoords="offset points",
                            xytext=(6, 4), fontsize=8, color="#52514e")
        ax.set_title(panel, fontsize=11, color="#1a1a19")
        ax.set_xlabel("Riders served (%)")
        ax.grid(color="#e6e5e0", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_ylabel(ylabel)
    axes[0][0].legend(frameon=False, fontsize=9)
    fig.suptitle("Grey line: runs no other run beats on both axes (labelled); other runs in the printed table", fontsize=9, color="#52514e")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


if __name__ == "__main__":
    runs = [Path(p) for p in sys.argv[1:] if (Path(p) / "simulation_stats.csv").exists()]
    df = pl.DataFrame([summarize(r) | describe(r) for r in runs]).with_columns(
        profit_k=pl.col("profit@0.25") / 1000,
        panel=pl.format("{}, {} vehicles", "day", "fleet"),
    )
    pl.Config.set_tbl_rows(-1)
    pl.Config.set_tbl_cols(12)
    print(df.select("day", "fleet", "family", "label", "served", "wait_s", "delay_s", "shared_trips", "profit_k")
          .sort("day", "fleet", "family", "served"))
    print(plot(df, "delay_s", "Mean rider delay (s)", True, "pareto_served_delay"))
    print(plot(df, "profit_k", "Profit at $0.25/km (thousand USD)", False, "pareto_served_profit"))
