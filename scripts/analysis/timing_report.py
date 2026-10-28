"""Where a run spent its time, or how two runs compare.

    uv run python scripts/analysis/timing_report.py reports/<run>
    uv run python scripts/analysis/timing_report.py reports/<base> reports/<new>
    uv run python scripts/analysis/timing_report.py reports/<base> reports/<a> reports/<b> ...

One run: the section tree from timing.csv (total, share of the run, calls, per
batch mean/p95/max). Two runs: section totals side by side with the speed-up,
and a check that the simulation results are identical, i.e. every column of
simulation_stats.csv except the time_* columns. More runs: one line per run
(run time, mean batch, speed-up and identical results versus the first).
"""

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.timing import format_summary, summarize  # noqa: E402


def load(run: Path) -> pl.DataFrame:
    return summarize(pl.read_csv(run / "timing.csv"))


def result_differences(base: Path, new: Path) -> list[str]:
    """Stats columns (time_* excluded) that differ, with the first batch that differs."""
    a = pl.read_csv(base / "simulation_stats.csv")
    b = pl.read_csv(new / "simulation_stats.csv")
    if a.height != b.height:
        return [f"batch count {a.height} vs {b.height}"]
    out = []
    for col in a.columns:
        if col.startswith("time_"):
            continue
        if col not in b.columns:
            out.append(f"{col}: missing in new run")
            continue
        diff = (a[col] != b[col]).fill_null(True) & ~(a[col].is_null() & b[col].is_null())
        if diff.any():
            k = int(diff.arg_max())
            out.append(f"{col}: first differs at batch {a['batch_id'][k]} ({a[col][k]} vs {b[col][k]})")
    return out


def compare(base: Path, new: Path) -> str:
    a, b = load(base), load(new)
    rows = a.select("section", "depth", base_s="total_s").join(
        b.select("section", new_s="total_s"), on="section", how="full", coalesce=True
    )
    order = {s: k for k, s in enumerate(b["section"])} | {s: k for k, s in enumerate(a["section"])}
    rows = rows.with_columns(pl.col("depth").fill_null(pl.col("section").str.count_matches("/")))
    lines = [f"{'section':<46} {'base s':>9} {'new s':>9} {'speed-up':>9}"]
    for r in sorted(rows.iter_rows(named=True), key=lambda r: order[r["section"]]):
        name = "  " * r["depth"] + r["section"].rpartition("/")[2]
        base_s, new_s = r["base_s"] or 0.0, r["new_s"] or 0.0
        speed = f"{base_s / new_s:.2f}x" if new_s > 0 else ""
        lines.append(f"{name:<46} {base_s:>9.2f} {new_s:>9.2f} {speed:>9}")
    total_a = a.filter(pl.col("depth") == 0)["total_s"].sum()
    total_b = b.filter(pl.col("depth") == 0)["total_s"].sum()
    lines.append(f"{'run':<46} {total_a:>9.2f} {total_b:>9.2f} {total_a / total_b:>8.2f}x")
    diffs = result_differences(base, new)
    lines.append("results: identical" if not diffs else "results DIFFER:\n  " + "\n  ".join(diffs))
    return "\n".join(lines)


def table(runs: list[Path]) -> str:
    lines = [f"{'run':<44} {'run s':>8} {'ms/batch':>9} {'speed-up':>9}  results vs first"]
    base = None
    for run in runs:
        s = load(run)
        total = s.filter(pl.col("depth") == 0)["total_s"].sum()
        batch = s.filter(pl.col("section") == "batch")["batch_mean_s"].item()
        base = base or total
        same = "identical" if not result_differences(runs[0], run) else "DIFFER"
        lines.append(f"{run.name:<44} {total:>8.1f} {batch * 1000:>9.1f} {base / total:>8.2f}x  {same}")
    return "\n".join(lines)


if __name__ == "__main__":
    runs = [Path(p) for p in sys.argv[1:]]
    if len(runs) == 1:
        print(format_summary(load(runs[0])))
    elif len(runs) == 2:
        print(compare(*runs))
    elif len(runs) > 2:
        print(table(runs))
    else:
        sys.exit(__doc__)
