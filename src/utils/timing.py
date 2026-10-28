"""Wall-clock timing of named sections of a simulation run.

    from src.utils.timing import timed

    with timed("rtv"):
        with timed("size_3"):  # recorded as "rtv/size_3"
            ...

A section opened inside another is recorded under the path "outer/inner", so a
report shows a total and its parts. The run loop calls `TIMER.take()` after
setup, after every batch and at the end, so every second of the run lands in
exactly one record (`timing.csv` in the run folder). Time phases with it, not
code that runs thousands of times per batch: a section costs about a microsecond.

    uv run python scripts/analysis/timing_report.py reports/<run>
"""

import time

import polars as pl

UNTIMED = "(untimed)"


class Section:
    """Context manager for one timed section; `seconds` is set when it closes."""

    __slots__ = ("_timer", "_name", "_path", "_t0", "seconds")

    def __init__(self, timer: "Timer", name: str) -> None:
        self._timer = timer
        self._name = name
        self.seconds = 0.0

    def __enter__(self) -> "Section":
        stack = self._timer._stack
        self._path = f"{stack[-1]}/{self._name}" if stack else self._name
        stack.append(self._path)
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.seconds = time.perf_counter() - self._t0
        self._timer._stack.pop()
        self._timer.add(self._path, self.seconds)


class Timer:
    """Accumulates seconds and calls per section path until `take()`."""

    def __init__(self) -> None:
        self._stack: list[str] = []
        self._totals: dict[str, list[float]] = {}

    def section(self, name: str) -> Section:
        return Section(self, name)

    def record(self, name: str, seconds: float) -> None:
        """Add time measured by hand as a section nested in the open one."""
        self.add(f"{self._stack[-1]}/{name}" if self._stack else name, seconds)

    def add(self, path: str, seconds: float, calls: int = 1) -> None:
        total = self._totals.get(path)
        if total is None:
            self._totals[path] = [seconds, calls]
        else:
            total[0] += seconds
            total[1] += calls

    def take(self) -> dict[str, tuple[float, int]]:
        """Sections recorded since the last call, then start afresh."""
        taken = {path: (s, int(c)) for path, (s, c) in self._totals.items()}
        self._totals = {}
        return taken


TIMER = Timer()


def timed(name: str) -> Section:
    """Time a block as a section of the run (nested in any open section)."""
    return TIMER.section(name)


def record(name: str, since: float) -> None:
    """Record the time since `since` (a time.perf_counter() value) as a section.

    For a stretch of code that a with-block would have to re-indent.
    """
    TIMER.record(name, time.perf_counter() - since)


class TimingLog:
    """One row per (record, section): phase is setup, warmup, measured or end."""

    def __init__(self) -> None:
        self._rows: list[tuple[str, int, str, float, int]] = []

    def add(self, phase: str, batch: int, sections: dict[str, tuple[float, int]]) -> None:
        self._rows.extend((phase, batch, p, s, c) for p, (s, c) in sections.items())

    def frame(self) -> pl.DataFrame:
        return pl.DataFrame(
            self._rows,
            schema={"phase": pl.Utf8, "batch": pl.Int64, "section": pl.Utf8,
                    "seconds": pl.Float64, "calls": pl.Int64},
            orient="row",
        )


def summarize(timing: pl.DataFrame) -> pl.DataFrame:
    """Per section: total, share of the run, calls, and per-batch mean/p95/max.

    Rows are in tree order (children under their parent, largest first). A
    parent whose children do not cover it gets a "(untimed)" child with the
    remainder, so gaps in the instrumentation show up.
    """
    batches = timing.filter(pl.col("phase").is_in(["warmup", "measured"]))
    n_batches = max(batches["batch"].n_unique(), 1)
    per_batch = (
        batches.group_by("section", "batch").agg(pl.col("seconds").sum())
        .group_by("section")
        .agg(
            (pl.col("seconds").sum() / n_batches).alias("batch_mean_s"),
            pl.col("seconds").quantile(0.95).alias("batch_p95_s"),
            pl.col("seconds").max().alias("batch_max_s"),
        )
    )
    totals = timing.group_by("section").agg(
        pl.col("seconds").sum().alias("total_s"), pl.col("calls").sum()
    )
    total = dict(zip(totals["section"], totals["total_s"]))

    children: dict[str, list[str]] = {}
    for path in total:
        parent = path.rpartition("/")[0]
        children.setdefault(parent, []).append(path)
    for parent, kids in list(children.items()):
        if not parent:
            continue
        rest = total[parent] - sum(total[k] for k in kids)
        if rest > 0.01 * total[parent]:
            path = f"{parent}/{UNTIMED}"
            total[path] = rest
            kids.append(path)
    run_s = sum(total[p] for p in children.get("", []))

    order: list[str] = []

    def walk(parent: str) -> None:
        for path in sorted(children.get(parent, []), key=lambda p: -total[p]):
            order.append(path)
            walk(path)

    walk("")
    rows = pl.DataFrame(
        {
            "section": order,
            "depth": [p.count("/") for p in order],
            "total_s": [total[p] for p in order],
            "share": [total[p] / run_s if run_s else 0.0 for p in order],
        }
    )
    return (
        rows.join(totals.select("section", "calls"), on="section", how="left")
        .join(per_batch, on="section", how="left")
        .with_columns(pl.col("calls").fill_null(0))
    )


def format_summary(summary: pl.DataFrame) -> str:
    lines = [
        f"{'section':<46} {'total s':>9} {'share':>6} {'calls':>8} "
        f"{'ms/batch':>9} {'p95 ms':>8} {'max ms':>8}"
    ]
    ms = lambda v: "" if v is None else f"{v * 1000:.1f}"
    for r in summary.iter_rows(named=True):
        name = "  " * r["depth"] + r["section"].rpartition("/")[2]
        lines.append(
            f"{name:<46} {r['total_s']:>9.2f} {r['share']:>6.1%} {r['calls']:>8} "
            f"{ms(r['batch_mean_s']):>9} {ms(r['batch_p95_s']):>8} {ms(r['batch_max_s']):>8}"
        )
    return "\n".join(lines)
