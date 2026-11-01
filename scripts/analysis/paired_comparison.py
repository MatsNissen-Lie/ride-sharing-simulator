"""Is configuration B better than A? Paired test over days.

    uv run python scripts/analysis/paired_comparison.py "<glob A with {day}>" "<glob B with {day}>" 16 23 28 29 30

Each day is simulated once per configuration, so the unit of comparison is
the day: the difference in riders served (B - A) per day is tested with a
paired t-test, an exact sign test and Wilcoxon's signed-rank test (all
two-sided). With few days the sign and Wilcoxon tests have a floor: with 5
days the smallest possible two-sided p is 0.0625.
"""

import glob
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_runs import summarize  # noqa: E402


def served(pattern: str, day: str) -> float:
    (run,) = glob.glob(pattern.format(day=day))
    return 100 * summarize(Path(run))["served"]


def main() -> None:
    a_pattern, b_pattern, *days = sys.argv[1:]
    a = np.array([served(a_pattern, d) for d in days])
    b = np.array([served(b_pattern, d) for d in days])
    diff = b - a
    for d, x, y, z in zip(days, a, b, diff):
        print(f"day {d}: A {x:.2f}%  B {y:.2f}%  B-A {z:+.2f}")
    t = stats.ttest_rel(b, a)
    sign = stats.binomtest(int((diff > 0).sum()), int((diff != 0).sum()))
    wil = stats.wilcoxon(diff)
    half = stats.t.ppf(0.975, len(diff) - 1) * diff.std(ddof=1) / np.sqrt(len(diff))
    print(f"mean B-A {diff.mean():+.2f} points, 95% CI [{diff.mean() - half:+.2f}, {diff.mean() + half:+.2f}]")
    print(f"paired t-test p = {t.pvalue:.3f}; sign test p = {sign.pvalue:.3f}; Wilcoxon p = {wil.pvalue:.3f}")


if __name__ == "__main__":
    main()
