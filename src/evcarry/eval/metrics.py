"""Pass rates and a McNemar hook for paired arm comparisons."""

from __future__ import annotations

from typing import Any

from evcarry.eval.flips import pairwise_flips


def pass_rate(rows: list[dict[str, Any]], arm: str) -> float:
    chosen = [row for row in rows if row.get("arm") == arm and row.get("passed") is not None]
    if not chosen:
        return 0.0
    return sum(1 for row in chosen if row["passed"]) / len(chosen)


def mcnemar_pvalue(rows: list[dict[str, Any]], arm_a: str, arm_b: str) -> float:
    """Exact McNemar p-value on the off-diagonal flip counts.

    The contingency table is ``[[0, a_only], [b_only, 0]]``. Concordant cells
    do not enter the statistic. Requires statsmodels, an analysis extra.
    """
    from statsmodels.stats.contingency_tables import mcnemar

    flips = pairwise_flips(rows, arm_a, arm_b)
    table = [[0, flips["a_only"]], [flips["b_only"], 0]]
    result = mcnemar(table, exact=True)
    return float(result.pvalue)
