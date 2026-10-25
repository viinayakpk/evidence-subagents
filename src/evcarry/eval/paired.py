"""Primary-endpoint statistics: paired task success on matched (task, seed) blocks (amendment A15).

A block is one (task, seed) that both arms finished with a grade. Seeds of one task are not independent
(temperature 0 makes them near-replicates), so the unit of resampling is the task: the confidence
interval comes from a bootstrap that draws tasks with replacement and keeps all of a task's blocks.
The exact McNemar test on blocks (``metrics.mcnemar_pvalue``) treats blocks as independent and is
reported as descriptive only. ``pass_hat_k`` is the tau-bench pass^k: the chance that k seeds of a
task, drawn without replacement, all pass. Success per reference dollar divides passes by the cost at
the reference price table, so provider routing does not move it.
"""

from __future__ import annotations

import random
from collections import defaultdict
from math import comb
from typing import Any

Row = dict[str, Any]


def matched_blocks(rows: list[Row], arm_a: str, arm_b: str,
                   exclude: frozenset[str] = frozenset()) -> dict[tuple[str, Any], tuple[bool, bool]]:
    """``(task, seed) -> (passed_a, passed_b)`` for blocks both arms graded, minus excluded tasks."""
    by_arm: dict[str, dict[tuple[str, Any], bool]] = defaultdict(dict)
    for row in rows:
        if row.get("passed") is None or row.get("task") in exclude:
            continue
        by_arm[row["arm"]][(row["task"], row.get("seed"))] = bool(row["passed"])
    a, b = by_arm.get(arm_a, {}), by_arm.get(arm_b, {})
    return {key: (a[key], b[key]) for key in sorted(set(a) & set(b), key=str)}


def _task_means(blocks: dict[tuple[str, Any], tuple[bool, bool]]) -> dict[str, tuple[float, float, int]]:
    sums: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0])
    for (task, _seed), (pa, pb) in blocks.items():
        sums[task][0] += pa
        sums[task][1] += pb
        sums[task][2] += 1
    return {task: (s[0], s[1], int(s[2])) for task, s in sums.items()}


def paired_difference(rows: list[Row], arm_a: str, arm_b: str, *, n_boot: int = 10000, seed: int = 0,
                      exclude: frozenset[str] = frozenset(), alpha: float = 0.05) -> dict[str, Any]:
    """Success of ``arm_b`` minus ``arm_a`` on matched blocks, with a task-clustered bootstrap CI.

    The point estimate weights every block equally; a bootstrap replicate draws tasks with replacement
    and pools their blocks. ``p_boot`` is the two-sided share of replicates on the far side of zero.
    """
    blocks = matched_blocks(rows, arm_a, arm_b, exclude)
    per_task = _task_means(blocks)
    tasks = sorted(per_task)
    if not tasks:
        return {"blocks": 0, "tasks": 0, "diff": None, "ci": None, "p_boot": None}

    def diff(chosen: list[str]) -> float:
        a = sum(per_task[t][0] for t in chosen)
        b = sum(per_task[t][1] for t in chosen)
        n = sum(per_task[t][2] for t in chosen)
        return (b - a) / n

    rng = random.Random(seed)
    reps = sorted(diff([rng.choice(tasks) for _ in tasks]) for _ in range(n_boot))
    low = reps[int(alpha / 2 * n_boot)]
    high = reps[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    below = sum(1 for r in reps if r <= 0) / n_boot
    above = sum(1 for r in reps if r >= 0) / n_boot
    a_only = sum(1 for pa, pb in blocks.values() if pa and not pb)
    b_only = sum(1 for pa, pb in blocks.values() if pb and not pa)
    return {
        "blocks": len(blocks),
        "tasks": len(tasks),
        "rate_a": sum(pa for pa, _ in blocks.values()) / len(blocks),
        "rate_b": sum(pb for _, pb in blocks.values()) / len(blocks),
        "diff": diff(tasks),
        "ci": (low, high),
        "p_boot": min(1.0, 2 * min(below, above)),
        "a_only": a_only,
        "b_only": b_only,
    }


def pass_hat_k(rows: list[Row], arm: str, k: int, exclude: frozenset[str] = frozenset()) -> float | None:
    """Mean over tasks with at least k graded seeds of C(c, k) / C(n, k)."""
    graded: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        if row.get("arm") == arm and row.get("passed") is not None and row.get("task") not in exclude:
            graded[row["task"]].append(bool(row["passed"]))
    values = [comb(sum(v), k) / comb(len(v), k) for v in graded.values() if len(v) >= k]
    return sum(values) / len(values) if values else None


def success_per_dollar(rows: list[Row], arm: str, cost_key: str = "reference_cost_usd",
                       exclude: frozenset[str] = frozenset()) -> float | None:
    """Passes per reference dollar over all of the arm's graded runs."""
    chosen = [r for r in rows if r.get("arm") == arm and r.get("passed") is not None and r.get("task") not in exclude]
    cost = sum(float(r.get(cost_key) or 0.0) for r in chosen)
    return sum(1 for r in chosen if r["passed"]) / cost if cost > 0 else None
