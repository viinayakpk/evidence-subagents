"""Pairwise outcome flips between two arms on matched (task, seed) rows."""

from __future__ import annotations

from typing import Any


def pairwise_flips(rows: list[dict[str, Any]], arm_a: str, arm_b: str) -> dict[str, int]:
    """Count pass/fail disagreements. ``a_only`` means A passed and B failed."""
    by_key: dict[tuple[str, int], dict[str, bool]] = {}
    for row in rows:
        if row.get("passed") is None:
            continue
        key = (str(row["task"]), int(row["seed"]))
        by_key.setdefault(key, {})[str(row["arm"])] = bool(row["passed"])
    both_pass = both_fail = a_only = b_only = 0
    for outcomes in by_key.values():
        if arm_a not in outcomes or arm_b not in outcomes:
            continue
        left, right = outcomes[arm_a], outcomes[arm_b]
        if left and right:
            both_pass += 1
        elif not left and not right:
            both_fail += 1
        elif left and not right:
            a_only += 1
        else:
            b_only += 1
    return {"both_pass": both_pass, "both_fail": both_fail, "a_only": a_only, "b_only": b_only}
