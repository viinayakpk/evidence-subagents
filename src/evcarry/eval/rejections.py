"""Evidence-arm rejections split into attributed and false alarms (amendment A16).

A rejected return is attributed when the audit found a real error: class A (it cited a failed call) or
B (a side detail was really wrong), the classes of ``docs/PILOT_DIAGNOSIS.md`` Appendix A. Class C
(only meta, over-broad or unverifiable claims, no error shown) and D (a check or harness bug) are false
alarms. Classes are assigned by hand from the traces, blind to the run's grade where possible.

The sensitivity analysis follows Maat's erratum (2609.34017): a run with any false-alarm rejection is
counted as failed, which bounds from below what the protocol is worth if its false alarms had cost the
task. It is reported next to the primary result, never instead of it.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

ATTRIBUTED = frozenset({"A", "B"})
FALSE_ALARM = frozenset({"C", "D"})


def split_rejections(classes: Iterable[str]) -> dict[str, int]:
    """Counts of attributed, false-alarm and unlabelled rejections."""
    counts = Counter()
    for cls in classes:
        key = "attributed" if cls in ATTRIBUTED else "false_alarm" if cls in FALSE_ALARM else "unlabelled"
        counts[key] += 1
    total = sum(counts.values())
    return {"attributed": counts["attributed"], "false_alarm": counts["false_alarm"],
            "unlabelled": counts["unlabelled"], "total": total,
            "false_alarm_share": counts["false_alarm"] / total if total else None}


def false_alarm_runs(rejections: Iterable[dict[str, Any]]) -> set[tuple[str, Any]]:
    """``(task, seed)`` of runs with at least one false-alarm rejection; rows carry task, seed, class."""
    return {(r["task"], r.get("seed")) for r in rejections if r.get("class") in FALSE_ALARM}


def with_false_alarms_as_failures(rows: list[dict[str, Any]], arm: str,
                                  flagged: set[tuple[str, Any]]) -> list[dict[str, Any]]:
    """``rows`` with ``arm``'s flagged runs set to failed (the lower-bound sensitivity)."""
    out = []
    for row in rows:
        if row.get("arm") == arm and (row.get("task"), row.get("seed")) in flagged and row.get("passed"):
            row = {**row, "passed": False, "sensitivity": "false_alarm_as_failure"}
        out.append(row)
    return out
