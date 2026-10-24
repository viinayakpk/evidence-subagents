"""Parquet index of run_end rows.

``passed`` is the result the paper counts; ``eval_passed`` is the grader's own.
They differ when a run is ``contaminated`` (it reached for ground truth) or
stopped at its cost ceiling: both count as failures.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from evcarry.trace.writer import read_events


def write_index(run_dir: Path) -> Path:
    rows = []
    for path in sorted(run_dir.glob("*.jsonl")):
        for event in read_events(path):
            if event.get("type") == "run_end":
                rows.append(
                    {
                        "task": event.get("task"),
                        "arm": event.get("arm"),
                        "model": event.get("model"),
                        "seed": event.get("seed"),
                        "passed": event.get("passed"),
                        "eval_passed": event.get("eval_passed", event.get("passed")),
                        "contaminated": bool(event.get("contaminated", False)),
                        "cost_ceiling": bool(event.get("cost_ceiling", False)),
                        "status": event.get("status"),
                        "agent_status": event.get("agent_status"),
                        "input_tokens": event.get("input_tokens"),
                        "output_tokens": event.get("output_tokens"),
                        "cost_usd": event.get("cost_usd"),
                        "judge_cost_usd": event.get("judge_cost_usd"),
                        "spend_usd": event.get("spend_usd"),
                    }
                )
    if not rows:
        table = pa.table(
            {
                "task": pa.array([], type=pa.string()),
                "arm": pa.array([], type=pa.string()),
                "model": pa.array([], type=pa.string()),
                "seed": pa.array([], type=pa.int64()),
                "passed": pa.array([], type=pa.bool_()),
                "eval_passed": pa.array([], type=pa.bool_()),
                "contaminated": pa.array([], type=pa.bool_()),
                "cost_ceiling": pa.array([], type=pa.bool_()),
                "status": pa.array([], type=pa.string()),
                "agent_status": pa.array([], type=pa.string()),
                "input_tokens": pa.array([], type=pa.int64()),
                "output_tokens": pa.array([], type=pa.int64()),
                "cost_usd": pa.array([], type=pa.float64()),
                "judge_cost_usd": pa.array([], type=pa.float64()),
                "spend_usd": pa.array([], type=pa.float64()),
            }
        )
    else:
        table = pa.Table.from_pylist(rows)
    dest = run_dir / "index.parquet"
    pq.write_table(table, dest)
    return dest
