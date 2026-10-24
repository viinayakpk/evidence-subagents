"""Parquet index of run_end rows."""

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
                        "status": event.get("status"),
                        "input_tokens": event.get("input_tokens"),
                        "output_tokens": event.get("output_tokens"),
                        "cost_usd": event.get("cost_usd"),
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
                "status": pa.array([], type=pa.string()),
                "input_tokens": pa.array([], type=pa.int64()),
                "output_tokens": pa.array([], type=pa.int64()),
                "cost_usd": pa.array([], type=pa.float64()),
            }
        )
    else:
        table = pa.Table.from_pylist(rows)
    dest = run_dir / "index.parquet"
    pq.write_table(table, dest)
    return dest
