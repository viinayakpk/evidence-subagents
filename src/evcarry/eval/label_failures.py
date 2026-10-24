"""Provisional failure labels. This is not an MS-Eval taxonomy.

Labels are written to a sidecar file. The trace JSONL is not modified.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TAXONOMY = (
    "delegated_failure_propagation",
    "specification_miss",
    "tool_error",
    "context_loss",
    "other",
)


def label_events(events: list[dict[str, Any]]) -> dict[str, str]:
    """Pick one provisional label from the trace. The mapping is a local stand-in."""
    types = [event.get("type") for event in events]
    policy_actions = [event.get("action") for event in events if event.get("type") == "policy"]
    tool_errors = [
        event
        for event in events
        if event.get("type") == "tool_call" and "error" in json.dumps(event.get("arguments", {})).lower()
    ]
    passed = None
    for event in events:
        if event.get("type") == "run_end":
            passed = event.get("passed")
    if "fallback" in policy_actions:
        label = "delegated_failure_propagation"
    elif tool_errors:
        label = "tool_error"
    elif passed is False:
        label = "specification_miss"
    elif "redelegate" in policy_actions:
        label = "context_loss"
    else:
        label = "other"
    if label not in TAXONOMY:
        label = "other"
    return {"label": label, "taxonomy": "provisional"}


def write_labels(trace_path: Path) -> Path:
    events = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    sidecar = trace_path.with_suffix(".labels.jsonl")
    record = {"trace": trace_path.name, **label_events(events)}
    sidecar.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    return sidecar
