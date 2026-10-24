"""One JSONL file per run. Events are sorted-key JSON."""

from __future__ import annotations

import json
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable

current_writer: ContextVar["JsonlWriter | None"] = ContextVar("evcarry_trace_writer", default=None)


class JsonlWriter:
    def __init__(self, path: Path, observer: Callable[[dict[str, Any]], None] | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._handle = path.open("a", encoding="utf-8")
        # Sees every event after it is written (the run's spend meter).
        self.observer = observer

    def write(self, event: dict[str, Any]) -> None:
        self._handle.write(json.dumps(event, sort_keys=True, default=str) + "\n")
        self._handle.flush()
        if self.observer is not None:
            self.observer(event)

    def close(self) -> None:
        self._handle.close()


def run_path(out_dir: Path, task: str, arm: str, model: str, seed: int) -> Path:
    safe_model = model.replace("/", "_")
    return out_dir / f"{task}__{arm}__{safe_model}__{seed}.jsonl"


def has_run_end(path: Path) -> bool:
    if not path.is_file():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "run_end":
            return True
    return False


def read_events(path: Path) -> list[dict[str, Any]]:
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events
