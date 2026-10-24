"""Persist a ledger next to a run trace."""

from __future__ import annotations

from pathlib import Path

from evcarry.ledger.ledger import Ledger


def write_ledger(path: Path, ledger: Ledger) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ledger.to_json(), encoding="utf-8")


def read_ledger(path: Path) -> Ledger:
    return Ledger.from_json(path.read_text(encoding="utf-8"))
