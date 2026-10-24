"""Repository paths."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Directory that contains this project's pyproject.toml."""
    override = os.environ.get("EVCARRY_ROOT")
    if override:
        return Path(override).resolve()
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "evcarry").is_dir():
            return candidate
    raise RuntimeError("Could not locate the evcarry repository root.")


def resolve_repo_path(path: str | Path, *, root: Path | None = None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (root or repo_root()) / candidate
