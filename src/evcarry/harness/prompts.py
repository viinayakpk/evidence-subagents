"""Frozen prompt files. The system prompt is the file bytes, unchanged."""

from __future__ import annotations

from pathlib import Path

from evcarry.paths import resolve_repo_path

_ARM_FILE = {
    "sa": "sa_system.md",
    "vanilla_mad": "vanilla_system.md",
    "self_verify": "self_verify_system.md",
    "typed": "typed_system.md",
    "evidence": "evidence_system.md",
    "evidence_random": "evidence_system.md",
    "evidence_risk": "evidence_system.md",
    "researcher": "researcher.md",
    "operator": "operator.md",
    "repair": "repair.md",
    "judge": "judge.md",
}


def prompt_path(prompts_dir: str | Path, name: str) -> Path:
    filename = _ARM_FILE.get(name, name)
    return resolve_repo_path(Path(prompts_dir) / filename)


def prompt_text(prompts_dir: str | Path, name: str) -> str:
    return prompt_path(prompts_dir, name).read_text(encoding="utf-8")
