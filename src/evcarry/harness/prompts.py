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


def subagent_instructions(prompts_dir: str | Path, role: str, schema: bool) -> str:
    """Role prompt, then the return-format file for the arm if the prompt set has one.

    ``return_claims.md`` (typed and evidence arms) or ``return_text.md`` (free-text
    arms) is appended after a blank line. ``prompts/v1`` has neither file, so the
    fixture prompts are the role files unchanged.
    """
    text = prompt_text(prompts_dir, role)
    suffix = prompt_path(prompts_dir, "return_claims.md" if schema else "return_text.md")
    if suffix.is_file():
        text = text.rstrip("\n") + "\n\n" + suffix.read_text(encoding="utf-8")
    return text
