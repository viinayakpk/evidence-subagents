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


# The sub-agent's reply format, per arm. Each file is appended to the role prompt;
# a prompt set without the file (prompts/v1) sends the role prompt alone.
_RETURN_FILE = {
    "vanilla_mad": "return_text.md",
    "self_verify": "return_text_self_verify.md",
    "typed": "return_typed.md",
    "evidence": "return_claims.md",
    "evidence_random": "return_claims.md",
    "evidence_risk": "return_claims.md",
}

# How the main agent sees its two sub-agent tools. Identical in every delegating arm.
SUBAGENT_TOOL_DESCRIPTIONS = {
    "researcher": "Researcher sub-agent: looks things up with read-only tools and reports what it found.",
    "operator": "Operator sub-agent: carries out changes with every tool, including writes and python_execute, and reports what it did.",
}

# Used when a prompt set has no repair.md (the repair is only sent in schema arms).
DEFAULT_REPAIR = "The previous return was not valid JSON for the claim schema. Return one JSON object and nothing else."


def subagent_instructions(prompts_dir: str | Path, role: str, arm: str) -> str:
    """Role prompt, then the arm's return-format file (``_RETURN_FILE``) after a blank line, if the set has it."""
    text = prompt_text(prompts_dir, role)
    filename = _RETURN_FILE.get(arm)
    suffix = prompt_path(prompts_dir, filename) if filename else None
    if suffix is not None and suffix.is_file():
        text = text.rstrip("\n") + "\n\n" + suffix.read_text(encoding="utf-8")
    return text


def repair_prompt(prompts_dir: str | Path | None, error: str) -> str:
    """``repair.md`` of the prompt set, then the parse error."""
    path = prompt_path(prompts_dir, "repair") if prompts_dir is not None else None
    base = path.read_text(encoding="utf-8").strip() if path is not None and path.is_file() else DEFAULT_REPAIR
    return f"{base}\n{error}"
