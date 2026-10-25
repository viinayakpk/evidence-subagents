"""Prompts differ across arms only where the arm requires it, and a frozen prompt set cannot drift."""

import shutil
from pathlib import Path

import pytest

from evcarry.harness import prompt_freeze
from evcarry.harness.arms import ARMS
from evcarry.harness.prompts import SUBAGENT_TOOL_DESCRIPTIONS, prompt_text, subagent_instructions
from evcarry.protocol.policy import PolicyDecision, fallback_message

GYM = "prompts/gym_v1"
DELEGATING = [name for name, spec in ARMS.items() if spec.delegate]
AUDITED = [name for name, spec in ARMS.items() if spec.checks]


def _preamble(text: str) -> str:
    marker = "format it needs.\n"
    return text[: text.index(marker) + len(marker)]


def test_delegating_main_prompts_share_one_preamble_and_differ_only_in_the_arm_block() -> None:
    preambles = {_preamble(prompt_text(GYM, arm)) for arm in DELEGATING}
    assert len(preambles) == 1
    (preamble,) = preambles
    sa = prompt_text(GYM, "sa")
    general = preamble.split("\n\n")[0]
    assert sa.startswith(general)  # workspace, autonomy and claim_done lines are the same for SA
    for arm in DELEGATING:
        block = prompt_text(GYM, arm)[len(preamble):]
        assert block.strip()
        # Only the arms that audit can produce FALLBACK, so only they are told about it.
        assert ("FALLBACK" in block) == (arm in AUDITED)
        assert ("cites the tool calls" in block) == (arm in AUDITED)


def test_sub_agent_prompts_follow_the_arm_definitions() -> None:
    for role in ("researcher", "operator"):
        role_text = prompt_text(GYM, role)
        vanilla = subagent_instructions(GYM, role, "vanilla_mad")
        verify = subagent_instructions(GYM, role, "self_verify")
        typed = subagent_instructions(GYM, role, "typed")
        evidence = subagent_instructions(GYM, role, "evidence_risk")
        for text in (vanilla, verify, typed, evidence):
            assert text.startswith(role_text.rstrip("\n") + "\n\n")
        # Vanilla: free text, no self-check, no citations.
        assert "plain-text report" in vanilla and "check every value" not in vanilla and "call_id\"" not in vanilla
        # Self-verify: sub-agents double-check and show the output behind each value.
        assert "check every value" in verify and "quote the part of the tool result" in verify
        # Typed: the claim structure without evidence.
        assert '"evidence": []' in typed and "Leave evidence empty." in typed and '"span"' not in typed
        # Evidence: claims cite calls and quote spans.
        assert '"span"' in evidence and "Cite the call" in evidence
        # Typed and evidence share every other rule word for word.
        for rule in ("When a claim is about a list or a set, include every item.", "Put anything you could not do in open_issues."):
            assert rule in typed and rule in evidence
    assert subagent_instructions(GYM, "researcher", "evidence") == subagent_instructions(GYM, "researcher", "evidence_random")


def test_sub_agent_tool_descriptions_are_neutral() -> None:
    for description in SUBAGENT_TOOL_DESCRIPTIONS.values():
        assert "claim" not in description.lower() and "evidence" not in description.lower()


def test_fallback_message_matches_what_the_main_agent_can_do() -> None:
    message = fallback_message("researcher", PolicyDecision("fallback", True, ["c1"], ["c1: L1: span not found"]))
    assert "yourself" not in message
    assert "delegate it with a more specific instruction" in message
    assert "delegate it with a more specific instruction" in prompt_text(GYM, "evidence_risk")


def _copy_prompts(tmp_path: Path) -> Path:
    target = tmp_path / "gym_v1"
    shutil.copytree(GYM, target)
    return target


def test_freeze_detects_file_and_code_changes(tmp_path: Path, monkeypatch) -> None:
    prompts = _copy_prompts(tmp_path)
    assert prompt_freeze.status(prompts).state == "unfrozen"
    prompt_freeze.freeze(prompts, git_commit="test")
    assert prompt_freeze.status(prompts).state == "match"
    (prompts / "return_text.md").write_text("Reply however you like.\n", encoding="utf-8")
    changed = prompt_freeze.status(prompts)
    assert changed.state == "changed"
    assert changed.changed == [
        "arm/budget_matched/operator", "arm/budget_matched/researcher",  # same return format as vanilla_mad (A21)
        "arm/vanilla_mad/operator", "arm/vanilla_mad/researcher",
    ]
    shutil.copy(Path(GYM) / "return_text.md", prompts / "return_text.md")
    assert prompt_freeze.status(prompts).state == "match"
    monkeypatch.setitem(SUBAGENT_TOOL_DESCRIPTIONS, "researcher", "Researcher: gathers evidence and returns claims.")
    assert prompt_freeze.status(prompts).changed == ["tool/researcher"]


async def test_a_paid_run_refuses_a_changed_frozen_prompt_set(tmp_path: Path, monkeypatch) -> None:
    from evcarry.runs.config import load_experiment
    from evcarry.runs.manager import FrozenInputMismatch, run_experiment

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    prompts = _copy_prompts(tmp_path)
    prompt_freeze.freeze(prompts)
    (prompts / "sa_system.md").write_text("Do the task.\n", encoding="utf-8")
    config = load_experiment("configs/experiments/openrouter_smoke.yaml")
    config.prompts = str(prompts)
    # Refused before the API key is even looked up, so no request can be sent.
    with pytest.raises(FrozenInputMismatch, match="sa_system|arm/sa/main"):
        await run_experiment(config, out_dir=tmp_path / "out")
    # A non-billable run still runs, and its run_start says the set changed.
    mock = load_experiment("configs/experiments/mock_e2e.yaml")
    mock.arms, mock.tasks, mock.prompts = ["sa"], "tests/fixtures/tasks/one_sections.txt", "prompts/v1"
    await run_experiment(mock, out_dir=tmp_path / "mock")
    from evcarry.trace.writer import read_events

    start = read_events(next((tmp_path / "mock").glob("*.jsonl")))[0]
    assert start["prompts_frozen"] == "unfrozen" and len(start["prompt_manifest_sha256"]) == 64


async def test_require_frozen_prompts_refuses_an_unfrozen_set(tmp_path: Path) -> None:
    from evcarry.runs.config import load_experiment
    from evcarry.runs.manager import FrozenInputMismatch, run_experiment

    mock = load_experiment("configs/experiments/mock_e2e.yaml")
    mock.require_frozen_prompts = True
    with pytest.raises(FrozenInputMismatch, match="requires a frozen prompt set"):
        await run_experiment(mock, out_dir=tmp_path / "out")


def test_every_return_format_asks_for_the_same_completeness() -> None:
    """Only the JSON formats asked for every list item and for what could not be done; vanilla was told to be short."""
    formats = {arm: subagent_instructions(GYM, "researcher", arm) for arm in ("vanilla_mad", "self_verify", "typed", "evidence_risk")}
    for arm, text in formats.items():
        assert "exactly as they appear" in text, arm
        assert "include every item" in text, arm
        assert "could not do" in text, arm
        assert "short plain-text" not in text and "short report" not in text, arm
