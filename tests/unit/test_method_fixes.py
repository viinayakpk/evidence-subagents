"""Pre-registered method amendments of 2026-10-25 (P1-8, P1-9, P1-10, P2-5) and handoff logging (P0-2)."""

from __future__ import annotations

import json
from pathlib import Path

from agents import Agent, Runner

from evcarry.harness.delegate import DelegateState, make_extractor
from evcarry.harness.prompts import prompt_text, subagent_instructions
from evcarry.ledger.ledger import Ledger
from evcarry.protocol.policy import PolicyDecision, decide, fallback_message, feedback_message
from evcarry.providers.scripted import FixtureModel, ScriptStep, _input_text

GYM = "prompts/gym_v1"


class _Events:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def write(self, event: dict) -> None:
        self.events.append(event)

    def of(self, kind: str) -> list[dict]:
        return [event for event in self.events if event["type"] == kind]


class _InputsModel(FixtureModel):
    def __init__(self, steps: list[ScriptStep]) -> None:
        super().__init__(steps, agent_name="researcher")
        self.inputs: list[str] = []

    def _next_step(self, input):
        self.inputs.append(_input_text(input))
        return super()._next_step(input)


def _claim(claim_id: str, value: int, span: str = "14 sections") -> dict:
    return {"claim_id": claim_id, "text": f"{claim_id} is {value}", "kind": "fact", "value": value,
            "evidence": [{"type": "tool_call", "call_id": "call_1", "span": span}]}


def _reply(*claims: dict, issues: list[str] | None = None) -> str:
    return json.dumps({"subagent_id": "researcher", "status": "complete", "claims": list(claims), "open_issues": issues or []})


def _ledger() -> Ledger:
    ledger = Ledger()
    ledger.record(call_id="call_1", tool_name="read_file", arguments={"path": "a"}, result_text="14 sections, 3 papers",
                  read_only=True)
    return ledger


async def _review(replies: list[str]) -> tuple[str, _Events, _InputsModel]:
    model = _InputsModel([ScriptStep(content=reply) for reply in replies])
    agent = Agent(name="researcher", instructions="You are the researcher.", model=model)
    writer = _Events()
    extractor = make_extractor(
        agent=agent, hooks=None, writer=writer, ledger=_ledger(), caller=None, task="t", seed=1, arm="evidence",
        checks=("L0", "L1"), sampler=None, rate=0.0, require_evidence=True, injector=None, state=DelegateState(),
        max_turns=5,
    )
    delivered = await extractor(await Runner.run(agent, "Count the sections and the papers."))
    return delivered, writer, model


# P1-8 -------------------------------------------------------------------------------------------------


def test_policy_decisions_per_claim() -> None:
    assert decide([], prior_redelegations=0).action == "accept"
    assert decide([("c2", "L1: x")], prior_redelegations=0, verified=["c1"]).action == "redelegate"
    kept = decide([("c2", "L1: x")], prior_redelegations=1, verified=["c1"])
    assert (kept.action, kept.accepted_claims, kept.failed_claims) == ("partial_accept", ["c1"], ["c2"])
    assert decide([("c2", "L1: x")], prior_redelegations=1).action == "fallback"
    assert kept.as_dict()["accepted_claims"] == ["c1"]


async def test_a_retry_resends_only_the_failed_claim_and_both_are_delivered() -> None:
    delivered, writer, model = await _review([
        _reply(_claim("c1", 14), _claim("c2", 4, "3 papers")),
        _reply(_claim("c2", 3, "3 papers")),
    ])
    assert [event["action"] for event in writer.of("policy")] == ["redelegate", "accept"]
    payload = json.loads(delivered)
    assert {claim["claim_id"]: claim["value"] for claim in payload["claims"]} == {"c1": 14, "c2": 3}
    assert payload["open_issues"] == []
    # The retry is told which claim failed and that c1 was accepted.
    retry = model.inputs[1]
    assert "c2: L1: value 4 is not in any cited result" in retry and "c1:" not in retry.split("could not verify", 1)[1]


async def test_after_the_retry_verified_claims_are_kept_and_the_rest_are_listed() -> None:
    delivered, writer, _model = await _review([
        _reply(_claim("c1", 14), _claim("c2", 4, "3 papers"), _claim("c3", 5, "3 papers")),
        _reply(_claim("c2", 7, "3 papers"), issues=["c3 could not be confirmed"]),
    ])
    policies = writer.of("policy")
    assert [event["action"] for event in policies] == ["redelegate", "partial_accept"]
    # c3 was left out of the retry: withdrawn, so rejected too.
    assert policies[-1]["accepted_claims"] == ["c1"] and policies[-1]["failed_claims"] == ["c2", "c3"]
    payload = json.loads(delivered)
    assert [claim["claim_id"] for claim in payload["claims"]] == ["c1"]
    issues = payload["open_issues"]
    assert "c3 could not be confirmed" in issues
    assert any(issue.startswith("REJECTED c2 (not verified; do not rely on it): L1: value 7") for issue in issues)
    assert any(issue.startswith("REJECTED c3") and "withdrawn after re-check" in issue for issue in issues)


async def test_fallback_only_when_nothing_was_verified() -> None:
    delivered, writer, _model = await _review([_reply(_claim("c1", 9)), _reply(_claim("c1", 8))])
    assert [event["action"] for event in writer.of("policy")] == ["redelegate", "fallback"]
    assert delivered.startswith("FALLBACK: none of sub-agent researcher's claims could be verified")
    # Withdrawing every failed claim in the retry is not an accept either.
    delivered, writer, _model = await _review([_reply(_claim("c1", 9)), _reply(issues=["c1 could not be supported"])])
    assert [event["action"] for event in writer.of("policy")] == ["redelegate", "fallback"]
    assert delivered.startswith("FALLBACK")


async def test_a_failed_resend_does_not_undo_a_verified_claim() -> None:
    delivered, writer, _model = await _review([
        _reply(_claim("c1", 14), _claim("c2", 4, "3 papers")),
        _reply(_claim("c1", 15), _claim("c2", 3, "3 papers")),  # c1 sent again, now wrong; c2 fixed
    ])
    policies = writer.of("policy")
    assert [event["action"] for event in policies] == ["redelegate", "accept"]
    assert policies[-1]["failed_claims"] == [] and policies[-1]["accepted_claims"] == ["c1", "c2"]
    payload = json.loads(delivered)
    assert {claim["claim_id"]: claim["value"] for claim in payload["claims"]} == {"c1": 14, "c2": 3}
    assert not any(issue.startswith("REJECTED") for issue in payload["open_issues"])
    # The check event still records the failed resend.
    assert any(failure["claim_id"] == "c1" for failure in writer.of("check")[-1]["failures"])


# P1-9 -------------------------------------------------------------------------------------------------


def test_redelegation_says_not_to_repeat_succeeded_actions() -> None:
    decision = PolicyDecision("redelegate", True, ["c2"], ["c2: L2: read-back of emails-send_email does not show the write's subject"])
    message = feedback_message(decision)
    assert "Do not repeat actions that already succeeded" in message
    assert "contains only these claims" in message and "c2: L2" in message
    fallback = fallback_message("operator", PolicyDecision("fallback", True, ["c1"], ["c1: L1: x"]))
    assert "does not repeat them" in fallback and "delegate it with a more specific instruction" in fallback


# P1-10 ------------------------------------------------------------------------------------------------


def test_claim_scope_is_in_the_evidence_return_format_only() -> None:
    evidence = subagent_instructions(GYM, "researcher", "evidence_risk")
    assert "which tools you have" in evidence and "One checkable fact per claim" in evidence
    for arm in ("vanilla_mad", "self_verify", "typed"):
        assert "which tools you have" not in subagent_instructions(GYM, "researcher", arm), arm
    main = prompt_text(GYM, "evidence_risk")
    assert "REJECTED" in main and "FALLBACK" in main


# P0-2 -------------------------------------------------------------------------------------------------


async def test_the_harness_messages_are_logged() -> None:
    _delivered, writer, _model = await _review([_reply(_claim("c1", 14), _claim("c2", 4, "3 papers")), _reply(_claim("c2", 3, "3 papers"))])
    handoffs = writer.of("handoff")
    assert [event["kind"] for event in handoffs] == ["subagent_output", "redelegation", "subagent_output"]
    assert '"value": 4' in handoffs[0]["text"] and "could not verify some claims" in handoffs[1]["text"]
    assert all(event["subagent"] == "researcher" and event["chars"] == len(event["text"]) for event in handoffs)


async def test_a_mock_run_logs_every_handoff(tmp_path: Path) -> None:
    from evcarry.runs.config import load_experiment
    from evcarry.runs.manager import run_experiment
    from evcarry.trace.writer import read_events

    config = load_experiment("configs/experiments/mock_e2e.yaml")
    config.arms = ["vanilla_mad", "evidence_risk"]
    config.tasks = "tests/fixtures/tasks/one_sections.txt"
    await run_experiment(config, out_dir=tmp_path)
    for path in sorted(tmp_path.glob("*.jsonl")):
        events = read_events(path)
        handoffs = [event for event in events if event["type"] == "handoff"]
        instructions = [event for event in handoffs if event["kind"] == "instruction"]
        delivered = [event for event in handoffs if event["kind"] == "delivered"]
        # The main agent called researcher ("collect") and operator ("finish"); both are logged with what came back.
        assert [(event["subagent"], event["text"]) for event in instructions] == [("researcher", "collect"), ("operator", "finish")]
        assert [event["call_id"] for event in instructions] == ["call_sub_research", "call_sub_operate"]
        assert [event["call_id"] for event in delivered] == ["call_sub_research", "call_sub_operate"]
        assert all(event["text"] for event in delivered)
        if "__evidence_risk__" in path.name:
            outputs = [event for event in handoffs if event["kind"] == "subagent_output"]
            assert [event["call_id"] for event in outputs] == ["call_sub_research", "call_sub_operate"]
            assert '"claim_id"' in outputs[0]["text"]


# P2-5 -------------------------------------------------------------------------------------------------


def test_every_delegating_main_agent_gets_the_same_tool_list(monkeypatch) -> None:
    from evcarry.harness import agents as agents_module
    from evcarry.harness.arms import arm_spec
    from evcarry.harness.hooks import LedgerHooks
    from evcarry.harness.tools import fixture_toolset
    from evcarry.providers.registry import load_models
    from evcarry.providers.usage import PriceTable

    monkeypatch.setattr(agents_module, "_model", lambda *args: FixtureModel([], agent_name=args[-1]))
    ledger, writer = Ledger(), _Events()
    toolset = fixture_toolset(None, ledger)
    hooks = LedgerHooks(ledger, writer, PriceTable(0, 0, 0, peak_multiplier=1), toolset.catalog)
    blocks = {}
    for arm in ("vanilla_mad", "self_verify", "evidence_risk", "sa"):
        main = agents_module.build_main_agent(
            arm=arm_spec(arm), task="t", seed=1, prompts_dir=GYM, model_config=load_models()["mock"], script=None,
            toolset=toolset, hooks=hooks, writer=writer, ledger=ledger, checks=(), sampler=None, rate=0.0,
            injector=None, subagent_max_turns=5,
        )
        own = prompt_text(GYM, arm).rstrip("\n")
        assert main.instructions.startswith(own)
        blocks[arm] = main.instructions[len(own):].strip()
    assert blocks["sa"] == ""
    assert blocks["vanilla_mad"] == blocks["self_verify"] == blocks["evidence_risk"]
    block = blocks["vanilla_mad"]
    assert "- researcher: local: query_rows, read_file, search_records" in block
    assert "- operator: local: create_event, list_events, read_file, write_answer" in block
