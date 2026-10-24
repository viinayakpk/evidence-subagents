import json

import pytest

from evcarry.protocol.policy import decide, feedback_message
from evcarry.protocol.sampler import RiskContext, budget_count, sample_random, sample_risk
from evcarry.protocol.schema import Claim, FailedReturn, SubagentReturn, ToolCallEvidence, parse_subagent_return


def _claim(claim_id: str, *, kind: str = "fact", value: object = 14, confidence: str | None = "high") -> Claim:
    return Claim(
        claim_id=claim_id,
        text=f"value {claim_id}",
        kind=kind,  # type: ignore[arg-type]
        value=value,
        evidence=[ToolCallEvidence(type="tool_call", call_id="c", span="14")],
        confidence=confidence,  # type: ignore[arg-type]
    )


def _payload() -> dict:
    return {
        "subagent_id": "researcher",
        "status": "complete",
        "claims": [
            {
                "claim_id": "c1",
                "text": "The record contains 14 sections.",
                "kind": "fact",
                "value": 14,
                "evidence": [{"type": "tool_call", "call_id": "call_search", "span": "14 sections"}],
            }
        ],
        "open_issues": [],
    }


def test_schema_round_trip_and_one_repair() -> None:
    raw = json.dumps(_payload())
    parsed, repairs = parse_subagent_return(raw)
    assert repairs == 0
    assert parsed.claims[0].evidence[0].call_id == "call_search"
    again = SubagentReturn.model_validate_json(parsed.model_dump_json())
    assert again.subagent_id == "researcher"
    with pytest.raises(FailedReturn) as rejected:
        parse_subagent_return("{")
    assert rejected.value.repairs == 0

    def repair(_text: str, _error: str) -> str:
        return raw

    fixed, repairs = parse_subagent_return("{", repair=repair)
    assert repairs == 1
    assert fixed.claims[0].claim_id == "c1"

    def still_bad(_text: str, _error: str) -> str:
        return "{"

    with pytest.raises(FailedReturn) as second:
        parse_subagent_return("{", repair=still_bad)
    assert second.value.repairs == 1


def test_policy_accept_redelegate_fallback() -> None:
    assert decide([], prior_redelegations=0).action == "accept"
    first = decide([("c1", "missing")], prior_redelegations=0)
    assert first.action == "redelegate"
    assert "c1" in feedback_message(first)
    second = decide([("c1", "missing")], prior_redelegations=1)
    assert second.action == "fallback"
    with pytest.raises(ValueError):
        decide([], prior_redelegations=-1)


def test_sampler_budget_and_risk_order() -> None:
    assert budget_count(10, 0) == 0
    assert budget_count(1, 0.3) == 0
    assert budget_count(10, 0.3) == 3
    claims = [
        _claim("b", kind="fact", value="plain", confidence="high"),
        _claim("a", kind="action_done", value=3, confidence="low"),
        _claim("c", kind="derived", value="x", confidence="high"),
    ]
    ctx = RiskContext(deliverable_ids={"a"}, failed_l0_l1_ids=set(), hit_limits=False)
    ranked = sample_risk(claims, rate=1.0, ctx=ctx)
    assert ranked[0] == "a"
    assert sample_risk(claims, rate=0.3, ctx=ctx) == ["a"]
    first = sample_random(claims, rate=1.0, seed=7)
    second = sample_random(claims, rate=1.0, seed=7)
    assert first == second
    assert len(sample_random(claims, rate=0.3, seed=7)) == 1
    assert budget_count(1, 0.3) == 0
