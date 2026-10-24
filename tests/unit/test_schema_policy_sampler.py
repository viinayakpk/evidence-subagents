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
    assert budget_count(10, 0.3) == 3
    assert budget_count(0, 0.3) == 0
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


def test_every_return_gets_at_least_one_audit_when_the_rate_is_above_zero() -> None:
    """Owner decision 1: 0.3 * 1 rounded to 0, so one-claim returns were never audited."""
    assert [budget_count(n, 0.3) for n in range(1, 8)] == [1, 1, 1, 1, 2, 2, 2]
    assert [budget_count(n, 0.1) for n in (1, 4, 10, 15)] == [1, 1, 1, 2]
    assert budget_count(1, 0) == 0  # rate 0 (the evidence arm) still audits nothing
    one = [_claim("only")]
    assert sample_risk(one, rate=0.3) == ["only"]
    assert sample_random(one, rate=0.3, seed="audit|t|1|0") == ["only"]
    two = [_claim("a", kind="fact", value="plain"), _claim("b", kind="derived")]
    assert sample_risk(two, rate=0.3) == ["b"]  # the higher-risk claim
    assert len(sample_random(two, rate=0.3, seed="audit|t|1|0")) == 1


@pytest.mark.parametrize("arm", ["evidence_risk", "evidence_random"])
async def test_both_audited_arms_audit_a_one_claim_return(arm: str) -> None:
    """The floor is applied by the shared budget, so the risk and random arms get the same number of audits."""
    from types import SimpleNamespace

    from evcarry.harness.arms import effective_audit
    from evcarry.harness.delegate import DelegateState, make_extractor
    from evcarry.ledger.ledger import Ledger

    checks, sampler, rate = effective_audit(arm, sampler="risk", rate=0.3, checks=["L0", "L1", "L2", "L3"])
    writer = _Events()
    extractor = make_extractor(
        agent=None, hooks=None, writer=writer, ledger=Ledger(), caller=None, task="t", seed=1, arm=arm,
        checks=("L2",), sampler=sampler, rate=rate, require_evidence=True, injector=None, state=DelegateState(),
        max_turns=5,
    )
    payload = SubagentReturn(subagent_id="researcher", status="complete", claims=[_claim("only")], open_issues=[])
    await extractor(SimpleNamespace(final_output=payload))
    assert [event["sampled"] for event in writer.events if event["type"] == "audit"] == [["only"]]


class _Events:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def write(self, event: dict) -> None:
        self.events.append(event)


async def test_random_audit_draws_again_for_each_return() -> None:
    """The random control used the run seed for every return, so it always audited the same position."""
    from types import SimpleNamespace

    from evcarry.harness.delegate import DelegateState, make_extractor
    from evcarry.ledger.ledger import Ledger

    writer = _Events()
    extractor = make_extractor(
        agent=None, hooks=None, writer=writer, ledger=Ledger(), caller=None, task="t", seed=1, arm="evidence_random",
        checks=("L2",), sampler="random", rate=0.5, require_evidence=True, injector=None, state=DelegateState(),
        max_turns=5,
    )
    payload = SubagentReturn(
        subagent_id="researcher", status="complete", claims=[_claim(f"c{i}") for i in range(4)], open_issues=[]
    )
    for _ in range(12):
        await extractor(SimpleNamespace(final_output=payload))
    draws = [tuple(event["sampled"]) for event in writer.events if event["type"] == "audit"]
    assert len(draws) == 12 and all(len(draw) == 2 for draw in draws)
    assert len(set(draws)) > 1
    # Still reproducible: the same task, seed and return index give the same draw.
    again = _Events()
    replay = make_extractor(
        agent=None, hooks=None, writer=again, ledger=Ledger(), caller=None, task="t", seed=1, arm="evidence_random",
        checks=("L2",), sampler="random", rate=0.5, require_evidence=True, injector=None, state=DelegateState(),
        max_turns=5,
    )
    for _ in range(12):
        await replay(SimpleNamespace(final_output=payload))
    assert [tuple(event["sampled"]) for event in again.events if event["type"] == "audit"] == draws
