"""Wire L0–L3, the sampler, the replayer, and the judge at one return."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from evcarry.ledger.ledger import Ledger
from evcarry.protocol.checks import CheckContext, CheckResult, run_l0_l1
from evcarry.protocol.judge import JudgeVerdict, LexicalJudge, run_judge
from evcarry.protocol.replayer import ReplayResult, ToolCaller, render_tool_result, replay_claim
from evcarry.protocol.sampler import RiskContext, sample_random, sample_risk
from evcarry.protocol.schema import Claim, SubagentReturn, ToolCallEvidence


@dataclass
class AuditReport:
    checks: list[CheckResult] = field(default_factory=list)
    sampled: list[str] = field(default_factory=list)
    replays: list[ReplayResult] = field(default_factory=list)
    verdicts: list[tuple[str, JudgeVerdict]] = field(default_factory=list)
    failures: list[tuple[str, str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "checks": [item.as_dict() for item in self.checks],
            "sampled": list(self.sampled),
            "replays": [item.as_dict() for item in self.replays],
            "verdicts": [{"claim_id": claim_id, **verdict.as_dict()} for claim_id, verdict in self.verdicts],
            "failures": [{"claim_id": claim_id, "reason": reason} for claim_id, reason in self.failures],
        }


def risk_context_for(payload: SubagentReturn, ledger: Ledger, l0_l1_failures: list[tuple[str, str]]) -> RiskContext:
    deliverable = {claim.claim_id for claim in payload.claims if claim.kind in {"artifact", "action_done"}}
    used_by_write: set[str] = set()
    for claim in payload.claims:
        for evidence in claim.evidence:
            if not isinstance(evidence, ToolCallEvidence):
                continue
            entry = ledger.get(evidence.call_id)
            if entry is not None and not entry.read_only:
                used_by_write.add(claim.claim_id)
    return RiskContext(
        deliverable_ids=deliverable,
        used_by_write_ids=used_by_write,
        failed_l0_l1_ids={claim_id for claim_id, _reason in l0_l1_failures},
    )


def _evidence_blob(claim: Claim, ledger: Ledger, replay_text: str | None) -> str:
    parts: list[str] = []
    if replay_text:
        parts.append(replay_text)
    for evidence in claim.evidence:
        if isinstance(evidence, ToolCallEvidence):
            entry = ledger.get(evidence.call_id)
            if entry is not None:
                parts.append(entry.result_text)
    return "\n".join(parts)


async def _replay_blob(claim: Claim, ledger: Ledger, caller: ToolCaller) -> tuple[ReplayResult, str]:
    result = await replay_claim(claim, ledger, caller)
    texts: list[str] = []
    for evidence in claim.evidence:
        if not isinstance(evidence, ToolCallEvidence):
            continue
        entry = ledger.get(evidence.call_id)
        if entry is None:
            continue
        if entry.read_only:
            texts.append(entry.result_text)
        elif entry.readback_tool:
            readback = await caller.call_tool(entry.readback_tool, dict(entry.readback_arguments or {}))
            texts.append(render_tool_result(readback))
    return result, "\n".join(texts)


async def audit_return(
    payload: SubagentReturn,
    *,
    ledger: Ledger,
    caller: ToolCaller | None,
    checks: tuple[str, ...] | list[str],
    sampler: str | None,
    rate: float,
    seed: int,
    judge: Any = None,
    require_evidence: bool = True,
    file_root=None,
) -> AuditReport:
    """L0/L1 failures reject even when the claim was not sampled.

    L2 and L3 run only on the sampled claim ids. A rate of 0 samples nothing.
    ``judge`` is a ``LexicalJudge`` (the default) or an ``LlmJudge``.
    """
    enabled = set(checks)
    report = AuditReport()
    support_judge = judge or LexicalJudge()
    ctx = CheckContext(ledger=ledger, file_root=file_root)
    l0_failures: list[tuple[str, str]] = []
    if "L0" in enabled or "L1" in enabled:
        for claim in payload.claims:
            if not claim.evidence and not require_evidence:
                continue
            for result in run_l0_l1(claim, ctx):
                if result.level == "L0" and "L0" not in enabled:
                    continue
                if result.level == "L1" and "L1" not in enabled:
                    continue
                report.checks.append(result)
                if not result.ok:
                    failure = (result.claim_id, f"{result.level}: {result.reason}")
                    report.failures.append(failure)
                    if result.level in {"L0", "L1"}:
                        l0_failures.append(failure)
    want_sample = "L2" in enabled or "L3" in enabled
    if want_sample and sampler and rate > 0:
        context = risk_context_for(payload, ledger, l0_failures)
        if sampler == "random":
            report.sampled = sample_random(payload.claims, rate=rate, seed=seed)
        else:
            report.sampled = sample_risk(payload.claims, rate=rate, ctx=context)
    sampled = set(report.sampled)
    for claim in payload.claims:
        if claim.claim_id not in sampled:
            continue
        replay_text = None
        if "L2" in enabled and caller is not None:
            replay, replay_text = await _replay_blob(claim, ledger, caller)
            report.replays.append(replay)
            if not replay.ok:
                report.failures.append((claim.claim_id, f"L2: {replay.reason}"))
        if "L3" in enabled:
            blob = _evidence_blob(claim, ledger, replay_text)
            verdict = await run_judge(support_judge, claim, blob)
            report.verdicts.append((claim.claim_id, verdict))
            if not verdict.supports:
                report.failures.append((claim.claim_id, f"L3: {verdict.reason}"))
    return report
