"""Wire L0–L3, the sampler, the replayer, and the judge at one return."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from evcarry.ledger.ledger import Ledger
from evcarry.protocol.checks import CheckContext, CheckResult, run_l0_l1
from evcarry.protocol.judge import JudgeVerdict, LexicalJudge, run_judge
from evcarry.protocol.replayer import ReplayResult, ToolCaller, replay_claim
from evcarry.protocol.sampler import RiskContext, sample_random, sample_risk
from evcarry.protocol.schema import Claim, SubagentReturn, ToolCallEvidence


_ARGUMENT_CHARS = 2000


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


def _evidence_blob(claim: Claim, ledger: Ledger, readbacks: dict[str, str] | None) -> str:
    """What the L3 judge reads: each cited call (tool and arguments), its logged result, and its read-back.

    The arguments matter: a ``python_execute`` call that only prints a number is
    visible as such. Read-only replays are not repeated here; L2 already checked
    they match the logged result.
    """
    parts: list[str] = []
    seen: set[str] = set()
    for evidence in claim.evidence:
        if not isinstance(evidence, ToolCallEvidence) or evidence.call_id in seen:
            continue
        seen.add(evidence.call_id)
        entry = ledger.get(evidence.call_id)
        if entry is None:
            continue
        arguments = json.dumps(entry.arguments, ensure_ascii=False, sort_keys=True, default=str)
        if len(arguments) > _ARGUMENT_CHARS:
            arguments = arguments[:_ARGUMENT_CHARS] + "...[cut]"
        parts.append(f"--- call {entry.call_id}: {entry.tool_name} {arguments}\n{entry.result_text}")
        readback = (readbacks or {}).get(entry.call_id)
        if readback is not None:
            parts.append(f"--- read-back of {entry.call_id} now ({entry.readback_tool}):\n{readback}")
    return "\n".join(parts)


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
        readbacks = None
        if "L2" in enabled and caller is not None:
            replay = await replay_claim(claim, ledger, caller)
            readbacks = replay.readbacks
            report.replays.append(replay)
            if not replay.ok:
                report.failures.append((claim.claim_id, f"L2: {replay.reason}"))
        if "L3" in enabled:
            blob = _evidence_blob(claim, ledger, readbacks)
            verdict = await run_judge(support_judge, claim, blob)
            report.verdicts.append((claim.claim_id, verdict))
            if not verdict.supports:
                report.failures.append((claim.claim_id, f"L3: {verdict.reason}"))
    return report
