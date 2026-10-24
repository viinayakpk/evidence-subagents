"""Wire L0–L3, the sampler, the replayer, and the judge at one return.

Changed 2026-10-25, before the headline (docs/PILOT_DIAGNOSIS.md P1-7, P1-11):

- L3 evidence is built per cited call: the call (tool and arguments), then its logged
  result in full when it fits, else the head of the result plus a window around every
  quoted span and value of the claim, with each gap marked
  ``[... N characters not shown ...]`` (``excerpt``). The judge prompt says text in a gap
  is unknown, not absent. Before, the joined evidence was cut blindly in the middle at
  12,000 characters, and the judge read the cut as a contradiction (19 pilot rejects).
- A sampled claim that already failed L0, L1 or L2 is not sent to the judge: it is
  rejected either way (40% of the pilot's judge calls were on rejected returns).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from evcarry.ledger.ledger import Ledger
from evcarry.protocol.checks import CheckContext, CheckResult, checkable_items, clean_span, run_l0_l1
from evcarry.protocol.judge import JudgeVerdict, LexicalJudge, run_judge
from evcarry.protocol.replayer import ReplayResult, ToolCaller, replay_claim
from evcarry.protocol.sampler import RiskContext, sample_random, sample_risk
from evcarry.protocol.schema import Claim, SubagentReturn, ToolCallEvidence

_ARGUMENT_CHARS = 2000
# What the judge reads per claim, over all its cited calls and read-backs.
EVIDENCE_CHARS = 16000
# Each cited result or read-back gets at least this much of that budget.
_MIN_PART_CHARS = 2000
_HEAD_CHARS = 1500
_WINDOW_CHARS = 700
_HITS_PER_ANCHOR = 3


@dataclass
class AuditReport:
    checks: list[CheckResult] = field(default_factory=list)
    sampled: list[str] = field(default_factory=list)
    replays: list[ReplayResult] = field(default_factory=list)
    verdicts: list[tuple[str, JudgeVerdict]] = field(default_factory=list)
    failures: list[tuple[str, str]] = field(default_factory=list)
    # Sampled claims not sent to the judge because L0, L1 or L2 had already failed them.
    skipped_l3: list[str] = field(default_factory=list)

    def failed_claim_ids(self) -> set[str]:
        return {claim_id for claim_id, _reason in self.failures}

    def as_dict(self) -> dict[str, object]:
        return {
            "checks": [item.as_dict() for item in self.checks],
            "sampled": list(self.sampled),
            "replays": [item.as_dict() for item in self.replays],
            "verdicts": [{"claim_id": claim_id, **verdict.as_dict()} for claim_id, verdict in self.verdicts],
            "failures": [{"claim_id": claim_id, "reason": reason} for claim_id, reason in self.failures],
            "skipped_l3": list(self.skipped_l3),
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


def _occurrences(text: str, anchor: str) -> list[tuple[int, int]]:
    """Where ``anchor`` occurs in ``text``: exact matches, else matches with any whitespace between its words."""
    found: list[tuple[int, int]] = []
    start = text.find(anchor)
    while start >= 0 and len(found) < _HITS_PER_ANCHOR:
        found.append((start, start + len(anchor)))
        start = text.find(anchor, start + 1)
    if found:
        return found
    words = anchor.split()
    if not words:
        return []
    pattern = re.compile(r"\s+".join(re.escape(word) for word in words), re.IGNORECASE)
    return [(match.start(), match.end()) for match in pattern.finditer(text)][:_HITS_PER_ANCHOR]


def _merge(windows: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(windows):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def excerpt(text: str, anchors: list[str], budget: int) -> str:
    """``text`` if it fits ``budget``; else its head plus a window around each anchor, gaps marked.

    Windows are taken in priority order (the head, then each anchor's matches in turn) while
    they fit the budget. Every left-out stretch is marked ``[... N characters not shown ...]``.
    """
    if len(text) <= budget:
        return text
    head = (0, min(len(text), _HEAD_CHARS, budget))
    candidates = [head]
    for anchor in anchors:
        for start, end in _occurrences(text, anchor):
            candidates.append((max(0, start - _WINDOW_CHARS), min(len(text), end + _WINDOW_CHARS)))
    chosen: list[tuple[int, int]] = []
    for window in candidates:
        trial = _merge([*chosen, window])
        if sum(end - start for start, end in trial) <= budget:
            chosen = trial
    pieces: list[str] = []
    position = 0
    for start, end in chosen:
        if start > position:
            pieces.append(f"[... {start - position} characters not shown ...]")
        pieces.append(text[start:end])
        position = end
    if position < len(text):
        pieces.append(f"[... {len(text) - position} characters not shown ...]")
    return "\n".join(pieces)


def _anchors(claim: Claim, call_id: str) -> list[str]:
    """The claim's quoted spans for this call, then its checkable values: where the judge must be able to look."""
    spans = [
        clean_span(evidence.span)
        for evidence in claim.evidence
        if isinstance(evidence, ToolCallEvidence) and evidence.call_id == call_id and evidence.span
    ]
    values = [written for _kind, _target, written in checkable_items(claim.value)]
    return [anchor for anchor in dict.fromkeys([*spans, *values]) if anchor]


def evidence_blob(
    claim: Claim, ledger: Ledger, readbacks: dict[str, str] | None, max_chars: int = EVIDENCE_CHARS
) -> str:
    """What the L3 judge reads: each cited call (tool and arguments), its logged result, and its read-back.

    The arguments matter: a ``python_execute`` call that only prints a number is
    visible as such. Read-only replays are not repeated here; L2 already checked
    they show the cited span. Long results are excerpted around the claim's spans and values.
    """
    entries = []
    seen: set[str] = set()
    for evidence in claim.evidence:
        if not isinstance(evidence, ToolCallEvidence) or evidence.call_id in seen:
            continue
        seen.add(evidence.call_id)
        entry = ledger.get(evidence.call_id)
        if entry is not None:
            entries.append(entry)
    parts_count = len(entries) + sum(1 for entry in entries if (readbacks or {}).get(entry.call_id) is not None)
    budget = max(_MIN_PART_CHARS, max_chars // max(1, parts_count))
    parts: list[str] = []
    for entry in entries:
        anchors = _anchors(claim, entry.call_id)
        arguments = json.dumps(entry.arguments, ensure_ascii=False, sort_keys=True, default=str)
        if len(arguments) > _ARGUMENT_CHARS:
            arguments = arguments[:_ARGUMENT_CHARS] + f"[... {len(arguments) - _ARGUMENT_CHARS} characters not shown ...]"
        parts.append(f"--- call {entry.call_id}: {entry.tool_name} {arguments}\n{excerpt(entry.result_text, anchors, budget)}")
        readback = (readbacks or {}).get(entry.call_id)
        if readback is not None:
            parts.append(f"--- read-back of {entry.call_id} now ({entry.readback_tool}):\n{excerpt(readback, anchors, budget)}")
    return "\n".join(parts)


# The old private name; scripts/judge_validation.py and tests import it.
_evidence_blob = evidence_blob


async def audit_return(
    payload: SubagentReturn,
    *,
    ledger: Ledger,
    caller: ToolCaller | None,
    checks: tuple[str, ...] | list[str],
    sampler: str | None,
    rate: float,
    seed: int | str,
    judge: Any = None,
    require_evidence: bool = True,
    file_root=None,
) -> AuditReport:
    """L0/L1 failures reject even when the claim was not sampled.

    L2 and L3 run only on the sampled claim ids, and L3 only on a sampled claim that
    passed L0, L1 and L2. A rate of 0 samples nothing.
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
            if claim.claim_id in report.failed_claim_ids():
                report.skipped_l3.append(claim.claim_id)
                continue
            blob = evidence_blob(claim, ledger, readbacks)
            verdict = await run_judge(support_judge, claim, blob)
            report.verdicts.append((claim.claim_id, verdict))
            if not verdict.supports:
                report.failures.append((claim.claim_id, f"L3: {verdict.reason}"))
    return report
