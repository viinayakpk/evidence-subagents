"""Five corruption types, each in a naive and a pointer-consistent form.

Naive corruptions change the claim and leave the pointer alone, so L0/L1 or
L2 can see the mismatch. Pointer-consistent corruptions also edit the pointer
so it still names a real call or a real span, and a later check has to notice
that the call does not support the claim.
"""

from __future__ import annotations

import re
from typing import Any

from evcarry.ledger.ledger import Ledger
from evcarry.protocol.schema import Claim, ToolCallEvidence

_NUMBER = re.compile(r"\d+(?:\.\d+)?")


class CorruptionError(Exception):
    """The claim did not have the structure this corruption needs."""


def apply_corruption(claim: Claim, kind: str, variant: str, ledger: Ledger) -> Claim:
    if variant not in {"naive", "pointer_consistent"}:
        raise CorruptionError(f"unknown variant {variant}")
    updated = claim.model_copy(deep=True)
    if kind == "numeric_perturbation":
        return _numeric(updated, variant)
    if kind == "fabricated_entity":
        return _fabricated(updated, variant)
    if kind == "omission":
        return _omission(updated, variant)
    if kind == "false_action_done":
        return _false_action(updated, variant, ledger)
    if kind == "mismatched_pointer":
        return _mismatched(updated, variant, ledger)
    raise CorruptionError(f"unknown corruption {kind}")


def _numeric(claim: Claim, variant: str) -> Claim:
    match = _NUMBER.search(claim.text)
    if match is None:
        raise CorruptionError("numeric perturbation needs a number in the claim text")
    original = match.group(0)
    number = float(original) if "." in original else int(original)
    shifted = number - 3 if number > 3 else number + 3
    rendered = str(int(shifted)) if float(shifted).is_integer() else str(shifted)
    claim.text = claim.text[: match.start()] + rendered + claim.text[match.end() :]
    claim.value = int(rendered) if rendered.isdigit() else rendered
    if variant == "pointer_consistent":
        for evidence in claim.evidence:
            if isinstance(evidence, ToolCallEvidence):
                evidence.span = rendered
    return claim


def _fabricated(claim: Claim, variant: str) -> Claim:
    if "paper-A" in claim.text:
        claim.text = claim.text.replace("paper-A", "student-Z")
    else:
        claim.text = claim.text.rstrip(".") + " student-Z."
    claim.value = "student-Z"
    if variant == "pointer_consistent":
        for evidence in claim.evidence:
            if isinstance(evidence, ToolCallEvidence):
                evidence.call_id = "missing-call"
                evidence.span = "student-Z"
    return claim


def _omission(claim: Claim, variant: str) -> Claim:
    if isinstance(claim.value, list) and len(claim.value) >= 2:
        dropped = claim.value[-1]
        claim.value = list(claim.value[:-1])
    else:
        dropped = "E"
        claim.value = ["A", "B", "C", "D"]
    claim.text = "Required papers: A, B, C, D"
    if variant == "pointer_consistent":
        for evidence in claim.evidence:
            if isinstance(evidence, ToolCallEvidence):
                evidence.span = "A, B, C, D"
    else:
        # Leave a span that still quotes the original full list when one exists.
        for evidence in claim.evidence:
            if isinstance(evidence, ToolCallEvidence) and evidence.span and str(dropped) not in (evidence.span or ""):
                evidence.span = f"{evidence.span}, {dropped}"
    return claim


def _false_action(claim: Claim, variant: str, ledger: Ledger) -> Claim:
    claim.kind = "action_done"
    claim.text = "Calendar event retro was created."
    claim.value = "retro"
    writes = [entry for entry in ledger.entries.values() if not entry.read_only]
    if variant == "pointer_consistent":
        if not writes:
            raise CorruptionError("pointer-consistent false action_done needs a write in the ledger")
        target = writes[0]
        for evidence in claim.evidence:
            if isinstance(evidence, ToolCallEvidence):
                evidence.call_id = target.call_id
                evidence.span = target.result_text[: max(1, min(len(target.result_text), 16))]
    return claim


def _mismatched(claim: Claim, variant: str, ledger: Ledger) -> Claim:
    if variant == "naive":
        claim.text = "There are 99 sections."
        claim.value = 99
        return claim
    others = [
        entry
        for entry in ledger.entries.values()
        if entry.read_only and "14 sections" in entry.result_text and entry.call_id != _first_call_id(claim)
    ]
    if not others:
        raise CorruptionError("pointer-consistent mismatch needs a second real call containing 14")
    target = sorted(others, key=lambda entry: entry.call_id)[0]
    claim.text = "student-Z scored 14"
    claim.value = "14"
    for evidence in claim.evidence:
        if isinstance(evidence, ToolCallEvidence):
            evidence.call_id = target.call_id
            evidence.span = "14 sections"
    return claim


def _first_call_id(claim: Claim) -> str | None:
    for evidence in claim.evidence:
        if isinstance(evidence, ToolCallEvidence):
            return evidence.call_id
    return None


CORRUPTION_TYPES = (
    "numeric_perturbation",
    "fabricated_entity",
    "omission",
    "false_action_done",
    "mismatched_pointer",
)
VARIANTS = ("naive", "pointer_consistent")


def describe(kind: str, variant: str) -> dict[str, Any]:
    return {"kind": kind, "variant": variant}
