"""Deterministic L0 existence and L1 content checks. No model tokens."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from evcarry.ledger.ledger import Ledger, LedgerEntry
from evcarry.protocol.schema import Claim, DbEvidence, FileEvidence, SearchEvidence, ToolCallEvidence

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass
class CheckResult:
    level: str
    claim_id: str
    ok: bool
    reason: str
    evidence_index: int | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "level": self.level,
            "claim_id": self.claim_id,
            "ok": self.ok,
            "reason": self.reason,
            "evidence_index": self.evidence_index,
        }


@dataclass
class CheckContext:
    ledger: Ledger
    file_root: Path | None = None
    files: dict[str, str] = field(default_factory=dict)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_numeric_value(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str) and _NUMBER.fullmatch(value.strip()):
        return True
    return False


def canonical_number(value: object) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, str) and _NUMBER.fullmatch(value.strip()):
        number = float(value) if "." in value else int(value)
        if isinstance(number, float) and number.is_integer():
            return str(int(number))
        return str(number)
    return str(value)


def value_in_text(value: object, text: str) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple)):
        return all(value_in_text(item, text) for item in value)
    if is_numeric_value(value):
        return canonical_number(value) in text
    return str(value) in text


def run_l0_l1(claim: Claim, ctx: CheckContext) -> list[CheckResult]:
    """Run L0 and L1 on every evidence pointer of one claim."""
    if not claim.evidence:
        return [
            CheckResult(
                level="L0",
                claim_id=claim.claim_id,
                ok=False,
                reason="claim has no evidence pointer",
            )
        ]
    results: list[CheckResult] = []
    for index, evidence in enumerate(claim.evidence):
        pointer = _resolve(evidence, ctx)
        if pointer.failure is not None:
            results.append(
                CheckResult("L0", claim.claim_id, False, pointer.failure, index)
            )
            continue
        results.append(CheckResult("L0", claim.claim_id, True, "pointer resolved", index))
        content_ok, content_reason = _l1(claim, evidence, pointer.text or "")
        results.append(CheckResult("L1", claim.claim_id, content_ok, content_reason, index))
    return results


@dataclass
class _Resolved:
    text: str | None = None
    failure: str | None = None
    entry: LedgerEntry | None = None


def _resolve(evidence: object, ctx: CheckContext) -> _Resolved:
    if isinstance(evidence, ToolCallEvidence):
        entry = ctx.ledger.get(evidence.call_id)
        if entry is None:
            return _Resolved(failure=f"call_id {evidence.call_id} is not in the ledger")
        return _Resolved(text=entry.result_text, entry=entry)
    if isinstance(evidence, FileEvidence):
        text = _file_text(evidence.path, ctx)
        if text is None:
            return _Resolved(failure=f"file {evidence.path} is not available")
        digest = sha256_text(text)
        if digest != evidence.sha256:
            return _Resolved(failure=f"sha256 mismatch for {evidence.path}")
        if evidence.lines:
            ranged = _line_span(text, evidence.lines)
            if ranged is None:
                return _Resolved(failure=f"line span {evidence.lines} is outside {evidence.path}")
            return _Resolved(text=ranged)
        return _Resolved(text=text)
    if isinstance(evidence, SearchEvidence):
        entry = _find_search(ctx.ledger, evidence.query)
        if entry is None:
            return _Resolved(failure=f"no ledger search for query {evidence.query!r}")
        if evidence.doc_id and evidence.doc_id not in entry.result_text:
            return _Resolved(failure=f"doc_id {evidence.doc_id} is not in the search result")
        if evidence.rank is not None and evidence.rank < 1:
            return _Resolved(failure="search rank must be >= 1")
        return _Resolved(text=entry.result_text, entry=entry)
    if isinstance(evidence, DbEvidence):
        entry = _find_sql(ctx.ledger, evidence.sql)
        if entry is None:
            return _Resolved(failure="no ledger query matches the cited sql")
        missing = [key for key in evidence.row_keys if key not in entry.result_text]
        if missing:
            return _Resolved(failure=f"row keys missing from result: {missing}")
        return _Resolved(text=entry.result_text, entry=entry)
    return _Resolved(failure=f"unsupported evidence type {type(evidence).__name__}")


def _l1(claim: Claim, evidence: object, text: str) -> tuple[bool, str]:
    span = getattr(evidence, "span", None)
    if isinstance(span, str) and span not in text:
        return False, "quoted span is not in the logged result"
    # A derived value (sum, rate, ranking) is computed from the evidence and need
    # not appear in it verbatim; L3 recomputes it. Its spans are still checked.
    if claim.kind == "derived":
        return True, "span matches the logged result; derived value left to L3"
    if is_numeric_value(claim.value) and canonical_number(claim.value) not in text:
        return False, f"numeric value {canonical_number(claim.value)} is not in the logged result"
    return True, "span and numeric value match the logged result"


def _file_text(path: str, ctx: CheckContext) -> str | None:
    if path in ctx.files:
        return ctx.files[path]
    if ctx.file_root is None:
        return None
    root = ctx.file_root.resolve()
    candidate = (root / path).resolve()
    if root != candidate and root not in candidate.parents:
        return None
    if not candidate.is_file():
        return None
    return candidate.read_text(encoding="utf-8")


def _line_span(text: str, spec: str) -> str | None:
    """``spec`` is ``start-end`` with 1-based inclusive line numbers."""
    match = re.fullmatch(r"(\d+)-(\d+)", spec.strip())
    if match is None:
        return None
    start, end = int(match.group(1)), int(match.group(2))
    lines = text.splitlines()
    if start < 1 or end < start or end > len(lines):
        return None
    return "\n".join(lines[start - 1 : end])


def _find_search(ledger: Ledger, query: str) -> LedgerEntry | None:
    for entry in ledger.entries.values():
        if entry.tool_name == "search_records" and entry.arguments.get("query") == query:
            return entry
    return None


def _find_sql(ledger: Ledger, sql: str) -> LedgerEntry | None:
    for entry in ledger.entries.values():
        if entry.tool_name == "query_rows" and entry.arguments.get("sql") == sql:
            return entry
    return None
