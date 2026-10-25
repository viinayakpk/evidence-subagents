"""Deterministic L0 existence and L1 content checks. No model tokens.

L0: every pointer resolves to a ledger entry, file or row.

L1 (changed 2026-10-25, before the headline; docs/PILOT_DIAGNOSIS.md P1-1, P1-5, P1-6):

- Every quoted span must be in its own pointer's logged text. Harness notes the model
  saw but the ledger does not store (the ``[call_id: ...]`` header, the
  ``...[truncated: ...]`` note, a ``view_overlong_output`` page header) are removed from
  the span first, and runs of whitespace are collapsed in both.
- The claim's value must be in at least one cited pointer's logged text, not in every
  one: a claim citing a ``send_email`` result and the read-back that shows the number
  failed on the send result in the pilot.
- One value rule for L1 and L2 (``value_supported``): a number matches as a number
  (``0.3`` matches ``0.30``; ``$1,234`` and ``21.20%`` are numbers); a one-token string (an
  ID, path, e-mail address, date) matches case-insensitively; a list matches item by item.
  Multi-word strings, dicts and booleans are not decided by a string match and are left
  to L3, as is a ``derived`` value. Before, L1 checked numbers only and L2 demanded every
  value verbatim.
- A write's logged text is its result plus its arguments (what was written), except for
  tools that run code (``python_execute``, the terminal server), whose arguments are code,
  not content: a value printed nowhere but in the code is not evidence.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path

from evcarry.ledger.ledger import Ledger, LedgerEntry
from evcarry.protocol.schema import Claim, DbEvidence, FileEvidence, SearchEvidence, ToolCallEvidence

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
# Numbers in tool output. Matches are maximal digit runs, so "14" is not found inside
# "140", "2014" or "0.145". The grouped form also reads "1,234.5" as 1234.5.
_TEXT_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_GROUPED_NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?")
_GROUPED_VALUE = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?")
_WHITESPACE = re.compile(r"\s+")
_UNICODE_ESCAPE = re.compile(r"\\u([0-9a-fA-F]{4})")

# Text the harness adds to what the model sees (harness.tools.display_text, overlong_view).
# The ledger holds the server's text without it, so a span that copies it cannot match.
_HARNESS_NOTES = (
    re.compile(r"\[call_id: [^\]\n]*\]"),
    re.compile(r"\.{0,3}\[truncated: showing [^\]]*(?:\]|$)"),
    re.compile(r"^\S+ page \d+ of \d+ \(chars \d+-\d+ of \d+\)$", re.MULTILINE),
)

# Tools whose arguments are code to run, not content written: their arguments are not evidence.
_CODE_TOOLS = ("python_execute",)
_CODE_TOOL_PREFIXES = ("terminal-",)
# Writes that change no state the agent reads (the gym's claim_done ends the run).
NO_EFFECT_WRITES = ("claim_done",)


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


def _decimal(value: object) -> Decimal | None:
    try:
        return Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return None


def as_number(value: object) -> Decimal | None:
    """The number a claim value stands for, or None. ``"$1,234.50"`` and ``"21.20%"`` are numbers."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return _decimal(canonical_number(value))
    if not isinstance(value, str):
        return None
    text = value.strip()
    text = re.sub(r"^[$€£]\s*", "", text)
    text = re.sub(r"\s*%$", "", text)
    if _GROUPED_VALUE.fullmatch(text):
        text = text.replace(",", "")
    if _NUMBER.fullmatch(text):
        return _decimal(canonical_number(text))
    return None


def _text_numbers(text: str) -> set[Decimal]:
    found: set[Decimal] = set()
    for pattern in (_TEXT_NUMBER, _GROUPED_NUMBER):
        for match in pattern.finditer(text):
            number = _decimal(match.group(0).replace(",", ""))
            if number is None:
                continue
            start = match.start()
            # A minus sign counts only when it is not a separator ("2026-10-11", "a-3").
            negative = start > 0 and text[start - 1] == "-" and (start < 2 or not text[start - 2].isalnum())
            found.add(-number if negative else number)
    return found


def number_in_text(value: object, text: str) -> bool:
    """A numeric claim value appears in ``text`` as a whole number, not as a substring of a longer one.

    ``0.3`` matches ``0.30`` and ``14`` matches ``14.0``. A value written with fewer
    decimals than the text matches the text's number rounded to those decimals
    (``3.14`` matches ``3.14159``); an integer value does not match ``14.7``.
    """
    target = as_number(value)
    if target is None:
        return canonical_number(value) in text
    exponent = target.as_tuple().exponent
    places = -exponent if isinstance(exponent, int) and exponent < 0 else 0
    for number in _text_numbers(text):
        if number == target:
            return True
        if places:
            try:
                if number.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP) == target:
                    return True
            except InvalidOperation:  # a number too long to round at that precision
                continue
    return False


def normalize_text(text: str) -> str:
    """Runs of whitespace (spaces, tabs, newlines) collapsed to one space; ends stripped."""
    return _WHITESPACE.sub(" ", text).strip()


def strip_harness_notes(text: str) -> str:
    for pattern in _HARNESS_NOTES:
        text = pattern.sub(" ", text)
    return text


def decode_escapes(text: str) -> str:
    """``\\uXXXX`` escapes as the characters they stand for (a JSON-escaped output or span reads the same)."""
    return _UNICODE_ESCAPE.sub(lambda match: chr(int(match.group(1), 16)), text)


def clean_span(span: str) -> str:
    """A quoted span as the check compares it: harness notes removed, ``\\uXXXX`` escapes decoded,
    one pair of surrounding quotes dropped, whitespace collapsed."""
    text = normalize_text(decode_escapes(strip_harness_notes(span)))
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'`":
        text = text[1:-1].strip()
    return text


def span_in_text(span: str, text: str) -> bool:
    """Whether a quoted span is in a logged text, both cleaned the same way (escapes decoded, whitespace collapsed).

    A span that is empty once harness notes are removed quotes nothing and passes.
    """
    wanted = clean_span(span)
    if not wanted:
        return True
    haystack = normalize_text(decode_escapes(text))
    if wanted in haystack:
        return True
    # A Python repr in the result doubles each backslash (``['\\\\Sent']``) and the quoted span often
    # has one (dev run 2026-10-25, sf-sales-trip evidence_risk return 9): compare with runs of backslashes as one.
    return "\\" in haystack + wanted and _BACKSLASHES.sub(r"\\", wanted) in _BACKSLASHES.sub(r"\\", haystack)


_BACKSLASHES = re.compile(r"\\{2,}")


def _edge(char: str, *, before: bool) -> str:
    """A digit must not run on into a digit, a letter not into a letter (``2026-10-24`` is found in ``2026-10-24T20:40``)."""
    if char.isdigit():
        return r"(?<!\d)" if before else r"(?!\d)"
    if char.isalpha():
        return r"(?<![^\W\d_])" if before else r"(?![^\W\d_])"
    return ""


def text_contains(text: str, wanted: str) -> bool:
    """Case-insensitive containment after whitespace collapsing (read-back effect fields, string values).

    An edge of ``wanted`` must not run on into more of the same kind of character in
    ``text``: ``US`` is not found in ``status``, ``ID 5`` is not found in ``ID 55``.
    """
    target = normalize_text(wanted).casefold()
    if not target:
        return True
    pattern = _edge(target[0], before=True) + re.escape(target) + _edge(target[-1], before=False)
    return re.search(pattern, normalize_text(text).casefold()) is not None


def checkable_items(value: object) -> list[tuple[str, object, str]]:
    """The parts of a claim value a string match can decide: ``(kind, target, as written)``.

    Numbers (``as_number``) and one-token strings are checked; a list contributes its items.
    Multi-word strings, dicts, booleans and None contribute nothing (L3 judges them).
    """
    if value is None or isinstance(value, (bool, dict)):
        return []
    if isinstance(value, (list, tuple)):
        return [item for element in value for item in checkable_items(element)]
    number = as_number(value)
    if number is not None:
        return [("number", value, str(value))]
    if isinstance(value, str):
        token = value.strip()
        if token and not _WHITESPACE.search(token) and len(token) <= 200:
            return [("token", token, token)]
    return []


def value_supported(value: object, texts: list[str]) -> tuple[bool, str]:
    """The one value rule of L1 and L2: every checkable part of ``value`` is in at least one of ``texts``."""
    items = checkable_items(value)
    if not items:
        return True, "value has nothing a string match can decide; left to L3"
    missing = []
    for kind, target, written in items:
        if kind == "number":
            found = any(number_in_text(target, text) for text in texts)
        else:
            found = any(text_contains(text, str(target)) for text in texts)
        if not found:
            missing.append(written)
    if missing:
        shown = ", ".join(missing[:5]) + (f" (+{len(missing) - 5} more)" if len(missing) > 5 else "")
        return False, f"value {shown} is not in any cited result"
    return True, "value is in a cited result"


def value_in_text(value: object, text: str) -> bool:
    """``value_supported`` against one text (used by the lexical judge)."""
    if isinstance(value, dict):
        return all(value_in_text(item, text) for item in value.values())
    if isinstance(value, str) and not checkable_items(value):
        return text_contains(text, value)
    return value_supported(value, [text])[0]


def is_code_tool(tool_name: str) -> bool:
    return tool_name in _CODE_TOOLS or tool_name.startswith(_CODE_TOOL_PREFIXES)


def logged_text(entry: LedgerEntry) -> str:
    """What a pointer to ``entry`` can be checked against: its result, plus a write's arguments."""
    if entry.read_only or is_code_tool(entry.tool_name) or not entry.arguments:
        return entry.result_text
    arguments = json.dumps(entry.arguments, ensure_ascii=False, sort_keys=True, default=str)
    return f"{entry.result_text}\n{arguments}"


def run_l0_l1(claim: Claim, ctx: CheckContext) -> list[CheckResult]:
    """L0 and the span part of L1 per evidence pointer, then the value part of L1 once per claim."""
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
    texts: list[str] = []
    for index, evidence in enumerate(claim.evidence):
        pointer = _resolve(evidence, ctx)
        if pointer.failure is not None:
            results.append(CheckResult("L0", claim.claim_id, False, pointer.failure, index))
            continue
        results.append(CheckResult("L0", claim.claim_id, True, "pointer resolved", index))
        text = pointer.text or ""
        texts.append(text)
        span = getattr(evidence, "span", None)
        if isinstance(span, str) and not span_in_text(span, text):
            results.append(CheckResult("L1", claim.claim_id, False, "quoted span is not in the logged result", index))
        else:
            results.append(CheckResult("L1", claim.claim_id, True, "span matches the logged result", index))
    if not texts:
        return results  # no pointer resolved: L0 already rejects
    # A derived value (sum, rate, ranking) is computed from the evidence and need not
    # appear in it verbatim; L3 recomputes it. Its spans are still checked above.
    if claim.kind == "derived":
        results.append(CheckResult("L1", claim.claim_id, True, "derived value left to L3"))
    else:
        ok, reason = value_supported(claim.value, texts)
        results.append(CheckResult("L1", claim.claim_id, ok, reason))
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
        return _Resolved(text=logged_text(entry), entry=entry)
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
