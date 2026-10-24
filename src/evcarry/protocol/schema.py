"""Pydantic claim and evidence schema for sub-agent returns.

The JSON Schema in the experiment plan is the contract. Invalid payloads get
one repair attempt; a second failure is a failed return.
"""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

EvidenceKind = Literal["tool_call", "file", "search", "db"]
ClaimKind = Literal["fact", "derived", "action_done", "artifact"]
Confidence = Literal["low", "medium", "high"]
ReturnStatus = Literal["complete", "partial", "failed"]


class ToolCallEvidence(BaseModel):
    type: Literal["tool_call"]
    call_id: str
    span: str | None = None


class FileEvidence(BaseModel):
    type: Literal["file"]
    path: str
    sha256: str
    lines: str | None = None


class SearchEvidence(BaseModel):
    type: Literal["search"]
    query: str
    rank: int | None = None
    doc_id: str | None = None


class DbEvidence(BaseModel):
    type: Literal["db"]
    sql: str
    row_keys: list[str] = Field(default_factory=list)


Evidence = Annotated[
    ToolCallEvidence | FileEvidence | SearchEvidence | DbEvidence,
    Field(discriminator="type"),
]


class Claim(BaseModel):
    claim_id: str
    text: str
    kind: ClaimKind
    value: Any = None
    evidence: list[Evidence]
    depends_on: list[str] = Field(default_factory=list)
    confidence: Confidence | None = None


class SubagentReturn(BaseModel):
    subagent_id: str
    status: ReturnStatus
    claims: list[Claim]
    open_issues: list[str] = Field(default_factory=list)


class FailedReturn(Exception):
    """A sub-agent return was invalid after the single repair attempt."""

    def __init__(self, message: str, *, repairs: int) -> None:
        super().__init__(message)
        self.repairs = repairs


_RETURN_ADAPTER = TypeAdapter(SubagentReturn)


def export_json_schema() -> dict[str, Any]:
    """JSON Schema generated from the Pydantic model."""
    return SubagentReturn.model_json_schema()


_FENCED = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)


def _validate(text: str) -> SubagentReturn:
    """Validate the text, else the JSON object inside a code fence or surrounding prose.

    Models asked for JSON in the prompt (no response_format) often wrap it in
    ```json fences; that is a formatting slip, not a schema failure, and costs no repair.
    """
    try:
        return _RETURN_ADAPTER.validate_json(text)
    except ValidationError as first:
        candidates = [match.group(1) for match in _FENCED.finditer(text)]
        start, end = text.find("{"), text.rfind("}")
        if 0 <= start < end:
            candidates.append(text[start : end + 1])
        for candidate in candidates:
            try:
                return _RETURN_ADAPTER.validate_json(candidate)
            except ValidationError:
                continue
        raise first


def parse_subagent_return(
    payload: str | bytes,
    repair: Any | None = None,
) -> tuple[SubagentReturn, int]:
    """Parse a return, allowing exactly one repair callback.

    ``repair`` is called as ``repair(raw_text, error_text) -> str`` at most once.
    """
    text = payload.decode() if isinstance(payload, bytes) else payload
    try:
        return _validate(text), 0
    except ValidationError as first:
        if repair is None:
            raise FailedReturn(str(first), repairs=0) from first
        repaired = repair(text, str(first))
        if not isinstance(repaired, str):
            raise FailedReturn("Repair callback did not return a string.", repairs=1)
        try:
            return _validate(repaired), 1
        except ValidationError as second:
            raise FailedReturn(str(second), repairs=1) from second
