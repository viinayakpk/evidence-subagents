"""L2 evidence replayer.

Read-only MCP calls are issued again through the MCP client and compared with
the ledger. Writes are never replayed: the replayer calls the tool's read-back
instead and checks the claim value against that state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from evcarry.ledger.ledger import Ledger, LedgerEntry
from evcarry.protocol.checks import value_in_text
from evcarry.protocol.schema import Claim, ToolCallEvidence


class ToolCaller(Protocol):
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any: ...


@dataclass
class ReplayResult:
    claim_id: str
    ok: bool
    reason: str
    replayed_tool: str | None = None
    write_replayed: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "claim_id": self.claim_id,
            "ok": self.ok,
            "reason": self.reason,
            "replayed_tool": self.replayed_tool,
            "write_replayed": self.write_replayed,
        }


def render_tool_result(result: Any) -> str:
    """Prefer the text blocks a model would read."""
    content = getattr(result, "content", None)
    if content:
        parts: list[str] = []
        for block in content:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)
        if parts:
            return "\n".join(parts)
    if isinstance(result, str):
        return result
    structured = getattr(result, "structured_content", None)
    if structured is not None:
        import json

        return json.dumps(structured, sort_keys=True)
    return str(result)


async def replay_claim(claim: Claim, ledger: Ledger, caller: ToolCaller) -> ReplayResult:
    pointers = [item for item in claim.evidence if isinstance(item, ToolCallEvidence)]
    if not pointers:
        return ReplayResult(claim.claim_id, True, "no tool-call pointer to replay")
    texts: list[str] = []
    for pointer in pointers:
        entry = ledger.get(pointer.call_id)
        if entry is None:
            return ReplayResult(claim.claim_id, False, f"call_id {pointer.call_id} is not in the ledger")
        if entry.read_only:
            result = await caller.call_tool(entry.tool_name, dict(entry.arguments))
            text = render_tool_result(result)
            if text.strip() != entry.result_text.strip():
                return ReplayResult(
                    claim.claim_id,
                    False,
                    f"replay of {entry.tool_name} drifted from the ledger",
                    replayed_tool=entry.tool_name,
                )
            texts.append(text)
        else:
            readback = _readback(entry)
            if readback is None:
                return ReplayResult(
                    claim.claim_id,
                    False,
                    f"write {entry.tool_name} has no read-back tool",
                    replayed_tool=None,
                    write_replayed=False,
                )
            tool_name, arguments = readback
            result = await caller.call_tool(tool_name, arguments)
            texts.append(render_tool_result(result))
    blob = "\n".join(texts)
    if claim.value is not None and not value_in_text(claim.value, blob):
        return ReplayResult(
            claim.claim_id,
            False,
            "claim value is not in the replayed or read-back text",
            write_replayed=False,
        )
    return ReplayResult(claim.claim_id, True, "replay matches", write_replayed=False)


def _readback(entry: LedgerEntry) -> tuple[str, dict[str, Any]] | None:
    if not entry.readback_tool:
        return None
    return entry.readback_tool, dict(entry.readback_arguments or {})
