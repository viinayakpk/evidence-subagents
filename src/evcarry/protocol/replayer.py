"""L2 evidence replayer.

Read-only MCP calls are issued again through the MCP client. Writes are never
replayed: the replayer calls the tool's read-back instead.

Rules (changed 2026-10-25, before the headline; docs/PILOT_DIAGNOSIS.md P1-2 to P1-5):

- A replayed read must still show the claim's quoted span (``checks.span_in_text``),
  not reproduce its whole text: a tool that prints a clock (``canvas_health_check``)
  or lists state the run itself changed later failed full-text equality in the pilot.
  Volatile patterns from the tool policy are masked in both texts first.
- A read whose span is gone from the replay is a failure unless a later write could have
  changed what it read: a write on the same server, a write by any server that edits the
  workspace's files when the read was of those files, or a code-running tool
  (``python_execute``, the terminal), which can change files and the database. Then
  the state it read may have changed legitimately: the pointer is ``stateful`` and its
  logged text is used.
- A write is checked by its read-back: the read-back must show the write's effect
  fields (recipient, subject, title: ``LedgerEntry.effect``), and the claim value is
  checked against the write's logged result and arguments plus the read-back, not
  the read-back alone (a read-back never contains "Email sent successfully").
- Pointers L2 cannot check (a write without a read-back tool, a nondeterministic read,
  ``python_execute``) are ``unverifiable``. Their logged text joins the value check when
  another pointer of the claim was replayed or read back; when none was, L2 defers the
  value to L1 (which already checked it) and L3.
- The value check is ``checks.value_supported``, the same rule as L1.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

from evcarry.ledger.ledger import Ledger, LedgerEntry
from evcarry.protocol.checks import NO_EFFECT_WRITES, is_code_tool, logged_text, span_in_text, text_contains, value_supported
from evcarry.protocol.schema import Claim, ToolCallEvidence

# Gym servers that read or write the agent workspace's files: a write by one changes what another reads.
_WORKSPACE_FILE_SERVERS = frozenset({"filesystem", "excel", "word", "pptx", "pdf-tools", "terminal"})


def _server(entry: LedgerEntry) -> str | None:
    """The entry's server; for ledgers written before 2026-10-25, the tool name's prefix."""
    if entry.server:
        return entry.server
    return entry.tool_name.split("-", 1)[0] if "-" in entry.tool_name else None


def _changed_later(ledger: Ledger, read: LedgerEntry) -> bool:
    """Whether a write recorded after ``read`` could have changed the state it read."""
    mine = _server(read)
    seen = False
    for key, entry in ledger.entries.items():
        if not seen:
            seen = key == read.call_id
            continue
        if entry.read_only or entry.tool_name in NO_EFFECT_WRITES:
            continue
        theirs = _server(entry)
        if is_code_tool(entry.tool_name) or mine is None or theirs is None or theirs == mine:
            return True
        if mine in _WORKSPACE_FILE_SERVERS and theirs in _WORKSPACE_FILE_SERVERS:
            return True
    return False


class ToolCaller(Protocol):
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any: ...


@dataclass
class ReplayResult:
    claim_id: str
    ok: bool
    reason: str
    replayed_tool: str | None = None
    write_replayed: bool = False
    unverifiable: list[str] | None = None
    # Read-back text per write call id, for the L3 evidence. Not written to the trace.
    readbacks: dict[str, str] | None = None
    # Reads whose replay no longer shows the span because the run wrote after them.
    stateful: list[str] | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "claim_id": self.claim_id,
            "ok": self.ok,
            "reason": self.reason,
            "replayed_tool": self.replayed_tool,
            "write_replayed": self.write_replayed,
            "unverifiable": list(self.unverifiable or []),
            "stateful": list(self.stateful or []),
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
    checked: list[str] = []  # texts L2 itself obtained (replays, read-backs) and the logged texts beside them
    deferred: list[str] = []  # logged texts of pointers L2 could not check
    unverifiable: list[str] = []
    stateful: list[str] = []
    readbacks: dict[str, str] = {}
    replayed_tool: str | None = None
    for pointer in pointers:
        entry = ledger.get(pointer.call_id)
        if entry is None:
            return ReplayResult(claim.claim_id, False, f"call_id {pointer.call_id} is not in the ledger")
        if entry.read_only and entry.nondeterministic:
            unverifiable.append(f"{entry.call_id}: {entry.tool_name} output is nondeterministic")
            deferred.append(logged_text(entry))
            continue
        if entry.read_only:
            replay = render_tool_result(await caller.call_tool(entry.tool_name, dict(entry.arguments)))
            replayed_tool = entry.tool_name
            masked = mask_volatile(replay, entry.volatile)
            changed_later = _changed_later(ledger, entry)
            span = pointer.span
            if isinstance(span, str) and not span_in_text(mask_volatile(span, entry.volatile), masked):
                if not changed_later:
                    return ReplayResult(
                        claim.claim_id,
                        False,
                        f"cited span is not in the replay of {entry.tool_name}",
                        replayed_tool=entry.tool_name,
                    )
                stateful.append(f"{entry.call_id}: {entry.tool_name} read state that a later write changed")
                checked.append(entry.result_text)
                continue
            checked.append(replay)
            if changed_later:
                checked.append(entry.result_text)
            continue
        readback = _readback(entry)
        if readback is None:
            unverifiable.append(f"{entry.call_id}: write {entry.tool_name} has no read-back tool")
            deferred.append(logged_text(entry))
            continue
        tool_name, arguments = readback
        text = render_tool_result(await caller.call_tool(tool_name, arguments))
        readbacks[entry.call_id] = text
        missing = [name for name, expected in (entry.effect or {}).items() if not text_contains(text, expected)]
        if missing:
            return ReplayResult(
                claim.claim_id,
                False,
                f"read-back of {entry.tool_name} does not show the write's {', '.join(missing)}",
                unverifiable=unverifiable or None,
                readbacks=readbacks,
            )
        checked.extend([logged_text(entry), text])
    if not checked:
        return ReplayResult(
            claim.claim_id,
            True,
            "unverifiable by replay: " + "; ".join(unverifiable),
            unverifiable=unverifiable,
            stateful=stateful or None,
        )
    # As in L1, a derived value is not expected verbatim; L3 recomputes it.
    if claim.kind != "derived":
        ok, reason = value_supported(claim.value, checked + deferred)
        if not ok:
            return ReplayResult(
                claim.claim_id,
                False,
                f"claim value is not in the replayed, read-back or logged text: {reason}",
                replayed_tool=replayed_tool,
                unverifiable=unverifiable or None,
                readbacks=readbacks,
                stateful=stateful or None,
            )
    return ReplayResult(
        claim.claim_id,
        True,
        "replay matches",
        replayed_tool=replayed_tool,
        unverifiable=unverifiable or None,
        readbacks=readbacks,
        stateful=stateful or None,
    )


def mask_volatile(text: str, patterns: list[str] | None) -> str:
    """Strip, then replace each volatile pattern match (multiline regex) with a placeholder."""
    masked = text.strip()
    for pattern in patterns or ():
        masked = re.sub(pattern, "<volatile>", masked, flags=re.MULTILINE)
    return masked


def _readback(entry: LedgerEntry) -> tuple[str, dict[str, Any]] | None:
    if not entry.readback_tool:
        return None
    return entry.readback_tool, dict(entry.readback_arguments or {})
