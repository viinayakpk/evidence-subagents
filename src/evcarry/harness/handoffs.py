"""Handoff content in the run trace (P0-2, 2026-10-25).

The trace had no message content: the main agent's instruction to a sub-agent,
the sub-agent's reply, and the harness's repair, re-delegation and FALLBACK
messages were not logged, which blocks the propagation labels (which instruction
led to which wrong value). Each is now a ``handoff`` event:

- ``instruction``: what the main agent asked (the agent tool's ``input``), with the main agent's tool-call id;
- ``subagent_output``: the sub-agent's raw final output, before parsing and checks (schema arms);
- ``repair``, ``redelegation``, ``fallback``: the harness's message, as sent;
- ``delivered``: what the main agent received as the tool result (every delegating arm).

Events carry the sub-agent name and the main agent's tool-call id, so one
delegation can be followed from instruction to delivery.
"""

from __future__ import annotations

import json
from contextvars import ContextVar
from typing import Any

# The delegation in progress in this task: {"subagent": name, "call_id": the main agent's tool-call id}.
current_handoff: ContextVar[dict[str, Any] | None] = ContextVar("evcarry_current_handoff", default=None)

# A longer text is cut here and flagged; sub-agent replies in the pilot were a few thousand characters.
HANDOFF_CHARS = 100_000


def write_handoff(writer: Any, kind: str, text: str, **fields: Any) -> None:
    if writer is None:
        return
    context = current_handoff.get() or {}
    body = text if isinstance(text, str) else str(text)
    writer.write(
        {
            "type": "handoff",
            "kind": kind,
            "subagent": fields.pop("subagent", context.get("subagent")),
            "call_id": context.get("call_id"),
            **fields,
            "chars": len(body),
            "truncated": len(body) > HANDOFF_CHARS,
            "text": body[:HANDOFF_CHARS],
        }
    )


def instruction_text(arguments: Any) -> str:
    """The ``input`` the main agent passed to an agent tool, else the raw arguments."""
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return arguments
    else:
        parsed = arguments
    if isinstance(parsed, dict) and isinstance(parsed.get("input"), str):
        return parsed["input"]
    return arguments if isinstance(arguments, str) else json.dumps(arguments, default=str)


def traced_handoff(tool: Any, writer: Any, subagent: str) -> Any:
    """Log an agent tool's instruction and what the main agent got back; mark the delegation for nested events."""
    invoke = tool.on_invoke_tool

    async def with_handoff_log(context: Any, arguments: str) -> Any:
        token = current_handoff.set({"subagent": subagent, "call_id": getattr(context, "tool_call_id", None)})
        try:
            write_handoff(writer, "instruction", instruction_text(arguments))
            result = await invoke(context, arguments)
            write_handoff(writer, "delivered", result if isinstance(result, str) else str(result))
            return result
        finally:
            current_handoff.reset(token)

    tool.on_invoke_tool = with_handoff_log
    return tool
