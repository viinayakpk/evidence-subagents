"""Ledger and llm_call trace events from Agents SDK run hooks."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from agents import RunHooks

from evcarry.harness.tools import meta_for
from evcarry.ledger.ledger import Ledger
from evcarry.providers.usage import PriceTable, cost_usd, is_peak
from evcarry.trace.writer import JsonlWriter


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class LedgerHooks(RunHooks):
    """Trace MCP tool calls and model calls for one run.

    The tools write the ledger themselves (``harness.tools``); this hook writes
    the ``tool_call`` and ``llm_call`` trace events. Agent-as-tool calls are
    skipped: the nested hooks see the real MCP calls.
    """

    def __init__(
        self,
        ledger: Ledger,
        writer: JsonlWriter,
        price: PriceTable,
        catalog: dict[str, Any] | None = None,
    ) -> None:
        self.ledger = ledger
        self.writer = writer
        self.price = price
        self.catalog = catalog

    async def on_llm_end(self, context: Any, agent: Any, response: Any) -> None:
        del context, response
        model = getattr(agent, "model", None)
        take = getattr(model, "take_record", None)
        record = take() if take else None
        if record is None:
            return
        now = datetime.now(timezone.utc)
        system = record.system_instructions or ""
        self.writer.write(
            {
                "type": "llm_call",
                "agent": getattr(agent, "name", ""),
                "configured_model": record.configured_model,
                **record.provenance(),
                "routing": record.routing,
                "router": record.identity.router,
                "attempts": record.attempts,
                "tokens": record.usage.as_dict(),
                "cost_usd": cost_usd(record.usage, self.price, now),
                "latency_ms": record.latency_ms,
                "peak": is_peak(now, self.price),
                "provider_reported_cost_usd": record.usage.provider_reported_cost_usd,
                "system_sha256": _sha256(system),
            }
        )

    async def on_tool_end(self, context: Any, agent: Any, tool: Any, result: object) -> None:
        if getattr(tool, "_is_agent_tool", False):
            return
        name = getattr(tool, "name", "")
        try:
            meta = meta_for(name, self.catalog)
        except KeyError:
            return
        call_id = getattr(context, "tool_call_id", "") or ""
        raw_args = getattr(context, "tool_arguments", "") or "{}"
        try:
            arguments = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
        except json.JSONDecodeError:
            arguments = {}
        entry = self.ledger.latest(call_id) if call_id else None
        self.writer.write(
            {
                "type": "tool_call",
                "agent": getattr(agent, "name", ""),
                "tool": name,
                "call_id": call_id,
                "ledger_call_id": entry.call_id if entry is not None else None,
                "read_only": meta.read_only,
                "arguments": arguments,
                "result_sha256": entry.sha256 if entry is not None else None,
                "result_chars": len(entry.result_text) if entry is not None else None,
            }
        )
