"""Persistent sub-agent conversations (owner decision 3, 2026-10-24).

In every delegating arm each sub-agent (researcher, operator) keeps one
conversation for the whole run. Every call from the main agent continues it,
and so do the harness's repair and re-delegation messages in the schema arms.
Before this, the evidence arms' harness re-delegation continued the sub-agent's
conversation while a main agent that asked again (vanilla, self-verify, or any
arm after a FALLBACK) reached a fresh sub-agent, so the retry context differed
by arm.

Calls to one sub-agent are serialised: two parallel calls would otherwise read
the same history and interleave their items in it. Calls to different
sub-agents still run in parallel.

The SDK de-duplicates tool items in a model input by ``call_id``. If a provider
ever sends a tool-call id that is already in the stored history, the stored
copy of the new pair gets an alias (``<id>r<n>``) so later calls see both
pairs; the run where the repeat happens still sees one of them merged away.
DeepInfra's ids are random, so this is not expected; ``subagent_call_id_reused``
in the trace says if it happened. Ids are otherwise stored unchanged.
"""

from __future__ import annotations

import asyncio
import copy
from typing import Any

from agents.items import TResponseInputItem
from agents.memory.session import SessionABC

from evcarry.trace.writer import current_writer


class SubagentSession(SessionABC):
    """In-memory history of one sub-agent's conversation in one run."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.session_settings = None
        self._items: list[TResponseInputItem] = []
        self._call_ids: set[str] = set()  # call ids of the stored tool calls
        self._alias: dict[str, str] = {}  # a reused provider id -> the id its newest pair is stored under

    async def get_items(self, limit: int | None = None) -> list[TResponseInputItem]:
        items = self._items if limit is None else self._items[-limit:] if limit > 0 else []
        return copy.deepcopy(items)

    async def add_items(self, items: list[TResponseInputItem]) -> None:
        for item in copy.deepcopy(items):
            self._items.append(self._unique_call_id(item))

    def _unique_call_id(self, item: TResponseInputItem) -> TResponseInputItem:
        if not isinstance(item, dict) or not isinstance(item.get("call_id"), str):
            return item
        raw = item["call_id"]
        if item.get("type") == "function_call":
            if raw in self._call_ids:
                n = 1
                while f"{raw}r{n}" in self._call_ids:
                    n += 1
                self._alias[raw] = f"{raw}r{n}"
                writer = current_writer.get()
                if writer is not None:
                    writer.write({"type": "subagent_call_id_reused", "session": self.session_id, "call_id": raw, "stored_as": self._alias[raw]})
            else:
                self._alias.pop(raw, None)
            item["call_id"] = self._alias.get(raw, raw)
            self._call_ids.add(item["call_id"])
        elif raw in self._alias:
            item["call_id"] = self._alias[raw]  # the output of the aliased call
        return item

    async def pop_item(self) -> TResponseInputItem | None:
        return self._items.pop() if self._items else None

    async def clear_session(self) -> None:
        self._items.clear()
        self._call_ids.clear()
        self._alias.clear()


def serialized(tool: Any) -> Any:
    """Let one invocation of an ``as_tool`` sub-agent run at a time; later calls wait their turn in call order."""
    lock = asyncio.Lock()
    invoke = tool.on_invoke_tool

    async def one_at_a_time(context: Any, arguments: str) -> Any:
        async with lock:
            return await invoke(context, arguments)

    tool.on_invoke_tool = one_at_a_time
    return tool
