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
"""

from __future__ import annotations

import asyncio
import copy
from typing import Any

from agents.items import TResponseInputItem
from agents.memory.session import SessionABC


class SubagentSession(SessionABC):
    """In-memory history of one sub-agent's conversation in one run."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.session_settings = None
        self._items: list[TResponseInputItem] = []

    async def get_items(self, limit: int | None = None) -> list[TResponseInputItem]:
        items = self._items if limit is None else self._items[-limit:] if limit > 0 else []
        return copy.deepcopy(items)

    async def add_items(self, items: list[TResponseInputItem]) -> None:
        self._items.extend(copy.deepcopy(items))

    async def pop_item(self) -> TResponseInputItem | None:
        return self._items.pop() if self._items else None

    async def clear_session(self) -> None:
        self._items.clear()


def serialized(tool: Any) -> Any:
    """Let one invocation of an ``as_tool`` sub-agent run at a time; later calls wait their turn in call order."""
    lock = asyncio.Lock()
    invoke = tool.on_invoke_tool

    async def one_at_a_time(context: Any, arguments: str) -> Any:
        async with lock:
            return await invoke(context, arguments)

    tool.on_invoke_tool = one_at_a_time
    return tool
