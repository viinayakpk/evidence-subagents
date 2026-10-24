"""Function tools over an MCP caller, the per-run tool catalog, and the ledger hook-in.

Each tool records its full result in the ledger before returning, so the
ledger holds what the server said even when the model is shown a shortened
copy. Gym runs show each result under a ``[call_id: ...]`` header, cut at
``max_chars``; the rest is paged with ``view_overlong_output``. Fixture runs
show the raw text.
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from agents import FunctionTool

from evcarry.ledger.ledger import Ledger
from evcarry.protocol.replayer import render_tool_result

_OBJECT = {"type": "object", "additionalProperties": False}

VIEW_OVERLONG = "view_overlong_output"


@dataclass(frozen=True)
class ToolMeta:
    name: str
    description: str
    schema: dict[str, Any]
    read_only: bool
    readback_tool: str | None = None
    readback_arguments: dict[str, Any] | None = None
    # Gym writes: read-back argument <- write argument, plus constant arguments.
    readback_from: dict[str, str] | None = None
    readback_fixed: dict[str, Any] | None = None
    # Read-back argument <- the first group of a regex over the write's result (an id the server assigned).
    readback_from_result: dict[str, str] | None = None
    # Write arguments whose values the read-back must show (L2 checks the write's effect, not its success text).
    effect: tuple[str, ...] = ()
    nondeterministic: bool = False
    # Regexes masked before an L2 replay is compared with the ledger.
    volatile: tuple[str, ...] = ()
    strict: bool = True
    server: str | None = None

    def readback_for(self, arguments: dict[str, Any], result_text: str = "") -> tuple[str, dict[str, Any]] | None:
        """The read-back call for this write, or None (no read-back tool, or the result has no id to read back)."""
        if not self.readback_tool:
            return None
        args = dict(self.readback_arguments or {})
        args.update(self.readback_fixed or {})
        for target, source in (self.readback_from or {}).items():
            if source in arguments:
                args[target] = arguments[source]
        for target, pattern in (self.readback_from_result or {}).items():
            match = re.search(pattern, result_text)
            if match is None:
                return None  # the write failed or answered in an unknown shape: nothing to read back
            args[target] = match.group(1)
        return self.readback_tool, args

    def effect_for(self, arguments: dict[str, Any]) -> dict[str, str] | None:
        """The effect fields of this write's arguments that its read-back must show."""
        found = {
            name: str(arguments[name]).strip()
            for name in self.effect
            if isinstance(arguments.get(name), (str, int, float)) and str(arguments[name]).strip()
        }
        return found or None


def _schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {**_OBJECT, "properties": properties, "required": required}


CATALOG: dict[str, ToolMeta] = {
    "search_records": ToolMeta(
        "search_records",
        "Search the fixture corpus.",
        _schema({"query": {"type": "string"}}, ["query"]),
        True,
    ),
    "query_rows": ToolMeta(
        "query_rows",
        "Run a read-only SQL query.",
        _schema({"sql": {"type": "string"}}, ["sql"]),
        True,
    ),
    "read_file": ToolMeta(
        "read_file",
        "Read a workspace file.",
        _schema({"path": {"type": "string"}}, ["path"]),
        True,
    ),
    "write_answer": ToolMeta(
        "write_answer",
        "Write answer.txt.",
        _schema({"text": {"type": "string"}}, ["text"]),
        False,
        "read_file",
        {"path": "answer.txt"},
        effect=("text",),
    ),
    "create_event": ToolMeta(
        "create_event",
        "Create a calendar event.",
        _schema({"title": {"type": "string"}}, ["title"]),
        False,
        "list_events",
        {},
        effect=("title",),
    ),
    "list_events": ToolMeta(
        "list_events",
        "List calendar event titles.",
        _schema({}, []),
        True,
    ),
}

RESEARCHER_TOOLS = ("query_rows", "read_file", "search_records")
OPERATOR_TOOLS = ("create_event", "list_events", "read_file", "write_answer")
ALL_TOOLS = tuple(sorted(CATALOG))


def meta_for(name: str, catalog: dict[str, ToolMeta] | None = None) -> ToolMeta:
    return (catalog if catalog is not None else CATALOG)[name]


@dataclass
class Toolset:
    """Which tools each role gets in one run, and how results are shown to the model."""

    catalog: dict[str, ToolMeta]
    caller: Any
    all_tools: tuple[str, ...]
    researcher: tuple[str, ...]
    operator: tuple[str, ...]
    main_extra: tuple[str, ...] = ()
    stop_tools: tuple[str, ...] = ()
    show_call_id: bool = False
    max_chars: int | None = None
    ledger: Ledger | None = None
    local: dict[str, Callable[[dict[str, Any]], Awaitable[str]]] = field(default_factory=dict)


def fixture_toolset(caller: Any, ledger: Ledger | None = None) -> Toolset:
    return Toolset(
        catalog=CATALOG,
        caller=caller,
        all_tools=ALL_TOOLS,
        researcher=RESEARCHER_TOOLS,
        operator=OPERATOR_TOOLS,
        ledger=ledger,
    )


def display_text(text: str, ledger_id: str, *, show_call_id: bool, max_chars: int | None) -> str:
    body = text
    if max_chars is not None and len(text) > max_chars:
        body = (
            text[:max_chars]
            + f"\n...[truncated: showing {max_chars} of {len(text)} chars. "
            f'Call {VIEW_OVERLONG} with call_id "{ledger_id}" and page=1 for the next part.]'
        )
    if show_call_id:
        return f"[call_id: {ledger_id}]\n{body}"
    return body


def overlong_view(ledger: Ledger, page_size: int) -> Callable[[dict[str, Any]], Awaitable[str]]:
    """``view_overlong_output``: page through a ledger entry's full text."""

    async def view(arguments: dict[str, Any]) -> str:
        call_id = str(arguments.get("call_id", ""))
        try:
            page = int(arguments.get("page", 0) or 0)
        except (TypeError, ValueError):
            return f"page must be an integer, got {arguments.get('page')!r}"
        entry = ledger.get(call_id)
        if entry is None:
            return f"No recorded tool output with call_id '{call_id}'."
        text = entry.result_text
        pages = max(1, -(-len(text) // page_size))
        if page < 0 or page >= pages:
            return f"Page {page} is out of range; {call_id} has pages 0 to {pages - 1}."
        start = page * page_size
        chunk = text[start : start + page_size]
        return f"{call_id} page {page} of {pages - 1} (chars {start}-{start + len(chunk)} of {len(text)})\n{chunk}"

    return view


OVERLONG_META = ToolMeta(
    VIEW_OVERLONG,
    "Read a long tool output that was truncated. Pages are counted from 0; page 0 is what you already saw.",
    _schema({"call_id": {"type": "string"}, "page": {"type": "integer"}}, ["call_id", "page"]),
    True,
)


class ToolRouter:
    """The caller the tools and the L2 replayer use: local tools first, then the environment."""

    def __init__(self, env: Any, local: dict[str, Callable[[dict[str, Any]], Awaitable[str]]]) -> None:
        self.env = env
        self.local = local

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        handler = self.local.get(name)
        if handler is not None:
            return await handler(arguments)
        return await self.env.call_tool(name, arguments)


def build_function_tool(name: str, caller: Any, toolset: Toolset | None = None) -> FunctionTool:
    catalog = toolset.catalog if toolset is not None else CATALOG
    meta = catalog[name]
    ledger = toolset.ledger if toolset is not None else None

    async def _invoke(ctx: Any, args_json: str) -> str:
        # A FunctionTool built directly has no failure handler: an exception here
        # would end the whole run. Bad arguments and tool errors go back to the model.
        try:
            arguments = json.loads(args_json) if args_json else {}
            result = await caller.call_tool(name, arguments)
        except Exception as exc:  # noqa: BLE001
            return f"Error calling {name}: {type(exc).__name__}: {exc}"
        text = render_tool_result(result)
        if ledger is None:
            return text
        ledger_id = ledger.unique_id(getattr(ctx, "tool_call_id", "") or name)
        readback = meta.readback_for(arguments, text)
        ledger.record(
            call_id=ledger_id,
            tool_name=name,
            arguments=arguments,
            result_text=text,
            read_only=meta.read_only,
            readback_tool=readback[0] if readback else None,
            readback_arguments=readback[1] if readback else None,
            nondeterministic=meta.nondeterministic,
            volatile=list(meta.volatile) or None,
            server=meta.server,
            effect=meta.effect_for(arguments) if not meta.read_only else None,
        )
        if toolset is None:
            return text
        # A page of an overlong output is already page-sized; cutting it again would hide its tail.
        limit = None if name == VIEW_OVERLONG else toolset.max_chars
        return display_text(text, ledger_id, show_call_id=toolset.show_call_id, max_chars=limit)

    return FunctionTool(
        name=meta.name,
        description=meta.description,
        params_json_schema=meta.schema,
        on_invoke_tool=_invoke,
        strict_json_schema=meta.strict,
    )


def build_tools(
    names: tuple[str, ...] | list[str],
    caller: Any,
    toolset: Toolset | None = None,
) -> list[FunctionTool]:
    return [build_function_tool(name, caller, toolset) for name in sorted(names)]
