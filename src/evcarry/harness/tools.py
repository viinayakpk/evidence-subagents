"""Function tools over the in-process MCP client, plus the read-back catalog."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from agents import FunctionTool

from evcarry.protocol.replayer import render_tool_result

_OBJECT = {"type": "object", "additionalProperties": False}


@dataclass(frozen=True)
class ToolMeta:
    name: str
    description: str
    schema: dict[str, Any]
    read_only: bool
    readback_tool: str | None = None
    readback_arguments: dict[str, Any] | None = None


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
    ),
    "create_event": ToolMeta(
        "create_event",
        "Create a calendar event.",
        _schema({"title": {"type": "string"}}, ["title"]),
        False,
        "list_events",
        {},
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


def meta_for(name: str) -> ToolMeta:
    return CATALOG[name]


def build_function_tool(name: str, caller: Any) -> FunctionTool:
    meta = CATALOG[name]

    async def _invoke(_ctx: Any, args_json: str) -> str:
        arguments = json.loads(args_json) if args_json else {}
        result = await caller.call_tool(name, arguments)
        return render_tool_result(result)

    return FunctionTool(
        name=meta.name,
        description=meta.description,
        params_json_schema=meta.schema,
        on_invoke_tool=_invoke,
        strict_json_schema=True,
    )


def build_tools(names: tuple[str, ...] | list[str], caller: Any) -> list[FunctionTool]:
    return [build_function_tool(name, caller) for name in sorted(names)]
