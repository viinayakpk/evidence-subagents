"""Which gym MCP tools are read-only, which replays may differ, and how writes are read back.

The policy is ``configs/gym_tools.yaml``. A gym tool is exposed to the model as
``{server}-{tool}`` so names from different servers cannot collide.
"""

from __future__ import annotations

import copy
import hashlib
import re
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

import yaml

from evcarry.harness.tools import ToolMeta
from evcarry.paths import resolve_repo_path

_VALID_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

CLAIM_DONE = "claim_done"
PYTHON_EXECUTE = "python_execute"

CLAIM_DONE_META = ToolMeta(
    CLAIM_DONE,
    "Call this tool once the task is fully completed. The run ends immediately.",
    {"type": "object", "additionalProperties": False, "properties": {}, "required": []},
    False,
)

PYTHON_EXECUTE_META = ToolMeta(
    PYTHON_EXECUTE,
    "Execute Python code in the agent workspace and return stdout and stderr.",
    {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python source code to execute."},
            "filename": {"type": "string", "description": "Optional file name ending in .py."},
            "timeout": {"type": "integer", "description": "Max seconds, capped at 120."},
        },
        "required": ["code"],
    },
    False,
    strict=False,
)


def normalize_schema(schema: Any) -> None:
    """Rewrite an MCP input schema in place the way the gym's own agent does before sending it.

    Adapted from ``_fix_schema`` in the gym's ``utils/roles/task_agent.py``
    (Apache-2.0), non-strict variant: a type list keeps its first non-null type,
    an ``anyOf`` of one type and null becomes that type, ``enum`` is dropped from
    non-string fields, ``additionalProperties: true`` is dropped, and array items
    without a type become strings. Keeps the tools the model sees identical to
    the gym's.
    """
    if not isinstance(schema, dict):
        return
    if isinstance(schema.get("type"), list):
        non_null = [kind for kind in schema["type"] if kind != "null"]
        schema["type"] = non_null[0] if non_null else "string"
    if "anyOf" in schema and not any(key in schema for key in ("type", "$ref")):
        variants = [item for item in schema["anyOf"] if item.get("type") != "null" and item != {"type": "null"}]
        if len(variants) == 1:
            schema.pop("anyOf")
            schema.update(variants[0])
    if "enum" in schema and schema.get("type", "string") != "string":
        del schema["enum"]
    if schema.get("type") == "object" and schema.get("additionalProperties") is True:
        del schema["additionalProperties"]
    if schema.get("type") == "array" and "items" in schema:
        items = schema["items"]
        if isinstance(items, dict) and "type" not in items and not any(key in items for key in ("anyOf", "oneOf", "allOf", "$ref")):
            items["type"] = "string"
        normalize_schema(items)
    for key in ("properties", "additionalProperties"):
        value = schema.get(key)
        if isinstance(value, dict):
            for item in value.values():
                normalize_schema(item)
    for key in ("allOf", "anyOf", "oneOf"):
        for item in schema.get(key, []):
            normalize_schema(item)


def exposed_name(server: str, tool: str) -> str:
    name = f"{server}-{tool}"
    if _VALID_NAME.match(name):
        return name
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "_", name)
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
    return cleaned if len(cleaned) <= 64 else f"{cleaned[:55]}_{digest}"


@dataclass(frozen=True)
class GymToolPolicy:
    read_only: dict[str, tuple[str, ...]]
    nondeterministic: dict[str, tuple[str, ...]]
    stateful_servers: frozenset[str]
    volatile: dict[str, tuple[str, ...]]
    readback: dict[str, dict[str, Any]]
    sha256: str

    @classmethod
    def load(cls, path: str | Path | None = None) -> GymToolPolicy:
        target = resolve_repo_path(path or "configs/gym_tools.yaml")
        raw_bytes = target.read_bytes()
        data = yaml.safe_load(raw_bytes) or {}
        return cls(
            read_only={k: tuple(v) for k, v in (data.get("read_only") or {}).items()},
            nondeterministic={k: tuple(v) for k, v in (data.get("nondeterministic") or {}).items()},
            stateful_servers=frozenset(data.get("stateful_servers") or ()),
            volatile={k: tuple(v) for k, v in (data.get("volatile") or {}).items()},
            readback=dict(data.get("readback") or {}),
            sha256=hashlib.sha256(raw_bytes).hexdigest(),
        )

    def is_read_only(self, server: str, tool: str, annotation: bool | None) -> bool:
        if server in self.stateful_servers or annotation is False:
            return False
        if annotation is True:
            return True
        return any(fnmatchcase(tool, pattern) for pattern in self.read_only.get(server, ()))

    def is_nondeterministic(self, server: str, tool: str) -> bool:
        return any(fnmatchcase(tool, pattern) for pattern in self.nondeterministic.get(server, ()))

    def meta(self, server: str, tool: Any) -> ToolMeta:
        """``ToolMeta`` for one ``mcp.types.Tool`` listed by ``server``."""
        annotations = getattr(tool, "annotations", None)
        hint = getattr(annotations, "read_only_hint", None) if annotations is not None else None
        name = exposed_name(server, tool.name)
        schema = copy.deepcopy(dict(getattr(tool, "input_schema", None) or {}))
        schema.setdefault("type", "object")
        schema.setdefault("properties", {})
        normalize_schema(schema)
        readback = self.readback.get(name) or {}
        read_only = self.is_read_only(server, tool.name, hint)
        return ToolMeta(
            name=name,
            description=(tool.description or tool.name).strip(),
            schema=schema,
            read_only=read_only,
            readback_tool=readback.get("tool"),
            readback_from=dict(readback.get("args") or {}) or None,
            readback_fixed=dict(readback.get("fixed") or {}) or None,
            nondeterministic=read_only and self.is_nondeterministic(server, tool.name),
            volatile=self.volatile.get(server, ()),
            strict=False,
            server=server,
        )
