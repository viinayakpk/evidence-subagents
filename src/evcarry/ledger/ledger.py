"""In-memory ledger of MCP calls."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class LedgerEntry:
    call_id: str
    tool_name: str
    arguments: dict[str, object]
    result_text: str
    sha256: str
    read_only: bool
    readback_tool: str | None = None
    readback_arguments: dict[str, object] | None = None
    nondeterministic: bool = False
    volatile: list[str] | None = None
    # The MCP server the tool belongs to (None for local tools). Absent in ledgers written before 2026-10-25.
    server: str | None = None
    # For a write: the argument values its read-back must show (recipient, subject, title), by argument name.
    effect: dict[str, str] | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class Ledger:
    entries: dict[str, LedgerEntry] = field(default_factory=dict)

    def unique_id(self, call_id: str) -> str:
        """``call_id`` if unused, else ``call_id#2``, ``call_id#3``, ... Provider ids can repeat."""
        if call_id not in self.entries:
            return call_id
        suffix = 2
        while f"{call_id}#{suffix}" in self.entries:
            suffix += 1
        return f"{call_id}#{suffix}"

    def record(
        self,
        *,
        call_id: str,
        tool_name: str,
        arguments: dict[str, object],
        result_text: str,
        read_only: bool,
        readback_tool: str | None = None,
        readback_arguments: dict[str, object] | None = None,
        nondeterministic: bool = False,
        volatile: list[str] | None = None,
        server: str | None = None,
        effect: dict[str, str] | None = None,
    ) -> LedgerEntry:
        if not call_id:
            raise ValueError("tool-call id is required")
        if call_id in self.entries:
            raise ValueError(f"call_id {call_id} is already in the ledger")
        entry = LedgerEntry(
            call_id=call_id,
            tool_name=tool_name,
            arguments=arguments,
            result_text=result_text,
            sha256=_sha256(result_text),
            read_only=read_only,
            readback_tool=readback_tool,
            readback_arguments=readback_arguments,
            nondeterministic=nondeterministic,
            volatile=volatile,
            server=server,
            effect=effect,
        )
        self.entries[call_id] = entry
        return entry

    def get(self, call_id: str) -> LedgerEntry | None:
        return self.entries.get(call_id)

    def latest(self, call_id: str) -> LedgerEntry | None:
        """The newest entry recorded under ``call_id`` or one of its ``#n`` variants."""
        found = None
        for key, entry in self.entries.items():
            if key == call_id or key.startswith(f"{call_id}#"):
                found = entry
        return found

    def to_json(self) -> str:
        payload = [entry.as_dict() for entry in self.entries.values()]
        return json.dumps(payload, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> Ledger:
        ledger = cls()
        for raw in json.loads(text):
            ledger.entries[raw["call_id"]] = LedgerEntry(**raw)
        return ledger
