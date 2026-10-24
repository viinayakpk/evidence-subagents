"""In-process MCP server used by mock runs and the replayer tests.

Tool results are plain text. The replayer reads ``TextContent``, not
``structured_content``, because the MCP SDK wraps string returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from mcp import Client
from mcp.server import MCPServer
from mcp.types import ToolAnnotations

_READ_ONLY = ToolAnnotations(read_only_hint=True)
_WRITE = ToolAnnotations(read_only_hint=False)


@dataclass
class FakeState:
    root: Path
    events: list[str] = field(default_factory=list)
    calls: dict[str, int] = field(default_factory=dict)

    def hit(self, name: str) -> None:
        self.calls[name] = self.calls.get(name, 0) + 1

    def read(self, path: str) -> str:
        candidate = self._resolve(path)
        if not candidate.is_file():
            return f"missing {path}"
        return candidate.read_text(encoding="utf-8")

    def write_answer(self, text: str) -> None:
        target = self.root / "answer.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")

    def _resolve(self, path: str) -> Path:
        if path.startswith("/") or ".." in Path(path).parts:
            raise ValueError(f"path traversal rejected: {path}")
        root = self.root.resolve()
        candidate = (root / path).resolve()
        if root != candidate and root not in candidate.parents:
            raise ValueError(f"path traversal rejected: {path}")
        return candidate


def build_fake_server(root: Path) -> tuple[MCPServer, FakeState]:
    state = FakeState(root=root)
    server = MCPServer("evcarry-fake")

    @server.tool(annotations=_READ_ONLY)
    def search_records(query: str) -> str:
        """Search the fixture corpus."""
        state.hit("search_records")
        if query == "other":
            return "note 14 sections"
        return f"Found 14 sections in paper-A for {query}"

    @server.tool(annotations=_READ_ONLY)
    def query_rows(sql: str) -> str:
        """Run a read-only SQL query against the fixture table."""
        state.hit("query_rows")
        return f"rows paper-A paper-B for {sql}"

    @server.tool(annotations=_READ_ONLY)
    def read_file(path: str) -> str:
        """Read a workspace file. Path traversal is rejected."""
        state.hit("read_file")
        try:
            return state.read(path)
        except ValueError as exc:
            return str(exc)

    @server.tool(annotations=_WRITE)
    def write_answer(text: str) -> str:
        """Write answer.txt. Read it back with read_file."""
        state.hit("write_answer")
        state.write_answer(text)
        return "wrote answer.txt"

    @server.tool(annotations=_WRITE)
    def create_event(title: str) -> str:
        """Create a calendar event. Read events back with list_events."""
        state.hit("create_event")
        state.events.append(title)
        return f"created {title}"

    @server.tool(annotations=_READ_ONLY)
    def list_events() -> str:
        """List calendar event titles."""
        state.hit("list_events")
        if not state.events:
            return "none"
        return " ".join(state.events)

    setattr(server, "evcarry_state", state)
    return server, state


class FakeMcp:
    """Async context that owns the in-process client."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.server, self.state = build_fake_server(root)
        self.client: Client | None = None

    async def __aenter__(self) -> FakeMcp:
        self.client = Client(self.server)
        await self.client.__aenter__()
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self.client is not None:
            await self.client.__aexit__(*exc)

    async def call_tool(self, name: str, arguments: dict[str, object]) -> object:
        if self.client is None:
            raise RuntimeError("Fake MCP client is not open")
        return await self.client.call_tool(name, arguments)
