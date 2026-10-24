"""Catch rates on a synthetic ledger, one corruption at a time.

This is the table notebook 03 prints. It does not run the agent loop.
The lexical L3 judge does not catch omissions: remaining words of the list
still appear in the evidence.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from evcarry.inject.corruptions import CORRUPTION_TYPES, VARIANTS, apply_corruption
from evcarry.ledger.ledger import Ledger
from evcarry.protocol.audit import audit_return
from evcarry.protocol.schema import Claim, SubagentReturn, ToolCallEvidence


class _Caller:
    def __init__(self, ledger: Ledger) -> None:
        self.ledger = ledger
        self.calls: list[str] = []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        self.calls.append(name)
        if name == "search_records":
            query = str(arguments.get("query"))
            if query == "other":
                return "note 14 sections"
            return "Found 14 sections in paper-A for sections\nA, B, C, D"
        if name == "read_file":
            return "14 sections"
        if name == "list_events":
            return "standup"
        return ""


def _base_claim() -> Claim:
    # The list value lets omission drop a token that is already inside the span,
    # so the naive variant does not retarget the pointer. Numeric corruptions
    # overwrite ``value`` and still read the number in ``text``.
    return Claim(
        claim_id="c1",
        text="The record contains 14 sections in paper-A.",
        kind="fact",
        value=["paper-A", "14"],
        evidence=[ToolCallEvidence(type="tool_call", call_id="call_search", span="14 sections")],
        confidence="high",
    )


def _ledger() -> Ledger:
    ledger = Ledger()
    ledger.record(
        call_id="call_search",
        tool_name="search_records",
        arguments={"query": "sections"},
        result_text="Found 14 sections in paper-A for sections\nA, B, C, D",
        read_only=True,
    )
    ledger.record(
        call_id="call_other",
        tool_name="search_records",
        arguments={"query": "other"},
        result_text="note 14 sections",
        read_only=True,
    )
    ledger.record(
        call_id="call_write",
        tool_name="write_answer",
        arguments={"text": "14 sections"},
        result_text="wrote answer.txt",
        read_only=False,
        readback_tool="read_file",
        readback_arguments={"path": "answer.txt"},
    )
    return ledger


def _earliest(report) -> str | None:
    failed = {reason.split(":", 1)[0] for _claim, reason in report.failures}
    for level in ("L0", "L1", "L2", "L3"):
        if level in failed:
            return level
    return None


def _run(coro):
    """Run a coroutine from a script or from a notebook that already has a loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def catch_table() -> list[dict[str, str | None]]:
    rows = []
    for kind in CORRUPTION_TYPES:
        for variant in VARIANTS:
            ledger = _ledger()
            claim = apply_corruption(_base_claim(), kind, variant, ledger)
            payload = SubagentReturn(
                subagent_id="researcher",
                status="complete",
                claims=[claim],
                open_issues=[],
            )
            report = _run(
                audit_return(
                    payload,
                    ledger=ledger,
                    caller=_Caller(ledger),
                    checks=("L0", "L1", "L2", "L3"),
                    sampler="risk",
                    rate=1.0,
                    seed=1,
                    require_evidence=True,
                )
            )
            rows.append({"kind": kind, "variant": variant, "earliest": _earliest(report)})
    return rows


def expected_catchers() -> dict[tuple[str, str], str | None]:
    return {(row["kind"], row["variant"]): row["earliest"] for row in catch_table()}
