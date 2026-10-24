import asyncio

import pytest

from evcarry.env.fake_mcp import FakeMcp
from evcarry.inject.corruptions import apply_corruption
from evcarry.inject.injector import FaultInjector
from evcarry.ledger.ledger import Ledger
from evcarry.protocol.checks import CheckContext, run_l0_l1, sha256_text
from evcarry.protocol.replayer import replay_claim
from evcarry.protocol.schema import (
    Claim,
    DbEvidence,
    FileEvidence,
    SearchEvidence,
    SubagentReturn,
    ToolCallEvidence,
)


def _ledger() -> Ledger:
    ledger = Ledger()
    ledger.record(
        call_id="call_search",
        tool_name="search_records",
        arguments={"query": "sections"},
        result_text="Found 14 sections in paper-A for sections",
        read_only=True,
    )
    ledger.record(
        call_id="call_rows",
        tool_name="query_rows",
        arguments={"sql": "select papers"},
        result_text="rows paper-A paper-B for select papers",
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


def test_l0_l1_pointers(tmp_path) -> None:
    ledger = _ledger()
    body = "line one\n14 sections\n"
    (tmp_path / "notes.txt").write_text(body, encoding="utf-8")
    ctx = CheckContext(ledger=ledger, file_root=tmp_path, files={})
    good = Claim(
        claim_id="tool",
        text="14 sections",
        kind="fact",
        value=14,
        evidence=[ToolCallEvidence(type="tool_call", call_id="call_search", span="14 sections")],
    )
    assert all(item.ok for item in run_l0_l1(good, ctx))
    missing = good.model_copy(deep=True)
    missing.evidence = [ToolCallEvidence(type="tool_call", call_id="nope", span="14")]
    assert any(not item.ok and item.level == "L0" for item in run_l0_l1(missing, ctx))
    bad_span = good.model_copy(deep=True)
    bad_span.evidence = [ToolCallEvidence(type="tool_call", call_id="call_search", span="no such span")]
    assert any(not item.ok and item.level == "L1" for item in run_l0_l1(bad_span, ctx))
    file_claim = Claim(
        claim_id="file",
        text="notes",
        kind="fact",
        value=14,
        evidence=[FileEvidence(type="file", path="notes.txt", sha256=sha256_text(body), lines="2-2")],
    )
    assert all(item.ok for item in run_l0_l1(file_claim, ctx))
    bad_hash = file_claim.model_copy(deep=True)
    bad_hash.evidence[0].sha256 = "0" * 64
    assert any(not item.ok and "sha256" in item.reason for item in run_l0_l1(bad_hash, ctx))
    search = Claim(
        claim_id="search",
        text="paper",
        kind="fact",
        value=None,
        evidence=[SearchEvidence(type="search", query="sections", rank=1, doc_id="paper-A")],
    )
    assert all(item.ok for item in run_l0_l1(search, ctx))
    db = Claim(
        claim_id="db",
        text="rows",
        kind="fact",
        value=None,
        evidence=[DbEvidence(type="db", sql="select papers", row_keys=["paper-A", "paper-B"])],
    )
    assert all(item.ok for item in run_l0_l1(db, ctx))
    empty = good.model_copy(deep=True)
    empty.evidence = []
    assert run_l0_l1(empty, ctx)[0].reason == "claim has no evidence pointer"


def test_replayer_does_not_replay_writes(tmp_path) -> None:
    async def scenario() -> None:
        async with FakeMcp(tmp_path) as mcp:
            await mcp.call_tool("search_records", {"query": "sections"})
            await mcp.call_tool("write_answer", {"text": "14 sections"})
            ledger = Ledger()
            ledger.record(
                call_id="call_search",
                tool_name="search_records",
                arguments={"query": "sections"},
                result_text="Found 14 sections in paper-A for sections",
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
            read = Claim(
                claim_id="r",
                text="14",
                kind="fact",
                value=14,
                evidence=[ToolCallEvidence(type="tool_call", call_id="call_search", span="14 sections")],
            )
            write = Claim(
                claim_id="w",
                text="wrote",
                kind="action_done",
                value="14 sections",
                evidence=[ToolCallEvidence(type="tool_call", call_id="call_write", span="wrote answer.txt")],
            )
            assert (await replay_claim(read, ledger, mcp)).ok
            assert (await replay_claim(write, ledger, mcp)).ok
            assert mcp.state.calls["search_records"] == 2
            assert mcp.state.calls["write_answer"] == 1
            assert mcp.state.calls["read_file"] == 1

    asyncio.run(scenario())


def _base() -> Claim:
    return Claim(
        claim_id="c1",
        text="The record contains 14 sections in paper-A.",
        kind="fact",
        value=14,
        evidence=[ToolCallEvidence(type="tool_call", call_id="call_search", span="14 sections")],
    )


def test_corruptions_are_stable_and_ignore_arm() -> None:
    ledger = _ledger()
    ledger.record(
        call_id="call_other",
        tool_name="search_records",
        arguments={"query": "other"},
        result_text="note 14 sections",
        read_only=True,
    )
    for kind in (
        "numeric_perturbation",
        "fabricated_entity",
        "omission",
        "false_action_done",
        "mismatched_pointer",
    ):
        for variant in ("naive", "pointer_consistent"):
            left = apply_corruption(_base(), kind, variant, ledger)
            right = apply_corruption(_base(), kind, variant, ledger)
            assert left.model_dump() == right.model_dump()
    numeric = apply_corruption(_base(), "numeric_perturbation", "naive", ledger)
    assert numeric.value == 11
    injector = FaultInjector(probability=1, types=("numeric_perturbation",), variant="naive")
    payload = SubagentReturn(subagent_id="researcher", status="complete", claims=[_base()], open_issues=[])
    first = injector.decide(task="fixture-sections", seed=1, return_index=0, claims=payload.claims)
    second = injector.decide(task="fixture-sections", seed=1, return_index=0, claims=payload.claims)
    assert first is not None and second is not None
    assert first.as_dict() == second.as_dict()
    assert "arm" not in first.as_dict()
    off = FaultInjector(probability=0)
    assert off.decide(task="t", seed=1, return_index=0, claims=payload.claims) is None


def test_path_traversal_rejected(tmp_path) -> None:
    async def scenario() -> None:
        async with FakeMcp(tmp_path) as mcp:
            result = await mcp.call_tool("read_file", {"path": "../secret"})
            from evcarry.protocol.replayer import render_tool_result

            assert "rejected" in render_tool_result(result)

    asyncio.run(scenario())
