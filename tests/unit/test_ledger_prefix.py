"""A cited call id without the provider's prefix (dev run, 2026-10-25, sf-support-metrics evidence_risk)."""

from evcarry.ledger.ledger import Ledger


def _ledger() -> Ledger:
    ledger = Ledger()
    ledger.record(call_id="chatcmpl-tool-814208009fe40ab0", tool_name="python_execute", arguments={},
                  result_text="exists: True\nsize_bytes: 2993", read_only=False)
    ledger.record(call_id="chatcmpl-tool-a048455ee680b7cb", tool_name="filesystem-get_file_info", arguments={},
                  result_text="size: 2993", read_only=True)
    return ledger


def test_exact_ids_resolve_as_before() -> None:
    ledger = _ledger()
    assert ledger.get("chatcmpl-tool-814208009fe40ab0").tool_name == "python_execute"


def test_an_id_cited_without_its_prefix_resolves_to_the_one_call() -> None:
    ledger = _ledger()
    assert ledger.get("814208009fe40ab0").call_id == "chatcmpl-tool-814208009fe40ab0"
    assert ledger.get("a048455ee680b7cb").tool_name == "filesystem-get_file_info"


def test_partial_short_or_ambiguous_ids_do_not_resolve() -> None:
    ledger = _ledger()
    assert ledger.get("08009fe40ab0") is None  # not at a separator
    assert ledger.get("40ab0") is None  # too short
    assert ledger.get("deadbeefdeadbeef") is None
    ledger.record(call_id="other-814208009fe40ab0", tool_name="x", arguments={}, result_text="", read_only=True)
    assert ledger.get("814208009fe40ab0") is None  # two calls end in it


def test_l0_and_l1_accept_the_dev_run_claim_cited_without_prefixes() -> None:
    from evcarry.protocol.checks import CheckContext, run_l0_l1
    from evcarry.protocol.schema import Claim, ToolCallEvidence

    claim = Claim(
        claim_id="c1", text="The PDF exists with a size of 2993 bytes.", kind="artifact", value=2993,
        evidence=[ToolCallEvidence(type="tool_call", call_id="a048455ee680b7cb", span="size: 2993"),
                  ToolCallEvidence(type="tool_call", call_id="814208009fe40ab0", span="exists: True")],
    )
    results = run_l0_l1(claim, CheckContext(ledger=_ledger()))
    assert all(r.ok for r in results), [(r.level, r.reason) for r in results]
