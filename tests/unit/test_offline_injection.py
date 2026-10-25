import asyncio

from evcarry.analysis.offline_injection import FAULTS, Item, first_failing_level, inject, study
from evcarry.ledger.ledger import Ledger
from evcarry.protocol.schema import Claim, ToolCallEvidence


def _ledger() -> Ledger:
    ledger = Ledger()
    ledger.record(call_id="call_count", tool_name="query_rows", arguments={"sql": "count"},
                  result_text="order_count 4100 for Europe", read_only=True)
    ledger.record(call_id="call_names", tool_name="query_rows", arguments={"sql": "names"},
                  result_text="customers Ethan Noah Harper", read_only=True)
    ledger.record(call_id="call_send", tool_name="send_email", arguments={"to": "eu@company.com"},
                  result_text="Email sent successfully to eu@company.com", read_only=False)
    return ledger


def _claim(value, call_id="call_count", span="order_count 4100", kind="fact") -> Claim:
    return Claim(claim_id="c1", text=f"Europe has {value} orders", kind=kind, value=value,
                 evidence=[ToolCallEvidence(type="tool_call", call_id=call_id, span=span)])


def test_unmodified_claim_passes_and_numeric_change_is_caught_at_l1():
    ledger = _ledger()
    claim = _claim("4100")
    assert asyncio.run(first_failing_level(claim, ledger, {}))[0] is None
    bad = inject(claim, ledger, "numeric_change", "naive")
    assert bad.value != "4100"
    assert asyncio.run(first_failing_level(bad, ledger, {}))[0] == "L1"


def test_fabricated_row_and_false_done_are_caught():
    ledger = _ledger()
    claim = _claim(["Ethan", "Noah"], call_id="call_names", span="customers Ethan Noah")
    for fault, variant in (("fabricated_row", "naive"), ("false_done", "naive"), ("false_done", "consistent")):
        bad = inject(claim, ledger, fault, variant)
        assert asyncio.run(first_failing_level(bad, ledger, {}))[0] == "L1", fault


def test_wrong_pointer_naive_caught_consistent_only_by_value():
    ledger = _ledger()
    claim = _claim("4100")
    naive = inject(claim, ledger, "wrong_pointer", "naive")
    assert naive.evidence[0].call_id != "call_count"
    assert asyncio.run(first_failing_level(naive, ledger, {}))[0] == "L1"
    consistent = inject(claim, ledger, "wrong_pointer", "consistent")
    assert asyncio.run(first_failing_level(consistent, ledger, {}))[0] == "L1"  # 4100 is not in the new call
    words = _claim("several orders")  # multi-word value: nothing a string match can decide
    moved = inject(words, ledger, "wrong_pointer", "consistent")
    assert asyncio.run(first_failing_level(moved, ledger, {}))[0] is None


def test_study_reports_every_fault_without_l3():
    ledger = _ledger()
    report = asyncio.run(study([Item("r/t", 0, _claim("4100"), ledger)]))
    assert report["levels"] == ["L0", "L1", "L2"]
    assert set(report["faults"]) == {f"{f}/{v}" for f, v in FAULTS}
    assert report["unmodified"]["cum_L2"] == 0.0
