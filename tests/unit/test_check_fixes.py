"""Regression tests for the L0-L3 fixes of 2026-10-25, built from claims and ledger entries of the pilot.

Each case names the pilot return it comes from (docs/PILOT_DIAGNOSIS.md, Appendix A). Ledger texts are
copied from runs/pilot_deepseek/ledgers; long ones are shortened where the cut does not matter.
"""

from __future__ import annotations

from types import SimpleNamespace

from evcarry.env.gym_tools import GymToolPolicy
from evcarry.ledger.ledger import Ledger
from evcarry.protocol.audit import audit_return, evidence_blob, excerpt
from evcarry.protocol.checks import (
    CheckContext,
    as_number,
    clean_span,
    run_l0_l1,
    span_in_text,
    text_contains,
    value_supported,
)
from evcarry.protocol.judge import LexicalJudge, judge_request
from evcarry.protocol.replayer import replay_claim
from evcarry.protocol.schema import Claim, SubagentReturn, ToolCallEvidence

# sf-sales-region-analysis-gsheet-email, evidence_risk, return 8, claim c1.
SEND_ID = "chatcmpl-tool-a07a0c660acff998"
READ_ID = "chatcmpl-tool-9c91355be7818991"
SEND_RESULT = "Email sent successfully to europe.manager@company.com"
SEND_ARGS = {
    "body": "Hello,\n\nPlease find the all-time sales performance figures for Europe below:\n\nTotal Revenue: 617263.14\n"
    "Order Count: 3914\nAverage Order Value: 157.71\nRevenue Share: 21.20%\n\nBest regards,\nSales Analytics Team",
    "subject": "Europe Sales Performance Report",
    "to": "europe.manager@company.com",
}
READ_RESULT = (
    "Email ID: 5954\nSubject: Europe Sales Performance Report\nFrom: PG Email User <user@example.com>\n"
    "To: europe.manager@company.com\nDate: 2026-10-24T20:40:22.693401+00:00\n"
    "Message-ID: <179287442267.90.10295360784264183213@1580e6f254eb>\n\nText Content:\nHello,\n\n"
    "Please find the all-time sales performance figures for Europe below:\n\nTotal Revenue: 617263.14\n"
    "Order Count: 3914\nAverage Order Value: 157.71\nRevenue Share: 21.20%\n\nBest regards,\nSales Analytics Team\n\n"
)


def _ptr(call_id: str, span: str | None = None) -> ToolCallEvidence:
    return ToolCallEvidence(type="tool_call", call_id=call_id, span=span)


def _claim(claim_id: str, kind: str, value: object, *pointers: ToolCallEvidence, text: str = "claim") -> Claim:
    return Claim(claim_id=claim_id, text=text, kind=kind, value=value, evidence=list(pointers))


def _failures(claim: Claim, ledger: Ledger) -> list[str]:
    return [f"{r.level}: {r.reason}" for r in run_l0_l1(claim, CheckContext(ledger=ledger)) if not r.ok]


class _Caller:
    def __init__(self, texts: dict[str, str]) -> None:
        self.texts = texts
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, arguments: dict) -> str:
        self.calls.append((name, dict(arguments)))
        return self.texts[name]


def _sales_ledger() -> Ledger:
    ledger = Ledger()
    ledger.record(call_id=SEND_ID, tool_name="emails-send_email", arguments=SEND_ARGS, result_text=SEND_RESULT,
                  read_only=False, readback_tool="emails-search_emails",
                  readback_arguments={"query": SEND_ARGS["subject"], "folder": "Sent"}, server="emails",
                  effect={"subject": SEND_ARGS["subject"]})
    ledger.record(call_id=READ_ID, tool_name="emails-read_email", arguments={"email_id": "5954"}, result_text=READ_RESULT,
                  read_only=True, server="emails")
    return ledger


# P1-1 -------------------------------------------------------------------------------------------------


def test_l1_value_needs_one_cited_result_not_every_one() -> None:
    """sf-sales r8 c1: the send result lacks the number, the read-back shows it; L1 failed it per pointer."""
    claim = _claim(
        "c1", "action_done", "617263.14",
        _ptr(SEND_ID, SEND_RESULT),
        _ptr(READ_ID, "To: europe.manager@company.com"),
        _ptr(READ_ID, "Total Revenue: 617263.14\nOrder Count: 3914\nAverage Order Value: 157.71\nRevenue Share: 21.20%"),
    )
    assert _failures(claim, _sales_ledger()) == []
    wrong = claim.model_copy(update={"value": "617263.15"})
    assert _failures(wrong, _sales_ledger()) == ["L1: value 617263.15 is not in any cited result"]


def test_a_write_pointer_is_checked_against_what_was_written() -> None:
    """A claim citing only the send call: the number is in the send's arguments (the e-mail body)."""
    claim = _claim("c1", "action_done", "617263.14", _ptr(SEND_ID, SEND_RESULT))
    assert _failures(claim, _sales_ledger()) == []


def test_a_span_from_a_crashed_python_call_is_still_caught() -> None:
    """terminal-arxiv r2 (genuine catch): the span is in the code, the output is a traceback."""
    ledger = Ledger()
    ledger.record(
        call_id="chatcmpl-tool-aecd125430ff5010", tool_name="python_execute",
        arguments={"code": "raw = b'/F2 18 Tf 22 TL'\ndec = zlib.decompress(base64.a85decode(raw, adobe=False))"},
        result_text="=== STDERR ===\nTraceback (most recent call last):\n  File \"x.py\", line 11, in <module>\n"
        "    dec = zlib.decompress(base64.a85decode(raw, adobe=False))\nValueError: Non-Ascii85 digit found: ~\n"
        "=== INFO ===\nReturn code: 1\nTime: 0.07s / 30s limit",
        read_only=False,
    )
    claim = _claim("c3", "fact", None, _ptr("chatcmpl-tool-aecd125430ff5010", "/F2 18 Tf 22 TL"))
    assert _failures(claim, ledger) == ["L1: quoted span is not in the logged result"]


# P1-5 -------------------------------------------------------------------------------------------------


def test_one_value_rule() -> None:
    assert as_number("21.20%") == as_number(21.2) and as_number("$1,234.50") == as_number(1234.5)
    assert as_number("2026-10-31") is None and as_number(True) is None
    assert value_supported("21.20%", ["Revenue Share: 21.20%"])[0]
    assert value_supported("europe.manager@company.com", [READ_RESULT])[0]
    assert value_supported("EUROPE.MANAGER@COMPANY.COM", [READ_RESULT])[0]  # case-insensitive
    # One-token strings must stand alone: US is not found in "status", 55 is not 5.
    assert not value_supported("US", ['{"status": "ok"}'])[0]
    assert text_contains("ID 55", "ID 55") and not text_contains("ID 55 x", "ID 5")
    # Multi-word strings, dicts and booleans are left to L3 (summary strings and derived keys failed L2 in the pilot).
    assert value_supported("read-only exploration only", ["nothing"])[0]
    assert value_supported({"directories": 0, "files": 0}, ["empty"])[0]
    assert value_supported(True, ["no"])[0]
    # A list is checked item by item, across all cited texts.
    assert value_supported(["5954", "5955"], ["ID: 5954", "ID: 5955"])[0]
    ok, reason = value_supported(["5954", "5956"], ["ID: 5954", "ID: 5955"])
    assert not ok and "5956" in reason


def test_l1_now_checks_one_token_strings_on_every_claim() -> None:
    """Before, L1 skipped every non-numeric value; a fabricated ID passed L1 and waited for a sampled L2."""
    ledger = _sales_ledger()
    assert _failures(_claim("c", "fact", "5954", _ptr(READ_ID)), ledger) == []
    assert _failures(_claim("c", "fact", "5999", _ptr(READ_ID)), ledger) == ["L1: value 5999 is not in any cited result"]
    assert _failures(_claim("c", "derived", "5999", _ptr(READ_ID)), ledger) == []


# P1-6 -------------------------------------------------------------------------------------------------


def test_spans_drop_harness_notes_and_whitespace() -> None:
    """canvas-quiz r2/r6: spans quoting the truncation note the model saw; whitespace-collapsed table rows."""
    text = "id  name     size\n23  quiz.csv 1824\n24  roster.csv 990\n" + "x" * 50
    shown_note = '\n...[truncated: showing 8000 of 279168 chars. Call view_overlong_output with call_id "c9" and page=1 for the next part.]'
    assert span_in_text("24  roster.csv 990" + shown_note, text)
    assert span_in_text("[call_id: c9]\nid name size", text)
    assert span_in_text("id name size 23 quiz.csv 1824", text)  # collapsed table whitespace
    assert span_in_text('"23  quiz.csv"', text)  # one pair of quotes around the quote
    assert span_in_text("caf\\u00e9", "menu: café")  # an escaped character
    assert span_in_text(shown_note, text)  # nothing left to quote
    assert not span_in_text("25 grades.csv", text)
    assert clean_span("c9 page 1 of 3 (chars 8000-16000 of 279168)\nrow") == "row"


# P1-2 -------------------------------------------------------------------------------------------------


async def test_l2_write_claim_uses_the_logged_write_plus_the_read_back() -> None:
    """wc-shipping r4 c1: the value is the send's success text, which the Sent-folder search never shows."""
    ledger = Ledger()
    ledger.record(
        call_id="chatcmpl-tool-8796b2a61607454e", tool_name="emails-send_email",
        arguments={"body": "Shipping Zone Configuration Review\n\n- California: 2 shipping methods", "subject": "Shipping Zone Configuration Review",
                   "to": "logistics@shop.com"},
        result_text="Email sent successfully to logistics@shop.com", read_only=False,
        readback_tool="emails-search_emails", readback_arguments={"query": "Shipping Zone Configuration Review", "folder": "Sent"},
        server="emails", effect={"subject": "Shipping Zone Configuration Review"},
    )
    readback = (
        "Search query: Shipping Zone Configuration Review\nFolder: Sent\nPage: 1/1\nTotal results: 2\n\n"
        "1. ID: 5950\n   Subject: Shipping Zone Configuration Review\n   From: PG Email User <user@example.com>\n"
        "   Date: 2026-10-24T20:57:34.674922+00:00\n\n2. ID: 5949\n   Subject: Shipping Zone Configuration Review\n"
    )
    claim = _claim("c1", "action_done", "Email sent successfully to logistics@shop.com",
                   _ptr("chatcmpl-tool-8796b2a61607454e", "Email sent successfully to logistics@shop.com"))
    caller = _Caller({"emails-search_emails": readback})
    result = await replay_claim(claim, ledger, caller)
    assert result.ok, result.reason
    assert caller.calls == [("emails-search_emails", {"query": "Shipping Zone Configuration Review", "folder": "Sent"})]
    # The write's effect field must show in the read-back: a send that never reached the Sent folder fails.
    empty = await replay_claim(claim, ledger, _Caller({"emails-search_emails": "Total results: 0"}))
    assert not empty.ok and "does not show the write's subject" in empty.reason


def test_write_read_backs_for_calendar_notion_and_sheets() -> None:
    policy = GymToolPolicy.load()

    def meta(server: str, tool: str):
        return policy.meta(server, SimpleNamespace(name=tool, description=tool, input_schema={"type": "object"}, annotations=None))

    event = meta("google_calendar", "create_event")
    created = "Event created with ID: 04645dc1-0134-40f6-83c3-efa7ce5233b3\nTitle: CCC Module Review Meeting"
    assert event.readback_for({"summary": "CCC Module Review Meeting"}, created) == (
        "google_calendar-get_event", {"eventId": "04645dc1-0134-40f6-83c3-efa7ce5233b3"}
    )
    assert event.readback_for({"summary": "x"}, "Error: calendar not found") is None  # nothing to read back
    assert event.effect_for({"summary": "CCC Module Review Meeting", "description": "agenda"}) == {"summary": "CCC Module Review Meeting"}
    page = meta("notion", "API-post-page")
    assert page.readback_for({}, '{"id":"17a1370e-36fc-41ad-9c6a-67cc36941c47","object":"page"}') == (
        "notion-API-retrieve-a-page", {"page_id": "17a1370e-36fc-41ad-9c6a-67cc36941c47"}
    )
    cells = meta("google_sheet", "update_cells")
    assert cells.readback_for({"spreadsheet_id": "s1", "sheet": "Regional Performance", "range": "A1:E6", "data": []}) == (
        "google_sheet-get_sheet_data", {"spreadsheet_id": "s1", "sheet": "Regional Performance", "range": "A1:E6"}
    )
    send = meta("emails", "send_email")
    assert send.effect_for({"subject": "Report", "to": "a@b.c"}) == {"subject": "Report"}


# P1-3 -------------------------------------------------------------------------------------------------


async def test_l2_reads_compare_the_span_and_mask_the_canvas_clock() -> None:
    """canvas-module r0 and canvas-quiz r0: canvas_health_check answered with a new timestamp."""
    volatile = list(GymToolPolicy.load().volatile["canvas"])
    first = '{\n  "status": "ok",\n  "timestamp": "2026-10-24T19:52:32.453Z",\n  "user": {\n    "id": 1,\n    "name": "Dr. Oliver Murphy"\n  }\n}'
    later = first.replace("19:52:32.453Z", "19:58:50.696Z")
    ledger = Ledger()
    ledger.record(call_id="h", tool_name="canvas-canvas_health_check", arguments={}, result_text=first, read_only=True,
                  volatile=volatile, server="canvas")
    span_claim = _claim("c1", "fact", 1, _ptr("h", '"name": "Dr. Oliver Murphy"'))
    clock_claim = _claim("c2", "fact", None, _ptr("h", '"timestamp": "2026-10-24T19:52:32.453Z"'))
    caller = _Caller({"canvas-canvas_health_check": later})
    assert (await replay_claim(span_claim, ledger, caller)).ok
    assert (await replay_claim(clock_claim, ledger, caller)).ok  # the clock is masked on both sides


async def test_l2_excuses_drift_only_when_the_run_wrote_after_the_read() -> None:
    """canvas-module r16/r17, sf-sales r11: folder counts changed by the sub-agent's own later sends."""
    before, after = "Folders:\n- INBOX (12)\n- Sent (3)", "Folders:\n- INBOX (12)\n- Sent (5)"
    claim = _claim("c1", "fact", 3, _ptr("f", "Sent (3)"))
    caller = _Caller({"emails-get_folders": after})
    quiet = Ledger()
    quiet.record(call_id="f", tool_name="emails-get_folders", arguments={}, result_text=before, read_only=True, server="emails")
    drift = await replay_claim(claim, quiet, caller)
    assert not drift.ok and "cited span is not in the replay" in drift.reason
    busy = Ledger()
    busy.record(call_id="f", tool_name="emails-get_folders", arguments={}, result_text=before, read_only=True, server="emails")
    busy.record(call_id="s", tool_name="emails-send_email", arguments={"subject": "x"}, result_text="Email sent", read_only=False)
    stateful = await replay_claim(claim, busy, caller)
    assert stateful.ok and stateful.stateful and "a later write changed" in stateful.stateful[0]
    # Only a write that can change what was read counts: not claim_done, not a write on another server's state.
    for tool, server, excused in (
        ("claim_done", None, False),
        ("notion-API-post-page", "notion", False),
        ("python_execute", None, True),  # code can change files and the database
    ):
        other = Ledger()
        other.record(call_id="f", tool_name="emails-get_folders", arguments={}, result_text=before, read_only=True, server="emails")
        other.record(call_id="w", tool_name=tool, arguments={}, result_text="ok", read_only=False, server=server)
        assert (await replay_claim(claim, other, caller)).ok is excused, tool
    # Files: an Excel write changes what a filesystem read showed.
    tree = Ledger()
    tree.record(call_id="t", tool_name="filesystem-directory_tree", arguments={}, result_text='[{"name": "a.txt"}]',
                read_only=True, server="filesystem")
    tree.record(call_id="x", tool_name="excel-create_workbook", arguments={"filepath": "b.xlsx"}, result_text="Created",
                read_only=False, server="excel")
    tree_claim = _claim("c2", "fact", "a.txt", _ptr("t", '[{"name": "a.txt"}]'))
    assert (await replay_claim(tree_claim, tree, _Caller({"filesystem-directory_tree": '[{"name": "a.txt"}, {"name": "b.xlsx"}]'}))).ok


# P1-4 -------------------------------------------------------------------------------------------------


async def test_l2_counts_the_logged_text_of_a_non_replayable_pointer() -> None:
    """sf-hr-compensation r12: 11 is in the python output, not in the replayed Notion read."""
    ledger = Ledger()
    ledger.record(call_id="py", tool_name="python_execute", arguments={"code": "print(...)"},
                  result_text="=== STDOUT ===\nBlocks on final page: 11 | expected: 11", read_only=False)
    ledger.record(call_id="nb", tool_name="notion-API-get-block-children", arguments={"block_id": "p1"},
                  result_text='{"results": [{"type": "paragraph"}], "has_more": false}', read_only=True, server="notion")
    claim = _claim("c1", "fact", 11, _ptr("py", "Blocks on final page: 11"), _ptr("nb", '"has_more": false'))
    result = await replay_claim(claim, ledger, _Caller({"notion-API-get-block-children": '{"results": [{"type": "paragraph"}], "has_more": false}'}))
    assert result.ok and result.unverifiable and "python_execute" in result.unverifiable[0]
    absent = claim.model_copy(update={"value": 12})
    assert not (await replay_claim(absent, ledger, _Caller({"notion-API-get-block-children": "{}"}))).ok


# P1-7 -------------------------------------------------------------------------------------------------


def test_judge_evidence_keeps_the_cited_passage_and_marks_every_gap() -> None:
    """canvas-quiz r7: the old 12,000-character middle cut removed account 4's output, and the judge read it as missing."""
    body = "".join(f"account {i}: " + "row, " * 400 + "\n" for i in range(1, 5))
    middle = body.index("account 3")
    text = body[:middle] + "TOTAL QUIZZES: 87\n" + body[middle:]
    shown = excerpt(text, ["TOTAL QUIZZES: 87"], 4000)
    assert "TOTAL QUIZZES: 87" in shown and shown.startswith(text[:1500])
    assert "characters not shown ...]" in shown and len(shown) < 4200
    # Every left-out character is accounted for by a marker.
    gaps = sum(int(part.split(" characters", 1)[0]) for part in shown.split("[... ")[1:])
    assert gaps == len(text) - 1500 - (min(len(text), middle + len("TOTAL QUIZZES: 87") + 700) - (middle - 700))
    assert excerpt("short", ["x"], 100) == "short"
    ledger = Ledger()
    ledger.record(call_id="big", tool_name="canvas-canvas_list_account_courses", arguments={"account_id": 3},
                  result_text=text, read_only=True)
    blob = evidence_blob(_claim("c7", "fact", 87, _ptr("big", "TOTAL QUIZZES: 87")), ledger, None, max_chars=3000)
    assert "TOTAL QUIZZES: 87" in blob and blob.startswith("--- call big: canvas-canvas_list_account_courses")


def test_judge_request_has_no_middle_cut() -> None:
    claim = _claim("c1", "fact", 1, _ptr("x"))
    request = judge_request(claim, "H" * 900 + "T" * 900, 400)
    assert "H" * 400 + "\n[... 1400 characters not shown ...]" in request and "T" not in request.split("EVIDENCE")[1]


# P1-11 ------------------------------------------------------------------------------------------------


async def test_claims_that_failed_l0_to_l2_are_not_sent_to_the_judge() -> None:
    judged: list[str] = []

    class Judge(LexicalJudge):
        def judge(self, claim, evidence_text):
            judged.append(claim.claim_id)
            return super().judge(claim, evidence_text)

    ledger = _sales_ledger()
    claims = [
        _claim("ok", "fact", "5954", _ptr(READ_ID, "Email ID: 5954")),
        _claim("bad", "fact", "5999", _ptr(READ_ID, "Email ID: 5954")),
    ]
    report = await audit_return(SubagentReturn(subagent_id="researcher", status="complete", claims=claims), ledger=ledger,
                                caller=_Caller({"emails-read_email": READ_RESULT}), checks=("L0", "L1", "L2", "L3"),
                                sampler="risk", rate=1.0, seed=1, judge=Judge())
    assert judged == ["ok"] and report.skipped_l3 == ["bad"]
    assert report.failed_claim_ids() == {"bad"}
    assert report.as_dict()["skipped_l3"] == ["bad"]
