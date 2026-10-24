"""scripts/recheck_pilot.py on a two-return run directory shaped like runs/pilot_deepseek."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from evcarry.ledger.ledger import Ledger
from evcarry.ledger.store import write_ledger
from evcarry.paths import repo_root

TASK = "sf-sales-region-analysis-gsheet-email"


def _script():
    spec = importlib.util.spec_from_file_location("recheck_pilot", repo_root() / "scripts" / "recheck_pilot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _claim(claim_id: str, value: str, *pointers: tuple[str, str]) -> dict:
    return {"claim_id": claim_id, "text": "t", "kind": "action_done", "value": value,
            "evidence": [{"type": "tool_call", "call_id": call_id, "span": span} for call_id, span in pointers]}


def _run_dir(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    ledger = Ledger()
    ledger.record(call_id="send", tool_name="emails-send_email", arguments={"subject": "Europe", "body": "x"},
                  result_text="Email sent successfully to europe.manager@company.com", read_only=False,
                  readback_tool="emails-search_emails", readback_arguments={"query": "Europe", "folder": "Sent"})
    ledger.record(call_id="read", tool_name="emails-read_email", arguments={"email_id": "5954"},
                  result_text="Subject: Europe\nTotal Revenue: 617263.14", read_only=True)
    write_ledger(run / "ledgers" / f"{TASK}__evidence_risk__1.json", ledger)
    ledger.record(call_id="gt", tool_name="python_execute", arguments={"code": "open('/x/groundtruth_workspace/a.xlsx')"},
                  result_text="FileNotFoundError", read_only=False)
    write_ledger(run / "ledgers" / f"{TASK}__sa__1.json", ledger)
    good = _claim("c1", "617263.14", ("send", "Email sent successfully"), ("read", "Total Revenue: 617263.14"))
    fabricated = _claim("c2", "999.99", ("read", "Total Revenue: 999.99"))
    request = "CLAIM\n...\nEVIDENCE\n--- call send: emails-send_email {}\nEmail sent\n--- read-back of send now (emails-search_emails):\n1. Subject: Europe\n"
    evidence_events = [
        {"type": "run_start", "task": TASK, "arm": "evidence_risk", "seed": 1},
        # Return 0: the pilot rejected c1 on the send pointer (the per-pointer L1 rule); c2 is fabricated.
        {"type": "check", "return_index": 0, "failures": [
            {"claim_id": "c1", "reason": "L1: numeric value 617263.14 is not in the logged result"},
            {"claim_id": "c2", "reason": "L1: quoted span is not in the logged result"}], "checks": []},
        {"type": "judge_call", "return_index": 0, "claim_id": "c1", "claim": good, "request": request},
        {"type": "judge_call", "return_index": 0, "claim_id": "c2", "claim": fabricated, "request": request},
        # Return 1: rejected only by an unsampled claim whose other pointer passed L1.
        {"type": "check", "return_index": 1, "failures": [
            {"claim_id": "c9", "reason": "L1: numeric value 3914 is not in the logged result"}],
         "checks": [{"claim_id": "c9", "level": "L1", "ok": True, "evidence_index": 1}]},
        {"type": "run_end", "task": TASK, "arm": "evidence_risk", "seed": 1, "passed": False, "eval_passed": False},
    ]
    sa_events = [
        {"type": "run_start", "task": TASK, "arm": "sa", "seed": 1},
        {"type": "run_end", "task": TASK, "arm": "sa", "seed": 1, "passed": False, "eval_passed": True, "contaminated": True},
    ]
    for arm, events in (("evidence_risk", evidence_events), ("sa", sa_events)):
        path = run / f"{TASK}__{arm}__model__1.jsonl"
        path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
    return run


async def test_recheck_clears_the_check_bugs_and_keeps_the_fabrication(tmp_path: Path) -> None:
    script = _script()
    result = await script.recheck(_run_dir(tmp_path), {("sf-sales", 0): "D", ("sf-sales", 1): "D"})
    outcomes = {row["return"]: row for row in result["returns"]}
    assert outcomes[0]["outcome"] == "remains" and outcomes[0]["class"] == "D"
    statuses = {claim["claim_id"]: claim["status"] for claim in outcomes[0]["claims"]}
    assert statuses == {"c1": "cleared", "c2": "fails L0-L2"}
    assert outcomes[1]["outcome"] == "cleared"
    assert outcomes[1]["claims"] == [{"claim_id": "c9", "old": ["L1: numeric value 3914 is not in the logged result"],
                                      "status": "unsampled cleared", "new": []}]


def test_contamination_is_rescored_in_two_tiers(tmp_path: Path) -> None:
    rows = {row["arm"]: row for row in _script().contamination_table(_run_dir(tmp_path))}
    assert rows["sa"]["tier"] == "attempted" and rows["sa"]["passed_registered"] is False
    assert rows["sa"]["passed_obtained_only"] is True
    assert rows["evidence_risk"]["tier"] == "none"


def test_appendix_a_is_read_from_the_diagnosis() -> None:
    classes = _script().appendix_classes(repo_root() / "docs" / "PILOT_DIAGNOSIS.md")
    assert len(classes) == 64 and classes[("terminal", 2)] == "A" and classes[("sf-sales", 8)] == "D"
