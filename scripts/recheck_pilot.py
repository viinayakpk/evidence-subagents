"""Re-run the fixed checks over a finished run directory, offline: no model, no Docker, no API call.

    uv run python scripts/recheck_pilot.py runs/pilot_deepseek [--appendix docs/PILOT_DIAGNOSIS.md] [--json out.json]

Two parts.

1. Evidence-arm rejections under the L0-L2 fixes of 2026-10-25 (docs/STATUS_fixes.md). For every
   ``check`` event of an evidence trace that had a failure (a rejected return):

   - A claim the pilot sampled for L2/L3 is known in full (its ``judge_call`` carries the claim). It
     is checked again with ``protocol.checks.run_l0_l1`` and ``protocol.replayer.replay_claim`` on the
     run's ledger. Replays are offline: a read answers with its logged text (drift cannot be observed
     again) and a write's read-back with the read-back text the pilot's judge request shows. Ledgers
     written before 2026-10-25 have no effect fields and no read-backs for calendar, Notion or Sheets
     writes, so those parts of the new L2 cannot run here.
   - The pilot's L3 verdict on such a claim is kept, except that a claim now failing L0-L2 is not
     judged (P1-11), and a verdict given on evidence the old 12,000-character middle cut had cut
     ("characters omitted" in the request) is marked ``re-judge``: it needs a new judge call.
   - A claim that was not sampled is known only by its check results. An L1 value failure is cleared
     when another pointer of the same claim passed L1 (P1-1); any other failure is kept.

   A rejected return is ``cleared`` when none of its claims still fails, ``re-judge`` when only
   re-judge verdicts stand in the way, else ``remains``. Sampled claims of accepted returns are checked
   too, to count rejections the new rules would add.

2. Contamination in two tiers (``env.contamination``) for every finished run, against the pilot's flag.

Prints markdown; ``--json`` also writes every per-claim and per-run row.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evcarry.env.contamination import contamination, removed_fingerprints, tier  # noqa: E402
from evcarry.ledger.store import read_ledger  # noqa: E402
from evcarry.protocol.checks import CheckContext, run_l0_l1  # noqa: E402
from evcarry.protocol.replayer import replay_claim  # noqa: E402
from evcarry.protocol.schema import Claim  # noqa: E402

_READBACK = re.compile(r"--- read-back of (\S+) now \(([^)]*)\):\n(.*?)(?=\n--- call |\n--- read-back of |\Z)", re.DOTALL)
# Short task names used in docs/PILOT_DIAGNOSIS.md Appendix A.
_SHORT = {
    "canvas-module-completion-notion-gcal": "canvas-module",
    "canvas-quiz-analysis-ppt": "canvas-quiz",
    "fetch-sf-hr-training-notion-gcal-excel": "fetch-sf-hr",
    "howtocook-weekly-gsheet-gcal": "howtocook",
    "sf-hr-compensation-pdf-notion": "sf-hr-comp",
    "sf-sales-region-analysis-gsheet-email": "sf-sales",
    "terminal-arxiv-latex-excel-word-notion": "terminal",
    "wc-shipping-analysis": "wc-shipping",
}


def _events(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def appendix_classes(path: Path | None) -> dict[tuple[str, int], str]:
    """``(short task, return index) -> class`` from the diagnosis' Appendix A table."""
    if path is None or not path.is_file():
        return {}
    classes: dict[tuple[str, int], str] = {}
    in_table = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## Appendix A"):
            in_table = True
            continue
        if in_table and line.startswith("## "):
            break
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if in_table and len(cells) == 5 and cells[1].isdigit():
            classes[(cells[0], int(cells[1]))] = cells[3]
    return classes


class OfflineCaller:
    """A replay answers with the logged text; a read-back with the text the pilot's judge request shows."""

    def __init__(self, ledger: Any, readbacks: dict[str, str]) -> None:
        self.reads = {(entry.tool_name, json.dumps(entry.arguments, sort_keys=True, default=str)): entry.result_text
                      for entry in ledger.entries.values() if entry.read_only}
        self.readbacks = {}
        for entry in ledger.entries.values():
            if entry.readback_tool and entry.call_id in readbacks:
                key = (entry.readback_tool, json.dumps(entry.readback_arguments or {}, sort_keys=True, default=str))
                self.readbacks[key] = readbacks[entry.call_id]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        key = (name, json.dumps(arguments, sort_keys=True, default=str))
        if key in self.reads:
            return self.reads[key]
        if key in self.readbacks:
            return self.readbacks[key]
        raise LookupError(f"no offline answer for {name}")


async def _new_l012(claim: Claim, ledger: Any, readbacks: dict[str, str]) -> tuple[list[str], str | None]:
    """The new L0-L2 failures of a known claim, and a note when L2 could not run offline."""
    failures = [f"{r.level}: {r.reason}" for r in run_l0_l1(claim, CheckContext(ledger=ledger)) if not r.ok]
    try:
        replay = await replay_claim(claim, ledger, OfflineCaller(ledger, readbacks))
    except LookupError as exc:
        return failures, f"L2 not checkable offline ({exc})"
    if not replay.ok:
        failures.append(f"L2: {replay.reason}")
    return failures, None


def _unsampled_status(claim_id: str, reasons: list[str], checks: list[dict[str, Any]]) -> tuple[str, list[str]]:
    """Kept or cleared, for a failing claim whose content the trace does not have."""
    l1_passed = any(c.get("claim_id") == claim_id and c.get("level") == "L1" and c.get("ok") for c in checks)
    kept = [reason for reason in reasons if not (reason.startswith("L1: numeric value") and l1_passed)]
    return ("unsampled kept" if kept else "unsampled cleared"), kept


async def recheck(run_dir: Path, classes: dict[tuple[str, int], str]) -> dict[str, Any]:
    returns: list[dict[str, Any]] = []
    added: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("*__evidence*.jsonl")):
        events = _events(path)
        start = next((e for e in events if e.get("type") == "run_start"), None)
        if start is None or not any(e.get("type") == "run_end" for e in events):
            continue
        task, arm, seed = start["task"], start["arm"], start["seed"]
        ledger_path = run_dir / "ledgers" / f"{task}__{arm}__{seed}.json"
        if not ledger_path.is_file():
            continue
        ledger = read_ledger(ledger_path)
        judged: dict[tuple[int, str], dict[str, Any]] = {}
        readbacks: dict[str, str] = {}
        for event in events:
            if event.get("type") == "judge_call" and event.get("claim"):
                judged[(int(event.get("return_index", -1)), str(event.get("claim_id")))] = event
                for call_id, _tool, text in _READBACK.findall(event.get("request") or ""):
                    readbacks.setdefault(call_id, text)
        for event in events:
            if event.get("type") != "check":
                continue
            index = int(event.get("return_index", -1))
            reasons: dict[str, list[str]] = defaultdict(list)
            for failure in event.get("failures") or []:
                reasons[str(failure.get("claim_id"))].append(str(failure.get("reason")))
            claims_here = {cid for (ri, cid) in judged if ri == index}
            rows = []
            for claim_id in sorted(set(reasons) | claims_here):
                old = reasons.get(claim_id, [])
                call = judged.get((index, claim_id))
                if call is None:
                    status, kept = _unsampled_status(claim_id, old, event.get("checks") or [])
                    rows.append({"claim_id": claim_id, "old": old, "status": status, "new": kept})
                    continue
                claim = Claim.model_validate(call["claim"])
                new, note = await _new_l012(claim, ledger, readbacks)
                old_l3 = [reason for reason in old if reason.startswith("L3:")]
                if new:
                    status = "fails L0-L2"
                elif old_l3 and "characters omitted" in (call.get("request") or ""):
                    status = "re-judge"
                elif old_l3:
                    status = "L3 kept"
                else:
                    status = "cleared"
                rows.append({"claim_id": claim_id, "old": old, "status": status, "new": new + old_l3 * (status == "L3 kept"),
                             "note": note})
            if not reasons:
                added.extend({"task": task, "return": index, **row} for row in rows if row["status"] == "fails L0-L2")
                continue
            failing = {row["status"] for row in rows if row["claim_id"] in reasons or row["status"] == "fails L0-L2"}
            if failing & {"fails L0-L2", "L3 kept", "unsampled kept"}:
                outcome = "remains"
            elif "re-judge" in failing:
                outcome = "re-judge"
            else:
                outcome = "cleared"
            short = _SHORT.get(task, task)
            returns.append({"task": short, "return": index, "class": classes.get((short, index), "?"), "outcome": outcome,
                            "claims": rows})
    return {"returns": returns, "added": added}


def contamination_table(run_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(run_dir.glob("*.jsonl")):
        events = _events(path)
        end = next((e for e in events if e.get("type") == "run_end"), None)
        if end is None:
            continue
        stem = f"{end['task']}__{end['arm']}__{end['seed']}"
        ledger_path = run_dir / "ledgers" / f"{stem}.json"
        if not ledger_path.is_file():
            continue
        found = contamination(read_ledger(ledger_path), task=end["task"],
                              fingerprints=removed_fingerprints(run_dir / "logs" / stem / "sealed_task"))
        grader = end.get("eval_passed", end.get("passed"))
        new_tier = tier(found)
        ceiling = end.get("status") == "cost_ceiling"
        rows.append({
            "task": end["task"], "arm": end["arm"], "seed": end["seed"], "grader": grader, "ceiling": ceiling,
            "old_flagged": bool(end.get("contaminated")), "tier": new_tier,
            "attempted": [f"{row['tool']} ({row['markers']})" for row in found["attempted"]],
            "obtained": [row["tool"] for row in found["obtained"]],
            "passed_old": end.get("passed"),
            "passed_registered": False if grader and new_tier != "none" else (None if grader is None else bool(grader) and not ceiling),
            "passed_obtained_only": False if grader and new_tier == "obtained" else (None if grader is None else bool(grader) and not ceiling),
        })
    return rows


def _md(header: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(" --- " for _ in header) + "|"]
    return "\n".join([*lines, *("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--appendix", type=Path, default=Path(__file__).resolve().parents[1] / "docs" / "PILOT_DIAGNOSIS.md")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result = asyncio.run(recheck(args.run_dir, appendix_classes(args.appendix)))
    returns = result["returns"]
    by_outcome = Counter(row["outcome"] for row in returns)
    print(f"## Rejected evidence returns re-checked offline ({len(returns)})\n")
    print(_md(["outcome", "returns"], [[key, value] for key, value in sorted(by_outcome.items())]))
    by_class: dict[str, Counter] = defaultdict(Counter)
    for row in returns:
        by_class[row["class"]][row["outcome"]] += 1
    print("\n" + _md(["class", *sorted(by_outcome)], [[cls, *(by_class[cls][key] for key in sorted(by_outcome))] for cls in sorted(by_class)]))
    print("\n" + _md(
        ["task", "return", "class", "outcome", "claims still failing (status: reasons)"],
        [[row["task"], row["return"], row["class"], row["outcome"],
          "; ".join(f"{c['claim_id']} {c['status']}: {' / '.join(c['new'])[:160]}" for c in row["claims"]
                    if c["status"] not in ("cleared", "unsampled cleared")) or "-"]
         for row in returns],
    ))
    print(f"\nSampled claims of accepted returns that the new L0-L2 would reject: {len(result['added'])}")
    for row in result["added"]:
        print(f"- {row['task']} r{row['return']} {row['claim_id']}: {' / '.join(row['new'])[:200]}")
    table = contamination_table(args.run_dir)
    print(f"\n## Contamination, two tiers ({len(table)} finished runs)\n")
    flagged = [row for row in table if row["old_flagged"] or row["tier"] != "none"]
    print(_md(
        ["task", "arm", "grader", "old flag", "tier", "attempted calls", "counted pass (registered)", "counted pass (obtained only)"],
        [[r["task"], r["arm"], r["grader"], r["old_flagged"], r["tier"], "; ".join(r["attempted"])[:200] or "-",
          r["passed_registered"], r["passed_obtained_only"]] for r in flagged],
    ))
    arms = sorted({row["arm"] for row in table})
    counts = [[arm,
               sum(1 for r in table if r["arm"] == arm and r["passed_old"]),
               sum(1 for r in table if r["arm"] == arm and r["passed_registered"]),
               sum(1 for r in table if r["arm"] == arm and r["passed_obtained_only"]),
               sum(1 for r in table if r["arm"] == arm)] for arm in arms]
    print("\n" + _md(["arm", "passed (pilot rule)", "passed (two tiers, registered)", "passed (obtained only)", "finished"], counts))
    if args.json:
        args.json.write_text(json.dumps({"recheck": result, "contamination": table}, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
