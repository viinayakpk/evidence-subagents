"""Primary endpoint report (amendments A15-A17): paired success, clustered bootstrap, pass^k, success per
reference dollar, attributed vs false-alarm rejections and the exclusion sensitivity. Reads traces only.

    uv run python scripts/paired_report.py runs/pilot_deepseek --classes docs/PILOT_DIAGNOSIS.md \
        --exclude tasks/grader_exclusions.txt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evcarry.eval.paired import paired_difference, pass_hat_k, success_per_dollar  # noqa: E402
from evcarry.eval.rejections import false_alarm_runs, split_rejections, with_false_alarms_as_failures  # noqa: E402

ARMS = ("sa", "vanilla_mad", "self_verify", "evidence_risk")
_ROW = re.compile(r"^\| ([\w-]+) \| (\d+) \| (redelegate|fallback) \| ([ABCD]) \|", re.MULTILINE)


def load_rows(run_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(run_dir.glob("*.jsonl")):
        events = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        end = next((e for e in events if e.get("type") == "run_end"), None)
        if end is None:
            continue
        reference = sum(float(e.get("price_table_cost_usd") or 0.0) for e in events if e.get("type") == "llm_call")
        rows.append({"task": end["task"], "arm": end["arm"], "seed": end.get("seed", 1),
                     "passed": end.get("passed"), "cost_usd": end.get("cost_usd"), "reference_cost_usd": reference})
    return rows


def short_to_task(rows: list[dict]) -> dict[str, str]:
    out = {}
    for row in rows:
        parts = row["task"].split("-")
        out.setdefault("-".join(parts[:2]), row["task"])
    out.setdefault("sf-hr-comp", "sf-hr-compensation-pdf-notion")
    out.setdefault("terminal", "terminal-arxiv-latex-excel-word-notion")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--classes", type=Path, help="markdown with an Appendix A table of rejection classes")
    parser.add_argument("--exclude", type=Path, help="task list (one per line, # comments)")
    parser.add_argument("--n-boot", type=int, default=10000)
    args = parser.parse_args()
    rows = load_rows(args.run_dir)
    exclude = frozenset()
    if args.exclude and args.exclude.is_file():
        exclude = frozenset(l.split("#")[0].strip() for l in args.exclude.read_text().splitlines() if l.split("#")[0].strip())
    report: dict = {"runs": len(rows), "excluded_tasks": sorted(exclude), "arms": {}, "vs_vanilla": {}}
    for arm in ARMS:
        report["arms"][arm] = {"pass^1": pass_hat_k(rows, arm, 1), "pass^2": pass_hat_k(rows, arm, 2),
                               "success_per_ref_dollar": success_per_dollar(rows, arm)}
    for arm in ARMS:
        if arm == "vanilla_mad":
            continue
        report["vs_vanilla"][arm] = {
            "all_tasks": paired_difference(rows, "vanilla_mad", arm, n_boot=args.n_boot),
            "minus_exclusions": paired_difference(rows, "vanilla_mad", arm, n_boot=args.n_boot, exclude=exclude),
        }
    if args.classes:
        names = short_to_task(rows)
        rej = [{"task": names.get(t, t), "seed": 1, "return": int(r), "class": c}
               for t, r, _a, c in _ROW.findall(args.classes.read_text(encoding="utf-8"))]
        report["evidence_rejections"] = split_rejections(r["class"] for r in rej)
        lower = with_false_alarms_as_failures(rows, "evidence_risk", false_alarm_runs(rej))
        report["vs_vanilla"]["evidence_risk"]["false_alarms_as_failures"] = paired_difference(
            lower, "vanilla_mad", "evidence_risk", n_boot=args.n_boot)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
