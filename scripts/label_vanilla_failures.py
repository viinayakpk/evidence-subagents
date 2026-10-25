"""List vanilla_mad failures in run directories with their grader evidence, joined to hand labels.

Free and offline: reads traces only. Labels and the codebook are in validation/vanilla_failure_labels.json
and docs/FAILURE_CODEBOOK.md; a failure without a label is printed as UNLABELLED so none is skipped.

    uv run python scripts/label_vanilla_failures.py runs/pilot_deepseek runs/dev_verify_v1 runs/dev_verify_v2
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "validation" / "vanilla_failure_labels.json"


def failures(run_dir: Path, arm: str) -> list[dict]:
    out = []
    for path in sorted(run_dir.glob(f"*__{arm}__*.jsonl")):
        events = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        end = next((e for e in events if e.get("type") == "run_end"), None)
        if end is None or end.get("passed"):
            continue
        tail = (end.get("eval") or {}).get("stdout_tail") or ""
        out.append({
            "run": run_dir.name, "task": end["task"], "seed": end.get("seed", 1), "status": end.get("status"),
            "eval_passed": end.get("eval_passed"), "contaminated": end.get("contaminated"),
            "subagent_tool_calls": sum(1 for e in events if e.get("type") == "tool_call" and e.get("agent") != "main"),
            "subagent_errors": sum(1 for e in events if e.get("type") == "subagent_error"),
            "grader_fails": [l.strip()[:160] for l in tail.splitlines() if "FAIL" in l][:4],
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--arm", default="vanilla_mad")
    args = parser.parse_args()
    labels = {(l["run"], l["task"], l["seed"]): l for l in json.loads(LABELS.read_text())["labels"]}
    counts: Counter = Counter()
    for run_dir in args.run_dirs:
        for row in failures(run_dir, args.arm):
            label = labels.get((row["run"], row["task"], row["seed"]))
            name = label["label"] if label else "UNLABELLED"
            counts[name] += 1
            print(json.dumps({**row, "label": name, "subtype": label and label["subtype"],
                              "mast": label and label["mast"], "ms_eval_dfp": label and label["ms_eval_dfp"]}))
    print(json.dumps({"counts": dict(counts)}))


if __name__ == "__main__":
    main()
