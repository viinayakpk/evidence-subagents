"""Write the pre-registered list of broken-grader tasks (amendment A17, 2026-10-25), or check it.

A task is listed when a static grader rule flags it: A13 (the grader imports a module the grading Python
lacks, ``task_rules.grader_missing_modules``) or the docx-table rule (``task_rules.docx_table_defect``).
The primary analysis keeps every headline60 task; the sensitivity analysis drops listed tasks. The list
is written from code before any headline run and is not edited after outcomes are seen.

    uv run python scripts/grader_exclusions.py          # write tasks/grader_exclusions.txt
    uv run python scripts/grader_exclusions.py --check  # fail if the file is out of date
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evcarry.env.task_rules import docx_table_defect, grader_missing_modules  # noqa: E402
from evcarry.env.toolathlon import gym_task_root  # noqa: E402

LISTS = ("headline60", "pilot10", "dev")
OUT = ROOT / "tasks" / "grader_exclusions.txt"


def _names(name: str) -> list[str]:
    text = (ROOT / "tasks" / f"{name}.txt").read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]


def render() -> str:
    gym = gym_task_root()
    lines = ["# Broken-grader exclusions (plan amendment A17, 2026-10-25), from scripts/grader_exclusions.py.",
             "# Rules: A13 grader imports a module the grading Python lacks; docx-table rule (A10/P2-4).",
             "# Primary analysis keeps all tasks; the sensitivity analysis drops the tasks listed here."]
    for name in LISTS:
        flagged = []
        for task in _names(name):
            reasons = []
            missing = grader_missing_modules(gym / task)
            if missing:
                reasons.append("A13 missing " + ",".join(missing))
            if docx_table_defect(gym / task):
                reasons.append("docx table not read")
            if reasons:
                flagged.append(f"{task}  # {name}: {'; '.join(reasons)}")
        lines.append(f"# {name}: {len(flagged)} flagged")
        lines.extend(flagged)
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = render()
    if args.check:
        if not OUT.is_file() or OUT.read_text(encoding="utf-8") != text:
            print("tasks/grader_exclusions.txt is out of date", file=sys.stderr)
            return 1
        return 0
    OUT.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
