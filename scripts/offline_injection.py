"""Offline fault injection over logged evidence-arm claims, L0-L2 by default (no model or tool calls).

    uv run python scripts/offline_injection.py runs/dev_verify_v1 runs/dev_verify_v2 runs/pilot_deepseek \
        --out results/offline_injection.json

``--l3`` would add the same-model judge, which makes paid calls; it is off by default and this script
refuses it unless ``--allow-paid-judge`` is also given, which needs the owner's approval of the spend.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evcarry.analysis.offline_injection import load_items, study  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--l3", action="store_true", help="also run L3 (paid judge calls); off by default")
    parser.add_argument("--allow-paid-judge", action="store_true")
    args = parser.parse_args()
    if args.l3:
        if not args.allow_paid_judge:
            sys.exit("--l3 makes paid judge calls; add --allow-paid-judge only with the owner's approval")
        sys.exit("--l3: wire evcarry.protocol.judge into study(l3=...) for the approved judge config first")
    report = {}
    for run_dir in args.run_dirs:
        report[run_dir.name] = asyncio.run(study(load_items(run_dir)))
    everything = [item for run_dir in args.run_dirs for item in load_items(run_dir)]
    report["all"] = asyncio.run(study(everything))
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
