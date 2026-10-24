"""Hand-label validation of the L3 judge (logic in ``evcarry.eval.judge_validation``).

    # 1. sample 100 judged claims from finished runs (no API call)
    uv run python scripts/judge_validation.py export runs/pilot_deepseek --out validation/judge_v1
    # 2. synthetic faults have no verdict yet: let the run's judge rate them (paid, small)
    uv run python scripts/judge_validation.py judge validation/judge_v1 --model openrouter_ds_1017 --only-missing --hit
    # 3. the owner fills human_label in validation/judge_v1/items.csv (blind: no verdicts in that file)
    # 4. agreement, kappa, precision and recall
    uv run python scripts/judge_validation.py score validation/judge_v1
    # M6: the local model re-judges the same rows as a second judge
    uv run python scripts/judge_validation.py judge validation/judge_v1 --model local_primary --column local
    uv run python scripts/judge_validation.py score validation/judge_v1

``judge`` refuses a billable model without ``--hit`` and prints a worst-case cost first.
Its ``judge_call`` events (provider, tokens, cost) go to ``<dir>/judge_calls_<column>.jsonl``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from evcarry.eval import judge_validation as jv
from evcarry.paths import resolve_repo_path


def export(args: argparse.Namespace) -> int:
    items = jv.collect([resolve_repo_path(path) for path in args.run_dirs])
    injected = sum(1 for item in items if item.origin == "injected")
    wanted = round(jv.DEFAULT_ALLOCATION["fault"] * args.n / sum(jv.DEFAULT_ALLOCATION.values()))
    faults = jv.synthesize_faults(items, max(0, wanted - injected) if args.faults is None else args.faults, args.seed)
    sample = jv.stratified_sample([*items, *faults], n=args.n, seed=args.seed)
    items_path, key_path = jv.write_export(sample, resolve_repo_path(args.out), seed=args.seed)
    strata: dict[str, int] = {}
    for item in sample:
        strata[item.stratum] = strata.get(item.stratum, 0) + 1
    print(f"judged claims found: {len(items)} ({injected} injected faults); synthetic faults made: {len(faults)}")
    print(f"sample: {len(sample)} {strata}")
    missing = sum(1 for item in sample if not item.judge_verdict)
    if missing:
        print(f"{missing} rows have no judge verdict yet (synthetic faults): run `judge ... --only-missing`")
    print(f"label: {items_path}\nkey (do not show the labeller): {key_path}")
    return 0


def judge(args: argparse.Namespace) -> int:
    from evcarry.protocol.judge import build_judge
    from evcarry.providers.registry import load_models
    from evcarry.trace.writer import JsonlWriter

    folder = resolve_repo_path(args.dir)
    rows = jv.read_key(folder / "key.csv")
    config = load_models()[args.model]
    todo = [row for row in rows if not args.only_missing or not (
        row.judge_verdict if args.column == "judge" else row.extra.get(f"{args.column}_verdict"))]
    price = config.price_table()
    chars = sum(len(row.request) for row in todo) + 4000 * len(todo)  # + the system prompt per call
    worst = (chars / 2 * price.input_cache_miss + 2000 * len(todo) * price.output) / 1_000_000
    print(f"{len(todo)} rows to judge with {args.model}; worst-case cost ${worst:.4f}")
    if config.billable and not args.hit:
        print("billable model: pass --hit to make the calls")
        return 2
    writer = JsonlWriter(folder / f"judge_calls_{args.column}.jsonl")
    try:
        judge_model = build_judge(args.model, writer=writer)
        asked = asyncio.run(jv.judge_items(rows, judge_model, column=args.column, only_missing=args.only_missing))
    finally:
        writer.close()
    jv.write_key(rows, folder / "key.csv")
    print(f"judged {asked} rows; cost ${getattr(judge_model, 'cost_usd', 0.0):.4f}; key updated")
    return 0


def score(args: argparse.Namespace) -> int:
    folder = resolve_repo_path(args.dir)
    report = jv.score(jv.read_key(folder / "key.csv"), jv.read_labels(folder / "items.csv"))
    text = json.dumps(report, indent=2)
    (folder / "score.json").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("export", help="sample judged claims into items.csv (blind) and key.csv")
    exp.add_argument("run_dirs", nargs="+")
    exp.add_argument("--out", required=True)
    exp.add_argument("--n", type=int, default=100)
    exp.add_argument("--seed", type=int, default=20261024)
    exp.add_argument("--faults", type=int, default=None, help="synthetic faults to make (default: fill the fault quota)")
    jud = sub.add_parser("judge", help="run a judge model on the exported rows")
    jud.add_argument("dir")
    jud.add_argument("--model", required=True)
    jud.add_argument("--column", default="judge", help="'judge' fills judge_verdict; any other name adds <name>_verdict")
    jud.add_argument("--only-missing", action="store_true")
    jud.add_argument("--hit", action="store_true", help="allow calls to a billable model")
    sco = sub.add_parser("score", help="compare judge verdicts with human_label")
    sco.add_argument("dir")
    args = parser.parse_args(argv)
    return {"export": export, "judge": judge, "score": score}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
