"""``evcarry run|summarize|check``."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import pandas as pd

from evcarry.env.isolation import EnvironmentUnavailable
from evcarry.harness.arms import ARMS
from evcarry.providers.registry import MissingApiKey, load_models
from evcarry.runs.budget import PeakWindowError, SpendCapExceeded
from evcarry.runs.config import load_experiment
from evcarry.runs.manager import dry_run_report, run_experiment
from evcarry.trace.writer import read_events


def summarize(run_dir: Path) -> str:
    rows = []
    for path in sorted(run_dir.glob("*.jsonl")):
        for event in read_events(path):
            if event.get("type") == "run_end":
                rows.append(event)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return "no runs"
    columns = [
        column
        for column in ("task", "arm", "seed", "passed", "status", "cost_usd", "input_tokens", "output_tokens")
        if column in frame.columns
    ]
    return frame[columns].to_string(index=False)


def check_config(path: Path) -> str:
    config = load_experiment(path)
    models = load_models()
    if config.model not in models:
        raise SystemExit(f"unknown model {config.model}")
    unknown = [arm for arm in config.arms if arm not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms: {unknown}")
    from evcarry.paths import resolve_repo_path

    tasks = resolve_repo_path(config.tasks)
    if not tasks.is_file():
        raise SystemExit(f"missing task list {tasks}")
    report = dry_run_report(config)
    return "ok\n" + report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="evcarry")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run an experiment config")
    run.add_argument("config")
    run.add_argument("--dry-run", action="store_true")
    summary = sub.add_parser("summarize", help="print a table of run_end rows")
    summary.add_argument("run_dir")
    check = sub.add_parser("check", help="validate a config without calling a model")
    check.add_argument("config")
    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            print(check_config(Path(args.config)))
            return
        if args.command == "summarize":
            print(summarize(Path(args.run_dir)))
            return
        config = load_experiment(args.config)
        if args.dry_run:
            print(dry_run_report(config))
            return
        asyncio.run(run_experiment(config))
    except SpendCapExceeded as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    except PeakWindowError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(3) from exc
    except MissingApiKey as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(4) from exc
    except EnvironmentUnavailable as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(5) from exc
