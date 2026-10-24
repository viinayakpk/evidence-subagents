"""``evcarry run|summarize|check|gym``."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import pandas as pd

from evcarry.env.isolation import EnvironmentUnavailable
from evcarry.harness.arms import ARMS
from evcarry.providers.pin import ProviderMismatch
from evcarry.providers.registry import MissingApiKey, load_models
from evcarry.runs.budget import PeakWindowError, SpendCapExceeded
from evcarry.runs.config import load_experiment
from evcarry.runs.manager import FrozenInputMismatch, dry_run_report, run_experiment
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


def gym_tasks_report(task_list: Path, parallel: int) -> tuple[str, bool]:
    """Start and stop every listed task's environment; one line per task. 128 is a common tool limit."""
    from evcarry.env.gym_docker import smoke_task
    from evcarry.env.toolathlon import gym_task_root, load_task, load_task_list
    from evcarry.paths import resolve_repo_path

    names = load_task_list(resolve_repo_path(task_list))
    log_root = resolve_repo_path("runs/gym_check")

    async def run_all() -> list[dict]:
        gate = asyncio.Semaphore(parallel)

        async def one(name: str) -> dict:
            async with gate:
                return await smoke_task(load_task(name, gym_task_root()), log_dir=log_root / name)

        return list(await asyncio.gather(*(one(name) for name in names)))

    rows = asyncio.run(run_all())
    lines = []
    for row in rows:
        if not row["ok"]:
            lines.append(f"FAIL {row['task']}: {row['error']}")
            continue
        flags = []
        if row["preprocess_rc"] not in (0, None):
            flags.append(f"preprocess rc={row['preprocess_rc']}")
        if row["tools"] + 1 > 128:  # + view_overlong_output
            flags.append(f"{row['tools'] + 1} tools for the SA agent (over 128)")
        status = "WARN" if flags else "ok  "
        detail = f" [{'; '.join(flags)}]" if flags else ""
        lines.append(f"{status} {row['task']}: {row['tools']} tools, setup {row['timings'].get('setup_s')}s{detail}")
    failed = sum(1 for row in rows if not row["ok"])
    lines.append(f"{len(rows) - failed}/{len(rows)} task environments started; logs in {log_root}")
    return "\n".join(lines), failed == 0


def gym_command(action: str, tasks: Path | None = None, parallel: int = 4) -> str:
    from evcarry.env.gym_docker import GymImages, clean_leftovers, leftover_resources, preflight

    if action == "clean":
        removed = clean_leftovers()
        return f"removed containers {removed['containers']} and networks {removed['networks']}"
    images = GymImages.default()
    docker = preflight(images)
    left = leftover_resources(docker)
    lines = [f"docker: {docker}", f"agent image: {images.agent}", f"postgres image: {images.postgres}"]
    lines.append(f"leftover containers: {left['containers'] or 'none'}; networks: {left['networks'] or 'none'}")
    lines.append("ok: docker-environment experiments can start")
    if tasks is not None:
        report, ok = gym_tasks_report(tasks, parallel)
        lines.append(report)
        if not ok:
            raise EnvironmentUnavailable("\n".join(lines))
    return "\n".join(lines)


def prompts_command(action: str, prompts_dir: str) -> str:
    import json

    from evcarry.harness import prompt_freeze
    from evcarry.runs.manager import git_commit

    if action == "freeze":
        path = prompt_freeze.freeze(prompts_dir, git_commit=git_commit())
        return f"froze {prompts_dir}: {json.loads(path.read_text(encoding='utf-8'))['manifest_sha256']} -> {path}"
    return json.dumps(prompt_freeze.status(prompts_dir).as_dict(), indent=2)


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
    gym = sub.add_parser(
        "gym",
        help="gym Docker environment: `check` (preflight) or `clean` (remove every evcarry-labelled "
        "container and network, including those of an experiment still running in another process)",
    )
    gym.add_argument("action", choices=["check", "clean"])
    gym.add_argument("--tasks", type=Path, help="with check: start and stop each listed task's environment")
    gym.add_argument("--parallel", type=int, default=4)
    prompts = sub.add_parser(
        "prompts",
        help="prompt-set freeze: `status` (manifest hash and what changed) or `freeze` (write FROZEN.json; owner only)",
    )
    prompts.add_argument("action", choices=["status", "freeze"])
    prompts.add_argument("prompts_dir", nargs="?", default="prompts/gym_v1")
    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            print(check_config(Path(args.config)))
            return
        if args.command == "summarize":
            print(summarize(Path(args.run_dir)))
            return
        if args.command == "gym":
            print(gym_command(args.action, args.tasks, args.parallel))
            return
        if args.command == "prompts":
            print(prompts_command(args.action, args.prompts_dir))
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
    except FrozenInputMismatch as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(6) from exc
    except ProviderMismatch as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(7) from exc
