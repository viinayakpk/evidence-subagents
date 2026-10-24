"""Expand an experiment grid, resume finished runs, and write JSONL traces.

Parallel billable runs check the spend cap before they start and add cost
after they finish, so they can overshoot the cap by about one batch.
Off-peak is checked at the start of each run, not mid-run.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents import RunConfig, Runner, set_trace_processors

from evcarry.env.fake_mcp import FakeMcp
from evcarry.env.isolation import DockerIsolation
from evcarry.env.toolathlon import load_task, load_task_list, prepare_workspace
from evcarry.eval.evaluate import evaluate_task
from evcarry.harness.agents import build_main_agent
from evcarry.harness.arms import arm_spec, effective_audit
from evcarry.harness.hooks import LedgerHooks
from evcarry.harness.prompts import prompt_path
from evcarry.inject.injector import FaultInjector
from evcarry.ledger.ledger import Ledger
from evcarry.paths import repo_root, resolve_repo_path
from evcarry.providers.registry import load_models, resolve_api_key
from evcarry.runs.budget import BudgetGuard
from evcarry.runs.config import ExperimentConfig
from evcarry.trace.index import write_index
from evcarry.trace.processor import JsonlTraceProcessor
from evcarry.trace.writer import JsonlWriter, current_writer, has_run_end, read_events, run_path

_PROCESSORS_READY = False


def _install_trace_processors() -> None:
    global _PROCESSORS_READY
    if _PROCESSORS_READY:
        return
    set_trace_processors([JsonlTraceProcessor()])
    _PROCESSORS_READY = True


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root(),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_script_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def default_script_path(arm_name: str) -> Path:
    return resolve_repo_path("tests/fixtures/scripts") / arm_spec(arm_name).script


def grid(config: ExperimentConfig) -> list[tuple[str, str, int]]:
    tasks = load_task_list(resolve_repo_path(config.tasks))
    return [(task, arm, seed) for task in tasks for arm in config.arms for seed in config.seeds]


def dry_run_report(config: ExperimentConfig) -> str:
    model = load_models().get(config.model)
    lines = [
        f"experiment: {config.experiment}",
        f"model: {config.model}",
        f"environment: {config.environment}",
        f"schedule: {config.budget.schedule}",
        f"cap_usd: {config.budget.cap_usd}",
        f"parallel_tasks: {config.parallel_tasks}",
        "grid:",
    ]
    for task, arm, seed in grid(config):
        lines.append(f"  {task} {arm} seed={seed}")
    if config.environment == "docker":
        lines.append("docker: " + " ".join(DockerIsolation().plan()))
        lines.append("docker up: not called (dry-run)")
    if model is not None:
        lines.append(f"billable: {model.billable}")
        lines.append(f"kind: {model.kind}")
        if model.routing:
            lines.append("routing: " + json.dumps(model.routing, sort_keys=True))
    lines.append("api keys: not required for dry-run")
    return "\n".join(lines)


async def _run_one(
    *,
    config: ExperimentConfig,
    task_name: str,
    arm_name: str,
    seed: int,
    out_dir: Path,
    script: dict[str, Any] | None,
    model_config: Any,
    price: Any,
) -> float:
    _install_trace_processors()
    spec = arm_spec(arm_name)
    checks, sampler, rate = effective_audit(
        arm_name,
        sampler=config.audit.sampler,
        rate=config.audit.rate,
        checks=config.audit.checks,
    )
    task_root = resolve_repo_path(config.task_root) if config.task_root else resolve_repo_path("tests/fixtures/gym/tasks/finalpool")
    task = load_task(task_name, task_root)
    workspace = prepare_workspace(task, out_dir / "workspaces" / f"{task_name}__{arm_name}__{seed}")
    path = run_path(out_dir, task_name, arm_name, model_config.model or model_config.name, seed)
    if path.exists() and not has_run_end(path):
        path.unlink()
    writer = JsonlWriter(path)
    token = current_writer.set(writer)
    try:
        prompt_file = prompt_path(config.prompts, arm_name)
        routing = model_config.routing
        served_model = script.get("served_model") if script and model_config.kind == "scripted" else None
        writer.write(
            {
                "type": "run_start",
                "experiment": config.experiment,
                "task": task_name,
                "arm": arm_name,
                "model": model_config.name,
                "configured_model": model_config.model,
                "seed": seed,
                "config_sha256": config.source_sha256,
                "git_commit": git_commit(),
                "prompt_sha256": _sha256_file(prompt_file),
                "task_list_sha256": _sha256_file(resolve_repo_path(config.tasks)),
                "task_prompt_sha256": task.agent_prompt_sha256,
                "served_model": served_model,
                "routing": routing,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
        )
        injector = None
        if config.inject and config.inject.probability > 0:
            injector = FaultInjector(
                probability=config.inject.probability,
                types=tuple(config.inject.types),
                variant=config.inject.variant,
                max_per_run=config.inject.max_per_run,
            )
        async with FakeMcp(workspace) as mcp:
            ledger = Ledger()
            hooks = LedgerHooks(ledger, writer, price)
            agent = build_main_agent(
                arm=spec,
                task=task_name,
                seed=seed,
                prompts_dir=config.prompts,
                model_config=model_config,
                script=script,
                caller=mcp,
                hooks=hooks,
                writer=writer,
                ledger=ledger,
                checks=checks,
                sampler=sampler,
                rate=rate,
                injector=injector,
                max_turns=config.limits.max_steps,
            )
            await asyncio.wait_for(
                Runner.run(
                    agent,
                    task.task_text,
                    hooks=hooks,
                    max_turns=config.limits.max_steps,
                    run_config=RunConfig(tracing_disabled=True),
                ),
                timeout=config.limits.max_seconds,
            )
        evaluation = evaluate_task(task, workspace)
        events = read_events(path)
        input_tokens = 0
        output_tokens = 0
        cost = 0.0
        for event in events:
            if event.get("type") != "llm_call":
                continue
            tokens = event.get("tokens") or {}
            input_tokens += int(tokens.get("input_tokens") or 0)
            output_tokens += int(tokens.get("output_tokens") or 0)
            cost += float(event.get("cost_usd") or 0)
        writer.write(
            {
                "type": "run_end",
                "experiment": config.experiment,
                "task": task_name,
                "arm": arm_name,
                "model": model_config.model,
                "seed": seed,
                "passed": evaluation.get("passed"),
                "status": evaluation.get("status"),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
        )
        return cost
    finally:
        current_writer.reset(token)
        writer.close()


def _script_for(arm_name: str, overrides: dict[str, Path] | None, model_kind: str) -> dict[str, Any] | None:
    if model_kind != "scripted":
        return None
    path = (overrides or {}).get(arm_name) or default_script_path(arm_name)
    if not path.is_file():
        raise FileNotFoundError(path)
    return load_script_file(path)


async def run_experiment(
    config: ExperimentConfig,
    *,
    stop_after: int | None = None,
    script_overrides: dict[str, Path] | None = None,
    out_dir: Path | None = None,
    dry_run: bool = False,
) -> list[Path]:
    if dry_run:
        print(dry_run_report(config))
        return []
    if config.environment == "docker":
        DockerIsolation().up()
    models = load_models()
    if config.model not in models:
        raise KeyError(f"unknown model {config.model}")
    model_config = models[config.model]
    resolve_api_key(model_config)
    price = model_config.price_table()
    guard = BudgetGuard(
        cap_usd=config.budget.cap_usd,
        schedule=config.budget.schedule,
        price=price,
    )
    destination = out_dir or resolve_repo_path(config.out)
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    pending = []
    for task_name, arm_name, seed in grid(config):
        path = run_path(destination, task_name, arm_name, model_config.model or model_config.name, seed)
        if has_run_end(path):
            continue
        pending.append((task_name, arm_name, seed, path))
    if config.parallel_tasks <= 1 or stop_after is not None:
        finished = 0
        for task_name, arm_name, seed, path in pending:
            guard.check(billable=model_config.billable, now=datetime.now(timezone.utc))
            script = _script_for(arm_name, script_overrides, model_config.kind)
            cost = await _run_one(
                config=config,
                task_name=task_name,
                arm_name=arm_name,
                seed=seed,
                out_dir=destination,
                script=script,
                model_config=model_config,
                price=price,
            )
            guard.add(cost)
            written.append(path)
            finished += 1
            if stop_after is not None and finished >= stop_after:
                break
    else:
        semaphore = asyncio.Semaphore(config.parallel_tasks)

        async def _bounded(item: tuple[str, str, int, Path]) -> Path:
            task_name, arm_name, seed, path = item
            async with semaphore:
                guard.check(billable=model_config.billable, now=datetime.now(timezone.utc))
                script = _script_for(arm_name, script_overrides, model_config.kind)
                cost = await _run_one(
                    config=config,
                    task_name=task_name,
                    arm_name=arm_name,
                    seed=seed,
                    out_dir=destination,
                    script=script,
                    model_config=model_config,
                    price=price,
                )
                guard.add(cost)
                return path

        written = list(await asyncio.gather(*[_bounded(item) for item in pending]))
    if written or any(destination.glob("*.jsonl")):
        write_index(destination)
    return written
