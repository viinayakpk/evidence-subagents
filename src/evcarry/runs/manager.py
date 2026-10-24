"""Expand an experiment grid, resume finished runs, and write JSONL traces.

Parallel billable runs check the spend cap before they start and add cost
after they finish, so they can overshoot the cap by about one batch.
Off-peak is checked at the start of each run, not mid-run.

``environment: in_process`` runs the fixture tasks against the in-process
fake MCP server. ``environment: docker`` runs real Toolathlon-Gym tasks, one
isolated set of containers per run (``evcarry.env.gym_docker``), graded by the
task's own evaluation script.
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
from agents.exceptions import MaxTurnsExceeded

from evcarry.env.fake_mcp import FakeMcp
from evcarry.env.isolation import DockerIsolation
from evcarry.env.toolathlon import TaskSpec, load_task, load_task_list, prepare_workspace
from evcarry.eval.evaluate import evaluate_task
from evcarry.harness.agents import build_main_agent
from evcarry.harness.arms import arm_spec, effective_audit
from evcarry.harness.hooks import LedgerHooks
from evcarry.harness.prompts import prompt_path
from evcarry.harness.tools import OVERLONG_META, VIEW_OVERLONG, ToolRouter, Toolset, fixture_toolset, overlong_view
from evcarry.inject.injector import FaultInjector
from evcarry.ledger.ledger import Ledger
from evcarry.ledger.store import write_ledger
from evcarry.paths import repo_root, resolve_repo_path
from evcarry.protocol.judge import build_judge
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


def default_script_path(arm_name: str, script_dir: str = "tests/fixtures/scripts") -> Path:
    return resolve_repo_path(script_dir) / arm_spec(arm_name).script


def grid(config: ExperimentConfig) -> list[tuple[str, str, int]]:
    tasks = load_task_list(resolve_repo_path(config.tasks))
    return [(task, arm, seed) for task in tasks for arm in config.arms for seed in config.seeds]


def dry_run_report(config: ExperimentConfig) -> str:
    model = load_models().get(config.model)
    lines = [
        f"experiment: {config.experiment}",
        f"model: {config.model}",
        f"judge_model: {config.judge_model}",
        f"environment: {config.environment}",
        f"schedule: {config.budget.schedule}",
        f"cap_usd: {config.budget.cap_usd}",
        f"parallel_tasks: {config.parallel_tasks}",
        "grid:",
    ]
    for task, arm, seed in grid(config):
        lines.append(f"  {task} {arm} seed={seed}")
    if config.environment == "docker":
        lines.extend("docker " + step for step in DockerIsolation(_gym_images(config)).plan())
        lines.append("docker up: not called (dry-run)")
    if model is not None:
        lines.append(f"billable: {model.billable}")
        lines.append(f"kind: {model.kind}")
        if model.routing:
            lines.append("routing: " + json.dumps(model.routing, sort_keys=True))
    lines.append("api keys: not required for dry-run")
    return "\n".join(lines)


def _gym_images(config: ExperimentConfig):
    from evcarry.env.gym_docker import GymImages

    default = GymImages.default()
    return GymImages(
        agent=config.gym.agent_image or default.agent,
        postgres=config.gym.postgres_image or default.postgres,
    )


async def _drive(agent: Any, text: str, hooks: LedgerHooks, config: ExperimentConfig) -> dict[str, Any]:
    """Run the main agent. Turn and time limits end the run; other errors propagate."""
    try:
        result = await asyncio.wait_for(
            Runner.run(
                agent,
                text,
                hooks=hooks,
                max_turns=config.limits.max_steps,
                run_config=RunConfig(tracing_disabled=True),
            ),
            timeout=config.limits.max_seconds,
        )
    except MaxTurnsExceeded:
        return {"agent_status": "max_turns_reached"}
    except TimeoutError:
        return {"agent_status": "timeout"}
    return {"agent_status": "completed", "final_output": str(result.final_output)[-2000:]}


async def _run_fixture(
    *,
    task: TaskSpec,
    out_dir: Path,
    stem: str,
    build: Any,
    writer: JsonlWriter,
    price: Any,
    config: ExperimentConfig,
) -> dict[str, Any]:
    workspace = prepare_workspace(task, out_dir / "workspaces" / stem)
    async with FakeMcp(workspace) as mcp:
        ledger = Ledger()
        toolset = fixture_toolset(mcp, ledger)
        hooks = LedgerHooks(ledger, writer, price, toolset.catalog)
        agent = build(toolset=toolset, hooks=hooks, ledger=ledger)
        outcome = await _drive(agent, task.task_text, hooks, config)
    evaluation = evaluate_task(task, workspace)
    return {**outcome, "passed": evaluation.get("passed"), "status": evaluation.get("status")}


async def _run_gym(
    *,
    task: TaskSpec,
    out_dir: Path,
    stem: str,
    build: Any,
    writer: JsonlWriter,
    price: Any,
    config: ExperimentConfig,
) -> dict[str, Any]:
    from evcarry.env.gym_docker import CLAIM_DONE, PYTHON_EXECUTE, GymTaskEnv
    from evcarry.env.gym_tools import GymToolPolicy

    gym = config.gym
    policy = GymToolPolicy.load(gym.tool_policy)
    env = GymTaskEnv(
        task,
        images=_gym_images(config),
        policy=policy,
        log_dir=out_dir / "logs" / stem,
        setup_timeout=gym.setup_timeout_s,
        eval_timeout=gym.eval_timeout_s,
    )
    ledger = Ledger()
    async with env:
        mcp_tools = sorted(name for name in env.catalog if name not in (CLAIM_DONE, PYTHON_EXECUTE))
        catalog = {**env.catalog, VIEW_OVERLONG: OVERLONG_META}
        router = ToolRouter(env, {VIEW_OVERLONG: overlong_view(ledger, gym.tool_output_chars)})
        toolset = Toolset(
            catalog=catalog,
            caller=router,
            all_tools=tuple(sorted([*mcp_tools, PYTHON_EXECUTE, VIEW_OVERLONG, CLAIM_DONE])),
            researcher=tuple(sorted([*(name for name in mcp_tools if catalog[name].read_only), VIEW_OVERLONG])),
            operator=tuple(sorted([*mcp_tools, PYTHON_EXECUTE, VIEW_OVERLONG])),
            main_extra=(CLAIM_DONE,),
            stop_tools=(CLAIM_DONE,),
            show_call_id=True,
            max_chars=gym.tool_output_chars,
            ledger=ledger,
        )
        writer.write(
            {
                "type": "environment",
                "kind": "docker",
                **env.info.as_dict(),
                "tool_policy_sha256": policy.sha256,
                "tools": {
                    "all": len(toolset.all_tools),
                    "researcher": len(toolset.researcher),
                    "operator": len(toolset.operator),
                },
            }
        )
        hooks = LedgerHooks(ledger, writer, price, catalog)
        agent = build(toolset=toolset, hooks=hooks, ledger=ledger)
        outcome = await _drive(agent, task.task_text, hooks, config)
        evaluation = await env.evaluate()
        if gym.export_workspace:
            await env.export_workspace(out_dir / "workspaces" / stem)
    write_ledger(out_dir / "ledgers" / f"{stem}.json", ledger)
    return {
        **outcome,
        **evaluation,
        "claimed_done": env.claimed_done,
        "gym_run_id": env.info.run_id,
        "timings": {key: round(value, 3) for key, value in env.info.timings.items()},
    }


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
    path = run_path(out_dir, task_name, arm_name, model_config.model or model_config.name, seed)
    stem = f"{task_name}__{arm_name}__{seed}"
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
                "environment": config.environment,
                "judge_model": config.judge_model,
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
        judge = build_judge(config.judge_model, writer=writer)

        def build(*, toolset: Toolset, hooks: LedgerHooks, ledger: Ledger):
            return build_main_agent(
                arm=spec,
                task=task_name,
                seed=seed,
                prompts_dir=config.prompts,
                model_config=model_config,
                script=script,
                toolset=toolset,
                hooks=hooks,
                writer=writer,
                ledger=ledger,
                checks=checks,
                sampler=sampler,
                rate=rate,
                injector=injector,
                max_turns=config.limits.max_steps,
                judge=judge,
            )

        runner = _run_gym if config.environment == "docker" else _run_fixture
        outcome = await runner(
            task=task, out_dir=out_dir, stem=stem, build=build, writer=writer, price=price, config=config
        )
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
        judge_cost = float(getattr(judge, "cost_usd", 0.0))
        writer.write(
            {
                "type": "run_end",
                "experiment": config.experiment,
                "task": task_name,
                "arm": arm_name,
                "model": model_config.model,
                "seed": seed,
                **outcome,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost,
                "judge_calls": int(getattr(judge, "calls", 0)),
                "judge_cost_usd": judge_cost,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
        )
        return cost + judge_cost
    finally:
        current_writer.reset(token)
        writer.close()


def _script_for(arm_name: str, overrides: dict[str, Path] | None, model_kind: str, script_dir: str) -> dict[str, Any] | None:
    if model_kind != "scripted":
        return None
    path = (overrides or {}).get(arm_name) or default_script_path(arm_name, script_dir)
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
    models = load_models()
    if config.model not in models:
        raise KeyError(f"unknown model {config.model}")
    model_config = models[config.model]
    resolve_api_key(model_config)
    if config.judge_model in models:
        resolve_api_key(models[config.judge_model])
    if config.environment == "docker":
        DockerIsolation(_gym_images(config)).up()
    price = model_config.price_table()
    guard = BudgetGuard(
        cap_usd=config.budget.cap_usd,
        schedule=config.budget.schedule,
        price=price,
    )
    destination = out_dir or resolve_repo_path(config.out)
    destination.mkdir(parents=True, exist_ok=True)
    billable = model_config.billable or (config.judge_model in models and models[config.judge_model].billable)
    written: list[Path] = []
    pending = []
    for task_name, arm_name, seed in grid(config):
        path = run_path(destination, task_name, arm_name, model_config.model or model_config.name, seed)
        if has_run_end(path):
            continue
        pending.append((task_name, arm_name, seed, path))

    async def _one(task_name: str, arm_name: str, seed: int) -> None:
        guard.check(billable=billable, now=datetime.now(timezone.utc))
        script = _script_for(arm_name, script_overrides, model_config.kind, config.script_dir)
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

    if config.parallel_tasks <= 1 or stop_after is not None:
        finished = 0
        for task_name, arm_name, seed, path in pending:
            await _one(task_name, arm_name, seed)
            written.append(path)
            finished += 1
            if stop_after is not None and finished >= stop_after:
                break
    else:
        semaphore = asyncio.Semaphore(config.parallel_tasks)

        async def _bounded(item: tuple[str, str, int, Path]) -> Path:
            task_name, arm_name, seed, path = item
            async with semaphore:
                await _one(task_name, arm_name, seed)
                return path

        tasks = [asyncio.create_task(_bounded(item)) for item in pending]
        try:
            written = list(await asyncio.gather(*tasks))
        except BaseException:
            # Stop the other runs and let their environments tear down before re-raising.
            for item in tasks:
                item.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
    if written or any(destination.glob("*.jsonl")):
        write_index(destination)
    return written
