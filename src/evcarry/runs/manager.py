"""Expand an experiment grid, resume finished runs, and write JSONL traces.

Parallel billable runs check the spend cap before they start and add cost
after they finish, so they can overshoot the cap by about one batch.
Off-peak is checked at the start of each run, not mid-run. The cap counts
everything already logged in the output directory, including earlier
invocations and aborted runs (kept as ``*.jsonl.aborted-<n>``).

``environment: in_process`` runs the fixture tasks against the in-process
fake MCP server. ``environment: docker`` runs real Toolathlon-Gym tasks, one
isolated set of containers per run (``evcarry.env.gym_docker``), graded by the
task's own evaluation script.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents import Runner, set_trace_processors
from agents.exceptions import MaxTurnsExceeded

from evcarry.env.fake_mcp import FakeMcp
from evcarry.env.isolation import DockerIsolation, EnvironmentUnavailable
from evcarry.env.toolathlon import TaskSpec, load_task, load_task_list, prepare_workspace
from evcarry.eval.evaluate import evaluate_task
from evcarry.harness.agents import build_main_agent
from evcarry.harness.arms import arm_spec, effective_audit
from evcarry.harness.hooks import LedgerHooks
from evcarry.harness.prompts import prompt_path
from evcarry.harness.run_settings import harness_run_config
from evcarry.harness.tools import OVERLONG_META, VIEW_OVERLONG, ToolRouter, Toolset, fixture_toolset, overlong_view
from evcarry.inject.injector import FaultInjector
from evcarry.ledger.ledger import Ledger
from evcarry.ledger.store import write_ledger
from evcarry.paths import repo_root, resolve_repo_path
from evcarry.protocol.judge import build_judge
from evcarry.providers.pin import PinGuard, current_pin_guard
from evcarry.providers.registry import load_models, resolve_api_key
from evcarry.runs.budget import BudgetError, BudgetGuard
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


SHA256SUMS = "tasks/SHA256SUMS"
GYM_SUBMODULE = "third_party/toolathlon_gym"


class FrozenInputMismatch(RuntimeError):
    """A frozen task list or the gym checkout differs from what was frozen."""


def check_frozen_inputs(config: ExperimentConfig) -> None:
    """Refuse to run on a frozen task list whose bytes changed, or on a gym checkout other than the frozen commit.

    Task lists not named in ``tasks/SHA256SUMS`` (the test fixtures) are not checked.
    Docker runs send the host's ``docs/task.md`` while the images hold the frozen
    commit's task files, so the submodule must be at ``tasks/FROZEN.txt``'s commit.
    """
    tasks = resolve_repo_path(config.tasks)
    sums = resolve_repo_path(SHA256SUMS)
    if sums.is_file():
        for line in sums.read_text(encoding="utf-8").splitlines():
            digest, _, name = line.strip().partition("  ")
            if name and resolve_repo_path(name.strip()) == tasks and _sha256_file(tasks) != digest:
                raise FrozenInputMismatch(f"{config.tasks} does not match its hash in {SHA256SUMS}; the frozen list was edited.")
    if config.environment == "docker":
        from evcarry.env.gym_docker import gym_commit

        checkout = resolve_repo_path(GYM_SUBMODULE)
        if (checkout / ".git").exists():
            head = subprocess.run(
                ["git", "-C", str(checkout), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
            ).stdout.strip()
            if head and head != gym_commit():
                raise FrozenInputMismatch(f"{GYM_SUBMODULE} is at {head}, not the frozen gym commit {gym_commit()}.")


def check_prompt_freeze(config: ExperimentConfig, *, billable: bool) -> dict[str, object]:
    """Refuse a run whose prompt set changed after it was frozen (billable runs), or is not frozen when required."""
    from evcarry.harness import prompt_freeze

    found = prompt_freeze.status(config.prompts)
    if found.state == "changed" and (billable or config.require_frozen_prompts):
        raise FrozenInputMismatch(
            f"{config.prompts} changed after it was frozen ({prompt_freeze.FREEZE_FILE}): {', '.join(found.changed) or 'manifest'}"
        )
    if found.state != "match" and config.require_frozen_prompts:
        raise FrozenInputMismatch(f"{config.experiment} requires a frozen prompt set; {config.prompts} is {found.state}.")
    return found.as_dict()


def load_script_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


_SPEND_EVENTS = ("llm_call", "judge_call", "llm_error")


def event_spend(event: dict[str, Any]) -> float:
    """Dollars one trace event spent, for the cap: the larger of the price-table cost and the provider's bill.

    ``llm_error`` counts too: a call that was answered and then failed in the SDK
    was billed. The larger figure keeps a price change or an unexpected endpoint
    tier from letting the cap under-count.
    """
    if event.get("type") not in _SPEND_EVENTS:
        return 0.0
    table = float(event.get("cost_usd") or 0)
    reported = event.get("provider_reported_cost_usd")
    return max(table, float(reported)) if isinstance(reported, (int, float)) else table


def trace_spend(path: Path) -> float:
    total = 0.0
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue  # a line cut short when the process was killed
        if isinstance(event, dict):
            total += event_spend(event)
    return total


def logged_spend(out_dir: Path) -> float:
    """Dollars already logged in ``out_dir`` (``event_spend`` over every trace, finished or aborted).

    The spend cap starts from this, so a restart (the plan runs one seed per
    weekend) does not reset what was already spent.
    """
    return sum(trace_spend(path) for path in [*out_dir.glob("*.jsonl"), *out_dir.glob("*.jsonl.aborted-*")])


def _set_aside(path: Path) -> Path:
    """Keep an unfinished trace as ``<name>.aborted-<n>``: its calls were paid for and count toward the cap."""
    n = 1
    while (target := path.with_name(f"{path.name}.aborted-{n}")).exists():
        n += 1
    path.rename(target)
    return target


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
        f"prompts: {config.prompts} {_prompt_state(config)}",
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


def _prompt_state(config: ExperimentConfig) -> str:
    from evcarry.harness import prompt_freeze

    found = prompt_freeze.status(config.prompts)
    required = ", frozen set required" if config.require_frozen_prompts else ""
    return f"({found.state}, manifest {found.manifest_sha256[:12]}{required})"


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
                run_config=harness_run_config(),
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
    for stale in (out_dir / "workspaces" / stem, out_dir / "logs" / stem):
        shutil.rmtree(stale, ignore_errors=True)  # left by an unfinished earlier attempt
    env = GymTaskEnv(
        task,
        images=_gym_images(config),
        policy=policy,
        log_dir=out_dir / "logs" / stem,
        setup_timeout=gym.setup_timeout_s,
        eval_timeout=gym.eval_timeout_s,
    )
    ledger = Ledger()
    try:
        await env.__aenter__()
    except (EnvironmentUnavailable, RuntimeError) as exc:
        # This task's environment could not start (a server, preprocess, or docker step failed).
        # The run ends here and is not graded; other runs continue.
        writer.write({"type": "environment", "kind": "docker", **env.info.as_dict(), "error": str(exc)[:2000]})
        return {
            "agent_status": "not_started",
            "status": "env_error",
            "passed": None,
            "eval_ran": False,
            "error": str(exc)[:1000],
            "gym_run_id": env.info.run_id,
            "timings": {key: round(value, 3) for key, value in env.info.timings.items()},
        }
    try:
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
    finally:
        await env.__aexit__(None, None, None)
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
    prompt_status: dict[str, object] | None = None,
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
        _set_aside(path)
    writer = JsonlWriter(path)
    token = current_writer.set(writer)
    pin_guard = PinGuard()
    pin_token = current_pin_guard.set(pin_guard)
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
                # Every model-visible text of the prompt set (evcarry.harness.prompt_freeze) and its freeze state.
                "prompt_manifest_sha256": (prompt_status or {}).get("manifest_sha256"),
                "prompts_frozen": (prompt_status or {}).get("state"),
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
        # A sub-agent tool or the judge may have caught the mismatch; the run still fails, without run_end.
        pin_guard.raise_if_violated()
        events = read_events(path)
        input_tokens = 0
        output_tokens = 0
        cost = 0.0
        reported = 0.0
        for event in events:
            if event.get("type") in _SPEND_EVENTS and isinstance(event.get("provider_reported_cost_usd"), (int, float)):
                reported += float(event["provider_reported_cost_usd"])
            if event.get("type") not in ("llm_call", "llm_error"):
                continue
            tokens = event.get("tokens") or {}
            input_tokens += int(tokens.get("input_tokens") or 0)
            output_tokens += int(tokens.get("output_tokens") or 0)
            cost += float(event.get("cost_usd") or 0)
        judge_cost = float(getattr(judge, "cost_usd", 0.0))
        spend = sum(event_spend(event) for event in events)
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
                "judge_fallbacks": int(getattr(judge, "fallbacks", 0)),
                "judge_cost_usd": judge_cost,
                # What the cap counts for this run (event_spend), and the provider's own bill.
                "spend_usd": spend,
                "provider_reported_cost_usd": reported,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
        )
        return spend
    finally:
        current_pin_guard.reset(pin_token)
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
    check_frozen_inputs(config)
    billable = model_config.billable or (config.judge_model in models and models[config.judge_model].billable)
    prompt_status = check_prompt_freeze(config, billable=billable)
    resolve_api_key(model_config)
    if config.judge_model in models:
        resolve_api_key(models[config.judge_model])
    if config.environment == "docker":
        DockerIsolation(_gym_images(config)).up()
    price = model_config.price_table()
    destination = out_dir or resolve_repo_path(config.out)
    destination.mkdir(parents=True, exist_ok=True)
    guard = BudgetGuard(
        cap_usd=config.budget.cap_usd,
        schedule=config.budget.schedule,
        price=price,
        spent_usd=logged_spend(destination),
    )
    written: list[Path] = []
    pending = []
    for task_name, arm_name, seed in grid(config):
        path = run_path(destination, task_name, arm_name, model_config.model or model_config.name, seed)
        if has_run_end(path):
            continue
        pending.append((task_name, arm_name, seed, path))

    async def _one(task_name: str, arm_name: str, seed: int) -> None:
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
            prompt_status=prompt_status,
        )
        guard.add(cost)

    if config.parallel_tasks <= 1 or stop_after is not None:
        finished = 0
        for task_name, arm_name, seed, path in pending:
            guard.check(billable=billable, now=datetime.now(timezone.utc))
            await _one(task_name, arm_name, seed)
            written.append(path)
            finished += 1
            if stop_after is not None and finished >= stop_after:
                break
    else:
        semaphore = asyncio.Semaphore(config.parallel_tasks)
        refusals: list[BudgetError] = []

        async def _bounded(item: tuple[str, str, int, Path]) -> Path | None:
            task_name, arm_name, seed, path = item
            async with semaphore:
                if refusals:
                    return None
                try:
                    guard.check(billable=billable, now=datetime.now(timezone.utc))
                except BudgetError as exc:
                    # Start nothing new; runs already going (and paid for) finish.
                    refusals.append(exc)
                    return None
                await _one(task_name, arm_name, seed)
                return path

        tasks = [asyncio.create_task(_bounded(item)) for item in pending]
        try:
            results = await asyncio.gather(*tasks)
        except BaseException:
            # A run failed: stop the others and let their environments tear down before re-raising.
            for item in tasks:
                item.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        written = [path for path in results if path is not None]
        if refusals:
            write_index(destination)
            raise refusals[0]
    if written or any(destination.glob("*.jsonl")):
        write_index(destination)
    return written
