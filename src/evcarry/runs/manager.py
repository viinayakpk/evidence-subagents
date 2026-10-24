"""Expand an experiment grid, resume finished runs, and write JSONL traces.

Billable runs check the spend cap before they start. With a per-run cost
ceiling (``budget.run_ceiling_usd``) the check reserves the ceiling for every
run in flight, so parallel runs cannot overshoot the cap by a batch; each
run's meter refuses its model calls at the ceiling, and every run's calls at
the cap (``evcarry.runs.budget``). Off-peak is checked at the start of each
run, not mid-run. The cap counts everything already logged in the output
directory, including earlier invocations and aborted runs (kept as
``*.jsonl.aborted-<n>``), and every run in flight as it spends.

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
from evcarry.harness.run_settings import TEMPERATURE, harness_run_config
from evcarry.harness.tools import OVERLONG_META, VIEW_OVERLONG, ToolRouter, Toolset, fixture_toolset, overlong_view
from evcarry.inject.injector import FaultInjector
from evcarry.ledger.ledger import Ledger
from evcarry.ledger.store import write_ledger
from evcarry.paths import repo_root, resolve_repo_path
from evcarry.protocol.judge import build_judge
from evcarry.providers.pin import PinGuard, current_pin_guard
from evcarry.providers.registry import load_models, resolve_api_key
from evcarry.runs.budget import (
    _SPEND_EVENTS,
    BudgetError,
    BudgetGuard,
    CostCeilingExceeded,
    RunMeter,
    current_run_meter,
    event_spend,
)
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
        f"run_ceiling_usd: {config.budget.run_ceiling_usd}",
        f"max_steps (main agent): {config.limits.max_steps}",
        f"subagent_max_turns: {subagent_turns(config)}",
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


def subagent_turns(config: ExperimentConfig) -> int:
    return config.limits.subagent_max_turns or config.limits.max_steps


async def _drive(agent: Any, text: str, hooks: LedgerHooks, config: ExperimentConfig) -> dict[str, Any]:
    """Run the main agent. Turn, time and per-run cost limits end the run; other errors propagate."""
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
    except CostCeilingExceeded:
        return {"agent_status": "cost_ceiling"}
    return {"agent_status": "completed", "final_output": str(result.final_output)[-2000:]}


# A run stopped at its cost ceiling is not graded: it counts as failed (owner decision 5).
_NOT_GRADED_AT_CEILING = {"status": "cost_ceiling", "passed": False, "eval_ran": False}


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
    if outcome["agent_status"] == "cost_ceiling":
        return {**outcome, **_NOT_GRADED_AT_CEILING}
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
    from evcarry.env.gym_docker import CLAIM_DONE, PYTHON_EXECUTE, GymTaskEnv, ground_truth_suspects
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
        if outcome["agent_status"] == "cost_ceiling":
            evaluation = dict(_NOT_GRADED_AT_CEILING)
        else:
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
        "sealed": env.info.sealed,
        # Tool calls that reached for answers (removed task files, the gym's git history or public repo).
        "ground_truth_suspects": ground_truth_suspects(ledger),
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
    meter: RunMeter | None = None,
) -> float:
    _install_trace_processors()
    meter = meter or RunMeter(config.budget.run_ceiling_usd)
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
    writer = JsonlWriter(path, observer=meter.observe)
    token = current_writer.set(writer)
    pin_guard = PinGuard()
    pin_token = current_pin_guard.set(pin_guard)
    meter_token = current_run_meter.set(meter)
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
                # Sent on every agent and judge call of every arm (llm_call/judge_call log what each request carried).
                "temperature": TEMPERATURE,
                "judge_temperature": TEMPERATURE,
                "max_steps": config.limits.max_steps,
                "subagent_max_turns": subagent_turns(config),
                "run_ceiling_usd": meter.ceiling_usd,
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
                subagent_max_turns=subagent_turns(config),
                judge=judge,
            )

        runner = _run_gym if config.environment == "docker" else _run_fixture
        outcome = await runner(
            task=task, out_dir=out_dir, stem=stem, build=build, writer=writer, price=price, config=config
        )
        # A sub-agent tool or the judge may have caught the mismatch; the run still fails, without run_end.
        pin_guard.raise_if_violated()
        # A call refused at the experiment cap fails the run without run_end, like a pin violation.
        meter.raise_if_cap_reached()
        verdict = final_verdict(outcome, ceiling_hit=meter.ceiling_hit)
        events = read_events(path)
        input_tokens = 0
        output_tokens = 0
        cost = 0.0
        reported = 0.0
        judge_error_cost = 0.0
        for event in events:
            if event.get("type") in _SPEND_EVENTS and isinstance(event.get("provider_reported_cost_usd"), (int, float)):
                reported += float(event["provider_reported_cost_usd"])
            if event.get("type") == "llm_error" and event.get("role") == "judge":
                judge_error_cost += float(event.get("cost_usd") or 0)
                continue
            if event.get("type") not in ("llm_call", "llm_error"):
                continue
            tokens = event.get("tokens") or {}
            input_tokens += int(tokens.get("input_tokens") or 0)
            output_tokens += int(tokens.get("output_tokens") or 0)
            cost += float(event.get("cost_usd") or 0)
        judge_cost = float(getattr(judge, "cost_usd", 0.0)) + judge_error_cost
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
                **verdict,
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
    except Exception as exc:
        # The run ends without run_end (a restart runs it again); say why in its trace.
        writer.write({"type": "run_error", "error_type": type(exc).__name__, "error": str(exc)[:2000]})
        raise
    finally:
        current_run_meter.reset(meter_token)
        current_pin_guard.reset(pin_token)
        current_writer.reset(token)
        writer.close()


def final_verdict(outcome: dict[str, Any], *, ceiling_hit: bool = False) -> dict[str, Any]:
    """``passed`` as the paper counts it, next to the grader's own ``eval_passed``.

    Pre-registered (owner decision 4, 2026-10-24): a run whose tool calls reached for
    ground truth (``ground_truth_suspects``: removed task files, the gym's git history,
    or its public repo) is ``contaminated`` and counts as a failure, whatever the
    grader said. An ungraded run (``passed: null``) stays ungraded.

    Owner decision 5: a run in which any model call was refused at the per-run cost
    ceiling has status ``cost_ceiling`` and counts as failed, even if the main agent
    went on to finish (a sub-agent or judge call was cut short).
    """
    eval_passed = outcome.get("passed")
    contaminated = bool(outcome.get("ground_truth_suspects"))
    passed = False if contaminated and eval_passed is not None else eval_passed
    verdict: dict[str, Any] = {"eval_passed": eval_passed, "contaminated": contaminated, "passed": passed, "cost_ceiling": ceiling_hit}
    if ceiling_hit:
        verdict.update(passed=False, status="cost_ceiling")
    return verdict


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
        run_ceiling_usd=config.budget.run_ceiling_usd,
    )
    written: list[Path] = []
    pending = []
    for task_name, arm_name, seed in grid(config):
        path = run_path(destination, task_name, arm_name, model_config.model or model_config.name, seed)
        if has_run_end(path):
            continue
        pending.append((task_name, arm_name, seed, path))

    def _admit() -> RunMeter:
        """Check the cap and the schedule, then count the new run as in flight (no await in between)."""
        guard.check(billable=billable, now=datetime.now(timezone.utc))
        meter = RunMeter(config.budget.run_ceiling_usd, guard if billable else None)
        guard.open(meter)
        return meter

    async def _one(task_name: str, arm_name: str, seed: int, meter: RunMeter) -> None:
        try:
            script = _script_for(arm_name, script_overrides, model_config.kind, config.script_dir)
            await _run_one(
                config=config,
                task_name=task_name,
                arm_name=arm_name,
                seed=seed,
                out_dir=destination,
                script=script,
                model_config=model_config,
                price=price,
                prompt_status=prompt_status,
                meter=meter,
            )
        finally:
            # Finished, failed or aborted: what the run spent stays counted.
            guard.close(meter)

    if config.parallel_tasks <= 1 or stop_after is not None:
        finished = 0
        for task_name, arm_name, seed, path in pending:
            await _one(task_name, arm_name, seed, _admit())
            written.append(path)
            finished += 1
            if stop_after is not None and finished >= stop_after:
                break
    else:
        semaphore = asyncio.Semaphore(config.parallel_tasks)
        refusals: list[BudgetError] = []
        failures: list[Exception] = []

        async def _bounded(item: tuple[str, str, int, Path]) -> Path | None:
            task_name, arm_name, seed, path = item
            async with semaphore:
                if refusals or failures:
                    return None
                try:
                    meter = _admit()
                except BudgetError as exc:
                    # Start nothing new; runs already going (and paid for) finish.
                    refusals.append(exc)
                    return None
                try:
                    await _one(task_name, arm_name, seed, meter)
                except Exception as exc:  # noqa: BLE001 - re-raised once the runs in flight are done
                    # A failed run (provider error, provider mismatch, harness bug) stops new runs, but the
                    # runs already going were paid for and finish. The failed run has no run_end, so a
                    # restart runs it again; its spend is counted from its trace.
                    failures.append(exc)
                    return None
                return path

        tasks = [asyncio.create_task(_bounded(item)) for item in pending]
        try:
            results = await asyncio.gather(*tasks)
        except BaseException:
            # Interrupted (Ctrl-C): stop the others and let their environments tear down before re-raising.
            for item in tasks:
                item.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        written = [path for path in results if path is not None]
        if failures or refusals:
            write_index(destination)
        if failures:
            raise failures[0]
        if refusals:
            raise refusals[0]
    if written or any(destination.glob("*.jsonl")):
        write_index(destination)
    return written
