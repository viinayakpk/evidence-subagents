import json
from pathlib import Path

import pytest

from evcarry.analysis.fault_injection import catch_table
from evcarry.runs.cli import main
from evcarry.runs.config import InjectConfig, load_experiment
from evcarry.runs.manager import run_experiment
from evcarry.trace.writer import read_events


def _logical(path: Path) -> list[tuple]:
    rows = []
    for event in read_events(path):
        rows.append(
            (
                event.get("type"),
                event.get("action"),
                event.get("ok"),
                event.get("passed"),
                event.get("status"),
                event.get("arm"),
            )
        )
    return rows


async def test_mock_e2e_writes_jsonl(tmp_path: Path) -> None:
    config = load_experiment("configs/experiments/mock_e2e.yaml")
    await run_experiment(config, out_dir=tmp_path)
    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 9
    for path in files:
        events = read_events(path)
        kinds = [event["type"] for event in events]
        assert kinds[0] == "run_start"
        assert kinds[-1] == "run_end"
        assert any(event["type"] == "llm_call" and event["served_provider"] == "scripted" for event in events)
        end = events[-1]
        assert end["passed"] is True
        assert end["cost_usd"] == 0
        start = events[0]
        assert start["served_model"] == "mock-scripted"
        assert len(start["git_commit"]) >= 7
    sections = next(path for path in files if path.name.startswith("fixture-sections__evidence_risk"))
    evidence = read_events(sections)
    assert [event["action"] for event in evidence if event["type"] == "policy"] == ["accept", "accept"]
    writes = [event for event in evidence if event.get("tool") == "write_answer"]
    assert len(writes) == 1


async def test_corrupt_return_redelegates(tmp_path: Path) -> None:
    config = load_experiment("configs/experiments/mock_e2e.yaml")
    config.arms = ["evidence_risk"]
    config.tasks = "tests/fixtures/tasks/one_sections.txt"
    await run_experiment(
        config,
        out_dir=tmp_path,
        script_overrides={"evidence_risk": Path("tests/fixtures/scripts/evidence_corrupt.json")},
    )
    events = read_events(next(tmp_path.glob("*.jsonl")))
    actions = [event["action"] for event in events if event["type"] == "policy"]
    checks = [event["ok"] for event in events if event["type"] == "check"]
    assert actions[:2] == ["redelegate", "accept"]
    assert checks[0] is False
    assert checks[1] is True
    assert events[-1]["passed"] is True


async def test_injector_numeric_redelegates_once(tmp_path: Path) -> None:
    config = load_experiment("configs/experiments/mock_e2e.yaml")
    config.arms = ["evidence_risk"]
    config.tasks = "tests/fixtures/tasks/one_sections.txt"
    config.inject = InjectConfig(
        probability=1,
        types=["numeric_perturbation"],
        variant="naive",
        max_per_run=1,
    )
    await run_experiment(
        config,
        out_dir=tmp_path,
        script_overrides={"evidence_risk": Path("tests/fixtures/scripts/evidence_reinject.json")},
    )
    events = read_events(next(tmp_path.glob("*.jsonl")))
    injections = [event for event in events if event["type"] == "injection"]
    assert len(injections) == 1
    assert injections[0]["kind"] == "numeric_perturbation"
    checks = [event for event in events if event["type"] == "check"]
    assert checks[0]["ok"] is False
    assert any("11" in failure["reason"] for failure in checks[0]["failures"])
    assert [event["action"] for event in events if event["type"] == "policy"][:2] == ["redelegate", "accept"]
    assert events[-1]["passed"] is True


async def test_resume_skips_finished_file(tmp_path: Path) -> None:
    config = load_experiment("configs/experiments/mock_e2e.yaml")
    first_dir = tmp_path / "partial"
    await run_experiment(config, out_dir=first_dir, stop_after=1)
    finished = next(first_dir.glob("*.jsonl"))
    before = finished.read_bytes()
    await run_experiment(config, out_dir=first_dir)
    assert finished.read_bytes() == before
    fresh = tmp_path / "fresh"
    await run_experiment(config, out_dir=fresh)
    partial = {path.name: _logical(path) for path in first_dir.glob("*.jsonl")}
    full = {path.name: _logical(path) for path in fresh.glob("*.jsonl")}
    assert partial.keys() == full.keys()
    assert partial == full


async def test_restart_counts_spend_already_logged_and_keeps_aborted_traces(tmp_path: Path, monkeypatch) -> None:
    """The cap used to start at $0 on every invocation, and an unfinished trace (paid calls) was deleted."""
    from evcarry.runs.budget import BudgetGuard
    from evcarry.trace.writer import run_path

    config = load_experiment("configs/experiments/mock_e2e.yaml")
    config.arms = ["sa", "vanilla_mad"]
    config.tasks = "tests/fixtures/tasks/one_sections.txt"
    finished = run_path(tmp_path, "fixture-sections", "sa", "mock-scripted", 1)
    finished.write_text(
        "\n".join(
            json.dumps(event)
            for event in (
                {"type": "run_start"},
                {"type": "llm_call", "cost_usd": 0.6},
                {"type": "run_end", "task": "fixture-sections", "arm": "sa", "seed": 1, "passed": True, "cost_usd": 0.6},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    unfinished = run_path(tmp_path, "fixture-sections", "vanilla_mad", "mock-scripted", 1)
    unfinished.write_text(
        '{"type": "run_start"}\n{"type": "llm_call", "cost_usd": 0.5}\n{"type": "judge_call", "cost_usd": 0.1}\n{"type": "llm_',
        encoding="utf-8",
    )
    seen: list[float] = []
    original = BudgetGuard.check

    def check(self, *, billable, now):
        seen.append(self.spent_usd)
        return original(self, billable=billable, now=now)

    monkeypatch.setattr(BudgetGuard, "check", check)
    await run_experiment(config, out_dir=tmp_path)
    assert seen == [pytest.approx(1.2)]
    aborted = tmp_path / f"{unfinished.name}.aborted-1"
    assert aborted.is_file() and '"cost_usd": 0.5' in aborted.read_text(encoding="utf-8")
    assert read_events(unfinished)[-1]["type"] == "run_end"


async def test_an_edited_frozen_task_list_is_refused(tmp_path: Path, monkeypatch) -> None:
    from evcarry.runs import manager

    edited = tmp_path / "subset.txt"
    edited.write_text("fixture-sections\n", encoding="utf-8")
    sums = tmp_path / "SHA256SUMS"
    sums.write_text(f"{'0' * 64}  {edited}\n", encoding="utf-8")
    monkeypatch.setattr(manager, "SHA256SUMS", str(sums))
    config = load_experiment("configs/experiments/mock_e2e.yaml")
    config.tasks = str(edited)
    with pytest.raises(manager.FrozenInputMismatch):
        await run_experiment(config, out_dir=tmp_path / "out")
    assert not (tmp_path / "out").exists()
    sums.write_text(f"{manager._sha256_file(edited)}  {edited}\n", encoding="utf-8")
    config.arms = ["sa"]
    await run_experiment(config, out_dir=tmp_path / "out")
    # The real frozen lists match their recorded hashes, and the gym checkout is the frozen commit.
    monkeypatch.setattr(manager, "SHA256SUMS", "tasks/SHA256SUMS")
    manager.check_frozen_inputs(load_experiment("configs/experiments/pilot_deepseek.yaml"))


def test_cli_dry_run_and_exit_codes(capsys, monkeypatch) -> None:
    main(["run", "configs/experiments/mock_e2e.yaml", "--dry-run"])
    out = capsys.readouterr().out
    assert "fixture-sections" in out
    assert "api keys: not required" in out
    main(["run", "configs/experiments/headline_deepseek.yaml", "--dry-run"])
    headline = capsys.readouterr().out
    assert "run_parallel.sh" in headline
    assert "not called" in headline
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(SystemExit) as missing:
        main(["run", "configs/experiments/deepseek_smoke.yaml"])
    assert missing.value.code == 4
    with pytest.raises(SystemExit) as missing_gym:
        main(["run", "configs/experiments/pilot_deepseek.yaml"])  # OpenRouter route: needs OPENROUTER_API_KEY
    assert missing_gym.value.code == 4
    # The key check passes with a placeholder; the Docker preflight then refuses
    # before any run starts, so nothing is sent to the provider.
    monkeypatch.setenv("OPENROUTER_API_KEY", "placeholder-not-a-key")
    monkeypatch.setenv("EVCARRY_DOCKER", "/nonexistent/docker")
    with pytest.raises(SystemExit) as docker:
        main(["run", "configs/experiments/pilot_deepseek.yaml"])
    assert docker.value.code == 5
    # The headline needs a frozen prompt set; gym_v1 is frozen only after the pilot.
    with pytest.raises(SystemExit) as unfrozen:
        main(["run", "configs/experiments/headline_deepseek.yaml"])
    assert unfrozen.value.code == 6
    main(["check", "configs/experiments/openrouter_debug.yaml"])
    checked = capsys.readouterr().out
    assert "openrouter_pinned" in checked or "openrouter" in checked


def test_fault_table_expected_catchers() -> None:
    found = {(row["kind"], row["variant"]): row["earliest"] for row in catch_table()}
    expected = {
        ("numeric_perturbation", "naive"): "L1",
        ("numeric_perturbation", "pointer_consistent"): "L1",
        ("fabricated_entity", "naive"): "L2",
        ("fabricated_entity", "pointer_consistent"): "L0",
        ("omission", "naive"): None,
        ("omission", "pointer_consistent"): None,
        ("false_action_done", "naive"): "L2",
        ("false_action_done", "pointer_consistent"): "L2",
        ("mismatched_pointer", "naive"): "L1",
        ("mismatched_pointer", "pointer_consistent"): "L3",
    }
    assert found == expected


def test_summarize_cli(tmp_path: Path, capsys) -> None:
    import asyncio

    config = load_experiment("configs/experiments/mock_e2e.yaml")
    config.arms = ["sa"]
    config.tasks = "tests/fixtures/tasks/one_sections.txt"
    asyncio.run(run_experiment(config, out_dir=tmp_path))
    main(["summarize", str(tmp_path)])
    table = capsys.readouterr().out
    assert "fixture-sections" in table
    assert "True" in table


async def test_llm_judge_verdicts_are_traced(tmp_path: Path) -> None:
    config = load_experiment("configs/experiments/mock_e2e.yaml")
    config.arms = ["evidence_risk"]
    config.tasks = "tests/fixtures/tasks/one_sections.txt"
    config.judge_model = "mock_judge"
    await run_experiment(config, out_dir=tmp_path)
    events = read_events(next(tmp_path.glob("*.jsonl")))
    calls = [event for event in events if event["type"] == "judge_call"]
    assert calls and all(call["source"] == "llm" and call["verdict"] == "supported" for call in calls)
    verdicts = [v for event in events if event["type"] == "check" for v in event["verdicts"]]
    assert verdicts and all(v["source"] == "llm" and v["detail"]["model"] == "mock_judge" for v in verdicts)
    assert events[0]["judge_model"] == "mock_judge"
    assert events[-1]["judge_calls"] == len(calls)
    assert events[-1]["passed"] is True


async def test_budget_refusal_lets_running_runs_finish(tmp_path: Path, monkeypatch) -> None:
    from evcarry.runs.budget import BudgetGuard, PeakWindowError

    calls = {"n": 0}

    def check(self, *, billable, now):
        calls["n"] += 1
        if calls["n"] == 3:
            raise PeakWindowError("peak window reached")

    monkeypatch.setattr(BudgetGuard, "check", check)
    config = load_experiment("configs/experiments/mock_e2e.yaml")
    config.parallel_tasks = 2
    with pytest.raises(PeakWindowError):
        await run_experiment(config, out_dir=tmp_path)
    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 2
    assert all(read_events(path)[-1]["type"] == "run_end" for path in files)


async def test_a_swallowed_provider_mismatch_still_fails_the_run(tmp_path: Path, monkeypatch) -> None:
    """A sub-agent tool or the judge can catch ProviderMismatch; the run must still fail, without run_end."""
    from evcarry.providers.pin import ProviderMismatch, current_pin_guard
    from evcarry.runs import manager

    async def off_pin_run(**_kwargs):
        current_pin_guard.get().violate("served by Novita, pinned to ['deepinfra/fp8']")
        return {"agent_status": "completed", "passed": True, "status": "ok"}

    monkeypatch.setattr(manager, "_run_fixture", off_pin_run)
    config = load_experiment("configs/experiments/mock_e2e.yaml")
    config.arms = ["sa"]
    config.tasks = "tests/fixtures/tasks/one_sections.txt"
    with pytest.raises(ProviderMismatch, match="Novita"):
        await run_experiment(config, out_dir=tmp_path)
    (trace,) = tmp_path.glob("*.jsonl")
    events = read_events(trace)
    assert [event["type"] for event in events] == ["run_start", "run_error"]
    assert events[-1]["error_type"] == "ProviderMismatch"
    assert current_pin_guard.get() is None  # the guard does not leak into the next run


def test_cli_exit_code_for_a_provider_mismatch(monkeypatch, capsys) -> None:
    from evcarry.providers.pin import ProviderMismatch
    from evcarry.runs import cli

    async def refuse(_config):
        raise ProviderMismatch("served by Novita")

    monkeypatch.setattr(cli, "run_experiment", refuse)
    with pytest.raises(SystemExit) as stopped:
        main(["run", "configs/experiments/mock_e2e.yaml"])
    assert stopped.value.code == 7 and "Novita" in capsys.readouterr().err


async def test_a_failed_run_stops_new_runs_but_lets_paid_runs_finish(tmp_path: Path, monkeypatch) -> None:
    """One provider error used to cancel every run in flight, losing work already paid for."""
    import asyncio

    from evcarry.runs import manager

    original = manager._run_fixture
    started = {"n": 0}

    async def flaky(**kwargs):
        started["n"] += 1
        if started["n"] == 1:
            await asyncio.sleep(0.3)  # still running when its sibling fails
            return await original(**kwargs)
        raise RuntimeError("provider returned 503 after retries")

    monkeypatch.setattr(manager, "_run_fixture", flaky)
    config = load_experiment("configs/experiments/mock_e2e.yaml")
    config.parallel_tasks = 2
    with pytest.raises(RuntimeError, match="503"):
        await run_experiment(config, out_dir=tmp_path)
    ends = {path.name: read_events(path)[-1]["type"] for path in tmp_path.glob("*.jsonl")}
    assert sorted(ends.values()) == ["run_end", "run_error"]  # nothing new started after the failure
    assert started["n"] == 2


async def test_run_end_splits_agent_and_judge_cost_and_counts_every_billed_event(tmp_path: Path, monkeypatch) -> None:
    from evcarry.runs import manager
    from evcarry.trace.writer import current_writer

    async def billed_run(**_kwargs):
        writer = current_writer.get()
        writer.write({"type": "llm_call", "cost_usd": 0.01, "provider_reported_cost_usd": 0.01, "tokens": {"input_tokens": 100, "output_tokens": 10}})
        writer.write({"type": "llm_error", "role": "agent", "cost_usd": 0.002, "provider_reported_cost_usd": 0.002})
        writer.write({"type": "llm_error", "role": "judge", "cost_usd": 0.003, "provider_reported_cost_usd": 0.003})
        writer.write({"type": "judge_call", "cost_usd": 0.004, "provider_reported_cost_usd": 0.005})
        return {"agent_status": "completed", "passed": True, "status": "ok"}

    monkeypatch.setattr(manager, "_run_fixture", billed_run)
    config = load_experiment("configs/experiments/mock_e2e.yaml")
    config.arms, config.tasks = ["sa"], "tests/fixtures/tasks/one_sections.txt"
    await run_experiment(config, out_dir=tmp_path)
    end = read_events(next(tmp_path.glob("*.jsonl")))[-1]
    assert end["cost_usd"] == pytest.approx(0.012)  # agent calls, including the answered-then-failed one
    assert end["judge_cost_usd"] == pytest.approx(0.003)  # the lexical judge object costs 0; the failed judge call counts
    assert end["spend_usd"] == pytest.approx(0.01 + 0.002 + 0.003 + 0.005)  # the larger of table and bill, per event
    assert end["provider_reported_cost_usd"] == pytest.approx(0.02)
