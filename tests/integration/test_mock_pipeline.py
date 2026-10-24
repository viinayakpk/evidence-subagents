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
    with pytest.raises(SystemExit) as missing:
        main(["run", "configs/experiments/deepseek_smoke.yaml"])
    assert missing.value.code == 4
    with pytest.raises(SystemExit) as docker:
        main(["run", "configs/experiments/headline_deepseek.yaml"])
    assert docker.value.code == 5
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
