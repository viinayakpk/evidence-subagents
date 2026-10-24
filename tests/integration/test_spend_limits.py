"""Owner decision 5: per-run cost ceiling, turn limits, and a hard experiment cap."""

from datetime import datetime, time, timezone
from pathlib import Path

import pytest

from evcarry.providers.usage import PriceTable
from evcarry.runs.budget import (
    BudgetGuard,
    CostCeilingExceeded,
    RunMeter,
    SpendCapExceeded,
)
from evcarry.runs.config import load_experiment
from evcarry.runs.manager import logged_spend, run_experiment
from evcarry.trace.writer import read_events

# Every scripted call has 20 cache-miss input tokens: $0.02 a call at $1000 per million.
COSTLY = PriceTable(1000.0, 0.0, 0.0, peak_multiplier=1, peak_windows_utc=())
NOW = datetime(2026, 10, 24, 12, 0, tzinfo=timezone.utc)


def _costly_mock(monkeypatch, *, billable: bool = False) -> None:
    from evcarry.providers import registry
    from evcarry.runs import manager

    real = registry.load_models

    def models(path=None):
        loaded = real(path)
        loaded["mock"].billable = billable
        return loaded

    monkeypatch.setattr(manager, "load_models", models)
    monkeypatch.setattr(registry.ModelConfig, "price_table", lambda self: COSTLY)


def _spend(event: dict) -> float:
    from evcarry.runs.budget import event_spend

    return event_spend(event)


def test_meter_refuses_at_the_ceiling_and_at_the_cap() -> None:
    meter = RunMeter(0.05)
    meter.observe({"type": "llm_call", "cost_usd": 0.03})
    meter.admit()  # 0.03 < 0.05
    meter.observe({"type": "judge_call", "cost_usd": 0.01, "provider_reported_cost_usd": 0.02})  # the larger counts
    with pytest.raises(CostCeilingExceeded):
        meter.admit()
    with pytest.raises(CostCeilingExceeded):
        meter.admit()  # every later call too
    assert meter.ceiling_hit and meter.spent_usd == pytest.approx(0.05)
    meter.raise_if_cap_reached()  # a ceiling is not a cap refusal

    guard = BudgetGuard(cap_usd=1.0, schedule="any", price=COSTLY, spent_usd=0.99)
    capped = RunMeter(None, guard)
    guard.open(capped)
    capped.admit()
    capped.observe({"type": "llm_call", "cost_usd": 0.02})
    with pytest.raises(SpendCapExceeded):
        capped.admit()
    with pytest.raises(SpendCapExceeded):
        capped.raise_if_cap_reached()
    guard.close(capped)
    assert guard.spent_usd == pytest.approx(1.01) and guard.live == []


def test_admission_reserves_the_ceiling_for_runs_in_flight() -> None:
    guard = BudgetGuard(cap_usd=5.0, schedule="any", price=COSTLY, spent_usd=4.0, run_ceiling_usd=0.25)
    assert guard.reserve_usd() == pytest.approx(0.30)
    first, second = RunMeter(0.25, guard), RunMeter(0.25, guard)
    for meter in (first, second):
        guard.open(meter)
        meter.observe({"type": "llm_call", "cost_usd": 0.10})
    # spent 4.0 + live 0.2 + reserved (0.2 + 0.2) + this run 0.3 = 4.9
    guard.check(billable=True, now=NOW)
    third = RunMeter(0.25, guard)
    guard.open(third)
    # spent 4.0 + live 0.2 + reserved (0.2 + 0.2 + 0.3) + this run 0.3 = 5.2
    with pytest.raises(SpendCapExceeded, match="reserved"):
        guard.check(billable=True, now=NOW)
    guard.check(billable=False, now=NOW)  # non-billable runs skip the cap
    # Without a ceiling only the spend so far counts, as before.
    loose = BudgetGuard(cap_usd=5.0, schedule="any", price=COSTLY, spent_usd=4.9)
    loose.check(billable=True, now=NOW)


async def test_judge_refused_at_the_ceiling_does_not_fall_back_to_lexical() -> None:
    from evcarry.protocol.judge import LlmJudge
    from evcarry.protocol.schema import Claim

    class _Refusing:
        async def get_response(self, **_kwargs):
            raise CostCeilingExceeded("run stopped at its cost ceiling")

    judge = LlmJudge(_Refusing(), model_name="m", prompt="p", price=COSTLY)
    claim = Claim(claim_id="c1", text="14 sections", kind="fact", value=14, evidence=[])
    with pytest.raises(CostCeilingExceeded):
        await judge.ajudge(claim, "14 sections")
    assert judge.fallbacks == 0


async def test_a_run_at_its_cost_ceiling_stops_fails_and_keeps_its_spend(tmp_path: Path, monkeypatch) -> None:
    _costly_mock(monkeypatch)
    config = load_experiment("configs/experiments/mock_e2e.yaml")
    config.arms, config.tasks = ["sa"], "tests/fixtures/tasks/one_sections.txt"
    config.budget.run_ceiling_usd = 0.03
    await run_experiment(config, out_dir=tmp_path)
    events = read_events(next(tmp_path.glob("*.jsonl")))
    start, end = events[0], events[-1]
    assert start["run_ceiling_usd"] == 0.03
    assert (end["type"], end["agent_status"], end["status"], end["passed"]) == ("run_end", "cost_ceiling", "cost_ceiling", False)
    assert end["cost_ceiling"] is True and end["eval_ran"] is False and end["eval_passed"] is None  # never graded
    calls = [event for event in events if event["type"] == "llm_call"]
    assert len(calls) == 2  # $0.02 each: the second crosses $0.03, the third is refused
    assert end["spend_usd"] == pytest.approx(0.04) == logged_spend(tmp_path)
    # A finished run: a restart does not run it again, and its spend stays counted.
    before = next(tmp_path.glob("*.jsonl")).read_bytes()
    await run_experiment(config, out_dir=tmp_path)
    assert next(tmp_path.glob("*.jsonl")).read_bytes() == before


async def test_a_sub_agent_refused_at_the_ceiling_fails_the_run(tmp_path: Path, monkeypatch) -> None:
    """The sub-agent tool turns the refusal into tool-error text; the main agent's next call is refused too."""
    _costly_mock(monkeypatch)
    config = load_experiment("configs/experiments/mock_e2e.yaml")
    config.arms, config.tasks = ["vanilla_mad"], "tests/fixtures/tasks/one_sections.txt"
    config.budget.run_ceiling_usd = 0.03  # main $0.02, then the researcher's first call crosses it; its second is refused
    await run_experiment(config, out_dir=tmp_path)
    events = read_events(next(tmp_path.glob("*.jsonl")))
    errors = [event for event in events if event["type"] == "subagent_error"]
    assert errors and errors[0]["error_type"] == "CostCeilingExceeded"
    end = events[-1]
    assert (end["status"], end["agent_status"], end["passed"], end["cost_ceiling"]) == ("cost_ceiling", "cost_ceiling", False, True)
    assert len([event for event in events if event["type"] == "llm_call"]) == 2


async def test_turn_limits_reach_the_sub_agents(tmp_path: Path, monkeypatch) -> None:
    from evcarry.harness import agents as agents_module

    seen: dict[str, int] = {}
    original = agents_module.build_main_agent

    def spy(**kwargs):
        seen["subagent_max_turns"] = kwargs["subagent_max_turns"]
        return original(**kwargs)

    monkeypatch.setattr("evcarry.runs.manager.build_main_agent", spy)
    config = load_experiment("configs/experiments/mock_e2e.yaml")
    config.arms, config.tasks = ["vanilla_mad"], "tests/fixtures/tasks/one_sections.txt"
    config.limits.subagent_max_turns = 7
    await run_experiment(config, out_dir=tmp_path)
    start = read_events(next(tmp_path.glob("*.jsonl")))[0]
    assert seen == {"subagent_max_turns": 7}
    assert (start["max_steps"], start["subagent_max_turns"]) == (12, 7)
    for name in ("pilot_deepseek", "headline_deepseek"):
        paid = load_experiment(f"configs/experiments/{name}.yaml")
        assert paid.limits.max_steps == 40 and paid.limits.subagent_max_turns == 40
        assert paid.budget.run_ceiling_usd == 0.25
    # Pilot: the owner's $5 budget, run with the stop rule "no run may take spend past $4.50".
    assert load_experiment("configs/experiments/pilot_deepseek.yaml").budget.cap_usd == 4.5
    assert load_experiment("configs/experiments/headline_deepseek.yaml").budget.cap_usd == 45


async def test_parallel_billable_runs_stay_under_the_cap(tmp_path: Path, monkeypatch) -> None:
    """Admission reserves 1.2 x the ceiling per run in flight, so a parallel grid stops before the cap, not a batch after."""
    _costly_mock(monkeypatch, billable=True)
    config = load_experiment("configs/experiments/mock_e2e.yaml")
    config.parallel_tasks = 3
    config.budget.cap_usd = 0.30
    config.budget.run_ceiling_usd = 0.05
    with pytest.raises(SpendCapExceeded):
        await run_experiment(config, out_dir=tmp_path)
    total = logged_spend(tmp_path)
    assert 0 < total <= 0.30
    ends = [read_events(path)[-1] for path in tmp_path.glob("*.jsonl")]
    assert ends and all(end["type"] == "run_end" for end in ends)
    assert sum(end["spend_usd"] for end in ends) == pytest.approx(total)


async def test_a_reservation_refusal_waits_for_runs_in_flight_instead_of_stopping_the_grid(tmp_path: Path, monkeypatch) -> None:
    """Only the reserves of runs in flight blocked the third run; it used to stop every remaining run (exit 2)."""
    _costly_mock(monkeypatch, billable=True)
    config = load_experiment("configs/experiments/mock_e2e.yaml")
    config.parallel_tasks = 3
    config.budget.run_ceiling_usd = 1.0  # reserve $1.20 a run; the nine runs spend well under $1 together
    config.budget.cap_usd = 2.5  # two reserves fit, a third does not
    await run_experiment(config, out_dir=tmp_path)
    ends = [read_events(path)[-1] for path in tmp_path.glob("*.jsonl")]
    assert len(ends) == 9 and all(end["type"] == "run_end" and end["passed"] for end in ends)
    assert logged_spend(tmp_path) < 2.5
