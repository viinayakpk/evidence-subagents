"""P2-1 (one reference-priced ceiling, 100 turns), P3-2 (the context window is enforced), P3-3 (judge reasoning)."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from agents.run_config import CallModelData, ModelInputData

from evcarry.harness.context import OUTPUT_RESERVE_TOKENS, ContextTrimmer, current_trimmer
from evcarry.harness.run_settings import harness_run_config
from evcarry.providers.registry import load_models
from evcarry.providers.usage import PriceTable, Rates, TokenUsage, call_cost
from evcarry.runs.budget import BudgetGuard, CostCeilingExceeded, RunMeter, event_reference_spend
from evcarry.runs.config import load_experiment
from evcarry.trace.writer import current_writer

NOON = datetime(2026, 10, 25, 12, 0, tzinfo=timezone.utc)
PRICE = load_models()["openrouter_ds_1017"].price_table()


# P2-1 -------------------------------------------------------------------------------------------------


def test_every_call_is_also_priced_at_the_reference_endpoint() -> None:
    assert PRICE.reference_endpoint == "deepinfra/fp8"
    assert PRICE.reference_rates() == Rates(0.098, 0.00294, 0.294)
    tokens = TokenUsage(input_tokens=1_100_000, cache_hit_tokens=1_000_000, cache_miss_tokens=100_000, output_tokens=100_000)
    served = call_cost(tokens, PRICE, NOON, "InferenceNet")
    assert served.priced_as == "inference-net/fp8|inference-net/us"
    assert served.usd == pytest.approx((100_000 * 0.08 + 1_000_000 * 0.008 + 100_000 * 0.50) / 1e6)
    assert served.reference_usd == pytest.approx((100_000 * 0.098 + 1_000_000 * 0.00294 + 100_000 * 0.294) / 1e6)
    assert served.fields()["reference_cost_usd"] == served.reference_usd
    # The worst component ratio over the set: InferenceNet's cache read is 2.72x DeepInfra's.
    assert PRICE.billed_per_reference_bound() == pytest.approx(0.008 / 0.00294)
    plain = PriceTable(1.0, 0.1, 2.0, peak_multiplier=1, peak_windows_utc=())
    assert plain.reference_rates() is None and call_cost(tokens, plain, NOON, None).fields()["reference_cost_usd"] == pytest.approx(
        call_cost(tokens, plain, NOON, None).table_usd
    )
    with pytest.raises(ValueError, match="reference_endpoint"):
        PriceTable.from_mapping({"reference_endpoint": "nope/fp8", "endpoints": {"a/fp8": {"input_cache_miss": 1, "input_cache_hit": 0.1, "output": 2}}})


def test_a_reference_ceiling_counts_reference_dollars_and_the_cap_counts_the_bill() -> None:
    meter = RunMeter(0.15, basis="reference")
    meter.observe({"type": "llm_call", "cost_usd": 0.20, "reference_cost_usd": 0.10})
    meter.admit()  # billed 0.20 is above the ceiling, but the reference spend is 0.10
    assert (meter.spent_usd, meter.reference_usd) == (pytest.approx(0.20), pytest.approx(0.10))
    meter.observe({"type": "judge_call", "cost_usd": 0.12, "reference_cost_usd": 0.06})
    with pytest.raises(CostCeilingExceeded, match="reference"):
        meter.admit()
    assert meter.ceiling_hit
    # An event without a reference cost counts its billed spend.
    assert event_reference_spend({"type": "llm_call", "cost_usd": 0.01, "provider_reported_cost_usd": 0.02}) == 0.02
    assert event_reference_spend({"type": "check"}) == 0.0
    with pytest.raises(ValueError):
        RunMeter(0.1, basis="tokens")
    guard = BudgetGuard(cap_usd=45, schedule="any", price=PRICE, run_ceiling_usd=0.17, reserve_billed_per_reference=1.6)
    assert guard.reserve_usd() == pytest.approx(1.2 * 0.17 * 1.6)


def test_headline_and_dev_configs() -> None:
    headline = load_experiment("configs/experiments/headline_deepseek.yaml")
    assert headline.tasks == "tasks/headline60.txt" and headline.seeds == [1, 2]
    assert headline.arms == ["sa", "vanilla_mad", "self_verify", "evidence_risk"]
    assert headline.gym.network_internal and headline.gym.hide_task_dir and headline.require_frozen_prompts
    dev = load_experiment("configs/experiments/dev_verify.yaml")
    assert dev.tasks == "tasks/dev.txt" and dev.seeds == [1] and dev.arms == headline.arms
    assert dev.budget.cap_usd == 1.5 and dev.limits == headline.limits and dev.gym == headline.gym
    assert (dev.model, dev.judge_model, dev.audit) == (headline.model, headline.judge_model, headline.audit)


async def test_a_mock_run_logs_the_ceiling_basis_and_reference_spend(tmp_path) -> None:
    from evcarry.runs.manager import run_experiment
    from evcarry.trace.writer import read_events

    config = load_experiment("configs/experiments/mock_e2e.yaml")
    config.arms, config.tasks = ["sa"], "tests/fixtures/tasks/one_sections.txt"
    await run_experiment(config, out_dir=tmp_path)
    events = read_events(next(tmp_path.glob("*.jsonl")))
    assert (events[0]["ceiling_basis"], events[0]["context_tokens"]) == ("billed", 32000)
    assert events[-1]["reference_spend_usd"] == 0
    assert all("reference_cost_usd" in event for event in events if event["type"] == "llm_call")


# P3-2 -------------------------------------------------------------------------------------------------


def _items(outputs: list[str]) -> list[dict]:
    items: list[dict] = [{"role": "user", "content": "Collect the numbers."}]
    for index, text in enumerate(outputs):
        items.append({"type": "function_call", "call_id": f"t{index}", "name": "snowflake-read_query", "arguments": "{}"})
        items.append({"type": "function_call_output", "call_id": f"t{index}", "output": f"[call_id: c{index}]\n{text}"})
    return items


class _Events:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def write(self, event: dict) -> None:
        self.events.append(event)


def _call(trimmer: ContextTrimmer, agent, items: list[dict]) -> list[dict]:
    return trimmer(CallModelData(model_data=ModelInputData(input=items, instructions="system"), agent=agent, context=None)).input


def test_old_tool_outputs_are_stubbed_in_blocks_that_stay_put() -> None:
    trimmer = ContextTrimmer(10_000)
    assert trimmer.budget_tokens == 10_000 - OUTPUT_RESERVE_TOKENS
    agent = SimpleNamespace(name="researcher", tools=[])
    writer = _Events()
    token = current_writer.set(writer)
    try:
        items = _items(["x" * 2000 for _ in range(5)])
        first = _call(trimmer, agent, items)
        stubbed = [item for item in first if item.get("type") == "function_call_output" and item["output"].startswith("[Output of")]
        assert len(stubbed) == 4  # all but the newest
        assert first[-1] == items[-1]
        assert 'view_overlong_output with call_id "c0"' in stubbed[0]["output"] and "10000 tokens" in stubbed[0]["output"]
        assert writer.events[-1]["type"] == "context_trim" and writer.events[-1]["newly_stubbed"] == 4
        # The next call keeps the same stubs, byte for byte, and adds no new trim while the input fits.
        second = _call(trimmer, agent, _items(["x" * 2000 for _ in range(5)] + ["small"]))
        assert second[: len(first) - 1] == first[:-1]
        assert len(writer.events) == 1
    finally:
        current_writer.reset(token)
    # Another agent has its own state.
    other = SimpleNamespace(name="operator", tools=[])
    assert _call(trimmer, other, _items(["y" * 10])) == _items(["y" * 10])


def test_the_estimate_learns_the_tokens_per_character_from_the_provider() -> None:
    trimmer = ContextTrimmer(10_000)
    agent = SimpleNamespace(name="main", tools=[])
    items = _items(["x" * 2000 for _ in range(5)])
    small = _items(["x" * 10])
    _call(trimmer, agent, small)
    trimmer.calibrate(agent, 1)  # the provider counted far fewer tokens than the 1/3 default assumed
    assert _call(trimmer, agent, items) == items  # so the same input now fits


def test_every_runner_run_in_a_run_uses_the_runs_trimmer() -> None:
    assert harness_run_config().call_model_input_filter is None
    trimmer = ContextTrimmer(131_072)
    token = current_trimmer.set(trimmer)
    try:
        assert harness_run_config().call_model_input_filter is trimmer
    finally:
        current_trimmer.reset(token)


# P3-3 -------------------------------------------------------------------------------------------------


async def test_the_judge_sends_its_fixed_reasoning_setting(monkeypatch) -> None:
    from evcarry.protocol.judge import LlmJudge, build_judge
    from evcarry.protocol.schema import Claim

    assert load_models()["openrouter_ds_1017"].judge_extra_body == {"reasoning": {"effort": "low"}}
    monkeypatch.setattr("evcarry.providers.registry.async_openai_client", lambda **_kwargs: object())
    monkeypatch.setenv("OPENROUTER_API_KEY", "placeholder-not-a-key")
    assert build_judge("openrouter_ds_1017").extra_body == {"reasoning": {"effort": "low"}}
    assert build_judge("mock_judge").extra_body is None
    seen = []

    class Model:
        async def get_response(self, **kwargs):
            seen.append(kwargs["model_settings"])
            raise ConnectionError("not sent")

    judge = LlmJudge(Model(), model_name="m", prompt="p", price=PRICE, extra_body={"reasoning": {"effort": "low"}})
    await judge.ajudge(Claim(claim_id="c", text="t", kind="fact", value=1, evidence=[]), "1")
    assert seen[0].extra_body == {"reasoning": {"effort": "low"}} and seen[0].temperature == 0.0
