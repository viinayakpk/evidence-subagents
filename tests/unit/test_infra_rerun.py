"""A19 (2026-10-25): infrastructure errors end the run in every arm, and hit (task, seed) blocks run again whole (C1, C2).

Also C6 (agent reasoning pinned), C8 (read_email never replayed) and C9 (finish_reason on model-call events).
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx2 as httpx
import pyarrow.parquet as pq
import pytest
from openai import APIConnectionError, APIStatusError, InternalServerError, RateLimitError

from evcarry.providers.pin import ProviderMismatch
from evcarry.runs.budget import CostCeilingExceeded
from evcarry.runs.config import load_experiment
from evcarry.runs.infra import InfrastructureError, is_infrastructure_error
from evcarry.runs.manager import run_experiment
from evcarry.trace.writer import read_events

_REQUEST = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")


def _status_error(cls, code: int):
    return cls(f"Error code: {code}", response=httpx.Response(code, request=_REQUEST), body=None)


def test_what_counts_as_an_infrastructure_error() -> None:
    assert is_infrastructure_error(_status_error(RateLimitError, 429))
    assert is_infrastructure_error(_status_error(InternalServerError, 500))
    assert is_infrastructure_error(_status_error(APIStatusError, 502))
    assert is_infrastructure_error(APIConnectionError(request=_REQUEST))
    assert is_infrastructure_error(ProviderMismatch("served by Novita"))
    try:
        try:
            raise _status_error(RateLimitError, 429)
        except RateLimitError as inner:
            raise RuntimeError("wrapped by the SDK") from inner
    except RuntimeError as outer:
        assert is_infrastructure_error(outer)
    assert not is_infrastructure_error(_status_error(APIStatusError, 400))
    assert not is_infrastructure_error(RuntimeError("harness bug"))
    assert not is_infrastructure_error(CostCeilingExceeded("ceiling"))


def _config(arms: list[str], attempts: int = 2):
    config = load_experiment("configs/experiments/mock_e2e.yaml")
    config.arms = arms
    config.tasks = "tests/fixtures/tasks/one_sections.txt"
    config.max_block_attempts = attempts
    return config


def _raise_for(monkeypatch, agent_name: str, times: int | None) -> dict[str, int]:
    """Make the scripted model of ``agent_name`` raise a 429 (after its retries) ``times`` times (None: always)."""
    from evcarry.providers.scripted import FixtureModel

    original = FixtureModel.get_response
    raised = {"n": 0}

    async def flaky(self, *args, **kwargs):
        if self.agent_name == agent_name and (times is None or raised["n"] < times):
            raised["n"] += 1
            raise _status_error(RateLimitError, 429)
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(FixtureModel, "get_response", flaky)
    return raised


def _blocks(out: Path) -> list[dict]:
    return [json.loads(line) for line in (out / "blocks.ndjson").read_text(encoding="utf-8").splitlines()]


async def test_a_sub_agent_429_ends_the_run_and_the_whole_block_runs_again(tmp_path: Path, monkeypatch) -> None:
    """Before A19 the 429 reached vanilla's main agent as a tool error and the run was graded."""
    _raise_for(monkeypatch, "researcher", times=1)
    out = tmp_path / "out"
    await run_experiment(_config(["sa", "vanilla_mad"]), out_dir=out)
    first = read_events(next(out.glob("*vanilla_mad*.jsonl.aborted-1")))
    errors = [event for event in first if event["type"] == "subagent_error"]
    assert errors and errors[0]["infrastructure"] is True and errors[0]["error_type"] == "RateLimitError"
    end = first[-1]
    assert end["type"] == "run_end" and end["status"] == "infra_error" and end["passed"] is None
    assert end["error_type"] == "RateLimitError" and end["eval_ran"] is False
    # The block ran again in every arm, sa included, and both second attempts finished normally.
    assert list(out.glob("*__sa__*.jsonl.aborted-1"))
    for path in out.glob("*.jsonl"):
        last = read_events(path)[-1]
        assert last["type"] == "run_end" and last["attempt"] == 2 and last["status"] != "infra_error"
        assert last["previous_attempts"][0]["n"] == 1
    assert [(row["action"], row["hit"]) for row in _blocks(out)] == [("rerun", {"vanilla_mad": "infra_error"})]
    index = pq.read_table(out / "index.parquet").to_pylist()
    assert {row["attempt"] for row in index} == {2} and not any(row["block_infra"] for row in index)


async def test_a_main_agent_429_is_treated_the_same_way_in_sa(tmp_path: Path, monkeypatch) -> None:
    _raise_for(monkeypatch, "main", times=None)
    out = tmp_path / "out"
    await run_experiment(_config(["sa"], attempts=1), out_dir=out)
    end = read_events(next(out.glob("*.jsonl")))[-1]
    assert end["type"] == "run_end" and end["status"] == "infra_error" and end["agent_status"] == "infra_error"
    assert end["passed"] is None and end["attempt"] == 1


async def test_a_block_still_hit_after_max_attempts_is_dropped_in_every_arm(tmp_path: Path, monkeypatch) -> None:
    _raise_for(monkeypatch, "researcher", times=None)
    out = tmp_path / "out"
    await run_experiment(_config(["sa", "vanilla_mad"], attempts=2), out_dir=out)
    actions = [row["action"] for row in _blocks(out)]
    assert actions == ["rerun", "dropped"]
    index = pq.read_table(out / "index.parquet").to_pylist()
    assert sorted(row["arm"] for row in index) == ["sa", "vanilla_mad"]
    assert all(row["block_infra"] for row in index) and all(row["attempt"] == 2 for row in index)
    # A further invocation does not run the dropped block again.
    await run_experiment(_config(["sa", "vanilla_mad"], attempts=2), out_dir=out)
    assert [row["action"] for row in _blocks(out)] == ["rerun", "dropped"]


async def test_three_infra_errors_stop_new_runs(tmp_path: Path, monkeypatch) -> None:
    _raise_for(monkeypatch, "main", times=None)
    config = load_experiment("configs/experiments/mock_e2e.yaml")  # 3 tasks x 3 arms
    config.max_block_attempts = 1
    with pytest.raises(InfrastructureError, match="new runs stopped"):
        await run_experiment(config, out_dir=tmp_path)
    assert len(list(tmp_path.glob("*.jsonl"))) == 3


async def test_the_judge_does_not_fall_back_on_an_infrastructure_error() -> None:
    from evcarry.protocol.judge import LlmJudge
    from evcarry.protocol.schema import Claim
    from evcarry.providers.usage import PriceTable

    class Down:
        async def get_response(self, **_kwargs):
            raise _status_error(RateLimitError, 429)

    judge = LlmJudge(Down(), model_name="x", prompt="p", price=PriceTable(0, 0, 0, peak_multiplier=1))
    claim = Claim(claim_id="c1", text="t", kind="fact", value="v", evidence=[])
    with pytest.raises(RateLimitError):
        await judge.ajudge(claim, "evidence")
    assert judge.fallbacks == 0


def _recording(**kwargs):
    from openai import AsyncOpenAI

    from evcarry.providers.openai_compat import RecordingModel

    return RecordingModel("m", AsyncOpenAI(base_url="http://127.0.0.1:9", api_key="x"), kind="openrouter", **kwargs)


def test_agent_calls_carry_the_pinned_reasoning_and_judge_calls_their_own() -> None:
    from agents import ModelSettings

    from evcarry.providers.registry import load_models

    config = load_models()["openrouter_ds_1017"]
    assert config.agent_extra_body == {"reasoning": {"effort": "high"}}
    assert config.judge_extra_body == {"reasoning": {"effort": "low"}}
    model = _recording(agent_extra_body=config.agent_extra_body, routing=config.routing)
    merged = model._merge_settings(ModelSettings(temperature=0.0))
    assert merged.extra_body["reasoning"] == {"effort": "high"} and "provider" in merged.extra_body
    model.role = "judge"
    judged = model._merge_settings(ModelSettings(temperature=0.0, extra_body=config.judge_extra_body))
    assert judged.extra_body["reasoning"] == {"effort": "low"}


def test_finish_reason_and_truncation_are_recorded() -> None:
    model = _recording()
    body = {
        "model": "m",
        "choices": [{"finish_reason": "length", "native_finish_reason": "length", "message": {"content": ""}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "completion_tokens_details": {"reasoning_tokens": 4}},
    }
    record = model._record([{"status": 200, "body": body, "temperature": 0.0, "reasoning": {"effort": "high"}}], 0.0, None)
    fields = record.stop_fields()
    assert fields == {
        "finish_reason": "length",
        "native_finish_reason": "length",
        "truncated": True,
        "reasoning": {"effort": "high"},
        "reasoning_tokens": 4,
    }
    body["choices"][0].update(finish_reason="tool_calls", native_finish_reason="tool_calls")
    assert model._record([{"status": 200, "body": body}], 0.0, None).truncated is False


async def test_llm_call_events_carry_finish_reason(tmp_path: Path) -> None:
    out = tmp_path / "out"
    await run_experiment(_config(["sa"]), out_dir=out)
    calls = [event for event in read_events(next(out.glob("*.jsonl"))) if event["type"] == "llm_call"]
    assert calls and all("finish_reason" in event and "truncated" in event for event in calls)


def test_read_email_is_never_replayed() -> None:
    from types import SimpleNamespace

    from evcarry.env.gym_tools import GymToolPolicy

    tool = SimpleNamespace(name="read_email", description="read", input_schema={}, annotations=None)
    meta = GymToolPolicy.load().meta("emails", tool)
    assert meta.read_only and meta.nondeterministic
