"""Served-provider logging and the provider pin, on the response OpenRouter actually sent on 2026-10-24."""

import copy
import json
from pathlib import Path

import httpx2
import pytest
from agents.model_settings import ModelSettings
from agents.models.interface import ModelTracing
from openai import AsyncOpenAI

from evcarry.paths import repo_root
from evcarry.protocol.judge import LlmJudge
from evcarry.protocol.schema import Claim
from evcarry.providers.identity import served_identity
from evcarry.providers.openai_compat import RecordingModel, capture_http_body
from evcarry.providers.pin import PinGuard, ProviderMismatch, check_pin, current_pin_guard, pin_from_routing
from evcarry.providers.registry import load_models
from evcarry.runs.manager import event_spend, logged_spend
from evcarry.trace.writer import JsonlWriter, current_writer, read_events

PINNED = {"order": ["deepinfra/fp8"], "allow_fallbacks": False, "require_parameters": True}


def _real_body() -> dict:
    report = json.loads((repo_root() / "docs/evidence/tool_limit_probe_2026-10-24.json").read_text(encoding="utf-8"))
    body = copy.deepcopy(report["response_body"])
    # A plain text reply, so the SDK does not need the probe's tools.
    body["choices"][0]["message"] = {"role": "assistant", "content": "ok"}
    body["choices"][0]["finish_reason"] = "stop"
    return body


def _off_pin(body: dict, provider: str = "Novita") -> dict:
    body = copy.deepcopy(body)
    body["provider"] = provider
    for item in body["openrouter_metadata"]["endpoints"]["available"]:
        item["provider"] = provider
    return body


def test_identity_reads_generation_id_provider_and_tier_from_the_real_response() -> None:
    identity = served_identity(_real_body(), kind="openrouter", configured_model="deepseek/deepseek-v4.1-flash-1017")
    assert identity.generation_id == "gen-1792856999-vskGWCqPzAS924Mo2RQK"
    assert identity.response_provider == "DeepInfra" and identity.served_provider == "DeepInfra"
    assert identity.served_model == "deepseek/deepseek-v4.1-flash-20261017"
    assert identity.service_tier == "default"
    assert identity.quantization is None  # OpenRouter does not send it
    assert check_pin(identity, PINNED).ok is True


def test_pin_check_cases() -> None:
    real = served_identity(_real_body(), kind="openrouter", configured_model="m")
    assert check_pin(real, {**PINNED, "allow_fallbacks": True}).ok is None  # not pinned
    assert check_pin(real, None).ok is None
    wrong = served_identity(_off_pin(_real_body()), kind="openrouter", configured_model="m")
    result = check_pin(wrong, PINNED)
    assert result.ok is False and "Novita" in result.reason
    # The top-level provider and the metadata disagree: still a mismatch.
    mixed = _real_body()
    mixed["provider"] = "Novita"
    assert check_pin(served_identity(mixed, kind="openrouter", configured_model="m"), PINNED).ok is False
    # Neither name present: unobserved, not a failure.
    bare = {key: value for key, value in _real_body().items() if key not in ("provider", "openrouter_metadata")}
    unobserved = check_pin(served_identity(bare, kind="openrouter", configured_model="m"), PINNED)
    assert unobserved.ok is None and unobserved.reason == "provider not observed"
    # A priority-tier reply on a default-tier pin, or another quantization, breaks the pin.
    priority = _real_body()
    priority["service_tier"] = "priority"
    assert check_pin(served_identity(priority, kind="openrouter", configured_model="m"), PINNED).ok is False
    quantized = _real_body()
    quantized["quantization"] = "fp4"
    assert check_pin(served_identity(quantized, kind="openrouter", configured_model="m"), PINNED).ok is False
    # Slugs with hyphens match display names without them.
    pin = pin_from_routing({"order": ["inference-net/fp8"], "allow_fallbacks": False})
    assert pin is not None and pin.providers == {"inferencenet"} and pin.quantizations == {"fp8"}
    assert pin_from_routing(load_models()["openrouter_ds_1017"].routing).providers == {"deepinfra", "inferencenet"}


def _model(handler, price=None) -> tuple[RecordingModel, httpx2.AsyncClient]:
    http = httpx2.AsyncClient(transport=httpx2.MockTransport(handler), event_hooks={"response": [capture_http_body]})
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key="test-not-real", http_client=http)
    return RecordingModel("deepseek/deepseek-v4.1-flash-1017", client, kind="openrouter", routing=PINNED, price=price), http


_ARGS = dict(
    system_instructions="sys", input="hi", model_settings=ModelSettings(), tools=[], output_schema=None, handoffs=[],
    tracing=ModelTracing.DISABLED, previous_response_id=None, conversation_id=None, prompt=None,
)


async def test_off_pin_reply_is_refused_logged_priced_and_stops_later_calls(tmp_path: Path) -> None:
    requests = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, json=_off_pin(_real_body()))

    price = load_models()["openrouter_ds_1017"].price_table()
    model, http = _model(handler, price)
    writer = JsonlWriter(tmp_path / "trace.jsonl")
    guard = PinGuard()
    tokens = (current_writer.set(writer), current_pin_guard.set(guard))
    try:
        with pytest.raises(ProviderMismatch, match="Novita"):
            await model.get_response(**_ARGS)
        assert model.take_record() is None  # the reply is never handed to the agent
        with pytest.raises(ProviderMismatch, match="run stopped"):
            await model.get_response(**_ARGS)
    finally:
        current_pin_guard.reset(tokens[1])
        current_writer.reset(tokens[0])
        writer.close()
        await http.aclose()
    assert len(requests) == 1  # the second call never reached the network
    assert guard.violation is not None
    mismatch, error = read_events(tmp_path / "trace.jsonl")
    assert mismatch["type"] == "provider_mismatch" and mismatch["response_provider"] == "Novita"
    assert mismatch["generation_id"].startswith("gen-") and mismatch["provider_check"]["ok"] is False
    assert error["type"] == "llm_error" and error["error_type"] == "ProviderMismatch"
    # The refused call was billed, so it counts toward the cap: Novita is not in the price table, so it is
    # priced at the table's upper bound, which here is above OpenRouter's own cost; the larger figure counts.
    assert error["priced_as"] == "upper_bound"
    assert error["provider_reported_cost_usd"] == pytest.approx(0.001042132)
    assert error["cost_usd"] == pytest.approx(max(error["price_table_cost_usd"], error["provider_reported_cost_usd"]))
    assert error["cost_usd"] >= error["provider_reported_cost_usd"]
    assert logged_spend(tmp_path) == pytest.approx(error["cost_usd"])


async def test_on_pin_reply_records_provenance(tmp_path: Path) -> None:
    model, http = _model(lambda _request: httpx2.Response(200, json=_real_body()))
    try:
        await model.get_response(**_ARGS)
    finally:
        await http.aclose()
    record = model.take_record()
    provenance = record.provenance()
    assert provenance["response_provider"] == "DeepInfra"
    assert provenance["generation_id"] == "gen-1792856999-vskGWCqPzAS924Mo2RQK"
    assert provenance["provider_check"]["ok"] is True


async def test_off_pin_judge_call_is_not_swallowed_by_the_lexical_fallback(tmp_path: Path) -> None:
    model, http = _model(lambda _request: httpx2.Response(200, json=_off_pin(_real_body(), "Fireworks")))
    judge = LlmJudge(model, model_name="openrouter_ds_1017", prompt="judge", price=load_models()["openrouter_ds_1017"].price_table())
    claim = Claim(claim_id="c1", text="There are 14 sections.", kind="fact", value=14, evidence=[])
    try:
        with pytest.raises(ProviderMismatch, match="Fireworks"):
            await judge.ajudge(claim, "14 sections")
    finally:
        await http.aclose()


def test_event_spend_counts_billed_errors_and_the_larger_bill() -> None:
    assert event_spend({"type": "llm_call", "cost_usd": 0.002, "provider_reported_cost_usd": 0.003}) == 0.003
    assert event_spend({"type": "judge_call", "cost_usd": 0.002, "provider_reported_cost_usd": None}) == 0.002
    assert event_spend({"type": "llm_error", "cost_usd": 0.001}) == 0.001
    assert event_spend({"type": "tool_call", "cost_usd": 5}) == 0.0
