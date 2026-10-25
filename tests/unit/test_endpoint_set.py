"""The FP8 endpoint set: pin check and per-endpoint pricing.

Bodies are the replies OpenRouter sent to the endpoint probe on 2026-10-24
(``docs/evidence/endpoint_probe_2026-10-24.json``), one per endpoint.
"""

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx2
import pytest
from agents.model_settings import ModelSettings
from agents.models.interface import ModelTracing
from openai import AsyncOpenAI

from evcarry.harness.hooks import LedgerHooks
from evcarry.ledger.ledger import Ledger
from evcarry.paths import repo_root
from evcarry.protocol.judge import LlmJudge
from evcarry.protocol.schema import Claim
from evcarry.providers.identity import served_identity
from evcarry.providers.openai_compat import RecordingModel, capture_http_body
from evcarry.providers.pin import check_pin
from evcarry.providers.registry import load_models
from evcarry.providers.usage import PriceTable, Rates, TokenUsage, call_cost, cost_usd, token_usage_from_body
from evcarry.runs.budget import event_spend
from evcarry.trace.writer import JsonlWriter, current_writer, read_events

AT = datetime(2026, 10, 24, 20, 0, tzinfo=timezone.utc)
SET_ROUTING = load_models()["openrouter_ds_1017"].routing
PRICE = load_models()["openrouter_ds_1017"].price_table()
DEEPINFRA = Rates(0.098, 0.00294, 0.294)
INFERENCENET = Rates(0.08, 0.008, 0.50)


def _probe_body(tag: str) -> dict:
    report = json.loads((repo_root() / "docs/evidence/endpoint_probe_2026-10-24.json").read_text(encoding="utf-8"))
    served = [item for item in report["results"] if item["tag"] == tag and item["http_status"] == 200]
    body = copy.deepcopy(served[-1]["response_body"])
    body["choices"][0]["message"] = {"role": "assistant", "content": "ok"}  # plain text, so the SDK needs no tools
    body["choices"][0]["finish_reason"] = "stop"
    return body


def _identity(body: dict):
    return served_identity(body, kind="openrouter", configured_model="deepseek/deepseek-v4.1-flash-1017")


# --- the pin ---------------------------------------------------------------------------------


def test_the_route_is_the_fixed_fp8_set_in_deepinfra_first_order() -> None:
    assert SET_ROUTING["only"] == SET_ROUTING["order"] == ["deepinfra/fp8", "inference-net/fp8", "inference-net/us"]
    assert SET_ROUTING["quantizations"] == ["fp8"]
    assert SET_ROUTING["allow_fallbacks"] is False and SET_ROUTING["service_tier"] == "default"


@pytest.mark.parametrize("tag", ["deepinfra/fp8", "inference-net/fp8", "inference-net/us"])
def test_every_endpoint_of_the_set_passes_the_pin_on_its_real_reply(tag: str) -> None:
    identity = _identity(_probe_body(tag))
    assert check_pin(identity, SET_ROUTING).ok is True


def test_a_429_moved_to_the_next_endpoint_inside_one_request_passes_and_is_logged() -> None:
    """The whole-set probe: DeepInfra answered 429 and OpenRouter served the same request from InferenceNet."""
    identity = _identity(_probe_body("whole-set"))
    assert identity.response_provider == identity.served_provider == "InferenceNet"
    assert identity.router["attempt"] == 2 and "attempts=2" in identity.router["summary"]
    assert check_pin(identity, SET_ROUTING).ok is True
    assert call_cost(token_usage_from_body(_probe_body("whole-set")), PRICE, AT, "InferenceNet").usd == pytest.approx(5.844e-05)


def test_the_set_still_refuses_other_providers_and_other_tiers() -> None:
    novita = _probe_body("inference-net/fp8")
    novita["provider"] = "Novita"
    for item in novita["openrouter_metadata"]["endpoints"]["available"]:
        item["provider"] = "Novita"
    assert check_pin(_identity(novita), SET_ROUTING).ok is False
    for provider_tag in ("deepinfra/fp8", "inference-net/us"):
        for tier in ("priority", "flex"):
            body = _probe_body(provider_tag)
            body["service_tier"] = tier
            result = check_pin(_identity(body), SET_ROUTING)
            assert result.ok is False and tier in result.reason, (provider_tag, tier)
    # InferenceNet answers service_tier null: not a flex or priority reply.
    assert _identity(_probe_body("inference-net/us")).service_tier is None


def test_quantization_is_matched_per_tag() -> None:
    routing = {"only": ["deepinfra/fp8", "inference-net"], "allow_fallbacks": False}
    deepinfra_fp4 = _probe_body("deepinfra/fp8")
    deepinfra_fp4["quantization"] = "fp4"
    assert check_pin(_identity(deepinfra_fp4), routing).ok is False
    # inference-net is pinned without a quantization, so another provider's fp8 does not constrain it.
    inference_fp4 = _probe_body("inference-net/fp8")
    inference_fp4["quantization"] = "fp4"
    assert check_pin(_identity(inference_fp4), routing).ok is True
    assert check_pin(_identity(inference_fp4), SET_ROUTING).ok is False  # the real set is fp8 everywhere


# --- pricing ---------------------------------------------------------------------------------


def test_the_price_table_has_one_row_per_endpoint_and_an_upper_bound() -> None:
    assert dict(PRICE.endpoints) == {"deepinfra/fp8": DEEPINFRA, "inference-net/fp8": INFERENCENET, "inference-net/us": INFERENCENET}
    assert PRICE.worst_rates() == Rates(0.098, 0.008, 0.50)  # the larger rate of each component
    assert PRICE.rates_for("DeepInfra") == (DEEPINFRA, "deepinfra/fp8")
    assert PRICE.rates_for("InferenceNet") == (INFERENCENET, "inference-net/fp8|inference-net/us")
    assert PRICE.rates_for(None) == (PRICE.worst_rates(), "upper_bound")
    assert PRICE.rates_for("Novita") == (PRICE.worst_rates(), "upper_bound")


def test_a_call_is_priced_at_the_endpoint_that_served_it_never_the_cheapest() -> None:
    """The SA canvas run's token mix (97.7% cache hits): InferenceNet costs 1.8x DeepInfra for it."""
    tokens = TokenUsage(799_880, 781_824, 18_056, 14_122)
    deepinfra = call_cost(tokens, PRICE, AT, "DeepInfra")
    inferencenet = call_cost(tokens, PRICE, AT, "InferenceNet")
    assert deepinfra.usd == pytest.approx(DEEPINFRA.usd(tokens)) == pytest.approx(0.008224, rel=1e-3)
    assert inferencenet.usd == pytest.approx(INFERENCENET.usd(tokens)) == pytest.approx(0.014760, rel=1e-3)
    assert inferencenet.usd > deepinfra.usd
    unknown = call_cost(tokens, PRICE, AT, None)
    assert unknown.usd >= inferencenet.usd and unknown.priced_as == "upper_bound"


def test_openrouter_cost_wins_when_it_is_larger_and_never_lowers_the_table() -> None:
    cheap = TokenUsage(1000, 0, 1000, 100, provider_reported_cost_usd=0.0)
    assert call_cost(cheap, PRICE, AT, "DeepInfra").usd == pytest.approx(DEEPINFRA.usd(cheap))
    dear = TokenUsage(1000, 0, 1000, 100, provider_reported_cost_usd=1.0)
    priced = call_cost(dear, PRICE, AT, "DeepInfra")
    assert priced.usd == 1.0 and priced.table_usd == pytest.approx(DEEPINFRA.usd(dear))
    assert event_spend({"type": "llm_call", **priced.fields()}) == 1.0


# --- the events that carry cost ----------------------------------------------------------------


def _model(handler) -> tuple[RecordingModel, httpx2.AsyncClient]:
    http = httpx2.AsyncClient(transport=httpx2.MockTransport(handler), event_hooks={"response": [capture_http_body]})
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key="test-not-real", http_client=http, max_retries=0)
    model = RecordingModel("deepseek/deepseek-v4.1-flash-1017", client, kind="openrouter", routing=SET_ROUTING, price=PRICE)
    return model, http


_ARGS = dict(
    system_instructions="sys", input="hi", model_settings=ModelSettings(temperature=0.0), tools=[], output_schema=None,
    handoffs=[], tracing=ModelTracing.DISABLED, previous_response_id=None, conversation_id=None, prompt=None,
)


class _Agent:
    name = "main"

    def __init__(self, model: RecordingModel) -> None:
        self.model = model


async def test_llm_call_logs_the_served_provider_and_its_price(tmp_path: Path) -> None:
    body = _probe_body("inference-net/fp8")
    body["usage"].pop("cost")  # no OpenRouter cost: the served endpoint's row decides
    model, http = _model(lambda _request: httpx2.Response(200, json=body))
    writer = JsonlWriter(tmp_path / "trace.jsonl")
    hooks = LedgerHooks(Ledger(), writer, PRICE)
    try:
        await model.get_response(**_ARGS)
        await hooks.on_llm_end(None, _Agent(model), None)
    finally:
        writer.close()
        await http.aclose()
    (event,) = read_events(tmp_path / "trace.jsonl")
    assert event["type"] == "llm_call" and event["response_provider"] == "InferenceNet"
    assert event["provider_check"]["ok"] is True
    assert event["priced_as"] == "inference-net/fp8|inference-net/us"
    assert event["provider_reported_cost_usd"] is None
    assert event["cost_usd"] == pytest.approx(293 * 0.08 / 1e6 + 74 * 0.50 / 1e6)  # not DeepInfra's $0.0000505


async def test_judge_call_is_priced_at_the_served_endpoint(tmp_path: Path) -> None:
    body = _probe_body("inference-net/us")
    body["choices"][0]["message"] = {"role": "assistant", "content": '{"verdict": "supported", "reason": "ok", "missing": []}'}
    model, http = _model(lambda _request: httpx2.Response(200, json=body))
    writer = JsonlWriter(tmp_path / "trace.jsonl")
    judge = LlmJudge(model, model_name="openrouter_ds_1017", prompt="judge", price=PRICE, writer=writer)
    claim = Claim(claim_id="c1", text="There are 14 sections.", kind="fact", value=14, evidence=[])
    try:
        await judge.ajudge(claim, "14 sections")
    finally:
        writer.close()
        await http.aclose()
    (event,) = read_events(tmp_path / "trace.jsonl")
    assert event["type"] == "judge_call" and event["response_provider"] == "InferenceNet"
    assert event["priced_as"] == "inference-net/fp8|inference-net/us"
    assert event["cost_usd"] == pytest.approx(2.6008e-05) == pytest.approx(judge.cost_usd)


async def test_an_answered_then_failed_call_is_priced_at_the_served_endpoint(tmp_path: Path) -> None:
    body = _probe_body("deepinfra/fp8")
    body["choices"] = []  # answered and billed, then the SDK cannot use the reply
    model, http = _model(lambda _request: httpx2.Response(200, json=body))
    writer = JsonlWriter(tmp_path / "trace.jsonl")
    token = current_writer.set(writer)
    try:
        with pytest.raises(Exception):
            await model.get_response(**_ARGS)
    finally:
        current_writer.reset(token)
        writer.close()
        await http.aclose()
    (event,) = read_events(tmp_path / "trace.jsonl")
    assert event["type"] == "llm_error" and event["response_provider"] == "DeepInfra"
    assert event["priced_as"] == "deepinfra/fp8" and event["cost_usd"] == pytest.approx(4.9882e-05)
