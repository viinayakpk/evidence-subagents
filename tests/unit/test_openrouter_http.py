import json
import random

import httpx2
from agents.model_settings import ModelSettings
from agents.models.interface import ModelTracing
from openai import AsyncOpenAI

from evcarry.providers.openai_compat import (
    RETRY_CAP_S,
    RETRY_WINDOW_S,
    RecordingModel,
    capture_http_body,
    jittered_waits,
    openrouter_headers,
)


async def test_openrouter_routing_headers_and_served_identity() -> None:
    body = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "deepseek/deepseek-chat",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "total_tokens": 110,
            "prompt_cache_hit_tokens": 80,
            "prompt_cache_miss_tokens": 20,
            "cost": 0.00000924,
        },
        "openrouter_metadata": {
            "requested": "deepseek/deepseek-chat",
            "strategy": "order",
            "summary": "pinned",
            "attempt": 1,
            "endpoints": {
                "available": [
                    {"provider": "DeepSeek", "model": "deepseek/deepseek-chat-v3", "selected": True}
                ]
            },
        },
    }
    seen: dict = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen["json"] = json.loads(request.content)
        seen["headers"] = {key.lower(): value for key, value in request.headers.items()}
        return httpx2.Response(200, json=body)

    http = httpx2.AsyncClient(transport=httpx2.MockTransport(handler), event_hooks={"response": [capture_http_body]})
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key="test-not-real", http_client=http)
    headers = openrouter_headers(http_referer="https://example.test", x_title="evcarry")
    model = RecordingModel(
        "deepseek/deepseek-chat",
        client,
        kind="openrouter",
        routing={"order": ["deepseek"], "allow_fallbacks": False, "require_parameters": True},
        extra_headers=headers,
    )
    await model.get_response(
        "sys",
        "hi",
        ModelSettings(),
        [],
        None,
        [],
        ModelTracing.DISABLED,
        previous_response_id=None,
        conversation_id=None,
        prompt=None,
    )
    await http.aclose()
    provider = seen["json"]["provider"]
    assert provider["order"] == ["deepseek"]
    assert provider["allow_fallbacks"] is False
    assert provider["require_parameters"] is True
    assert seen["headers"]["http-referer"] == "https://example.test"
    assert seen["headers"]["x-title"] == "evcarry"
    assert seen["headers"]["x-openrouter-title"] == "evcarry"
    assert seen["headers"]["x-openrouter-metadata"] == "enabled"
    record = model.take_record()
    assert record is model.history[-1]
    assert record.identity.served_provider == "DeepSeek"
    assert record.identity.served_model == "deepseek/deepseek-chat-v3"
    assert record.usage.cache_hit_tokens == 80
    assert record.usage.cache_miss_tokens == 20
    assert record.routing["allow_fallbacks"] is False


async def test_a_failed_call_does_not_take_the_next_calls_record(tmp_path) -> None:
    """on_llm_end only runs after a successful call; a queued failure shifted every later llm_call by one."""
    from evcarry.trace.writer import JsonlWriter, current_writer, read_events

    ok = {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "deepseek-flash-1017",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 10, "prompt_cache_hit_tokens": 80, "prompt_cache_miss_tokens": 20},
    }
    replies = iter([httpx2.Response(400, json={"error": {"message": "context too long"}}), httpx2.Response(200, json=ok)])
    http = httpx2.AsyncClient(
        transport=httpx2.MockTransport(lambda _request: next(replies)), event_hooks={"response": [capture_http_body]}
    )
    model = RecordingModel(
        "deepseek-flash", AsyncOpenAI(base_url="https://api.test/v1", api_key="test-not-real", http_client=http), kind="deepseek"
    )
    args = dict(
        system_instructions="sys", input="hi", model_settings=ModelSettings(), tools=[], output_schema=None, handoffs=[],
        tracing=ModelTracing.DISABLED, previous_response_id=None, conversation_id=None, prompt=None,
    )
    writer = JsonlWriter(tmp_path / "trace.jsonl")
    token = current_writer.set(writer)
    try:
        try:
            await model.get_response(**args)
        except Exception as exc:  # noqa: BLE001
            assert type(exc).__name__ == "BadRequestError"
        assert model.take_record() is None
        await model.get_response(**args)
    finally:
        current_writer.reset(token)
        writer.close()
        await http.aclose()
    record = model.take_record()
    assert record.identity.served_model == "deepseek-flash-1017"
    assert record.usage.cache_hit_tokens == 80 and record.usage.output_tokens == 10
    assert model.take_record() is None
    (error,) = read_events(tmp_path / "trace.jsonl")
    assert error["type"] == "llm_error" and error["statuses"] == [400]
    assert error["tokens"] is None and error["served_model"] is None


async def test_temperature_zero_is_sent_and_recorded() -> None:
    """Owner decision 2: every request carries temperature 0, and the record says what the request carried."""
    from evcarry.harness.run_settings import TEMPERATURE, agent_model_settings, harness_run_config

    body = {
        "id": "gen-1",
        "object": "chat.completion",
        "created": 0,
        "model": "deepseek/deepseek-v4.1-flash-1017",
        "provider": "DeepInfra",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
    }
    sent: list[dict] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        return httpx2.Response(200, json=body)

    http = httpx2.AsyncClient(transport=httpx2.MockTransport(handler), event_hooks={"response": [capture_http_body]})
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key="test-not-real", http_client=http)
    model = RecordingModel("deepseek/deepseek-v4.1-flash-1017", client, kind="openrouter")
    for settings in (agent_model_settings(), harness_run_config().model_settings, ModelSettings()):
        await model.get_response(
            "sys", "hi", settings, [], None, [], ModelTracing.DISABLED,
            previous_response_id=None, conversation_id=None, prompt=None,
        )
    await http.aclose()
    assert TEMPERATURE == 0.0
    assert [request.get("temperature") for request in sent] == [0.0, 0.0, None]
    assert [record.temperature for record in model.history] == [0.0, 0.0, None]  # None: not sent


def test_every_agent_of_every_arm_is_built_with_temperature_zero() -> None:
    from evcarry.harness.agents import build_main_agent
    from evcarry.harness.arms import ARMS
    from evcarry.harness.hooks import LedgerHooks
    from evcarry.harness.tools import fixture_toolset
    from evcarry.ledger.ledger import Ledger
    from evcarry.providers.registry import load_models
    from evcarry.providers.usage import PriceTable

    class _Writer:
        def write(self, event):
            pass

    for arm in ARMS.values():
        ledger = Ledger()
        toolset = fixture_toolset(None, ledger)
        hooks = LedgerHooks(ledger, _Writer(), PriceTable(0, 0, 0, peak_multiplier=1), toolset.catalog)
        main = build_main_agent(
            arm=arm, task="fixture-sections", seed=1, prompts_dir="prompts/v1", model_config=load_models()["mock"],
            script=None, toolset=toolset, hooks=hooks, writer=_Writer(), ledger=ledger, checks=arm.checks,
            sampler=arm.sampler, rate=arm.default_rate, injector=None, subagent_max_turns=5,
        )
        assert main.model_settings.temperature == 0.0, arm.name
        nested_agents = [tool._agent_instance for tool in main.tools if getattr(tool, "_agent_instance", None)]
        assert len(nested_agents) == (2 if arm.delegate else 0)
        for tool in main.tools:
            nested = getattr(tool, "_agent_instance", None)
            if nested is not None:
                assert nested.model_settings.temperature == 0.0, (arm.name, tool.name)


async def test_the_pinned_route_asks_for_the_standard_service_tier() -> None:
    """The first pilot attempt was served by deepinfra/fp8/priority under the deepinfra/fp8 slug; the pin stopped it."""
    from evcarry.providers.openai_compat import routing_extra_body
    from evcarry.providers.pin import pin_from_routing
    from evcarry.providers.registry import load_models

    routing = load_models()["openrouter_ds_1017"].routing
    body = routing_extra_body(routing)
    assert body["service_tier"] == "default"
    endpoint_set = ["deepinfra/fp8", "inference-net/fp8", "inference-net/us"]
    assert "service_tier" not in body["provider"]
    assert body["provider"]["order"] == endpoint_set and body["provider"]["only"] == endpoint_set
    assert pin_from_routing(routing).tiers == frozenset({"default"})  # what the pin check accepts
    assert routing_extra_body({"order": ["deepseek"]}) == {"provider": {"order": ["deepseek"]}}

    sent: list[dict] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        return httpx2.Response(200, json={
            "id": "gen-1", "object": "chat.completion", "created": 0, "model": "m", "provider": "DeepInfra",
            "service_tier": "default",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    http = httpx2.AsyncClient(transport=httpx2.MockTransport(handler), event_hooks={"response": [capture_http_body]})
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key="test-not-real", http_client=http)
    model = RecordingModel("deepseek/deepseek-v4.1-flash-1017", client, kind="openrouter", routing=routing)
    await model.get_response(
        "sys", "hi", ModelSettings(temperature=0.0), [], None, [], ModelTracing.DISABLED,
        previous_response_id=None, conversation_id=None, prompt=None,
    )
    await http.aclose()
    assert sent[0]["service_tier"] == "default" and sent[0]["provider"]["allow_fallbacks"] is False
    assert model.take_record().pin.ok is True


async def test_an_overloaded_provider_is_retried_before_the_call_fails() -> None:
    """The second pilot launch failed on 429 engine_overloaded after the client's ~4 s of quick retries."""
    statuses = iter([429, 429, 429, 429, 429, 429, 200])
    ok = {
        "id": "gen-2", "object": "chat.completion", "created": 0, "model": "m", "provider": "DeepInfra",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }

    def handler(request: httpx2.Request) -> httpx2.Response:
        status = next(statuses)
        body = ok if status == 200 else {"error": {"message": "Provider returned error", "code": 429}}
        return httpx2.Response(status, json=body)

    http = httpx2.AsyncClient(transport=httpx2.MockTransport(handler), event_hooks={"response": [capture_http_body]})
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key="test-not-real", http_client=http, max_retries=0)
    model = RecordingModel("deepseek/deepseek-v4.1-flash-1017", client, kind="openrouter")
    model.retry_waits = (0.0,) * 6
    await model.get_response(
        "sys", "hi", ModelSettings(temperature=0.0), [], None, [], ModelTracing.DISABLED,
        previous_response_id=None, conversation_id=None, prompt=None,
    )
    record = model.take_record()
    assert record.attempts == 7 and record.identity.response_provider == "DeepInfra"

    model.retry_waits = (0.0,)  # one wait, then it gives up and raises
    statuses = iter([429, 429])
    import openai
    import pytest

    with pytest.raises(openai.RateLimitError):
        await model.get_response(
            "sys", "hi", ModelSettings(temperature=0.0), [], None, [], ModelTracing.DISABLED,
            previous_response_id=None, conversation_id=None, prompt=None,
        )
    await http.aclose()


def test_jittered_waits_grow_from_two_seconds_cap_at_a_minute_and_span_ten_minutes() -> None:
    for seed in range(20):
        waits = jittered_waits(rng=random.Random(seed))
        for n, wait in enumerate(waits):
            ceiling = min(RETRY_CAP_S, 2.0 * 2**n)
            assert ceiling / 2 <= wait <= ceiling
        assert sum(waits) >= RETRY_WINDOW_S and sum(waits[:-1]) < RETRY_WINDOW_S
        assert max(waits) <= 60.0 and waits[0] <= 2.0
    assert jittered_waits(rng=random.Random(1)) != jittered_waits(rng=random.Random(2))  # runs do not retry in lockstep


async def test_each_call_draws_its_own_waits() -> None:
    drawn: list[int] = []

    def waits() -> list[float]:
        drawn.append(1)
        return [0.0]

    statuses = iter([429, 200, 429, 200])
    ok = {
        "id": "gen-3", "object": "chat.completion", "created": 0, "model": "m", "provider": "DeepInfra",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }

    def handler(_request: httpx2.Request) -> httpx2.Response:
        status = next(statuses)
        return httpx2.Response(status, json=ok if status == 200 else {"error": {"message": "Provider returned error", "code": 429}})

    http = httpx2.AsyncClient(transport=httpx2.MockTransport(handler), event_hooks={"response": [capture_http_body]})
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key="test-not-real", http_client=http, max_retries=0)
    model = RecordingModel("deepseek/deepseek-v4.1-flash-1017", client, kind="openrouter")
    model.retry_waits = waits
    try:
        for _ in range(2):
            await model.get_response(
                "sys", "hi", ModelSettings(temperature=0.0), [], None, [], ModelTracing.DISABLED,
                previous_response_id=None, conversation_id=None, prompt=None,
            )
    finally:
        await http.aclose()
    assert len(drawn) == 2 and [record.attempts for record in model.history] == [2, 2]
