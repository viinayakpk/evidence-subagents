import json

import httpx2
from agents.model_settings import ModelSettings
from agents.models.interface import ModelTracing
from openai import AsyncOpenAI

from evcarry.providers.openai_compat import RecordingModel, capture_http_body, openrouter_headers


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
    record = model.history[-1]
    assert record.identity.served_provider == "DeepSeek"
    assert record.identity.served_model == "deepseek/deepseek-chat-v3"
    assert record.usage.cache_hit_tokens == 80
    assert record.usage.cache_miss_tokens == 20
    assert record.routing["allow_fallbacks"] is False
