"""OpenAI-compatible chat model that records the raw HTTP response.

The Agents SDK copies ``provider_data.model`` from the model we requested and
drops DeepSeek cache fields from ``Usage``. Served provider, served model, and
cache tokens are taken from the response body captured by an httpx hook.

``stream_response`` is forwarded without that accounting. ``Runner.run`` uses
``get_response``.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import Callable, Sequence
from collections import deque
from datetime import datetime, timezone
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

import httpx2
from agents.model_settings import ModelSettings
from agents.models.interface import ModelTracing
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from openai import APIConnectionError, AsyncOpenAI, InternalServerError, RateLimitError
from openai.types.responses.response_prompt_param import ResponsePromptParam

from evcarry.providers.identity import ServedIdentity, served_identity
from evcarry.providers.pin import PinCheck, ProviderMismatch, check_pin, current_pin_guard, provider_slug
from evcarry.providers.usage import CallCost, PriceTable, TokenUsage, call_cost, token_usage_from_body
from evcarry.runs.budget import admit_model_call
from evcarry.trace.writer import current_writer

# One bucket per in-flight get_response, so concurrent calls do not share attempts.
_HTTP_BODIES: ContextVar[list[dict[str, Any]] | None] = ContextVar("evcarry_http_bodies", default=None)

# Waits between attempts of one model call after a 429, a 5xx or a dropped connection. Each attempt
# is the OpenAI client's request with its own two quick retries (about 4 s), and on the endpoint-set
# route OpenRouter already moves a 429 to the next endpoint inside one request. DeepInfra's shared pool
# answered "engine_overloaded" 429s for minutes at a time during the pilot (2026-10-24), and a run that
# fails mid-way is redone from scratch, so a call keeps trying for about 10 minutes. Waits grow
# exponentially from 2 s and are capped at 60 s, so a call never sleeps for minutes after the provider
# has recovered; each wait is drawn from [c/2, c] ("equal jitter") so parallel runs that hit the same
# 429 do not retry in lockstep. Unanswered requests are not billed. The same for every arm and the judge.
RETRY_BASE_S = 2.0
RETRY_CAP_S = 60.0
RETRY_WINDOW_S = 600.0
_RETRYABLE = (RateLimitError, InternalServerError, APIConnectionError)


def jittered_waits(
    *,
    base: float = RETRY_BASE_S,
    cap: float = RETRY_CAP_S,
    window: float = RETRY_WINDOW_S,
    rng: random.Random | None = None,
) -> list[float]:
    """Equal-jitter exponential backoff: wait n is drawn from [c/2, c], c = min(cap, base * 2**n), until the waits sum to ``window``."""
    draw = rng or random
    waits: list[float] = []
    while sum(waits) < window:
        ceiling = min(cap, base * 2 ** len(waits))
        waits.append(ceiling / 2 + draw.uniform(0, ceiling / 2))
    return waits


_ROUTING_KEYS = (
    "order",
    "allow_fallbacks",
    "require_parameters",
    "only",
    "ignore",
    "quantizations",
    "sort",
    "data_collection",
    "zdr",
)


@dataclass
class LlmCallRecord:
    configured_model: str
    kind: str
    identity: ServedIdentity
    usage: TokenUsage
    routing: dict[str, Any] | None
    attempts: int
    latency_ms: float
    system_instructions: str | None = None
    pin: PinCheck | None = None
    # The sampling temperature the request carried (None: not sent, so the provider default applied).
    temperature: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "configured_model": self.configured_model,
            "kind": self.kind,
            **self.identity.as_dict(),
            "routing": self.routing,
            "attempts": self.attempts,
            "latency_ms": self.latency_ms,
            "usage": self.usage.as_dict(),
            "provider_check": self.pin.as_dict() if self.pin is not None else None,
            "temperature": self.temperature,
        }

    def cost(self, price: PriceTable, at: datetime) -> CallCost:
        """Priced at the endpoint that served the call; two disagreeing provider names get the table's upper bound."""
        names = {provider_slug(name): name for name in (self.identity.response_provider, self.identity.served_provider) if name}
        provider = next(iter(names.values())) if len(names) == 1 else None
        return call_cost(self.usage, price, at, provider)

    def provenance(self) -> dict[str, object]:
        """Who served the call: the fields ``llm_call``, ``llm_error`` and ``judge_call`` events share."""
        return {
            "served_model": self.identity.served_model,
            "served_provider": self.identity.served_provider,
            "response_provider": self.identity.response_provider,
            "provider_unobserved": self.identity.provider_unobserved,
            "generation_id": self.identity.generation_id,
            "service_tier": self.identity.service_tier,
            "quantization": self.identity.quantization,
            "provider_check": self.pin.as_dict() if self.pin is not None else None,
        }


def routing_extra_body(routing: dict[str, Any] | None) -> dict[str, Any]:
    """OpenRouter ``provider`` object, plus a top-level ``service_tier`` when the routing names one. Unknown keys are dropped.

    ``service_tier: default`` matters for a pinned open-weight model: OpenRouter lets a
    provider's priority endpoint (``deepinfra/fp8/priority``) compete for default traffic
    under its base slug, at a higher price, unless the request pins the standard tier.
    """
    if not routing:
        return {}
    body: dict[str, Any] = {}
    provider = {key: routing[key] for key in _ROUTING_KEYS if key in routing}
    if provider:
        body["provider"] = provider
    if routing.get("service_tier"):
        body["service_tier"] = str(routing["service_tier"])
    return body


def openrouter_headers(*, http_referer: str | None, x_title: str | None) -> dict[str, str]:
    """Headers sent on every OpenRouter call.

    ``X-OpenRouter-Metadata: enabled`` makes the response include
    ``openrouter_metadata`` (omitted on cache hits). ``X-Title`` is the header
    requested for this harness; ``X-OpenRouter-Title`` is the name current
    OpenRouter docs use. Both carry ``x_title`` when it is set.
    """
    headers = {"X-OpenRouter-Metadata": "enabled"}
    if http_referer:
        headers["HTTP-Referer"] = http_referer
    if x_title:
        headers["X-Title"] = x_title
        headers["X-OpenRouter-Title"] = x_title
    return headers


def sent_temperature(request: httpx2.Request) -> float | None:
    """The ``temperature`` in a request's JSON body, or None when the request did not send one."""
    try:
        body = json.loads(request.content)
    except Exception:
        return None
    value = body.get("temperature") if isinstance(body, dict) else None
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


async def capture_http_body(response: httpx2.Response) -> None:
    """Append each HTTP response JSON, and the temperature its request sent, to the current call's bucket."""
    bucket = _HTTP_BODIES.get()
    if bucket is None:
        return
    try:
        await response.aread()
        payload = json.loads(response.content)
    except Exception:
        return
    if isinstance(payload, dict):
        bucket.append({"status": response.status_code, "body": payload, "temperature": sent_temperature(response.request)})


def http_client_with_capture(**kwargs: Any) -> httpx2.AsyncClient:
    hooks = dict(kwargs.pop("event_hooks", {}) or {})
    response_hooks = list(hooks.get("response", []))
    response_hooks.append(capture_http_body)
    hooks["response"] = response_hooks
    return httpx2.AsyncClient(event_hooks=hooks, **kwargs)


def async_openai_client(
    *,
    base_url: str,
    api_key: str,
    default_headers: dict[str, str] | None = None,
    http_client: httpx2.AsyncClient | None = None,
) -> AsyncOpenAI:
    client = http_client or http_client_with_capture()
    return AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        default_headers=default_headers or None,
        http_client=client,
    )


class RecordingModel(OpenAIChatCompletionsModel):
    """Chat-completions model that pins OpenRouter routing and logs who served the call."""

    def __init__(
        self,
        model: str,
        openai_client: AsyncOpenAI,
        *,
        kind: str,
        routing: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
        price: PriceTable | None = None,
    ) -> None:
        super().__init__(model=model, openai_client=openai_client)
        self.kind = kind
        # Prices a call that was answered (and billed) but then failed in the SDK.
        self.price = price
        # "agent" or "judge": which run_end cost a failed call belongs to.
        self.role = "agent"
        self.configured_model = model if isinstance(model, str) else str(model)
        self.routing = dict(routing) if routing else None
        self.extra_headers = dict(extra_headers or {})
        self.history: list[LlmCallRecord] = []
        self._pending: deque[LlmCallRecord] = deque()
        # Waits between attempts, drawn afresh for each call (tests set a fixed sequence).
        self.retry_waits: Callable[[], Sequence[float]] | Sequence[float] = jittered_waits

    def take_record(self) -> LlmCallRecord | None:
        if not self._pending:
            return None
        return self._pending.popleft()

    def _merge_settings(self, model_settings: ModelSettings) -> ModelSettings:
        pinned = routing_extra_body(self.routing)
        body = dict(model_settings.extra_body or {})
        if "provider" in pinned and isinstance(body.get("provider"), dict):
            # The model config's routing wins over a per-call provider object.
            provider = {**body["provider"], **pinned["provider"]}
            body = {**body, **pinned, "provider": provider}
        else:
            body = {**body, **pinned}
        headers = {**(model_settings.extra_headers or {}), **self.extra_headers}
        from dataclasses import replace

        return replace(
            model_settings,
            extra_body=body or None,
            extra_headers=headers or None,
        )

    def _record(self, bucket: list[dict[str, Any]], started: float, system_instructions: str | None) -> LlmCallRecord:
        successes = [item for item in bucket if 200 <= int(item.get("status", 0)) < 300]
        attempt = successes[-1] if successes else (bucket[-1] if bucket else None)
        chosen = attempt["body"] if attempt else None
        identity = served_identity(chosen, kind=self.kind, configured_model=self.configured_model)
        usage = token_usage_from_body(chosen if isinstance(chosen, dict) else None)
        record = LlmCallRecord(
            configured_model=self.configured_model,
            kind=self.kind,
            identity=identity,
            usage=usage,
            routing=self.routing,
            attempts=len(bucket),
            latency_ms=(time.perf_counter() - started) * 1000,
            system_instructions=system_instructions,
            pin=check_pin(identity, self.routing) if successes else None,
            temperature=attempt.get("temperature") if attempt else None,
        )
        self.history.append(record)
        return record

    def _trace_failure(self, record: LlmCallRecord, bucket: list[dict[str, Any]], error: BaseException) -> None:
        """An ``llm_error`` event for a call that raised. ``on_llm_end`` never sees it.

        Tokens and served model are reported only when a 2xx body was captured
        (the call was answered and billed, then failed in the SDK).
        """
        writer = current_writer.get()
        if writer is None:
            return
        answered = any(200 <= int(item.get("status", 0)) < 300 for item in bucket)
        # An answered call was billed: its cost counts toward the spend cap like an llm_call's.
        if answered and self.price:
            cost = record.cost(self.price, datetime.now(timezone.utc)).fields()
        else:
            reported = record.usage.provider_reported_cost_usd if answered else None
            cost = {"cost_usd": max(0.0, reported or 0.0), "provider_reported_cost_usd": reported}
        writer.write(
            {
                "type": "llm_error",
                "role": self.role,
                "configured_model": record.configured_model,
                "kind": record.kind,
                "error_type": type(error).__name__,
                "error": str(error)[:1000],
                "attempts": record.attempts,
                "statuses": [item.get("status") for item in bucket],
                **(record.provenance() if answered else {"served_model": None}),
                "tokens": record.usage.as_dict() if answered else None,
                "temperature": record.temperature,
                **cost,
                "latency_ms": record.latency_ms,
            }
        )

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[Any],
        model_settings: ModelSettings,
        tools: list[Any],
        output_schema: Any,
        handoffs: list[Any],
        tracing: ModelTracing,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        prompt: ResponsePromptParam | None = None,
    ) -> Any:
        guard = current_pin_guard.get()
        if guard is not None:
            guard.raise_if_violated()  # no further paid calls once the run broke its pin
        admit_model_call()  # no further calls once the run is at its cost ceiling or the experiment at its cap
        bucket: list[dict[str, Any]] = []
        token = _HTTP_BODIES.set(bucket)
        started = time.perf_counter()
        waits = self.retry_waits() if callable(self.retry_waits) else self.retry_waits
        try:
            for wait in (*waits, None):
                try:
                    response = await super().get_response(
                        system_instructions,
                        input,
                        self._merge_settings(model_settings),
                        tools,
                        output_schema,
                        handoffs,
                        tracing,
                        previous_response_id=previous_response_id,
                        conversation_id=conversation_id,
                        prompt=prompt,
                    )
                    break
                except _RETRYABLE:
                    if wait is None:
                        raise
                    await asyncio.sleep(wait)  # every HTTP status lands in the bucket: attempts and statuses show the retries
        except BaseException as error:
            # Not queued for take_record: on_llm_end does not run for a failed call, so a
            # queued record would be paired with the next successful call and that
            # call's tokens and served model would be logged as this one's (zero, requested id).
            failed = self._record(bucket, started, system_instructions)
            if failed.pin is not None and failed.pin.ok is False:
                self._refuse(failed, bucket)  # answered off-pin, then failed in the SDK
            self._trace_failure(failed, bucket, error)
            raise
        finally:
            _HTTP_BODIES.reset(token)
        record = self._record(bucket, started, system_instructions)
        if record.pin is not None and record.pin.ok is False:
            self._refuse(record, bucket)
        self._pending.append(record)
        return response

    def _refuse(self, record: LlmCallRecord, bucket: list[dict[str, Any]]) -> None:
        """Log the off-pin call (it was billed) and stop the run: this reply is never used."""
        reason = record.pin.reason if record.pin is not None else "provider pin broken"
        error = ProviderMismatch(f"{self.configured_model}: {reason}")
        writer = current_writer.get()
        if writer is not None:
            writer.write({"type": "provider_mismatch", "configured_model": self.configured_model, **record.provenance()})
        self._trace_failure(record, bucket, error)
        guard = current_pin_guard.get()
        if guard is not None:
            guard.violate(reason)
        raise error

    def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[Any],
        model_settings: ModelSettings,
        tools: list[Any],
        output_schema: Any,
        handoffs: list[Any],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: ResponsePromptParam | None,
    ) -> Any:
        """Delegate streaming. Token and provider logs are not recorded on this path."""
        return super().stream_response(
            system_instructions,
            input,
            self._merge_settings(model_settings),
            tools,
            output_schema,
            handoffs,
            tracing,
            previous_response_id=previous_response_id,
            conversation_id=conversation_id,
            prompt=prompt,
        )
