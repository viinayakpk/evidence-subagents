"""Which provider and model actually served a completion.

OpenRouter's Chat Completions response carries this only when the request
sends ``X-OpenRouter-Metadata: enabled``. The selected endpoint is the
``openrouter_metadata.endpoints.available[]`` item with ``selected: true``.
The Agents SDK's ``provider_data.model`` is the model we asked for, so it is
not used here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ServedIdentity:
    served_model: str
    served_provider: str | None
    provider_unobserved: bool
    router: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "served_model": self.served_model,
            "served_provider": self.served_provider,
            "provider_unobserved": self.provider_unobserved,
            "router": self.router,
        }


def served_identity(body: dict[str, Any] | None, *, kind: str, configured_model: str) -> ServedIdentity:
    payload = body or {}
    response_model = payload.get("model")
    model = response_model if isinstance(response_model, str) and response_model else configured_model
    if kind != "openrouter":
        provider = {"openai_compatible": None, "scripted": "scripted"}.get(kind, kind)
        if kind == "openai_compatible":
            provider = None
        return ServedIdentity(model, provider if isinstance(provider, str) else _static_provider(kind), False, None)
    return _from_openrouter(payload, fallback_model=model)


def _static_provider(kind: str) -> str | None:
    if kind in {"deepseek", "vllm", "scripted"}:
        return kind
    return None


def _from_openrouter(body: dict[str, Any], *, fallback_model: str) -> ServedIdentity:
    metadata = body.get("openrouter_metadata")
    if not isinstance(metadata, dict):
        top = body.get("provider")
        provider = top if isinstance(top, str) else None
        return ServedIdentity(
            fallback_model,
            provider,
            provider is None,
            None,
        )
    selected_provider, selected_model = _selected_endpoint(metadata)
    if selected_provider is None:
        selected_provider, selected_model = _successful_attempt(metadata)
    router = {
        "requested": metadata.get("requested"),
        "strategy": metadata.get("strategy"),
        "attempt": metadata.get("attempt"),
        "summary": metadata.get("summary"),
        "selected_provider": selected_provider,
        "selected_model": selected_model,
    }
    return ServedIdentity(
        selected_model or fallback_model,
        selected_provider,
        selected_provider is None,
        router,
    )


def _selected_endpoint(metadata: dict[str, Any]) -> tuple[str | None, str | None]:
    endpoints = metadata.get("endpoints")
    if not isinstance(endpoints, dict):
        return None, None
    available = endpoints.get("available")
    if not isinstance(available, list):
        return None, None
    for item in available:
        if isinstance(item, dict) and item.get("selected") is True:
            provider = item.get("provider")
            model = item.get("model")
            return (
                provider if isinstance(provider, str) else None,
                model if isinstance(model, str) else None,
            )
    return None, None


def _successful_attempt(metadata: dict[str, Any]) -> tuple[str | None, str | None]:
    attempts = metadata.get("attempts")
    if not isinstance(attempts, list):
        return None, None
    for item in reversed(attempts):
        if isinstance(item, dict) and item.get("status") == 200:
            provider = item.get("provider")
            model = item.get("model")
            return (
                provider if isinstance(provider, str) else None,
                model if isinstance(model, str) else None,
            )
    return None, None
