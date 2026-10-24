"""Fail a run whose OpenRouter call was served by a provider other than the pinned one.

A model entry pins its provider when its routing has ``allow_fallbacks: false``
and an ``only`` or ``order`` list of endpoint tags (``deepinfra/fp8``). The
first tag segment is the provider slug; it is compared with the provider names
in the response (``DeepInfra``) after lower-casing and dropping everything but
letters and digits, so ``inference-net`` matches ``InferenceNet``. A
quantization segment (``fp8``) is compared with an observed quantization, and
a ``flex`` / ``priority`` segment with the response's ``service_tier``; neither
is in OpenRouter's responses today, so in practice the provider name decides.

A call whose provider cannot be observed passes (``ok: null``) and is logged as
unobserved. The check runs on every call: ``llm_call`` and ``judge_call`` record
it, and a mismatch raises ``ProviderMismatch``. Sub-agent tools and the judge
catch exceptions, so a run-wide ``PinGuard`` also remembers the mismatch: every
later model call of the run raises at once, and the run manager re-raises it
after the agent stops. A run that failed this way gets no ``run_end``.
"""

from __future__ import annotations

import re
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from evcarry.providers.identity import ServedIdentity

_QUANTIZATIONS = {"fp4", "mxfp4", "nvfp4", "fp6", "fp8", "fp16", "bf16", "fp32", "int4", "int8"}
_TIERS = {"flex", "priority"}


class ProviderMismatch(RuntimeError):
    """A pinned OpenRouter call was served by another provider, quantization, or tier."""


def provider_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


@dataclass(frozen=True)
class Pin:
    tags: tuple[str, ...]
    providers: frozenset[str]
    quantizations: frozenset[str]
    tiers: frozenset[str]


def pin_from_routing(routing: dict[str, Any] | None) -> Pin | None:
    """The pin a routing object enforces, or None when fallbacks are allowed or no endpoint is named."""
    if not routing or routing.get("allow_fallbacks", True) is not False:
        return None
    tags = routing.get("only") or routing.get("order")
    if not tags:
        return None
    tags = tuple(str(tag) for tag in tags)
    parts = [tag.lower().split("/") for tag in tags]
    return Pin(
        tags=tags,
        providers=frozenset(provider_slug(segments[0]) for segments in parts),
        quantizations=frozenset(s for segments in parts for s in segments[1:] if s in _QUANTIZATIONS),
        # A tag without a tier segment is the provider's default tier.
        tiers=frozenset(next((s for s in segments[1:] if s in _TIERS), "default") for segments in parts),
    )


@dataclass(frozen=True)
class PinCheck:
    ok: bool | None
    expected: tuple[str, ...] = ()
    observed: dict[str, str | None] = field(default_factory=dict)
    reason: str = ""

    def as_dict(self) -> dict[str, object]:
        return {"ok": self.ok, "expected": list(self.expected), "observed": dict(self.observed), "reason": self.reason}


def check_pin(identity: ServedIdentity, routing: dict[str, Any] | None) -> PinCheck:
    pin = pin_from_routing(routing)
    observed = {
        "response_provider": identity.response_provider,
        "selected_provider": identity.served_provider,
        "quantization": identity.quantization,
        "service_tier": identity.service_tier,
    }
    if pin is None:
        return PinCheck(None, (), observed, "routing does not pin a provider")
    names = [name for name in (identity.response_provider, identity.served_provider) if name]
    if not names:
        return PinCheck(None, pin.tags, observed, "provider not observed")
    wrong = [name for name in names if provider_slug(name) not in pin.providers]
    if wrong:
        return PinCheck(False, pin.tags, observed, f"served by {wrong[0]}, pinned to {list(pin.tags)}")
    if pin.quantizations and identity.quantization and identity.quantization.lower() not in pin.quantizations:
        return PinCheck(False, pin.tags, observed, f"quantization {identity.quantization}, pinned to {list(pin.tags)}")
    tier = (identity.service_tier or "").lower()
    if tier in _TIERS | {"default"} and tier not in pin.tiers:
        return PinCheck(False, pin.tags, observed, f"service tier {identity.service_tier}, pinned to {list(pin.tags)}")
    return PinCheck(True, pin.tags, observed, "")


class PinGuard:
    """Per-run memory of the first pin violation; shared by every model in the run."""

    def __init__(self) -> None:
        self.violation: str | None = None

    def violate(self, reason: str) -> None:
        if self.violation is None:
            self.violation = reason

    def raise_if_violated(self) -> None:
        if self.violation is not None:
            raise ProviderMismatch(f"run stopped: an earlier call broke the provider pin ({self.violation})")


current_pin_guard: ContextVar[PinGuard | None] = ContextVar("evcarry_pin_guard", default=None)
