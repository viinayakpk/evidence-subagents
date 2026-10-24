"""Token and dollar accounting from a dated price table.

DeepSeek's cache-hit fields are not copied into the Agents SDK ``Usage``
object (cached_tokens stays 0). Callers pass the raw JSON usage body.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timezone
from typing import Any


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    cache_hit_tokens: int
    cache_miss_tokens: int
    output_tokens: int
    provider_reported_cost_usd: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "input_tokens": self.input_tokens,
            "cache_hit_tokens": self.cache_hit_tokens,
            "cache_miss_tokens": self.cache_miss_tokens,
            "output_tokens": self.output_tokens,
            "provider_reported_cost_usd": self.provider_reported_cost_usd,
        }


@dataclass(frozen=True)
class PriceTable:
    input_cache_miss: float
    input_cache_hit: float
    output: float
    peak_multiplier: float = 2.0
    peak_windows_utc: tuple[tuple[time, time], ...] = (
        (time(1, 0), time(4, 0)),
        (time(6, 0), time(10, 0)),
    )
    weekdays_only: bool = True
    currency: str = "USD"

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> PriceTable:
        per = data.get("per_million_tokens") or data
        windows = []
        for start, end in data.get("peak_windows_utc") or [["01:00", "04:00"], ["06:00", "10:00"]]:
            windows.append((_parse_hhmm(start), _parse_hhmm(end)))
        return cls(
            input_cache_miss=float(per["input_cache_miss"]),
            input_cache_hit=float(per["input_cache_hit"]),
            output=float(per["output"]),
            peak_multiplier=float(data.get("peak_multiplier", 2)),
            peak_windows_utc=tuple(windows),
            weekdays_only=bool(data.get("weekdays_only", True)),
            currency=str(data.get("currency", "USD")),
        )


def _parse_hhmm(value: str) -> time:
    hour, minute = value.split(":")
    return time(int(hour), int(minute))


def _as_utc(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def is_peak(moment: datetime, price: PriceTable) -> bool:
    """Weekday peak windows in UTC. The end of each window is exclusive."""
    current = _as_utc(moment)
    if price.weekdays_only and current.weekday() >= 5:
        return False
    clock = current.time()
    return any(start <= clock < end for start, end in price.peak_windows_utc)


def cost_usd(tokens: TokenUsage, price: PriceTable, at: datetime) -> float:
    multiplier = price.peak_multiplier if is_peak(at, price) else 1.0
    dollars = (
        tokens.cache_miss_tokens * price.input_cache_miss
        + tokens.cache_hit_tokens * price.input_cache_hit
        + tokens.output_tokens * price.output
    ) / 1_000_000
    return dollars * multiplier


def token_usage_from_body(body: dict[str, Any] | None) -> TokenUsage:
    """Read chat-completions usage, including DeepSeek cache fields.

    Accepted hit/miss names, in order:
    ``prompt_cache_hit_tokens`` / ``prompt_cache_miss_tokens``, then
    ``prompt_tokens_details.cached_tokens`` or
    ``input_tokens_details.cached_tokens``.
    """
    usage = {}
    if isinstance(body, dict):
        raw_usage = body.get("usage")
        if isinstance(raw_usage, dict):
            usage = raw_usage
    prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    completion = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    hit = usage.get("prompt_cache_hit_tokens")
    miss = usage.get("prompt_cache_miss_tokens")
    if hit is None:
        details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
        if isinstance(details, dict):
            hit = details.get("cached_tokens")
    hit_tokens = int(hit or 0)
    miss_tokens = int(miss) if miss is not None else max(prompt - hit_tokens, 0)
    reported = usage.get("cost")
    reported_cost = float(reported) if isinstance(reported, (int, float)) else None
    return TokenUsage(prompt, hit_tokens, miss_tokens, completion, reported_cost)


def token_usage_from_counts(
    *,
    input_tokens: int,
    cached_tokens: int,
    output_tokens: int,
) -> TokenUsage:
    hit = max(int(cached_tokens), 0)
    prompt = max(int(input_tokens), 0)
    return TokenUsage(prompt, hit, max(prompt - hit, 0), max(int(output_tokens), 0))
