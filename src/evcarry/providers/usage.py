"""Token and dollar accounting from a dated price table.

DeepSeek's cache-hit fields are not copied into the Agents SDK ``Usage``
object (cached_tokens stays 0). Callers pass the raw JSON usage body.

An OpenRouter route over several endpoints has one price row per endpoint tag
(``endpoints:`` in the price file). A call is priced at the row of the provider
that served it. Endpoints of one provider cannot be told apart in a response
(``inference-net/fp8`` and ``inference-net/us`` both answer ``InferenceNet``),
so such a provider is priced at the larger rate of each component; a provider
that is not in the table, or not observed, at the larger rate of each component
over the whole table. The cost a call counts is the larger of that price and
OpenRouter's own ``usage.cost`` (``call_cost``), so it is never below the bill.

A table may name a ``reference_endpoint`` (2026-10-25, P2-1). Every call then also
gets ``reference_cost_usd``: its tokens at that endpoint's rates, whichever endpoint
served it. The per-run cost ceiling can count that figure, so every arm gets the
same token budget whatever the provider mix; the cap keeps counting the bill.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timezone
from typing import Any

from evcarry.providers.pin import provider_slug


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
class Rates:
    """USD per million tokens."""

    input_cache_miss: float
    input_cache_hit: float
    output: float

    @classmethod
    def from_mapping(cls, per: dict[str, Any]) -> Rates:
        return cls(float(per["input_cache_miss"]), float(per["input_cache_hit"]), float(per["output"]))

    def usd(self, tokens: TokenUsage) -> float:
        return (
            tokens.cache_miss_tokens * self.input_cache_miss
            + tokens.cache_hit_tokens * self.input_cache_hit
            + tokens.output_tokens * self.output
        ) / 1_000_000


def _upper(rows: list[Rates]) -> Rates:
    """The larger rate of each component."""
    return Rates(
        max(row.input_cache_miss for row in rows),
        max(row.input_cache_hit for row in rows),
        max(row.output for row in rows),
    )


@dataclass(frozen=True)
class PriceTable:
    # With ``endpoints``: the larger rate of each component over them (and any top-level rates).
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
    # One row per OpenRouter endpoint tag (``deepinfra/fp8``); empty for a single-endpoint table.
    endpoints: tuple[tuple[str, Rates], ...] = ()
    # The endpoint tag whose rates price ``reference_cost_usd`` (None: the call's own price-table cost).
    reference_endpoint: str | None = None

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> PriceTable:
        endpoints = tuple((str(tag), Rates.from_mapping(per)) for tag, per in (data.get("endpoints") or {}).items())
        per = data.get("per_million_tokens") or ({} if endpoints else data)
        rows = [row for _, row in endpoints] + ([Rates.from_mapping(per)] if per else [])
        top = _upper(rows)
        windows = []
        # An explicit empty list means no peak windows (OpenRouter); only a missing key gets DeepSeek's.
        raw_windows = data["peak_windows_utc"] if data.get("peak_windows_utc") is not None else [["01:00", "04:00"], ["06:00", "10:00"]]
        for start, end in raw_windows:
            windows.append((_parse_hhmm(start), _parse_hhmm(end)))
        reference = data.get("reference_endpoint")
        if reference is not None and str(reference) not in {tag for tag, _ in endpoints}:
            raise ValueError(f"reference_endpoint {reference} is not one of the table's endpoints")
        return cls(
            input_cache_miss=top.input_cache_miss,
            input_cache_hit=top.input_cache_hit,
            output=top.output,
            peak_multiplier=float(data.get("peak_multiplier", 2)),
            peak_windows_utc=tuple(windows),
            weekdays_only=bool(data.get("weekdays_only", True)),
            currency=str(data.get("currency", "USD")),
            endpoints=endpoints,
            reference_endpoint=str(reference) if reference is not None else None,
        )

    def worst_rates(self) -> Rates:
        return Rates(self.input_cache_miss, self.input_cache_hit, self.output)

    def reference_rates(self) -> Rates | None:
        """The reference endpoint's rates, or None when the table names none."""
        return next((row for tag, row in self.endpoints if tag == self.reference_endpoint), None)

    def billed_per_reference_bound(self) -> float:
        """The largest ratio of any endpoint's rate to the reference rate, component by component (1.0 without a reference)."""
        reference = self.reference_rates()
        if reference is None:
            return 1.0
        ratios = [1.0]
        for _tag, row in self.endpoints:
            for mine, base in (
                (row.input_cache_miss, reference.input_cache_miss),
                (row.input_cache_hit, reference.input_cache_hit),
                (row.output, reference.output),
            ):
                if base > 0:
                    ratios.append(mine / base)
        return max(ratios)

    def rates_for(self, provider: str | None) -> tuple[Rates, str]:
        """Rates for a call served by ``provider`` (a response's provider name) and the row(s) they came from.

        A single-endpoint table prices every call at its rates. With endpoints: the
        rows of the provider that served the call (the larger rate of each component
        if it has several), else the larger rate of each component over the table.
        """
        if not self.endpoints:
            return self.worst_rates(), "table"
        slug = provider_slug(provider) if provider else ""
        matched = [(tag, row) for tag, row in self.endpoints if slug and provider_slug(tag.split("/")[0]) == slug]
        if not matched:
            return self.worst_rates(), "upper_bound"
        return _upper([row for _, row in matched]), "|".join(tag for tag, _ in matched)


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


def cost_usd(tokens: TokenUsage, price: PriceTable, at: datetime, provider: str | None = None) -> float:
    """Price-table cost of a call served by ``provider`` (see ``PriceTable.rates_for``)."""
    multiplier = price.peak_multiplier if is_peak(at, price) else 1.0
    rates, _ = price.rates_for(provider)
    return rates.usd(tokens) * multiplier


@dataclass(frozen=True)
class CallCost:
    """What one answered call cost. ``usd`` is what traces, ``run_end`` and the cap count."""

    usd: float
    table_usd: float
    priced_as: str
    reported_usd: float | None
    # The same tokens at the table's reference endpoint (None: no reference, the price-table cost is used).
    reference_usd: float | None = None

    def fields(self) -> dict[str, object]:
        return {
            "cost_usd": self.usd,
            "price_table_cost_usd": self.table_usd,
            "priced_as": self.priced_as,
            "provider_reported_cost_usd": self.reported_usd,
            "reference_cost_usd": self.reference_usd if self.reference_usd is not None else self.table_usd,
        }


def call_cost(tokens: TokenUsage, price: PriceTable, at: datetime, provider: str | None) -> CallCost:
    """The larger of the served endpoint's price-table cost and OpenRouter's ``usage.cost``, never the cheaper."""
    multiplier = price.peak_multiplier if is_peak(at, price) else 1.0
    rates, priced_as = price.rates_for(provider)
    table = rates.usd(tokens) * multiplier
    reported = tokens.provider_reported_cost_usd
    reference_rates = price.reference_rates()
    reference = reference_rates.usd(tokens) * multiplier if reference_rates is not None else None
    return CallCost(max(table, reported) if reported is not None else table, table, priced_as, reported, reference)


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
