"""Spend cap, per-run cost ceiling, and DeepSeek off-peak window.

Mock and other non-billable providers skip the window and the cap. A billable
run is refused when logged spend is already at the cap, or when the schedule is
``offpeak_only`` and the clock is inside a weekday peak window.

Per-run ceiling (owner decision 5, 2026-10-24): every run has a ``RunMeter``
fed by the trace events it writes (``event_spend``: agent calls, judge calls,
answered-then-failed calls). Once a run's spend reaches
``budget.run_ceiling_usd`` its next model call (agent, sub-agent or judge) is
refused with ``CostCeilingExceeded``; the run ends with status
``cost_ceiling`` and counts as failed. The call that crosses the ceiling
completes, so a run can end slightly above it.

With a ceiling, the experiment cap is hard: a run starts only if what is spent,
plus ``RESERVE_FACTOR`` times the ceiling for it and for every run in flight
(less what each has spent), stays within the cap. If the cap is still reached
mid-run, every later model call is refused with ``SpendCapExceeded``; that run
gets no ``run_end`` (a restart runs it again) and the grid stops.

Reference ceiling (2026-10-25, P2-1): with ``ceiling_basis: reference`` the ceiling
counts each call's ``reference_cost_usd`` (its tokens at DeepInfra fp8 rates,
whichever endpoint served it), so every arm and provider mix has the same token
budget. The cap still counts billed dollars, so the per-run reserve is
``RESERVE_FACTOR x ceiling x reserve_billed_per_reference`` billed dollars.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Not at import time: every model imports this module (admit_model_call), and importing
    # evcarry.providers here would import them back before this module is ready.
    from evcarry.providers.usage import PriceTable

# What a run in flight may still cost, as a multiple of the ceiling: the call that crosses
# the ceiling, and the calls already running beside it, complete.
RESERVE_FACTOR = 1.2


class BudgetError(Exception):
    """Base class for budget refusals."""


class SpendCapExceeded(BudgetError):
    """Logged spend is already at or above the cap."""


class PeakWindowError(BudgetError):
    """A billable run was asked to start during a peak window."""


class CostCeilingExceeded(BudgetError):
    """This run's spend reached its per-run ceiling; its later model calls are refused."""


_SPEND_EVENTS = ("llm_call", "judge_call", "llm_error")


def event_spend(event: dict[str, Any]) -> float:
    """Dollars one trace event spent, for the cap: the larger of the price-table cost and the provider's bill.

    ``llm_error`` counts too: a call that was answered and then failed in the SDK
    was billed. The larger figure keeps a price change or an unexpected endpoint
    tier from letting the cap under-count.
    """
    if event.get("type") not in _SPEND_EVENTS:
        return 0.0
    table = float(event.get("cost_usd") or 0)
    reported = event.get("provider_reported_cost_usd")
    return max(table, float(reported)) if isinstance(reported, (int, float)) else table


def event_reference_spend(event: dict[str, Any]) -> float:
    """What one trace event counts toward a reference-priced ceiling (its billed spend when it has no reference cost)."""
    if event.get("type") not in _SPEND_EVENTS:
        return 0.0
    reference = event.get("reference_cost_usd")
    if isinstance(reference, (int, float)) and not isinstance(reference, bool):
        return float(reference)
    return event_spend(event)


class RunMeter:
    """Live spend of one run; ``admit`` runs before each of its model calls."""

    def __init__(self, ceiling_usd: float | None = None, guard: BudgetGuard | None = None, basis: str = "billed") -> None:
        if basis not in ("billed", "reference"):
            raise ValueError(f"unknown ceiling basis {basis}")
        self.ceiling_usd = ceiling_usd
        # Set for billable runs only: the experiment cap is checked at each call too.
        self.guard = guard
        self.basis = basis
        self.spent_usd = 0.0
        self.reference_usd = 0.0
        self.refusal: BudgetError | None = None

    def observe(self, event: dict[str, Any]) -> None:
        self.spent_usd += event_spend(event)
        self.reference_usd += event_reference_spend(event)

    @property
    def ceiling_spend_usd(self) -> float:
        """What the ceiling counts: billed dollars, or the same tokens at the reference endpoint's rates."""
        return self.reference_usd if self.basis == "reference" else self.spent_usd

    def admit(self) -> None:
        if self.refusal is None:
            if self.ceiling_usd is not None and self.ceiling_spend_usd >= self.ceiling_usd:
                self.refusal = CostCeilingExceeded(
                    f"run stopped at its cost ceiling: spent ${self.ceiling_spend_usd:.6f} ({self.basis}) "
                    f"of ${self.ceiling_usd:.2f}"
                )
            elif self.guard is not None and self.guard.cap_reached():
                self.refusal = SpendCapExceeded(
                    f"run stopped: the experiment reached its cap (${self.guard.total_usd():.6f} of ${self.guard.cap_usd:.2f})"
                )
        if self.refusal is not None:
            raise type(self.refusal)(str(self.refusal))

    @property
    def ceiling_hit(self) -> bool:
        return isinstance(self.refusal, CostCeilingExceeded)

    def raise_if_cap_reached(self) -> None:
        """After the agent stops: a call refused at the experiment cap (and swallowed by a sub-agent tool) fails the run."""
        if isinstance(self.refusal, SpendCapExceeded):
            raise SpendCapExceeded(str(self.refusal))


current_run_meter: ContextVar[RunMeter | None] = ContextVar("evcarry_run_meter", default=None)


def admit_model_call() -> None:
    """Every model's ``get_response`` calls this first (agents, sub-agents, judge)."""
    meter = current_run_meter.get()
    if meter is not None:
        meter.admit()


@dataclass
class BudgetGuard:
    cap_usd: float | None
    schedule: str
    price: PriceTable
    # Spend of finished runs, including every trace already in the output directory.
    spent_usd: float = 0.0
    run_ceiling_usd: float | None = None
    live: list[RunMeter] = field(default_factory=list)
    # Billed dollars per ceiling dollar: 1.0 for a billed ceiling; the worst endpoint's ratio for a reference ceiling.
    reserve_billed_per_reference: float = 1.0

    def total_usd(self) -> float:
        return self.spent_usd + sum(meter.spent_usd for meter in self.live)

    def cap_reached(self) -> bool:
        return self.cap_usd is not None and self.total_usd() >= self.cap_usd

    def reserve_usd(self) -> float:
        """Billed dollars held back for one run in flight."""
        if self.run_ceiling_usd is None:
            return 0.0
        return RESERVE_FACTOR * self.run_ceiling_usd * self.reserve_billed_per_reference

    def check(self, *, billable: bool, now: datetime) -> None:
        from evcarry.providers.usage import is_peak

        if not billable:
            return
        if self.schedule == "offpeak_only" and is_peak(now, self.price):
            raise PeakWindowError(
                "Refusing to start a billable run during a weekday peak window "
                "(01:00-04:00 and 06:00-10:00 UTC)."
            )
        if self.cap_usd is None:
            return
        total = self.total_usd()
        if total >= self.cap_usd:
            raise SpendCapExceeded(
                f"Refusing to start a billable run: spent ${total:.6f} is at the cap ${self.cap_usd:.6f}."
            )
        if self.run_ceiling_usd is not None:
            reserve = self.reserve_usd()
            in_flight = sum(max(0.0, reserve - meter.spent_usd) for meter in self.live)
            if total + in_flight + reserve > self.cap_usd:
                raise SpendCapExceeded(
                    f"Refusing to start a billable run: spent ${total:.6f}, plus ${in_flight:.6f} reserved for "
                    f"{len(self.live)} run(s) in flight, plus ${reserve:.2f} for this one would pass the cap "
                    f"${self.cap_usd:.6f}."
                )

    def open(self, meter: RunMeter) -> None:
        self.live.append(meter)

    def close(self, meter: RunMeter) -> None:
        """A run ended (finished, failed or aborted): its spend becomes committed spend."""
        if meter in self.live:
            self.live.remove(meter)
            self.spent_usd += meter.spent_usd

    def add(self, amount: float) -> None:
        self.spent_usd += amount
