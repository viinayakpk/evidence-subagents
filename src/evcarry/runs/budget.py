"""Spend cap and DeepSeek off-peak window.

Mock and other non-billable providers skip both gates. A billable run is
refused when logged spend is already at the cap, or when the schedule is
``offpeak_only`` and the clock is inside a weekday peak window.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from evcarry.providers.usage import PriceTable, is_peak


class BudgetError(Exception):
    """Base class for budget refusals."""


class SpendCapExceeded(BudgetError):
    """Logged spend is already at or above the cap."""


class PeakWindowError(BudgetError):
    """A billable run was asked to start during a peak window."""


@dataclass
class BudgetGuard:
    cap_usd: float | None
    schedule: str
    price: PriceTable
    spent_usd: float = 0.0

    def check(self, *, billable: bool, now: datetime) -> None:
        if not billable:
            return
        if self.schedule == "offpeak_only" and is_peak(now, self.price):
            raise PeakWindowError(
                "Refusing to start a billable run during a weekday peak window "
                "(01:00-04:00 and 06:00-10:00 UTC)."
            )
        if self.cap_usd is not None and self.spent_usd >= self.cap_usd:
            raise SpendCapExceeded(
                f"Refusing to start a billable run: spent ${self.spent_usd:.6f} "
                f"is at the cap ${self.cap_usd:.6f}."
            )

    def add(self, amount: float) -> None:
        self.spent_usd += amount
