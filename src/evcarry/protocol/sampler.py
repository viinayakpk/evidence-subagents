"""Risk-based and random audit samplers.

Both samplers are seeded and budget-exact. The budget is
``round_half_up(rate * n)`` claims, capped at the number of claims.
A rate of 0 audits nothing. ``0.3 * 1`` rounds to 0.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP

from evcarry.protocol.checks import is_numeric_value
from evcarry.protocol.schema import Claim

_NUMBER_IN_TEXT = re.compile(r"-?\d+(?:\.\d+)?")


def budget_count(n_claims: int, rate: float) -> int:
    if n_claims <= 0 or rate <= 0:
        return 0
    count = int((Decimal(str(rate)) * Decimal(n_claims)).to_integral_value(ROUND_HALF_UP))
    return max(0, min(n_claims, count))


@dataclass
class RiskContext:
    """Features the sampler can see without calling a model."""

    deliverable_ids: set[str] = field(default_factory=set)
    used_by_write_ids: set[str] = field(default_factory=set)
    failed_l0_l1_ids: set[str] = field(default_factory=set)
    hit_limits: bool = False
    had_errors: bool = False
    finished_very_fast: bool = False


def risk_score(claim: Claim, ctx: RiskContext) -> int:
    score = 0
    if claim.claim_id in ctx.deliverable_ids or claim.claim_id in ctx.used_by_write_ids:
        score += 5
    if claim.kind in {"derived", "action_done"}:
        score += 4
    if is_numeric_value(claim.value) or _NUMBER_IN_TEXT.search(claim.text):
        score += 3
    if ctx.hit_limits or ctx.had_errors or ctx.finished_very_fast:
        score += 3
    if claim.confidence == "low":
        score += 2
    if claim.claim_id in ctx.failed_l0_l1_ids:
        score += 6
    return score


def sample_risk(
    claims: list[Claim],
    *,
    rate: float,
    ctx: RiskContext | None = None,
    budget: int | None = None,
) -> list[str]:
    """Highest risk first. Ties break by claim_id so the choice is stable."""
    context = ctx or RiskContext()
    k = budget_count(len(claims), rate) if budget is None else min(len(claims), budget)
    ranked = sorted(claims, key=lambda claim: (-risk_score(claim, context), claim.claim_id))
    return [claim.claim_id for claim in ranked[:k]]


def sample_random(claims: list[Claim], *, rate: float, seed: int, budget: int | None = None) -> list[str]:
    k = budget_count(len(claims), rate) if budget is None else min(len(claims), budget)
    if k <= 0:
        return []
    identifiers = sorted(claim.claim_id for claim in claims)
    rng = random.Random(seed)
    return rng.sample(identifiers, k)
