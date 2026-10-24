"""Seeded injector. Off unless an experiment config turns it on.

The decision key is (task, seed, return index). The arm is not part of the key,
so the same corruption is chosen for every arm.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from evcarry.inject.corruptions import CORRUPTION_TYPES, VARIANTS, apply_corruption
from evcarry.ledger.ledger import Ledger
from evcarry.protocol.schema import Claim, SubagentReturn


@dataclass
class InjectionDecision:
    task: str
    seed: int
    return_index: int
    kind: str
    variant: str
    claim_id: str

    def as_dict(self) -> dict[str, object]:
        return {
            "task": self.task,
            "seed": self.seed,
            "return_index": self.return_index,
            "kind": self.kind,
            "variant": self.variant,
            "claim_id": self.claim_id,
        }


@dataclass
class FaultInjector:
    probability: float = 0.0
    types: tuple[str, ...] = CORRUPTION_TYPES
    variant: str = "naive"
    max_per_run: int | None = None
    applied: int = 0

    def decide(
        self,
        *,
        task: str,
        seed: int,
        return_index: int,
        claims: list[Claim],
    ) -> InjectionDecision | None:
        if self.probability <= 0 or not claims:
            return None
        if self.max_per_run is not None and self.applied >= self.max_per_run:
            return None
        rng = random.Random(f"{task}|{seed}|{return_index}")
        if rng.random() >= self.probability:
            return None
        claim = claims[rng.randrange(len(claims))]
        kind = self.types[rng.randrange(len(self.types))]
        if self.variant == "either":
            variant = VARIANTS[rng.randrange(len(VARIANTS))]
        else:
            variant = self.variant
        return InjectionDecision(task, seed, return_index, kind, variant, claim.claim_id)

    def apply(
        self,
        payload: SubagentReturn,
        decision: InjectionDecision,
        ledger: Ledger,
    ) -> SubagentReturn:
        updated = payload.model_copy(deep=True)
        for index, claim in enumerate(updated.claims):
            if claim.claim_id == decision.claim_id:
                updated.claims[index] = apply_corruption(claim, decision.kind, decision.variant, ledger)
                break
        else:
            raise KeyError(decision.claim_id)
        self.applied += 1
        return updated
