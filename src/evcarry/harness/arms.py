"""Arm defaults. The arm name selects the sampler family when it conflicts with YAML."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArmSpec:
    name: str
    delegate: bool
    schema: bool
    checks: tuple[str, ...]
    sampler: str | None
    default_rate: float
    require_evidence: bool
    script: str
    # Blind re-delegations per sub-agent call (budget_matched, A21): the harness asks the sub-agent to
    # re-check and resend its whole return this many times, with no evidence, checks or verdicts.
    blind_redelegations: int = 0


ARMS: dict[str, ArmSpec] = {
    "sa": ArmSpec("sa", False, False, (), None, 0.0, False, "sa.json"),
    "vanilla_mad": ArmSpec("vanilla_mad", True, False, (), None, 0.0, False, "vanilla_mad.json"),
    "self_verify": ArmSpec("self_verify", True, False, (), None, 0.0, False, "vanilla_mad.json"),
    # Plan amendment A21 (2026-10-25): vanilla_mad plus one blind re-delegation per sub-agent call, the
    # compute control for evidence_risk's audit-driven re-delegation. Same prompts and tools as vanilla_mad.
    "budget_matched": ArmSpec("budget_matched", True, False, (), None, 0.0, False, "budget_matched.json", 1),
    "typed": ArmSpec("typed", True, True, (), None, 0.0, False, "typed.json"),
    "evidence": ArmSpec("evidence", True, True, ("L0", "L1"), "risk", 0.0, True, "evidence_risk.json"),
    "evidence_random": ArmSpec(
        "evidence_random",
        True,
        True,
        ("L0", "L1", "L2", "L3"),
        "random",
        0.3,
        True,
        "evidence_risk.json",
    ),
    "evidence_risk": ArmSpec(
        "evidence_risk",
        True,
        True,
        ("L0", "L1", "L2", "L3"),
        "risk",
        0.3,
        True,
        "evidence_risk.json",
    ),
}

# Design R (plan amendment A21, 2026-10-25). self_verify stays defined for the local grid.
HEADLINE_ARMS = ("sa", "vanilla_mad", "evidence_risk", "budget_matched")


def arm_spec(name: str) -> ArmSpec:
    if name not in ARMS:
        raise KeyError(f"unknown arm {name}")
    return ARMS[name]


def effective_audit(name: str, *, sampler: str, rate: float, checks: list[str]) -> tuple[tuple[str, ...], str | None, float]:
    """Arm name wins for sampler kind and for which check levels exist.

    YAML supplies the rate for arms that spot-check (evidence_risk, evidence_random).
    The evidence arm stays L0/L1 with rate 0. Typed, SA, and MAD do not audit.
    """
    spec = arm_spec(name)
    if not spec.checks:
        return (), None, 0.0
    if spec.name == "evidence":
        return spec.checks, spec.sampler, 0.0
    chosen = spec.sampler or sampler
    return spec.checks, chosen, rate if rate is not None else spec.default_rate
