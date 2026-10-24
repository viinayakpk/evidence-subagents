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


ARMS: dict[str, ArmSpec] = {
    "sa": ArmSpec("sa", False, False, (), None, 0.0, False, "sa.json"),
    "vanilla_mad": ArmSpec("vanilla_mad", True, False, (), None, 0.0, False, "vanilla_mad.json"),
    "self_verify": ArmSpec("self_verify", True, False, (), None, 0.0, False, "vanilla_mad.json"),
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

HEADLINE_ARMS = ("sa", "vanilla_mad", "self_verify", "evidence_risk")


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
