"""Accept, or reject and re-delegate once, then fall back to the main agent.

The caller invokes the sub-agent at most twice. ``decide`` never asks for a
third attempt.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PolicyDecision:
    action: str
    rejected: bool
    failed_claims: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "rejected": self.rejected,
            "failed_claims": list(self.failed_claims),
            "reasons": list(self.reasons),
        }


def decide(failures: list[tuple[str, str]], *, prior_redelegations: int) -> PolicyDecision:
    """``failures`` is a list of ``(claim_id, reason)``.

    ``prior_redelegations`` is how many re-delegations have already been
    issued for this sub-agent call. Zero means this is the first return.
    """
    if prior_redelegations < 0:
        raise ValueError("prior_redelegations cannot be negative")
    if not failures:
        return PolicyDecision(action="accept", rejected=False)
    claim_ids = list(dict.fromkeys(claim_id for claim_id, _reason in failures))
    reasons = [f"{claim_id}: {reason}" for claim_id, reason in failures]
    if prior_redelegations >= 1:
        return PolicyDecision(
            action="fallback",
            rejected=True,
            failed_claims=claim_ids,
            reasons=reasons,
        )
    return PolicyDecision(
        action="redelegate",
        rejected=True,
        failed_claims=claim_ids,
        reasons=reasons,
    )


def feedback_message(decision: PolicyDecision) -> str:
    lines = [
        "Your previous return was rejected. Fix only the failed claims and return the full structured result again.",
    ]
    lines.extend(f"- {reason}" for reason in decision.reasons)
    return "\n".join(lines)


def fallback_message(subagent_id: str, decision: PolicyDecision) -> str:
    detail = "; ".join(decision.reasons) or "checks failed"
    # The main agent has no workspace tools in any delegating arm, so it re-plans and delegates again
    # (prompts/gym_v1/evidence_system.md says the same); it cannot do the sub-task itself.
    return (
        f"FALLBACK: sub-agent {subagent_id} failed after one re-delegation ({detail}). "
        "Plan this step again and delegate it with a more specific instruction."
    )
