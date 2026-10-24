"""Accept, or re-delegate the failed claims once; then keep what was verified.

The caller invokes the sub-agent at most twice. ``decide`` never asks for a
third attempt.

Per-claim acceptance (pre-registered amendment, 2026-10-25, before the headline;
docs/plan/EXPERIMENT_PLAN.md section 2): the unit of acceptance is the claim, not
the return. A return whose claims all pass is accepted. Otherwise the sub-agent is
asked once to re-check only the failed claims. After that retry, the main agent
gets every claim that passed (from either attempt) and a list of the claims that
did not, as ``REJECTED`` lines in ``open_issues`` (``partial_accept``); only when no
claim passed does it get the ``FALLBACK`` message. Before, one failing claim out of
a median of 8 sent the whole return back, and a second failure discarded every
verified claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PolicyDecision:
    action: str
    rejected: bool
    failed_claims: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    # Claims passed on to the main agent (partial_accept), from this return and the one before the retry.
    accepted_claims: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "rejected": self.rejected,
            "failed_claims": list(self.failed_claims),
            "reasons": list(self.reasons),
            "accepted_claims": list(self.accepted_claims),
        }


def decide(
    failures: list[tuple[str, str]],
    *,
    prior_redelegations: int,
    verified: list[str] | tuple[str, ...] = (),
) -> PolicyDecision:
    """``failures`` is a list of ``(claim_id, reason)``; ``verified`` the claim ids that passed so far.

    ``prior_redelegations`` is how many re-delegations have already been
    issued for this sub-agent call. Zero means this is the first return.
    """
    if prior_redelegations < 0:
        raise ValueError("prior_redelegations cannot be negative")
    if not failures:
        return PolicyDecision(action="accept", rejected=False, accepted_claims=list(verified))
    claim_ids = list(dict.fromkeys(claim_id for claim_id, _reason in failures))
    reasons = [f"{claim_id}: {reason}" for claim_id, reason in failures]
    if prior_redelegations >= 1:
        return PolicyDecision(
            action="partial_accept" if verified else "fallback",
            rejected=True,
            failed_claims=claim_ids,
            reasons=reasons,
            accepted_claims=list(verified),
        )
    return PolicyDecision(
        action="redelegate",
        rejected=True,
        failed_claims=claim_ids,
        reasons=reasons,
        accepted_claims=list(verified),
    )


def feedback_message(decision: PolicyDecision) -> str:
    lines = [
        "The harness could not verify some claims in your previous return. The other claims were accepted; do not send them again.",
        "Do not repeat actions that already succeeded (sending, writing, creating or deleting): if a failed claim is about such an action, check its result with a read instead and cite that call.",
        "Re-check only the claims below. Reply with one JSON object in the same format that contains only these claims, corrected and re-cited, under the same claim_id; move any claim you cannot support to open_issues.",
    ]
    lines.extend(f"- {reason}" for reason in decision.reasons)
    return "\n".join(lines)


def rejected_note(claim_id: str, reason: str) -> str:
    """The open_issues line that tells the main agent a claim was dropped."""
    return f"REJECTED {claim_id} (not verified; do not rely on it): {reason}"


def fallback_message(subagent_id: str, decision: PolicyDecision) -> str:
    detail = "; ".join(decision.reasons) or "checks failed"
    # The main agent has no workspace tools in any delegating arm, so it re-plans and delegates again
    # (prompts/gym_v1/evidence_system.md says the same); it cannot do the sub-task itself.
    return (
        f"FALLBACK: none of sub-agent {subagent_id}'s claims could be verified after one re-delegation ({detail}). "
        "Plan this step again and delegate it with a more specific instruction. "
        "Tell the sub-agent which actions already succeeded, so it does not repeat them."
    )
