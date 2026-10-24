"""Sub-agent return boundary: one audit, one re-delegation of the failed claims, then what was verified.

The first nested run is the ``as_tool`` invocation. This extractor reviews that
result. On ``redelegate`` it calls ``Runner.run`` once more on the same agent,
continuing the sub-agent's conversation with the feedback (and, for the scripted
model, its queue), and reviews that reply. There is no third attempt. In a run
the conversation is the sub-agent's persistent session (``harness.sessions``),
the same one the main agent's later calls continue.

Per-claim acceptance (``protocol.policy``, 2026-10-25): the retry is asked for the
failed claims only, the claims that passed the first time are kept, and after the
retry the main agent gets every verified claim plus ``REJECTED`` lines in
``open_issues`` for the rest. Only a retry in which nothing is verified ends in
``FALLBACK``.

Strict ``output_type`` rejects invalid JSON inside the SDK before this
extractor runs. The one-repair guarantee for a string return lives here and in
``parse_subagent_return``. Every message the harness sends or receives here is a
``handoff`` trace event (``harness.handoffs``).
"""

from __future__ import annotations

import json
from typing import Any

from agents import Runner

from evcarry.harness.handoffs import write_handoff
from evcarry.harness.hooks import LedgerHooks
from evcarry.harness.prompts import repair_prompt
from evcarry.harness.run_settings import harness_run_config
from evcarry.inject.corruptions import CorruptionError
from evcarry.inject.injector import FaultInjector
from evcarry.protocol.audit import audit_return
from evcarry.protocol.judge import LexicalJudge, current_audit
from evcarry.protocol.policy import decide, fallback_message, feedback_message, rejected_note
from evcarry.protocol.schema import Claim, FailedReturn, SubagentReturn, parse_subagent_return
from evcarry.trace.writer import JsonlWriter


class DelegateState:
    def __init__(self) -> None:
        self.return_index = 0


def _as_text(output: object) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, SubagentReturn):
        return output.model_dump_json()
    return json.dumps(output, default=str)


def _continued(run_result: Any, message: str, session: Any = None) -> str | list[Any]:
    """The sub-agent's own conversation (instruction, tool calls, reply) plus one new user message.

    A repair or re-delegation that sends only ``message`` without the conversation starts a
    fresh run in which the sub-agent no longer knows what it was asked to do. With a session,
    ``Runner.run`` puts the stored conversation in front of ``message`` itself.
    """
    if session is not None:
        return message
    return [*run_result.to_input_list(), {"role": "user", "content": message}]


async def _coerce(
    run_result: Any,
    agent: Any,
    hooks: LedgerHooks,
    max_turns: int,
    prompts_dir: str | None = None,
    session: Any = None,
    writer: Any = None,
) -> tuple[SubagentReturn, Any]:
    """The parsed return and the run it came from (the repair run when one was needed)."""
    output = run_result.final_output
    write_handoff(writer, "subagent_output", _as_text(output), subagent=getattr(agent, "name", None))
    if isinstance(output, SubagentReturn):
        return output, run_result
    text = _as_text(output)
    try:
        parsed, _repairs = parse_subagent_return(text)
        return parsed, run_result
    except FailedReturn as first:
        message = repair_prompt(prompts_dir, str(first))
        write_handoff(writer, "repair", message, subagent=getattr(agent, "name", None))
        repaired = await Runner.run(
            agent,
            _continued(run_result, message, session),
            hooks=hooks,
            max_turns=max_turns,
            run_config=harness_run_config(),
            session=session,
        )
        write_handoff(writer, "subagent_output", _as_text(repaired.final_output), subagent=getattr(agent, "name", None), repair=True)
        parsed, _repairs = parse_subagent_return(_as_text(repaired.final_output))
        return parsed, repaired


def _delivered(
    payload: SubagentReturn,
    kept: dict[str, Claim],
    rejected: dict[str, str],
    earlier_issues: list[str],
) -> str:
    """The verified claims of both attempts, with a REJECTED line in open_issues for each claim left out."""
    issues = list(dict.fromkeys([*earlier_issues, *payload.open_issues]))
    issues += [rejected_note(claim_id, reason) for claim_id, reason in rejected.items()]
    merged = SubagentReturn(
        subagent_id=payload.subagent_id, status=payload.status, claims=list(kept.values()), open_issues=issues
    )
    return merged.model_dump_json()


def make_extractor(
    *,
    agent: Any,
    hooks: LedgerHooks,
    writer: JsonlWriter,
    ledger: Any,
    caller: Any,
    task: str,
    seed: int,
    arm: str,
    checks: tuple[str, ...],
    sampler: str | None,
    rate: float,
    require_evidence: bool,
    injector: FaultInjector | None,
    state: DelegateState,
    max_turns: int,
    judge: Any = None,
    prompts_dir: str | None = None,
    session: Any = None,
):
    async def _review(
        run_result: Any,
        prior: int,
        kept: dict[str, Claim] | None = None,
        first_failures: dict[str, str] | None = None,
        earlier_issues: list[str] | None = None,
    ) -> str:
        payload, run_result = await _coerce(run_result, agent, hooks, max_turns, prompts_dir, session, writer)
        index = state.return_index
        state.return_index += 1
        decision_inject = None
        injected: dict[str, object] | None = None
        if injector is not None:
            decision_inject = injector.decide(
                task=task,
                seed=seed,
                return_index=index,
                claims=payload.claims,
            )
            if decision_inject is not None:
                # A corruption that does not fit the chosen claim (no number to perturb, no
                # second call to point at) is skipped and logged; raising here threw away
                # the sub-agent's whole return.
                try:
                    payload = injector.apply(payload, decision_inject, ledger)
                except CorruptionError as exc:
                    writer.write(
                        {"type": "injection", "arm": arm, **decision_inject.as_dict(), "applied": False, "reason": str(exc)}
                    )
                else:
                    writer.write({"type": "injection", "arm": arm, **decision_inject.as_dict(), "applied": True})
                    injected = {"claim_id": decision_inject.claim_id, "kind": decision_inject.kind, "variant": decision_inject.variant}
        # Labels this return's judge_call events in the trace; nothing here reaches the judge.
        audit_token = current_audit.set(
            {"arm": arm, "return_index": index, "subagent_id": payload.subagent_id, "injected": injected}
        )
        try:
            report = await audit_return(
                payload,
                ledger=ledger,
                caller=caller,
                checks=checks,
                sampler=sampler,
                rate=rate,
                # One random-audit draw per return. The bare run seed gave every return the
                # same draw; the "audit|" prefix keeps it apart from the injector's stream.
                seed=f"audit|{task}|{seed}|{index}",
                judge=judge or LexicalJudge(),
                require_evidence=require_evidence,
            )
        finally:
            current_audit.reset(audit_token)
        writer.write(
            {
                "type": "subagent_return",
                "arm": arm,
                "subagent_id": payload.subagent_id,
                "return_index": index,
                "status": payload.status,
                "claim_ids": [claim.claim_id for claim in payload.claims],
            }
        )
        writer.write({"type": "check", "arm": arm, "return_index": index, "ok": not report.failures, **report.as_dict()})
        writer.write({"type": "audit", "arm": arm, "return_index": index, "sampled": report.sampled})
        failed = report.failed_claim_ids()
        # Verified claims: those kept from before the retry, then this return's, a retried claim replacing its first version.
        verified = dict(kept or {})
        for claim in payload.claims:
            if claim.claim_id not in failed:
                verified[claim.claim_id] = claim
        # A claim verified before the retry stays verified: a resend that fails does not undo it (the sub-agent
        # was told not to send it again). A claim that failed before and is missing from the retry was withdrawn.
        failures = [(claim_id, reason) for claim_id, reason in report.failures if claim_id not in (kept or {})]
        if prior:
            returned = {claim.claim_id for claim in payload.claims}
            for claim_id, reason in (first_failures or {}).items():
                if claim_id not in returned and claim_id not in verified:
                    failures.append((claim_id, f"withdrawn after re-check ({reason})"))
        policy = decide(failures, prior_redelegations=prior, verified=list(verified))
        writer.write({"type": "policy", "arm": arm, "return_index": index, "prior": prior, **policy.as_dict()})
        if policy.action == "redelegate":
            message = feedback_message(policy)
            write_handoff(writer, "redelegation", message, subagent=getattr(agent, "name", None), return_index=index)
            second = await Runner.run(
                agent,
                _continued(run_result, message, session),
                hooks=hooks,
                max_turns=max_turns,
                run_config=harness_run_config(),
                session=session,
            )
            first = {}
            for claim_id, reason in failures:
                first.setdefault(claim_id, reason)
            return await _review(second, prior=1, kept=verified, first_failures=first, earlier_issues=list(payload.open_issues))
        if policy.action == "fallback":
            message = fallback_message(payload.subagent_id, policy)
            write_handoff(writer, "fallback", message, subagent=getattr(agent, "name", None), return_index=index)
            return message
        if prior == 0:
            return payload.model_dump_json()  # every claim passed the first time
        rejected: dict[str, str] = {}
        for claim_id, reason in failures:
            rejected.setdefault(claim_id, reason)
        return _delivered(payload, verified, rejected, earlier_issues or [])

    async def extractor(run_result: Any) -> str:
        return await _review(run_result, prior=0)

    return extractor
