"""Sub-agent return boundary: one audit, one re-delegation, then fallback.

The first nested run is the ``as_tool`` invocation. This extractor reviews that
result. On ``redelegate`` it calls ``Runner.run`` once more on the same agent,
which continues that agent's scripted queue. There is no third attempt.

Strict ``output_type`` rejects invalid JSON inside the SDK before this
extractor runs. The one-repair guarantee for a string return lives here and in
``parse_subagent_return``.
"""

from __future__ import annotations

import json
from typing import Any

from agents import RunConfig, Runner

from evcarry.harness.hooks import LedgerHooks
from evcarry.inject.injector import FaultInjector
from evcarry.protocol.audit import audit_return
from evcarry.protocol.judge import LexicalJudge
from evcarry.protocol.policy import decide, fallback_message, feedback_message
from evcarry.protocol.schema import FailedReturn, SubagentReturn, parse_subagent_return
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


async def _coerce(run_result: Any, agent: Any, hooks: LedgerHooks, max_turns: int) -> SubagentReturn:
    output = run_result.final_output
    if isinstance(output, SubagentReturn):
        return output
    text = _as_text(output)
    try:
        parsed, _repairs = parse_subagent_return(text)
        return parsed
    except FailedReturn as first:
        repair_prompt = (
            "The previous structured return did not match the schema. "
            f"Return one JSON object only.\n{first}"
        )
        repaired = await Runner.run(
            agent,
            repair_prompt,
            hooks=hooks,
            max_turns=max_turns,
            run_config=RunConfig(tracing_disabled=True),
        )
        parsed, _repairs = parse_subagent_return(_as_text(repaired.final_output))
        return parsed


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
    judge: LexicalJudge | None = None,
):
    async def _review(run_result: Any, prior: int) -> str:
        payload = await _coerce(run_result, agent, hooks, max_turns)
        index = state.return_index
        state.return_index += 1
        decision_inject = None
        if injector is not None:
            decision_inject = injector.decide(
                task=task,
                seed=seed,
                return_index=index,
                claims=payload.claims,
            )
            if decision_inject is not None:
                payload = injector.apply(payload, decision_inject, ledger)
                writer.write({"type": "injection", "arm": arm, **decision_inject.as_dict()})
        report = await audit_return(
            payload,
            ledger=ledger,
            caller=caller,
            checks=checks,
            sampler=sampler,
            rate=rate,
            seed=seed,
            judge=judge or LexicalJudge(),
            require_evidence=require_evidence,
        )
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
        policy = decide(report.failures, prior_redelegations=prior)
        writer.write({"type": "policy", "arm": arm, "return_index": index, "prior": prior, **policy.as_dict()})
        if policy.action == "redelegate":
            second = await Runner.run(
                agent,
                feedback_message(policy),
                hooks=hooks,
                max_turns=max_turns,
                run_config=RunConfig(tracing_disabled=True),
            )
            return await _review(second, prior=1)
        if policy.action == "fallback":
            return fallback_message(payload.subagent_id, policy)
        return payload.model_dump_json()

    async def extractor(run_result: Any) -> str:
        return await _review(run_result, prior=0)

    return extractor
