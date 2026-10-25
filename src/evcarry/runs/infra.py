"""Infrastructure errors and the block re-run rule (C1, C2; plan amendment A19, 2026-10-25).

An infrastructure error is a failure of the provider route, not of the agent: a 429 that
outlasted the retry window, a dropped connection or timeout, a 5xx, or a reply from an
endpoint outside the pinned set. Before A19 the same error ended an SA run (discarded and
re-run from scratch) but reached a delegating arm's main agent as a sub-agent tool error
(the run went on and was graded). From A19 it ends the run in every arm, wherever it was
raised (main agent, sub-agent, repair, re-delegation, L3 judge): ``run_end`` has
``status: infra_error`` and ``passed: null``.

Block rule: the unit is the (task, seed) block, all arms together. A block in which any arm
ended ``infra_error``, or left a trace without ``run_end`` (a stopped process), is re-run whole:
every arm's trace is set aside (``*.jsonl.aborted-<n>``, its spend still counted) and every
arm runs again, up to ``max_block_attempts`` attempts. After that the block is dropped:
an arm without ``run_end`` gets an ``infra_error`` ``run_end``, and the index marks every arm
of the block ``block_infra: true`` so analysis drops the block in every arm.
"""

from __future__ import annotations

from typing import Any

from openai import APIConnectionError, APIStatusError, InternalServerError, RateLimitError

from evcarry.providers.pin import ProviderMismatch

INFRA_STATUS = "infra_error"
# Stop starting new runs after this many infrastructure errors in one invocation (runs in flight finish).
INFRA_STOP_AFTER = 3


class InfrastructureError(RuntimeError):
    """A provider-route failure; ends the run with status ``infra_error`` in every arm."""


def is_infrastructure_error(error: BaseException | None) -> bool:
    """Whether ``error``, or an exception it was raised from, is an infrastructure error."""
    seen: set[int] = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, (InfrastructureError, RateLimitError, APIConnectionError, InternalServerError, ProviderMismatch)):
            return True
        if isinstance(error, APIStatusError) and int(getattr(error, "status_code", 0) or 0) >= 500:
            return True
        error = error.__cause__ or error.__context__
    return False


def root_infrastructure_error(error: BaseException) -> BaseException:
    """The innermost infrastructure error in ``error``'s chain (for the trace), else ``error``."""
    found = error
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if is_infrastructure_error(current) and not isinstance(current, InfrastructureError):
            found = current
        current = current.__cause__ or current.__context__
    return found


def infra_outcome(error: BaseException) -> dict[str, Any]:
    root = root_infrastructure_error(error)
    return {
        "agent_status": INFRA_STATUS,
        "status": INFRA_STATUS,
        "passed": None,
        "eval_ran": False,
        "error_type": type(root).__name__,
        "error": str(root)[:1000],
    }
