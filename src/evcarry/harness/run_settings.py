"""The ``RunConfig`` for every ``Runner.run`` in a harness run (main agent, sub-agents, repairs, re-delegations)."""

from __future__ import annotations

from agents import RunConfig


def harness_run_config() -> RunConfig:
    """SDK tracing off (the JSONL trace is the record).

    A call to a tool the agent does not have goes back to the model as an error,
    as a tool error does. The SDK default raises ``ModelBehaviorError``, which
    ends the run and, through the run manager, cancels the whole grid; a main
    agent in a delegating arm that tries a workspace tool directly is a normal
    model mistake, not a harness failure.
    """
    return RunConfig(tracing_disabled=True, tool_not_found_behavior="return_error_to_model")
