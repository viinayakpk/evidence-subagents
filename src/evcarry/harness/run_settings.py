"""Settings shared by every ``Runner.run`` and every model call in a harness run (main agent, sub-agents, repairs, re-delegations, judge)."""

from __future__ import annotations

from agents import ModelSettings, RunConfig

# Sampling temperature for every model call in every arm and on every provider: main agent,
# sub-agents and the L3 judge (owner decision, 2026-10-24). It is sent explicitly, so a
# provider default never applies; llm_call, llm_error and judge_call log the value the request
# carried, run_start the value configured.
TEMPERATURE = 0.0


def agent_model_settings() -> ModelSettings:
    """``ModelSettings`` for the main agent and the sub-agents."""
    return ModelSettings(temperature=TEMPERATURE)


def harness_run_config() -> RunConfig:
    """SDK tracing off (the JSONL trace is the record).

    A call to a tool the agent does not have goes back to the model as an error,
    as a tool error does. The SDK default raises ``ModelBehaviorError``, which
    ends the run and, through the run manager, cancels the whole grid; a main
    agent in a delegating arm that tries a workspace tool directly is a normal
    model mistake, not a harness failure.

    The run-level ``model_settings`` overrides each agent's, so a ``Runner.run``
    on an agent built without ``agent_model_settings()`` still sends ``TEMPERATURE``.

    Inside a run (``runs.manager`` sets ``harness.context.current_trimmer``) every
    ``Runner.run`` enforces the configured context window with that run's trimmer.
    """
    from evcarry.harness.context import current_trimmer

    return RunConfig(
        tracing_disabled=True,
        tool_not_found_behavior="return_error_to_model",
        model_settings=agent_model_settings(),
        call_model_input_filter=current_trimmer.get(),
    )
