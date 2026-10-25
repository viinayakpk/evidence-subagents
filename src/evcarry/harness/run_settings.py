"""Settings shared by every ``Runner.run`` and every model call in a harness run (main agent, sub-agents, repairs, re-delegations, judge)."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass

from agents import ModelSettings, RunConfig

# Sampling temperature of the L3 judge, and of the agents unless a run sets ``current_sampling``
# (owner decision, 2026-10-24; agents amended by A22, 2026-10-25). It is sent explicitly, so a
# provider default never applies; llm_call, llm_error and judge_call log the value the request
# carried, run_start the values configured. The judge always uses this constant (A12 keeps it fixed).
TEMPERATURE = 0.0


@dataclass(frozen=True)
class AgentSampling:
    """What every agent call of one run sends (main agent, sub-agents, repairs, re-delegations).

    ``seed`` is sent as the request's ``seed`` parameter when it is not None. OpenRouter lists ``seed``
    for deepseek/deepseek-v4.1-flash-1017 and for every endpoint of the pinned set (A22).
    """

    temperature: float = TEMPERATURE
    seed: int | None = None


# Set by the run manager for the duration of one run (``runs.config.SamplingConfig``).
current_sampling: ContextVar[AgentSampling | None] = ContextVar("evcarry_agent_sampling", default=None)


def agent_model_settings() -> ModelSettings:
    """``ModelSettings`` for the main agent and the sub-agents: the run's sampling, else temperature 0."""
    sampling = current_sampling.get() or AgentSampling()
    extra = {"seed": sampling.seed} if sampling.seed is not None else None
    return ModelSettings(temperature=sampling.temperature, extra_args=extra)


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
