"""Build the main agent and, for delegating arms, researcher and operator tools."""

from __future__ import annotations

from typing import Any

from agents import Agent

from evcarry.harness.arms import ArmSpec
from evcarry.harness.delegate import DelegateState, make_extractor
from evcarry.harness.hooks import LedgerHooks
from evcarry.harness.prompts import prompt_text
from evcarry.harness.tools import ALL_TOOLS, OPERATOR_TOOLS, RESEARCHER_TOOLS, build_tools
from evcarry.inject.injector import FaultInjector
from evcarry.protocol.schema import SubagentReturn
from evcarry.providers.registry import ModelConfig, build_model
from evcarry.providers.scripted import ScriptStep
from evcarry.trace.writer import JsonlWriter


def _steps(script: dict[str, Any] | None, task: str, agent: str) -> list[ScriptStep]:
    if not script:
        return []
    raw = script.get("tasks", {}).get(task, {}).get("agents", {}).get(agent, [])
    return [ScriptStep.from_mapping(item) for item in raw]


def _model(
    config: ModelConfig,
    script: dict[str, Any] | None,
    task: str,
    agent_name: str,
):
    usage = None if script is None else script.get("usage")
    served = None if script is None else script.get("served_model")
    model_config = config
    if served and config.kind == "scripted":
        model_config = ModelConfig(
            name=config.name,
            kind=config.kind,
            model=str(served),
            billable=config.billable,
            prices=config.prices,
        )
    return build_model(
        model_config,
        steps=_steps(script, task, agent_name),
        usage=usage,
        agent_name=agent_name,
    )


def build_main_agent(
    *,
    arm: ArmSpec,
    task: str,
    seed: int,
    prompts_dir: str,
    model_config: ModelConfig,
    script: dict[str, Any] | None,
    caller: Any,
    hooks: LedgerHooks,
    writer: JsonlWriter,
    ledger: Any,
    checks: tuple[str, ...],
    sampler: str | None,
    rate: float,
    injector: FaultInjector | None,
    max_turns: int,
) -> Agent:
    tools = build_tools(ALL_TOOLS, caller)
    if not arm.delegate:
        return Agent(
            name="main",
            instructions=prompt_text(prompts_dir, arm.name),
            tools=tools,
            model=_model(model_config, script, task, "main"),
        )
    state = DelegateState()
    sub_tools = []
    for name, allow, description in (
        ("researcher", RESEARCHER_TOOLS, "Gather evidence with read-only tools and return claims."),
        ("operator", OPERATOR_TOOLS, "Apply writes and return claims about the actions."),
    ):
        sub = Agent(
            name=name,
            instructions=prompt_text(prompts_dir, name),
            tools=build_tools(allow, caller),
            model=_model(model_config, script, task, name),
            output_type=SubagentReturn if arm.schema else None,
        )
        extractor = None
        if arm.schema:
            extractor = make_extractor(
                agent=sub,
                hooks=hooks,
                writer=writer,
                ledger=ledger,
                caller=caller,
                task=task,
                seed=seed,
                arm=arm.name,
                checks=checks,
                sampler=sampler,
                rate=rate,
                require_evidence=arm.require_evidence,
                injector=injector,
                state=state,
                max_turns=max_turns,
            )
        sub_tools.append(
            sub.as_tool(
                tool_name=name,
                tool_description=description,
                custom_output_extractor=extractor,
                hooks=hooks,
            )
        )
    ordered = sorted(sub_tools, key=lambda tool: tool.name)
    return Agent(
        name="main",
        instructions=prompt_text(prompts_dir, arm.name),
        tools=ordered,
        model=_model(model_config, script, task, "main"),
    )
