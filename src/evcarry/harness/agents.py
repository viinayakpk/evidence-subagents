"""Build the main agent and, for delegating arms, researcher and operator tools."""

from __future__ import annotations

from typing import Any

from agents import Agent
from agents.agent import StopAtTools
from agents.tool import default_tool_error_function

from evcarry.harness.arms import ArmSpec
from evcarry.harness.delegate import DelegateState, make_extractor
from evcarry.harness.hooks import LedgerHooks
from evcarry.harness.prompts import SUBAGENT_TOOL_DESCRIPTIONS, prompt_text, subagent_instructions
from evcarry.harness.run_settings import harness_run_config
from evcarry.harness.tools import Toolset, build_tools
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
            json_schema_outputs=config.json_schema_outputs,
        )
    return build_model(
        model_config,
        steps=_steps(script, task, agent_name),
        usage=usage,
        agent_name=agent_name,
    )


def _logged_failure(writer: JsonlWriter, name: str):
    """``as_tool`` failure handler: trace the error, then send the SDK's default text to the model.

    Without it an exception in a sub-agent run (provider error, turn limit, a
    return still invalid after its repair, a failed replay) reaches the main
    agent as a generic tool error and leaves nothing in the trace.
    """

    def handle(ctx: Any, error: Exception) -> str:
        writer.write(
            {"type": "subagent_error", "subagent": name, "error_type": type(error).__name__, "error": str(error)[:2000]}
        )
        return default_tool_error_function(ctx, error)

    return handle


def build_main_agent(
    *,
    arm: ArmSpec,
    task: str,
    seed: int,
    prompts_dir: str,
    model_config: ModelConfig,
    script: dict[str, Any] | None,
    toolset: Toolset,
    hooks: LedgerHooks,
    writer: JsonlWriter,
    ledger: Any,
    checks: tuple[str, ...],
    sampler: str | None,
    rate: float,
    injector: FaultInjector | None,
    max_turns: int,
    judge: Any = None,
) -> Agent:
    caller = toolset.caller
    # The gym's claim_done ends the run, as in the gym's own agent loop.
    stop: Any = StopAtTools(stop_at_tool_names=list(toolset.stop_tools)) if toolset.stop_tools else "run_llm_again"
    if not arm.delegate:
        return Agent(
            name="main",
            instructions=prompt_text(prompts_dir, arm.name),
            tools=build_tools(toolset.all_tools, caller, toolset),
            model=_model(model_config, script, task, "main"),
            tool_use_behavior=stop,
        )
    state = DelegateState()
    sub_tools = []
    # Providers without json_schema response_format get the schema in the prompt
    # instead; the extractor parses the string with one repair.
    typed_output = SubagentReturn if arm.schema and model_config.json_schema_outputs else None
    for name, allow in (("researcher", toolset.researcher), ("operator", toolset.operator)):
        sub = Agent(
            name=name,
            instructions=subagent_instructions(prompts_dir, name, arm.name),
            tools=build_tools(allow, caller, toolset),
            model=_model(model_config, script, task, name),
            output_type=typed_output,
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
                judge=judge,
                prompts_dir=prompts_dir,
            )
        sub_tools.append(
            sub.as_tool(
                tool_name=name,
                tool_description=SUBAGENT_TOOL_DESCRIPTIONS[name],
                custom_output_extractor=extractor,
                hooks=hooks,
                # The SDK default is 10 turns per nested run, far below a gym task's budget.
                max_turns=max_turns,
                run_config=harness_run_config(),
                failure_error_function=_logged_failure(writer, name),
            )
        )
    ordered = sorted(sub_tools, key=lambda tool: tool.name) + build_tools(toolset.main_extra, caller, toolset)
    return Agent(
        name="main",
        instructions=prompt_text(prompts_dir, arm.name),
        tools=ordered,
        model=_model(model_config, script, task, "main"),
        tool_use_behavior=stop,
    )
