"""The sub-agent return boundary: repair, re-delegation, and what the sub-agent sees."""

import json
from typing import Any

from agents import Agent, Runner

from evcarry.harness.delegate import DelegateState, make_extractor
from evcarry.ledger.ledger import Ledger
from evcarry.providers.scripted import FixtureModel, ScriptStep, _input_text
from evcarry.protocol.schema import parse_subagent_return

INSTRUCTION = "Count the sections in paper-A and report the number."


class _InputsModel(FixtureModel):
    """A scripted model that also keeps the text of every input it was given."""

    def __init__(self, steps: list[ScriptStep]) -> None:
        super().__init__(steps, agent_name="researcher")
        self.inputs: list[str] = []

    def _next_step(self, input: Any) -> ScriptStep:
        self.inputs.append(_input_text(input))
        return super()._next_step(input)


class _Events:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def write(self, event: dict) -> None:
        self.events.append(event)


def _return(value: int, *, fenced: bool = False) -> str:
    body = json.dumps(
        {
            "subagent_id": "researcher",
            "status": "complete",
            "claims": [
                {
                    "claim_id": "c1",
                    "text": f"paper-A has {value} sections",
                    "kind": "fact",
                    "value": value,
                    "evidence": [{"type": "tool_call", "call_id": "call_1", "span": "14 sections"}],
                }
            ],
            "open_issues": [],
        }
    )
    return f"Here is my return:\n```json\n{body}\n```" if fenced else body


def _ledger() -> Ledger:
    ledger = Ledger()
    ledger.record(call_id="call_1", tool_name="read_file", arguments={"path": "a"}, result_text="14 sections", read_only=True)
    return ledger


def _extractor(agent: Agent, writer: _Events):
    return make_extractor(
        agent=agent, hooks=None, writer=writer, ledger=_ledger(), caller=None, task="t", seed=1, arm="evidence",
        checks=("L0", "L1"), sampler=None, rate=0.0, require_evidence=True, injector=None, state=DelegateState(),
        max_turns=5,
    )


async def test_redelegation_continues_the_sub_agents_conversation() -> None:
    model = _InputsModel([ScriptStep(content=_return(11)), ScriptStep(content=_return(14))])
    agent = Agent(name="researcher", instructions="You are the researcher.", model=model)
    writer = _Events()
    first = await Runner.run(agent, INSTRUCTION)
    accepted = await _extractor(agent, writer)(first)
    assert [event["action"] for event in writer.events if event["type"] == "policy"] == ["redelegate", "accept"]
    assert json.loads(accepted)["claims"][0]["value"] == 14
    # The re-delegated run still has the original instruction and the rejected reply, then the feedback.
    second_input = model.inputs[1]
    assert INSTRUCTION in second_input
    assert "paper-A has 11 sections" in second_input
    assert "could not verify some claims" in second_input


async def test_repair_continues_the_conversation_and_fences_cost_no_repair() -> None:
    model = _InputsModel([ScriptStep(content="I could not format it."), ScriptStep(content=_return(14))])
    agent = Agent(name="researcher", instructions="You are the researcher.", model=model)
    first = await Runner.run(agent, INSTRUCTION)
    await _extractor(agent, _Events())(first)
    assert INSTRUCTION in model.inputs[1]
    assert "not valid JSON for the claim schema" in model.inputs[1]  # the text of repair.md

    fenced = _InputsModel([ScriptStep(content=_return(14, fenced=True))])
    agent = Agent(name="researcher", instructions="You are the researcher.", model=fenced)
    writer = _Events()
    await _extractor(agent, writer)(await Runner.run(agent, INSTRUCTION))
    assert len(fenced.inputs) == 1  # no repair call
    assert [event["action"] for event in writer.events if event["type"] == "policy"] == ["accept"]
    parsed, repairs = parse_subagent_return(_return(14, fenced=True))
    assert repairs == 0 and parsed.claims[0].value == 14


async def test_a_corruption_that_does_not_fit_is_skipped_not_fatal() -> None:
    from types import SimpleNamespace

    from evcarry.inject.injector import FaultInjector
    from evcarry.protocol.schema import SubagentReturn

    payload = SubagentReturn.model_validate_json(_return(14))
    payload.claims[0].text = "paper-A lists its sections"  # nothing for numeric_perturbation to change
    payload.claims[0].value = None
    writer = _Events()
    extractor = make_extractor(
        agent=None, hooks=None, writer=writer, ledger=_ledger(), caller=None, task="t", seed=1, arm="evidence",
        checks=("L0", "L1"), sampler=None, rate=0.0, require_evidence=True,
        injector=FaultInjector(probability=1, types=("numeric_perturbation",)), state=DelegateState(), max_turns=5,
    )
    accepted = await extractor(SimpleNamespace(final_output=payload))
    assert json.loads(accepted)["claims"][0]["text"] == "paper-A lists its sections"
    injection = next(event for event in writer.events if event["type"] == "injection")
    assert injection["applied"] is False and "number" in injection["reason"]
    assert [event["action"] for event in writer.events if event["type"] == "policy"] == ["accept"]


async def test_an_unknown_tool_call_goes_back_to_the_model(tmp_path) -> None:
    """The SDK default raised ModelBehaviorError, which ended the run and cancelled the grid."""
    from evcarry.runs.config import load_experiment
    from evcarry.runs.manager import run_experiment
    from evcarry.trace.writer import read_events

    script = json.loads(open("tests/fixtures/scripts/sa.json", encoding="utf-8").read())
    steps = script["tasks"]["fixture-sections"]["agents"]["main"]
    steps.insert(0, {"tool": "filesystem-read_file", "arguments": {"path": "x"}, "call_id": "call_no_such_tool"})
    override = tmp_path / "sa_unknown_tool.json"
    override.write_text(json.dumps(script), encoding="utf-8")
    config = load_experiment("configs/experiments/mock_e2e.yaml")
    config.arms = ["sa"]
    config.tasks = "tests/fixtures/tasks/one_sections.txt"
    await run_experiment(config, out_dir=tmp_path / "out", script_overrides={"sa": override})
    end = read_events(next((tmp_path / "out").glob("*.jsonl")))[-1]
    assert end["type"] == "run_end" and end["agent_status"] == "completed" and end["passed"] is True


async def test_a_failing_sub_agent_run_is_traced(tmp_path) -> None:
    from evcarry.runs.config import load_experiment
    from evcarry.runs.manager import run_experiment
    from evcarry.trace.writer import read_events

    script = json.loads(open("tests/fixtures/scripts/vanilla_mad.json", encoding="utf-8").read())
    script["tasks"]["fixture-sections"]["agents"]["researcher"] = []  # the nested run has no step: it raises
    override = tmp_path / "vanilla_broken.json"
    override.write_text(json.dumps(script), encoding="utf-8")
    config = load_experiment("configs/experiments/mock_e2e.yaml")
    config.arms = ["vanilla_mad"]
    config.tasks = "tests/fixtures/tasks/one_sections.txt"
    await run_experiment(config, out_dir=tmp_path / "out", script_overrides={"vanilla_mad": override})
    events = read_events(next((tmp_path / "out").glob("*.jsonl")))
    errors = [event for event in events if event["type"] == "subagent_error"]
    assert len(errors) == 1
    assert errors[0]["subagent"] == "researcher" and errors[0]["error_type"] == "UnexpectedModelCall"
    assert events[-1]["type"] == "run_end"


def _typed_return(marker: str) -> str:
    return json.dumps(
        {
            "subagent_id": "researcher",
            "status": "complete",
            "claims": [{"claim_id": "c1", "text": marker, "kind": "fact", "value": marker, "evidence": []}],
            "open_issues": [],
        }
    )


import pytest  # noqa: E402


@pytest.mark.parametrize("arm_name", ["vanilla_mad", "self_verify", "typed"])
async def test_every_delegating_arm_keeps_one_conversation_per_sub_agent(arm_name: str, monkeypatch) -> None:
    """Owner decision 3: a second call to a sub-agent (a retry or a new step) continues its conversation in every arm."""
    from evcarry.harness import agents as agents_module
    from evcarry.harness.arms import arm_spec
    from evcarry.harness.hooks import LedgerHooks
    from evcarry.harness.run_settings import harness_run_config
    from evcarry.harness.tools import fixture_toolset
    from evcarry.providers.registry import load_models
    from evcarry.providers.usage import PriceTable

    spec = arm_spec(arm_name)
    reply = _typed_return if spec.schema else (lambda marker: marker)
    models: dict[str, _InputsModel] = {}

    def scripted(config, script, task, agent_name):
        if agent_name == "main":
            steps = [
                ScriptStep(tool="researcher", arguments={"input": "FIRST: count the sections"}, call_id="m1"),
                ScriptStep(tool="researcher", arguments={"input": "SECOND: check that count again"}, call_id="m2"),
                ScriptStep(content="done"),
            ]
        else:
            steps = [ScriptStep(content=reply("first-reply-14")), ScriptStep(content=reply("second-reply-14"))]
        models[agent_name] = _InputsModel(steps)
        return models[agent_name]

    monkeypatch.setattr(agents_module, "_model", scripted)
    ledger, writer = Ledger(), _Events()
    toolset = fixture_toolset(None, ledger)
    hooks = LedgerHooks(ledger, writer, PriceTable(0, 0, 0, peak_multiplier=1), toolset.catalog)
    main = agents_module.build_main_agent(
        arm=spec, task="t", seed=1, prompts_dir="prompts/v1", model_config=load_models()["mock"], script=None,
        toolset=toolset, hooks=hooks, writer=writer, ledger=ledger, checks=(), sampler=None, rate=0.0,
        injector=None, subagent_max_turns=5,
    )
    await Runner.run(main, "Solve the task.", hooks=hooks, max_turns=5, run_config=harness_run_config())
    first, second = models["researcher"].inputs
    assert "SECOND" not in first
    assert second.count("FIRST: count the sections") == 1  # the history once, not repeated
    assert "first-reply-14" in second and "SECOND: check that count again" in second
    assert models["operator"].inputs == []  # never called: its own conversation is untouched


async def test_redelegation_with_a_session_continues_it_without_repeating_the_history() -> None:
    from evcarry.harness.sessions import SubagentSession

    model = _InputsModel([ScriptStep(content=_return(11)), ScriptStep(content=_return(14))])
    agent = Agent(name="researcher", instructions="You are the researcher.", model=model)
    session = SubagentSession("t|1|researcher")
    writer = _Events()
    first = await Runner.run(agent, INSTRUCTION, session=session)
    extractor = make_extractor(
        agent=agent, hooks=None, writer=writer, ledger=_ledger(), caller=None, task="t", seed=1, arm="evidence",
        checks=("L0", "L1"), sampler=None, rate=0.0, require_evidence=True, injector=None, state=DelegateState(),
        max_turns=5, session=session,
    )
    accepted = await extractor(first)
    assert json.loads(accepted)["claims"][0]["value"] == 14
    second_input = model.inputs[1]
    assert second_input.count(INSTRUCTION) == 1
    assert "paper-A has 11 sections" in second_input and "could not verify some claims" in second_input
    # The session keeps both attempts, so the main agent's next call to this sub-agent sees them.
    kept = _input_text(await session.get_items())
    assert "paper-A has 11 sections" in kept and "paper-A has 14 sections" in kept


async def test_calls_to_one_sub_agent_run_one_at_a_time() -> None:
    import asyncio
    from types import SimpleNamespace

    from evcarry.harness.sessions import serialized

    running, peak, order = 0, 0, []

    async def invoke(_context, arguments):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.01)
        order.append(arguments)
        running -= 1
        return arguments

    tool = serialized(SimpleNamespace(on_invoke_tool=invoke))
    await asyncio.gather(*(tool.on_invoke_tool(None, f"call-{i}") for i in range(4)))
    assert peak == 1 and order == ["call-0", "call-1", "call-2", "call-3"]


async def test_a_reused_provider_call_id_does_not_drop_history_from_later_calls() -> None:
    """The SDK merges tool items with the same call_id; a repeated id used to drop an earlier result for good."""
    from agents import function_tool

    from evcarry.harness.sessions import SubagentSession
    from evcarry.trace.writer import current_writer

    @function_tool
    def lookup(q: str) -> str:
        return f"result for {q}"

    class _RawInputs(_InputsModel):
        def _next_step(self, input: Any) -> ScriptStep:
            self.raw.append(json.dumps(input, default=str))
            return super()._next_step(input)

    model = _RawInputs([
        ScriptStep(tool="lookup", arguments={"q": "first"}, call_id="dup"), ScriptStep(content="r1"),
        ScriptStep(tool="lookup", arguments={"q": "second"}, call_id="dup"), ScriptStep(content="r2"),
        ScriptStep(content="r3"),
    ])
    model.raw = []
    agent = Agent(name="researcher", instructions="You are the researcher.", model=model, tools=[lookup])
    session, writer = SubagentSession("t|1|researcher"), _Events()
    token = current_writer.set(writer)
    try:
        for instruction in ("one", "two", "three"):
            await Runner.run(agent, instruction, session=session)
    finally:
        current_writer.reset(token)
    stored = [item for item in await session.get_items() if item.get("type") in ("function_call", "function_call_output")]
    assert [item["call_id"] for item in stored] == ["dup", "dup", "dupr1", "dupr1"]
    reused = [event for event in writer.events if event["type"] == "subagent_call_id_reused"]
    assert [(event["call_id"], event["stored_as"]) for event in reused] == [("dup", "dupr1")]
    later = model.raw[-1]
    assert "result for first" in later and "result for second" in later
