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
    assert "Your previous return was rejected" in second_input


async def test_repair_continues_the_conversation_and_fences_cost_no_repair() -> None:
    model = _InputsModel([ScriptStep(content="I could not format it."), ScriptStep(content=_return(14))])
    agent = Agent(name="researcher", instructions="You are the researcher.", model=model)
    first = await Runner.run(agent, INSTRUCTION)
    await _extractor(agent, _Events())(first)
    assert INSTRUCTION in model.inputs[1]
    assert "did not match the schema" in model.inputs[1]

    fenced = _InputsModel([ScriptStep(content=_return(14, fenced=True))])
    agent = Agent(name="researcher", instructions="You are the researcher.", model=fenced)
    writer = _Events()
    await _extractor(agent, writer)(await Runner.run(agent, INSTRUCTION))
    assert len(fenced.inputs) == 1  # no repair call
    assert [event["action"] for event in writer.events if event["type"] == "policy"] == ["accept"]
    parsed, repairs = parse_subagent_return(_return(14, fenced=True))
    assert repairs == 0 and parsed.claims[0].value == 14
