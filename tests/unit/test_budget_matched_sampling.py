"""budget_matched (A21) and configurable agent sampling (A22)."""

import json

import httpx2
from agents import Agent, Runner
from agents.models.interface import ModelTracing
from openai import AsyncOpenAI

from evcarry.harness import agents as agents_module
from evcarry.harness.arms import ARMS, HEADLINE_ARMS, arm_spec
from evcarry.harness.delegate import DelegateState, make_blind_extractor
from evcarry.harness.prompts import prompt_text, subagent_instructions
from evcarry.harness.run_settings import TEMPERATURE, AgentSampling, agent_model_settings, current_sampling, harness_run_config
from evcarry.ledger.ledger import Ledger
from evcarry.protocol.policy import PolicyDecision, blind_redelegation_message, feedback_message
from evcarry.providers.openai_compat import RecordingModel, capture_http_body
from evcarry.providers.scripted import FixtureModel, ScriptStep, _input_text
from evcarry.runs.config import load_experiment


class _InputsModel(FixtureModel):
    def __init__(self, steps):
        super().__init__(steps, agent_name="researcher")
        self.inputs: list[str] = []

    def _next_step(self, input):
        self.inputs.append(_input_text(input))
        return super()._next_step(input)


class _Events:
    def __init__(self):
        self.events: list[dict] = []

    def write(self, event):
        self.events.append(event)


def test_budget_matched_differs_from_evidence_risk_only_in_evidence_and_audit():
    bm, ev, va = arm_spec("budget_matched"), arm_spec("evidence_risk"), arm_spec("vanilla_mad")
    assert bm.delegate and ev.delegate and bm.blind_redelegations == 1 and ev.blind_redelegations == 0
    assert (bm.schema, bm.checks, bm.sampler, bm.require_evidence) == (False, (), None, False)
    # Prompts are vanilla_mad's: no claim schema, no FALLBACK, no citations.
    assert prompt_text("prompts/gym_v1", "budget_matched") == prompt_text("prompts/gym_v1", "vanilla_mad")
    for role in ("researcher", "operator"):
        assert subagent_instructions("prompts/gym_v1", role, "budget_matched") == subagent_instructions(
            "prompts/gym_v1", role, "vanilla_mad")
    assert va.blind_redelegations == 0
    assert HEADLINE_ARMS == ("sa", "vanilla_mad", "evidence_risk", "budget_matched")
    assert "self_verify" in ARMS  # kept for the local grid


def test_blind_message_has_the_evidence_shape_and_no_verdict():
    blind = blind_redelegation_message()
    verdict = feedback_message(PolicyDecision("redelegate", True, ["c1"], ["c1: L1: value 7 is not in any cited result"]))
    assert "Do not repeat actions that already succeeded" in blind and "Do not repeat actions that already succeeded" in verdict
    for word in ("claim", "verify", "L1", "cited", "evidence", "JSON"):
        assert word.lower() not in blind.lower(), word


async def test_every_return_is_sent_back_once_and_the_resend_is_delivered():
    model = _InputsModel([ScriptStep(content="first: 11 sections"), ScriptStep(content="second: 14 sections")])
    agent = Agent(name="researcher", instructions="You are the researcher.", model=model)
    writer = _Events()
    first = await Runner.run(agent, "Count the sections.")
    extractor = make_blind_extractor(agent=agent, hooks=None, writer=writer, arm="budget_matched",
                                     state=DelegateState(), max_turns=5)
    delivered = await extractor(first)
    assert delivered == "second: 14 sections"
    assert "Count the sections." in model.inputs[1] and "first: 11 sections" in model.inputs[1]
    assert blind_redelegation_message() in model.inputs[1]
    kinds = [(e["kind"], e.get("blind")) for e in writer.events if e["type"] == "handoff"]
    assert kinds == [("subagent_output", None), ("redelegation", True), ("subagent_output", None)]
    assert [e["action"] for e in writer.events if e["type"] == "policy"] == ["blind_redelegate"]
    assert not [e for e in writer.events if e["type"] in ("check", "audit", "judge_call")]


async def test_budget_matched_main_agent_wires_the_blind_extractor(monkeypatch):
    from evcarry.harness.hooks import LedgerHooks
    from evcarry.harness.tools import fixture_toolset
    from evcarry.providers.registry import load_models
    from evcarry.providers.usage import PriceTable

    models: dict[str, _InputsModel] = {}

    def scripted(config, script, task, agent_name):
        if agent_name == "main":
            steps = [ScriptStep(tool="researcher", arguments={"input": "count"}, call_id="m1"), ScriptStep(content="done")]
        else:
            steps = [ScriptStep(content="reply-1"), ScriptStep(content="reply-2")]
        models[agent_name] = _InputsModel(steps)
        return models[agent_name]

    monkeypatch.setattr(agents_module, "_model", scripted)
    ledger, writer = Ledger(), _Events()
    toolset = fixture_toolset(None, ledger)
    hooks = LedgerHooks(ledger, writer, PriceTable(0, 0, 0, peak_multiplier=1), toolset.catalog)
    main = agents_module.build_main_agent(
        arm=arm_spec("budget_matched"), task="t", seed=1, prompts_dir="prompts/v1", model_config=load_models()["mock"],
        script=None, toolset=toolset, hooks=hooks, writer=writer, ledger=ledger, checks=(), sampler=None, rate=0.0,
        injector=None, subagent_max_turns=5,
    )
    await Runner.run(main, "Solve the task.", hooks=hooks, max_turns=5, run_config=harness_run_config())
    assert len(models["researcher"].inputs) == 2
    delivered = [e["text"] for e in writer.events if e["type"] == "handoff" and e["kind"] == "delivered"]
    assert delivered == ["reply-2"]


def test_default_sampling_is_temperature_zero_without_seed():
    settings = agent_model_settings()
    assert settings.temperature == 0.0 and settings.extra_args is None
    assert TEMPERATURE == 0.0  # the judge's setting is unchanged


async def test_configured_temperature_and_seed_reach_the_request_and_the_judge_keeps_zero():
    from evcarry.protocol.judge import TEMPERATURE as JUDGE_TEMPERATURE

    body = {"id": "g", "object": "chat.completion", "created": 0, "model": "deepseek/deepseek-v4.1-flash-1017",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
    sent: list[dict] = []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx2.Response(200, json=body)

    http = httpx2.AsyncClient(transport=httpx2.MockTransport(handler), event_hooks={"response": [capture_http_body]})
    model = RecordingModel("deepseek/deepseek-v4.1-flash-1017",
                           AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key="test-not-real", http_client=http),
                           kind="openrouter")
    token = current_sampling.set(AgentSampling(0.6, 2))
    try:
        for settings in (agent_model_settings(), harness_run_config().model_settings):
            await model.get_response("sys", "hi", settings, [], None, [], ModelTracing.DISABLED,
                                     previous_response_id=None, conversation_id=None, prompt=None)
    finally:
        current_sampling.reset(token)
        await http.aclose()
    assert [(r["temperature"], r["seed"]) for r in sent] == [(0.6, 2), (0.6, 2)]
    assert JUDGE_TEMPERATURE == 0.0
    assert agent_model_settings().temperature == 0.0  # reset after the run


def test_headline_r_config_validates():
    config = load_experiment("configs/experiments/headline_R_seed1.yaml")
    assert config.arms == list(HEADLINE_ARMS) and config.seeds == [1]
    assert config.sampling.temperature == 0.6 and config.sampling.send_seed is True
    assert config.budget.cap_usd == 30 and config.require_frozen_prompts is True
    assert config.tasks == "tasks/headline60.txt"
