"""L3 judge: parsing, the scripted provider, trace events, fallback, provider routing, omissions."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from evcarry.analysis.fault_injection import catch_table
from evcarry.paths import repo_root
from evcarry.protocol.judge import (
    JUDGE_PROMPT,
    LexicalJudge,
    LlmJudge,
    build_judge,
    judge_request,
    parse_judge_output,
)
from evcarry.protocol.schema import Claim, ToolCallEvidence
from evcarry.providers.openai_compat import RecordingModel
from evcarry.providers.scripted import KeyedFixtureModel
from evcarry.providers.usage import PriceTable, cost_usd
from evcarry.trace.writer import JsonlWriter, read_events

FLAT = PriceTable(0.15, 0.003, 0.60, peak_multiplier=1)


def _claim(text: str = "Required papers: A, B, C, D", value: object = None) -> Claim:
    return Claim(
        claim_id="c1",
        text=text,
        kind="fact",
        value=value if value is not None else ["A", "B", "C", "D"],
        evidence=[ToolCallEvidence(type="tool_call", call_id="call_search", span="A, B, C, D")],
    )


def _judge(script: dict, writer: JsonlWriter | None = None) -> LlmJudge:
    prompt = (repo_root() / JUDGE_PROMPT).read_text(encoding="utf-8")
    return LlmJudge(KeyedFixtureModel(script), model_name="mock_judge", prompt=prompt, price=FLAT, writer=writer)


def test_parse_judge_output_shapes() -> None:
    plain = parse_judge_output('{"verdict": "incomplete", "missing": ["E"], "reason": "E is left out."}')
    assert plain.verdict == "incomplete" and plain.missing == ["E"]
    fenced = parse_judge_output('Here you go:\n```json\n{"verdict": "Supported", "reason": "ok"}\n```')
    assert fenced.verdict == "supported" and fenced.missing == []
    legacy = parse_judge_output('{"supports": false, "reason": "no"}')
    assert legacy.verdict == "unverifiable"
    with pytest.raises(ValidationError):
        parse_judge_output('{"verdict": "maybe"}')
    with pytest.raises(ValueError):
        parse_judge_output("no json here")


def test_judge_request_fields_and_truncation() -> None:
    request = judge_request(_claim(), "Required papers: A, B, C, D, E", 1000)
    assert "kind: fact\ntext: Required papers: A, B, C, D\nvalue: [\"A\", \"B\", \"C\", \"D\"]" in request
    assert request.endswith("EVIDENCE\nRequired papers: A, B, C, D, E\n")
    long = judge_request(_claim(), "H" * 900 + "T" * 900, 400)
    assert "[1400 characters omitted]" in long
    assert "H" * 300 in long and "T" * 100 in long
    assert "(no evidence text)" in judge_request(_claim(), "  ", 100)


async def test_llm_judge_structured_verdict_and_trace(tmp_path: Path) -> None:
    script = {
        "served_model": "mock-judge-1",
        "rules": [{"match": "text: Required papers: A, B, C, D\n",
                   "reply": {"verdict": "incomplete", "missing": ["E"], "reason": "E is left out."}}],
        "default": {"verdict": "supported", "reason": "ok"},
    }
    writer = JsonlWriter(tmp_path / "run.jsonl")
    judge = _judge(script, writer)
    verdict = await judge.ajudge(_claim(), "Required papers: A, B, C, D, E")
    clean = await judge.ajudge(_claim("Paper A exists", "A"), "Required papers: A, B, C, D, E")
    writer.close()
    assert (verdict.supports, verdict.verdict, verdict.missing, verdict.source) == (False, "incomplete", ["E"], "llm")
    assert clean.supports and clean.verdict == "supported"
    prompt_sha = hashlib.sha256((repo_root() / JUDGE_PROMPT).read_bytes()).hexdigest()
    assert verdict.detail["prompt_sha256"] == prompt_sha
    assert verdict.as_dict()["verdict"] == "incomplete"
    events = read_events(tmp_path / "run.jsonl")
    assert [event["type"] for event in events] == ["judge_call", "judge_call"]
    first = events[0]
    assert first["verdict"] == "incomplete" and first["served_model"] == "mock-judge-1"
    assert first["prompt_sha256"] == prompt_sha and first["error"] is None
    assert '"incomplete"' in first["raw_reply"] and first["tokens"]["output_tokens"] == 20
    expected = cost_usd(judge.model.usage_template, FLAT, datetime.now(timezone.utc))
    assert judge.calls == 2 and judge.cost_usd == pytest.approx(2 * expected)
    # The judge saw the whole evidence list, including the item the claim leaves out.
    assert "EVIDENCE\nRequired papers: A, B, C, D, E" in judge.model.inputs[0]


async def test_llm_judge_falls_back_to_lexical() -> None:
    garbage = _judge({"default": "I think it is fine."})
    verdict = await garbage.ajudge(_claim(text="student-Z passed", value="student-Z"), "no such student")
    assert verdict.source == "lexical_fallback" and verdict.supports is False
    assert "unparseable judge reply" in verdict.detail["error"]

    class Broken:
        async def get_response(self, **_kwargs):
            raise ConnectionError("endpoint down")

    broken = LlmJudge(Broken(), model_name="x", prompt="p", price=FLAT)
    supported = await broken.ajudge(_claim(), "Required papers: A, B, C, D, E")
    assert supported.source == "lexical_fallback" and supported.supports is True
    assert "ConnectionError: endpoint down" in supported.detail["error"]
    assert broken.calls == 1 and broken.cost_usd == 0
    assert broken.fallbacks == 1 and garbage.fallbacks == 1  # run_end reports judge_fallbacks


def test_build_judge_routes_through_the_provider_layer(monkeypatch) -> None:
    assert isinstance(build_judge("lexical"), LexicalJudge)
    assert isinstance(build_judge(None), LexicalJudge)
    scripted = build_judge("mock_judge")
    assert isinstance(scripted, LlmJudge) and isinstance(scripted.model, KeyedFixtureModel)
    with pytest.raises(KeyError):
        build_judge("no_such_model")
    seen: list[dict] = []

    def fake_client(**kwargs):
        seen.append(kwargs)
        return object()

    # No request is made: building the client is intercepted.
    monkeypatch.setattr("evcarry.providers.registry.async_openai_client", fake_client)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "placeholder-not-a-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "placeholder-not-a-key")
    for name, base_url, model in (
        ("deepseek_flash", "https://api.deepseek.com", "deepseek-flash"),
        ("openrouter_pinned", "https://openrouter.ai/api/v1", "deepseek/deepseek-chat"),
        ("local_primary", "http://127.0.0.1:8000/v1", "primitive-ai/Qwen3.8-Flash-Next-NVFP4"),
        ("openrouter_ds_1017", "https://openrouter.ai/api/v1", "deepseek/deepseek-v4.1-flash-1017"),
    ):
        judge = build_judge(name)
        assert isinstance(judge, LlmJudge) and isinstance(judge.model, RecordingModel)
        assert judge.model.configured_model == model
        assert judge.model.role == "judge"  # its failed calls are judge cost in run_end
        assert seen[-1]["base_url"] == base_url


def test_scripted_judge_catches_omissions_on_the_synthetic_set() -> None:
    lexical = {(row["kind"], row["variant"]): row["earliest"] for row in catch_table(include_clean=True)}
    judged = {(row["kind"], row["variant"]): row["earliest"] for row in catch_table(build_judge("mock_judge"), include_clean=True)}
    assert lexical[("omission", "naive")] is None and lexical[("omission", "pointer_consistent")] is None
    assert judged[("omission", "naive")] == "L3" and judged[("omission", "pointer_consistent")] == "L3"
    assert judged[("clean", "none")] is None and judged[("clean_list", "none")] is None
    for key, level in lexical.items():
        if key[0] != "omission":
            assert judged[key] == level, key


def test_judge_reads_the_answer_not_the_reasoning() -> None:
    from openai.types.responses import ResponseOutputMessage, ResponseOutputText, ResponseReasoningItem
    from openai.types.responses.response_reasoning_item import Content

    from evcarry.protocol.judge import _response_text

    reasoning = ResponseReasoningItem(
        id="r", type="reasoning", summary=[],
        content=[Content(type="reasoning_text", text='Draft: {"verdict": "supported", "reason": "maybe"}')],
    )
    answer = ResponseOutputMessage(
        id="m", type="message", role="assistant", status="completed",
        content=[ResponseOutputText(type="output_text", annotations=[],
                                    text='{"verdict": "incomplete", "missing": ["E"], "reason": "E is left out."}')],
    )
    text = _response_text(type("Response", (), {"output": [reasoning, answer]})())
    assert "Draft" not in text
    assert parse_judge_output(text).verdict == "incomplete"
    last = parse_judge_output('Example: {"verdict": "supported"} Answer: {"verdict": "contradicted", "reason": "14, not 11"}')
    assert last.verdict == "contradicted"
    tolerant = parse_judge_output('{"verdict": "incomplete", "missing": [5, "E"], "reason": null}')
    assert tolerant.missing == ["5", "E"] and tolerant.reason == ""
    assert parse_judge_output('{"verdict": "supported", "missing": null}').missing == []
