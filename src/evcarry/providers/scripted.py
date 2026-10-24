"""Scripted chat model for $0 runs.

Each agent instance owns one queue. A re-delegation continues that queue, so a
script that is rejected and run again needs a second sequence with new call ids.
The Agents SDK 0.24 package does not ship ``agents.testing.ScriptedModel``;
this model implements the public ``Model`` interface directly.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from agents.items import ModelResponse
from agents.models.interface import Model, ModelTracing
from agents.usage import Usage
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText
from openai.types.responses.response_usage import InputTokensDetails

from evcarry.providers.identity import ServedIdentity
from evcarry.providers.openai_compat import LlmCallRecord
from evcarry.providers.usage import TokenUsage, token_usage_from_counts


class UnexpectedModelCall(RuntimeError):
    """The scripted queue was empty when the runner asked for another turn."""


@dataclass
class ScriptStep:
    tool: str | None = None
    arguments: dict[str, Any] | None = None
    call_id: str | None = None
    content: str | None = None

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> ScriptStep:
        if "tool" in raw:
            return cls(tool=str(raw["tool"]), arguments=dict(raw.get("arguments") or {}), call_id=str(raw["call_id"]))
        return cls(content=str(raw.get("content", "")))


def usage_from_script(raw: dict[str, Any] | None) -> TokenUsage:
    data = raw or {}
    return token_usage_from_counts(
        input_tokens=int(data.get("input_tokens", 100)),
        cached_tokens=int(data.get("cached_tokens", 80)),
        output_tokens=int(data.get("output_tokens", 20)),
    )


def sdk_usage(tokens: TokenUsage) -> Usage:
    return Usage(
        requests=1,
        input_tokens=tokens.input_tokens,
        output_tokens=tokens.output_tokens,
        total_tokens=tokens.input_tokens + tokens.output_tokens,
        input_tokens_details=InputTokensDetails.model_validate(
            {"cached_tokens": tokens.cache_hit_tokens, "cache_write_tokens": 0}
        ),
    )


class FixtureModel(Model):
    """One scripted agent. ``seen_instructions`` keeps the exact system prompt bytes."""

    def __init__(
        self,
        steps: list[ScriptStep],
        *,
        served_model: str = "mock-scripted",
        usage: TokenUsage | None = None,
        agent_name: str = "agent",
    ) -> None:
        self.steps = list(steps)
        self.index = 0
        self.served_model = served_model
        self.kind = "scripted"
        self.configured_model = served_model
        self.routing = None
        self.usage_template = usage or usage_from_script(None)
        self.agent_name = agent_name
        self.seen_instructions: list[str | None] = []
        self.history: list[LlmCallRecord] = []
        self._pending: deque[LlmCallRecord] = deque()

    def take_record(self) -> LlmCallRecord | None:
        if not self._pending:
            return None
        return self._pending.popleft()

    def _next_step(self, input: str | list[Any]) -> ScriptStep:
        del input
        if self.index >= len(self.steps):
            raise UnexpectedModelCall(
                f"Scripted model for {self.agent_name} has no step left (used {self.index})."
            )
        step = self.steps[self.index]
        self.index += 1
        return step

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[Any],
        model_settings: Any,
        tools: list[Any],
        output_schema: Any,
        handoffs: list[Any],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: Any,
    ) -> ModelResponse:
        del model_settings, tools, output_schema, handoffs, tracing
        del previous_response_id, conversation_id, prompt
        self.seen_instructions.append(system_instructions)
        step = self._next_step(input)
        if step.tool:
            output = [
                ResponseFunctionToolCall(
                    id=f"fc_{step.call_id}",
                    call_id=step.call_id or "",
                    name=step.tool,
                    arguments=json.dumps(step.arguments or {}, sort_keys=True),
                    type="function_call",
                )
            ]
        else:
            output = [
                ResponseOutputMessage(
                    id=f"msg_{self.agent_name}_{self.index}",
                    type="message",
                    role="assistant",
                    status="completed",
                    content=[
                        ResponseOutputText(
                            type="output_text",
                            text=step.content or "",
                            annotations=[],
                        )
                    ],
                )
            ]
        tokens = self.usage_template
        identity = ServedIdentity(self.served_model, "scripted", False, None)
        record = LlmCallRecord(
            configured_model=self.configured_model,
            kind="scripted",
            identity=identity,
            usage=tokens,
            routing=None,
            attempts=1,
            latency_ms=0.0,
            system_instructions=system_instructions,
        )
        self.history.append(record)
        self._pending.append(record)
        return ModelResponse(
            output=output,
            usage=sdk_usage(tokens),
            response_id=None,
        )

    async def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[Any],
        model_settings: Any,
        tools: list[Any],
        output_schema: Any,
        handoffs: list[Any],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: Any,
    ) -> AsyncIterator[Any]:
        del system_instructions, input, model_settings, tools, output_schema, handoffs, tracing
        del previous_response_id, conversation_id, prompt
        raise RuntimeError("FixtureModel only supports get_response. Runner.run does not stream.")
        yield None


def _input_text(input: str | list[Any]) -> str:
    if isinstance(input, str):
        return input
    parts: list[str] = []
    for item in input:
        content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(str(block.get("text", "")) for block in content if isinstance(block, dict))
    return "\n".join(parts)


class KeyedFixtureModel(FixtureModel):
    """Replies by content instead of by position: the first rule whose ``match`` occurs in the input.

    Used for a scripted judge, where which claims get judged depends on the sampler.
    Script format: ``{"served_model": str, "default": <reply>, "rules": [{"match": str, "reply": <reply>}]}``;
    a reply that is not a string is sent as JSON.
    """

    def __init__(self, script: dict[str, Any], *, agent_name: str = "judge") -> None:
        super().__init__([], served_model=str(script.get("served_model") or "mock-judge"), agent_name=agent_name)
        self.rules = [(str(rule["match"]), _reply_text(rule["reply"])) for rule in script.get("rules", [])]
        self.default = _reply_text(script["default"]) if "default" in script else None
        self.inputs: list[str] = []

    def _next_step(self, input: str | list[Any]) -> ScriptStep:
        text = _input_text(input)
        self.inputs.append(text)
        self.index += 1
        for match, reply in self.rules:
            if match in text:
                return ScriptStep(content=reply)
        if self.default is None:
            raise UnexpectedModelCall(f"Keyed script for {self.agent_name} has no rule for this input.")
        return ScriptStep(content=self.default)


def _reply_text(reply: Any) -> str:
    return reply if isinstance(reply, str) else json.dumps(reply, sort_keys=True)
