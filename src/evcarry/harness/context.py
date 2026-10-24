"""The configured context window, enforced on every model call of every agent (P3-2, 2026-10-25).

``limits.context`` (131,072 tokens in the pilot and headline configs) was never
enforced: persistent sub-agent sessions grew to 329k input tokens. ``ContextTrimmer``
is the SDK's ``call_model_input_filter`` for every ``Runner.run`` of a run (main
agent, sub-agents, repairs, re-delegations), the same in every arm:

- Before each call it estimates the input's tokens: instructions, input items and
  tool schemas in characters, times the agent's tokens-per-character ratio from its
  previous call (``calibrate``, fed the provider's reported input tokens), or 1/3
  before the first.
- If the estimate is above the budget (the context minus ``OUTPUT_RESERVE_TOKENS``),
  the oldest tool outputs are replaced by a fixed stub, oldest first, until the
  estimate is at most ``TRIM_TO`` of the budget. The newest output is never stubbed.
- Trimming is in blocks and only moves forward: an output stubbed once stays stubbed,
  with the same bytes, on every later call of that agent. So the input up to the first
  stub, and the stubs themselves, stay a stable prefix the provider can cache; the
  boundary moves rarely, each time by a block.
- A stub names the output's ledger call id when the output showed one, so a sub-agent
  can read it again with ``view_overlong_output``; the ledger and the evidence checks
  always have the full text.

Each trim is a ``context_trim`` trace event. A 64k window is a config change
(``limits.context: 65536``); it removes about 39% of input tokens but only about 7%
of cost at DeepInfra prices, because the trimmed tokens are mostly cache hits.
"""

from __future__ import annotations

import json
import re
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from agents.run_config import CallModelData, ModelInputData

from evcarry.trace.writer import current_writer

OUTPUT_RESERVE_TOKENS = 8192
TRIM_TO = 0.75
DEFAULT_TOKENS_PER_CHAR = 1 / 3
_CALL_ID = re.compile(r"^\[call_id: ([^\]\n]+)\]")

current_trimmer: ContextVar[ContextTrimmer | None] = ContextVar("evcarry_context_trimmer", default=None)


@dataclass
class _AgentState:
    name: str
    stubbed: int = 0  # how many of the oldest tool outputs are stubbed
    tokens_per_char: float = DEFAULT_TOKENS_PER_CHAR
    last_chars: int = 0
    tool_chars: int | None = None


def _output_text(item: dict[str, Any]) -> str:
    output = item.get("output", "")
    if isinstance(output, str):
        return output
    return json.dumps(output, ensure_ascii=False, default=str)


def _chars(value: Any) -> int:
    if isinstance(value, str):
        return len(value)
    return len(json.dumps(value, ensure_ascii=False, default=str))


class ContextTrimmer:
    """One per run; keeps a trimming state per agent (main, researcher, operator)."""

    def __init__(self, context_tokens: int) -> None:
        self.context_tokens = int(context_tokens)
        self.budget_tokens = max(1024, self.context_tokens - OUTPUT_RESERVE_TOKENS)
        self._states: dict[int, _AgentState] = {}

    def _state(self, agent: Any) -> _AgentState:
        key = id(agent)
        if key not in self._states:
            self._states[key] = _AgentState(name=str(getattr(agent, "name", "")))
        return self._states[key]

    def _tool_chars(self, state: _AgentState, agent: Any) -> int:
        if state.tool_chars is None:
            total = 0
            for tool in getattr(agent, "tools", None) or []:
                total += len(str(getattr(tool, "name", ""))) + len(str(getattr(tool, "description", "") or ""))
                total += _chars(getattr(tool, "params_json_schema", None) or {})
            state.tool_chars = total
        return state.tool_chars

    def stub(self, item: dict[str, Any], tool_name: str) -> dict[str, Any]:
        match = _CALL_ID.match(_output_text(item))
        where = (
            f' Its full text is in the ledger: call view_overlong_output with call_id "{match.group(1)}" and page=0.'
            if match
            else ""
        )
        trimmed = dict(item)
        trimmed["output"] = (
            f"[Output of this {tool_name or 'tool'} call removed from the context "
            f"to stay within {self.context_tokens} tokens.{where}]"
        )
        return trimmed

    def __call__(self, data: CallModelData[Any]) -> ModelInputData:
        agent = data.agent
        state = self._state(agent)
        items = list(data.model_data.input)
        instructions = data.model_data.instructions
        names = {
            item.get("call_id"): str(item.get("name") or "")
            for item in items
            if isinstance(item, dict) and item.get("type") == "function_call"
        }
        outputs = [
            index for index, item in enumerate(items) if isinstance(item, dict) and item.get("type") == "function_call_output"
        ]
        sizes = [_chars(item) for item in items]
        stubs = {index: self.stub(items[index], names.get(items[index].get("call_id"), "")) for index in outputs}
        saving = {index: sizes[index] - _chars(stubs[index]) for index in outputs}
        fixed = (len(instructions) if isinstance(instructions, str) else 0) + self._tool_chars(state, agent)
        stubbed = min(state.stubbed, max(0, len(outputs) - 1))

        def estimate(count: int) -> int:
            chars = fixed + sum(sizes) - sum(saving[index] for index in outputs[:count])
            return int(chars * state.tokens_per_char)

        before = estimate(stubbed)
        if before > self.budget_tokens:
            target = self.budget_tokens * TRIM_TO
            count = stubbed
            while estimate(count) > target and count < len(outputs) - 1:
                count += 1
            if count > stubbed:
                writer = current_writer.get()
                if writer is not None:
                    writer.write(
                        {
                            "type": "context_trim",
                            "agent": state.name,
                            "budget_tokens": self.budget_tokens,
                            "estimated_tokens_before": before,
                            "estimated_tokens_after": estimate(count),
                            "stubbed_outputs": count,
                            "newly_stubbed": count - stubbed,
                            "outputs": len(outputs),
                            "tokens_per_char": round(state.tokens_per_char, 4),
                        }
                    )
            stubbed = count
        state.stubbed = max(state.stubbed, stubbed)
        trimmed = list(items)
        for index in outputs[:stubbed]:
            trimmed[index] = stubs[index]
        state.last_chars = fixed + sum(sizes) - sum(saving[index] for index in outputs[:stubbed])
        return ModelInputData(input=trimmed, instructions=instructions)

    def calibrate(self, agent: Any, input_tokens: int | None) -> None:
        """Learn the agent's tokens per character from the provider's count for the call just made."""
        state = self._states.get(id(agent))
        if state is None or not input_tokens or state.last_chars <= 0:
            return
        state.tokens_per_char = input_tokens / state.last_chars
