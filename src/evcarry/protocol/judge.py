"""L3 support judge.

``LlmJudge`` asks a model from ``configs/models.yaml`` (DeepSeek, OpenRouter,
local vLLM, or a scripted fixture) whether a claim's evidence supports it. The
system prompt is the fixed file ``prompts/v1/judge.md``; the reply is one JSON
object with a verdict label, the left-out items, and a reason. Every call is
written to the run trace as a ``judge_call`` event with the prompt hash,
request hash, raw reply, tokens, and cost. If the call or the parse fails, the
lexical judge decides and the verdict is marked ``lexical_fallback``.

``LexicalJudge`` is the cheap deterministic judge (and the fallback):
- Every hyphenated token in the claim text (``student-Z``) must appear in the
  evidence blob.
- A non-numeric string value must appear in the evidence blob.
- Ordinary English words are not scored. Dropping one item from a list whose
  remaining words still appear in the evidence is not caught here.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from agents.model_settings import ModelSettings
from agents.models.interface import ModelTracing
from pydantic import BaseModel, Field, ValidationError, field_validator

from evcarry.paths import resolve_repo_path
from evcarry.protocol.checks import is_numeric_value, value_in_text
from evcarry.protocol.schema import Claim
from evcarry.providers.usage import PriceTable, cost_usd
from evcarry.trace.writer import JsonlWriter, current_writer

_HYPHENATED = re.compile(r"[A-Za-z]+-[A-Za-z0-9]+")

JUDGE_PROMPT = "prompts/v1/judge.md"
VerdictLabel = Literal["supported", "contradicted", "incomplete", "unverifiable"]


@dataclass
class JudgeVerdict:
    supports: bool
    reason: str
    source: str
    verdict: str | None = None
    missing: list[str] = field(default_factory=list)
    detail: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, object]:
        data: dict[str, object] = {"supports": self.supports, "reason": self.reason, "source": self.source}
        if self.verdict is not None:
            data["verdict"] = self.verdict
        if self.missing:
            data["missing"] = list(self.missing)
        if self.detail:
            data["detail"] = dict(self.detail)
        return data


class LexicalJudge:
    source = "lexical"

    def judge(self, claim: Claim, evidence_text: str) -> JudgeVerdict:
        blob = evidence_text.lower()
        missing = [token for token in _HYPHENATED.findall(claim.text) if token.lower() not in blob]
        if missing:
            return JudgeVerdict(False, f"entity tokens not in evidence: {missing}", self.source)
        if (
            claim.value is not None
            and not isinstance(claim.value, (list, tuple))
            and not is_numeric_value(claim.value)
            and not value_in_text(claim.value, evidence_text)
        ):
            return JudgeVerdict(False, "claim value is not in the evidence blob", self.source)
        if isinstance(claim.value, (list, tuple)):
            absent = [item for item in claim.value if str(item) not in evidence_text]
            if absent:
                return JudgeVerdict(False, f"listed values missing from evidence: {absent}", self.source)
        return JudgeVerdict(True, "lexical support holds", self.source)

    async def ajudge(self, claim: Claim, evidence_text: str) -> JudgeVerdict:
        return self.judge(claim, evidence_text)


class JudgeOutput(BaseModel):
    """The structured verdict the judge prompt asks for."""

    verdict: VerdictLabel
    reason: str = ""
    missing: list[str] = Field(default_factory=list)

    @field_validator("missing", mode="before")
    @classmethod
    def _missing_as_strings(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
        return [str(value)]

    @field_validator("reason", mode="before")
    @classmethod
    def _reason_as_string(cls, value: Any) -> str:
        return "" if value is None else str(value)


def parse_judge_output(text: str) -> JudgeOutput:
    """Parse the last JSON object in ``text`` that has a ``verdict`` (or ``supports``) key.

    The last one is the model's answer; drafts and quoted examples come earlier.
    Code fences and surrounding prose are ignored. The older
    ``{"supports": bool, "reason": str}`` shape maps to ``supported`` / ``unverifiable``.
    """
    decoder = json.JSONDecoder()
    data: dict[str, Any] | None = None
    position = text.find("{")
    while position >= 0:
        try:
            candidate, end = decoder.raw_decode(text, position)
        except json.JSONDecodeError:
            position = text.find("{", position + 1)
            continue
        if isinstance(candidate, dict) and ("verdict" in candidate or "supports" in candidate):
            data = candidate
        position = text.find("{", end)
    if data is None:
        raise ValueError("judge reply has no JSON object with a verdict")
    if "verdict" not in data and isinstance(data.get("supports"), bool):
        data = {
            "verdict": "supported" if data["supports"] else "unverifiable",
            "reason": data.get("reason", ""),
        }
    if isinstance(data.get("verdict"), str):
        data["verdict"] = data["verdict"].strip().lower()
    return JudgeOutput.model_validate(data)


def judge_request(claim: Claim, evidence_text: str, max_evidence_chars: int) -> str:
    """The user message: the claim fields, then the evidence (cut in the middle if too long)."""
    evidence = evidence_text
    if len(evidence) > max_evidence_chars:
        head = max_evidence_chars * 3 // 4
        tail = max_evidence_chars - head
        cut = len(evidence) - head - tail
        evidence = f"{evidence[:head]}\n...[{cut} characters omitted]...\n{evidence[-tail:]}"
    value = json.dumps(claim.value, ensure_ascii=False, sort_keys=True, default=str)
    return (
        "CLAIM\n"
        f"kind: {claim.kind}\n"
        f"text: {claim.text}\n"
        f"value: {value}\n"
        "\n"
        "EVIDENCE\n"
        f"{evidence if evidence.strip() else '(no evidence text)'}\n"
    )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _response_text(response: Any) -> str:
    """The assistant message's output text only; reasoning items and blocks are skipped."""
    parts: list[str] = []
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) != "message":
            continue
        for block in getattr(item, "content", None) or []:
            text = getattr(block, "text", None)
            if getattr(block, "type", None) == "output_text" and isinstance(text, str):
                parts.append(text)
    return "\n".join(parts)


class LlmJudge:
    """Model-backed L3 judge. Counts its own calls and dollars for the run."""

    source = "llm"

    def __init__(
        self,
        model: Any,
        *,
        model_name: str,
        prompt: str,
        price: PriceTable,
        writer: JsonlWriter | None = None,
        fallback: LexicalJudge | None = None,
        max_evidence_chars: int = 12000,
    ) -> None:
        self.model = model
        self.model_name = model_name
        self.prompt = prompt
        self.prompt_sha256 = _sha256(prompt)
        self.price = price
        self.writer = writer
        self.fallback = fallback if fallback is not None else LexicalJudge()
        self.max_evidence_chars = max_evidence_chars
        self.calls = 0
        self.cost_usd = 0.0

    async def ajudge(self, claim: Claim, evidence_text: str) -> JudgeVerdict:
        request = judge_request(claim, evidence_text, self.max_evidence_chars)
        started = time.perf_counter()
        raw: str | None = None
        parsed: JudgeOutput | None = None
        error: str | None = None
        try:
            response = await self.model.get_response(
                system_instructions=self.prompt,
                input=request,
                model_settings=ModelSettings(temperature=0.0),
                tools=[],
                output_schema=None,
                handoffs=[],
                tracing=ModelTracing.DISABLED,
                previous_response_id=None,
                conversation_id=None,
                prompt=None,
            )
            raw = _response_text(response)
            parsed = parse_judge_output(raw)
        except (ValidationError, ValueError) as exc:
            error = f"unparseable judge reply: {exc}"[:500]
        except Exception as exc:  # noqa: BLE001 - provider errors fall back to the lexical judge
            error = f"{type(exc).__name__}: {exc}"[:500]
        latency_ms = (time.perf_counter() - started) * 1000
        take = getattr(self.model, "take_record", None)
        record = take() if callable(take) else None
        tokens = record.usage.as_dict() if record is not None else None
        cost = cost_usd(record.usage, self.price, datetime.now(timezone.utc)) if record is not None else 0.0
        self.calls += 1
        self.cost_usd += cost
        detail = {
            "model": self.model_name,
            "served_model": record.identity.served_model if record is not None else None,
            "prompt_sha256": self.prompt_sha256,
            "request_sha256": _sha256(request),
            "latency_ms": round(latency_ms, 1),
            "cost_usd": cost,
        }
        if parsed is not None:
            verdict = JudgeVerdict(
                supports=parsed.verdict == "supported",
                reason=parsed.reason,
                source=self.source,
                verdict=parsed.verdict,
                missing=list(parsed.missing),
                detail=detail,
            )
        else:
            backup = self.fallback.judge(claim, evidence_text)
            verdict = JudgeVerdict(
                supports=backup.supports,
                reason=f"{backup.reason} (LLM judge failed: {error})",
                source="lexical_fallback",
                verdict=None,
                detail={**detail, "error": error},
            )
        writer = self.writer or current_writer.get()
        if writer is not None:
            writer.write(
                {
                    "type": "judge_call",
                    "claim_id": claim.claim_id,
                    "judge_model": self.model_name,
                    "served_model": detail["served_model"],
                    "prompt_sha256": self.prompt_sha256,
                    "request_sha256": detail["request_sha256"],
                    "request_chars": len(request),
                    "raw_reply": (raw or "")[:4000],
                    "verdict": verdict.verdict,
                    "supports": verdict.supports,
                    "missing": verdict.missing,
                    "reason": verdict.reason,
                    "source": verdict.source,
                    "error": error,
                    "tokens": tokens,
                    "cost_usd": cost,
                    "latency_ms": detail["latency_ms"],
                }
            )
        return verdict


async def run_judge(judge: Any, claim: Claim, evidence_text: str) -> JudgeVerdict:
    """Call ``ajudge`` when the judge has one, else the synchronous ``judge``."""
    ajudge = getattr(judge, "ajudge", None)
    if ajudge is not None:
        return await ajudge(claim, evidence_text)
    return judge.judge(claim, evidence_text)


def build_judge(
    name: str | None,
    *,
    writer: JsonlWriter | None = None,
    models: dict[str, Any] | None = None,
    prompt_file: str = JUDGE_PROMPT,
) -> LexicalJudge | LlmJudge:
    """``lexical`` (or empty) gives the lexical judge; any other name is a ``configs/models.yaml`` key."""
    if not name or name == "lexical":
        return LexicalJudge()
    from evcarry.providers.registry import build_model, load_models
    from evcarry.providers.scripted import KeyedFixtureModel

    catalog = models if models is not None else load_models()
    if name not in catalog:
        raise KeyError(f"unknown judge model {name}; use 'lexical' or a key from configs/models.yaml")
    config = catalog[name]
    if config.kind == "scripted":
        if not config.script:
            raise ValueError(f"scripted judge {name} needs a `script` file")
        script = json.loads(resolve_repo_path(config.script).read_text(encoding="utf-8"))
        model: Any = KeyedFixtureModel(script, agent_name="judge")
    else:
        model = build_model(config, agent_name="judge")
    prompt = resolve_repo_path(prompt_file).read_text(encoding="utf-8")
    return LlmJudge(model, model_name=name, prompt=prompt, price=config.price_table(), writer=writer)
