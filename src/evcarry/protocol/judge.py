"""L3 support judge.

The mock and DeepSeek runs in this milestone use a deterministic lexical judge
so the pipeline costs $0. A model-backed judge is the M6 hook: it calls whatever
model the provider layer was given and expects JSON ``{"supports": bool, "reason": str}``.

Lexical rule (intentional, and weak on omissions):
- Every hyphenated token in the claim text (``student-Z``) must appear in the
  evidence blob.
- A non-numeric string value must appear in the evidence blob.
- Ordinary English words are not scored. Dropping one item from a list whose
  remaining words still appear in the evidence is not caught here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from evcarry.protocol.checks import is_numeric_value, value_in_text
from evcarry.protocol.schema import Claim

_HYPHENATED = re.compile(r"[A-Za-z]+-[A-Za-z0-9]+")


@dataclass
class JudgeVerdict:
    supports: bool
    reason: str
    source: str

    def as_dict(self) -> dict[str, object]:
        return {"supports": self.supports, "reason": self.reason, "source": self.source}


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


class ModelJudge:
    """Call a Model-interface object. Used when a real judge model is configured."""

    source = "model"

    def __init__(self, complete: Any) -> None:
        self._complete = complete

    async def judge(self, claim: Claim, evidence_text: str) -> JudgeVerdict:
        raw = await self._complete(claim, evidence_text)
        text = raw if isinstance(raw, str) else str(raw)
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            return JudgeVerdict(False, f"judge returned non-json: {exc}", self.source)
        supports = bool(data.get("supports"))
        reason = str(data.get("reason") or "")
        return JudgeVerdict(supports, reason, self.source)
