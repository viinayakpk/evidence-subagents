"""Freeze the text a prompt set sends to models, and refuse runs after it changes.

The manifest is every model-visible text the harness builds from a prompt set,
rendered the way runs render it: each arm's main system prompt, each
sub-agent's instructions with the arm's return format, the sub-agent tool
descriptions, the repair, re-delegation and FALLBACK messages, the L3 judge
prompt, the local tools' descriptions and schemas, and the tool-result header.
Hashing rendered text (not files) also catches a change made in code.

``freeze`` writes ``FROZEN.json`` into the prompt directory. ``status`` compares
the current manifest with it. The run manager refuses a billable run when a
frozen set has changed, and any run with ``require_frozen_prompts: true``
unless the set is frozen and unchanged.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from evcarry.harness.arms import ARMS
from evcarry.harness.prompts import (
    SUBAGENT_TOOL_DESCRIPTIONS,
    prompt_text,
    repair_prompt,
    subagent_instructions,
    subagent_tool_block,
)
from evcarry.paths import resolve_repo_path

FREEZE_FILE = "FROZEN.json"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def rendered_prompts(prompts_dir: str | Path) -> dict[str, str]:
    from evcarry.env.gym_tools import CLAIM_DONE_META, PYTHON_EXECUTE_META
    from evcarry.harness.tools import OVERLONG_META, ToolMeta, display_text
    from evcarry.protocol.judge import JUDGE_PROMPT
    from evcarry.protocol.policy import (
        PolicyDecision,
        blind_redelegation_message,
        fallback_message,
        feedback_message,
        rejected_note,
    )

    texts: dict[str, str] = {}
    for name, spec in sorted(ARMS.items()):
        texts[f"arm/{name}/main"] = prompt_text(prompts_dir, name)
        if spec.delegate:
            for role in ("researcher", "operator"):
                texts[f"arm/{name}/{role}"] = subagent_instructions(prompts_dir, role, name)
    for role, description in sorted(SUBAGENT_TOOL_DESCRIPTIONS.items()):
        texts[f"tool/{role}"] = description
    for meta in (CLAIM_DONE_META, PYTHON_EXECUTE_META, OVERLONG_META):
        texts[f"tool/{meta.name}"] = json.dumps({"description": meta.description, "schema": meta.schema}, sort_keys=True)
    decision = PolicyDecision(action="redelegate", rejected=True, failed_claims=["<claim>"], reasons=["<claim>: <reason>"])
    texts["message/repair"] = repair_prompt(prompts_dir, "<parse error>")
    texts["message/redelegate"] = feedback_message(decision)
    texts["message/blind_redelegate"] = blind_redelegation_message()
    texts["message/fallback"] = fallback_message("<subagent>", decision)
    texts["message/rejected_claim"] = rejected_note("<claim>", "<reason>")
    example = {
        name: ToolMeta(name, "", {}, name != "srv-write", server=None if name == "python_execute" else "srv")
        for name in ("srv-read", "srv-write", "python_execute")
    }
    texts["message/subagent_tools"] = subagent_tool_block(("srv-read",), tuple(example), example)
    texts["message/tool_result"] = display_text("x" * 40, "<call_id>", show_call_id=True, max_chars=20)
    texts["judge"] = resolve_repo_path(JUDGE_PROMPT).read_text(encoding="utf-8")
    return texts


def manifest_sha256(texts: dict[str, str]) -> str:
    return _sha256(json.dumps(texts, sort_keys=True, ensure_ascii=False))


@dataclass
class FreezeStatus:
    state: str  # "unfrozen" | "match" | "changed"
    manifest_sha256: str
    frozen_sha256: str | None = None
    changed: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "manifest_sha256": self.manifest_sha256,
            "frozen_sha256": self.frozen_sha256,
            "changed": list(self.changed),
        }


def freeze_path(prompts_dir: str | Path) -> Path:
    return resolve_repo_path(prompts_dir) / FREEZE_FILE


def status(prompts_dir: str | Path) -> FreezeStatus:
    texts = rendered_prompts(prompts_dir)
    current = manifest_sha256(texts)
    path = freeze_path(prompts_dir)
    if not path.is_file():
        return FreezeStatus("unfrozen", current)
    frozen = json.loads(path.read_text(encoding="utf-8"))
    recorded = frozen.get("texts") or {}
    changed = sorted(
        key for key in set(recorded) | set(texts) if recorded.get(key) != (_sha256(texts[key]) if key in texts else None)
    )
    state = "match" if frozen.get("manifest_sha256") == current else "changed"
    return FreezeStatus(state, current, frozen.get("manifest_sha256"), changed)


def freeze(prompts_dir: str | Path, *, git_commit: str = "unknown") -> Path:
    texts = rendered_prompts(prompts_dir)
    path = freeze_path(prompts_dir)
    record = {
        "manifest_sha256": manifest_sha256(texts),
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit,
        "texts": {key: _sha256(value) for key, value in sorted(texts.items())},
    }
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
