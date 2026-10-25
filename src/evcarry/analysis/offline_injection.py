"""Offline fault injection over logged evidence-arm claims (fix 4, 2026-10-25).

Real claims from finished evidence-arm runs are checked again with L0-L2 against their own ledgers,
with no model or tool call: a replayed read is answered with the logged result, a write read-back with
the read-back text the judge request logged. Claims that pass L0-L2 unmodified get one fault each:

- ``numeric_change``: one number in the value (and text) is moved by 7%, at least 1.
- ``fabricated_row``: a made-up row shaped like the existing ones is added (a list gains one, a scalar is
  replaced); a multi-word name, like any multi-word value, is left to L3 by the L1 value rule.
- ``wrong_pointer``: the first tool-call pointer is moved to another logged call. ``naive`` keeps the
  span; ``consistent`` takes a span from the new call's result, so only the value can give it away.
- ``false_done``: the claim becomes ``action_done`` for a write that never happened. ``naive`` keeps
  the original (read) pointers; ``consistent`` cites a real logged write with a span from its result.

Unmodified claims that fail L0-L2 give the rejection rate per level; since some of those claims are
really wrong, it is an upper bound on false rejects. L3 (the paid judge) is off by default: pass an
``l3`` callable to include it. Limits: offline, a replayed read cannot drift, so L2 here adds the write
read-back and replay value checks only; ``derived`` claims skip the value check (left to L3).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from evcarry.ledger.store import read_ledger
from evcarry.protocol.checks import CheckContext, checkable_items, logged_text, run_l0_l1
from evcarry.protocol.replayer import replay_claim
from evcarry.protocol.schema import Claim, ToolCallEvidence, parse_subagent_return

LEVELS = ("L0", "L1", "L2", "L3")
FAULTS = (
    ("numeric_change", "naive"),
    ("fabricated_row", "naive"),
    ("wrong_pointer", "naive"),
    ("wrong_pointer", "consistent"),
    ("false_done", "naive"),
    ("false_done", "consistent"),
)
FAKE_ENTITY = "Jordan Quillfeather"
FAKE_RECIPIENT = "audit.fabricated@example.com"
_READBACK = re.compile(r"--- read-back of (\S+) now \(([^)]*)\):\n(.*?)(?=\n--- call |\n--- read-back of |\Z)", re.DOTALL)
_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])")

L3Check = Callable[[Claim, Any], Awaitable[bool]]


@dataclass
class Item:
    """One logged claim with what an offline check needs."""

    run: str
    return_index: int
    claim: Claim
    ledger: Any
    readbacks: dict[str, str] = field(default_factory=dict)


class OfflineCaller:
    """Answers a replayed read with its logged result, a read-back with the logged read-back text."""

    def __init__(self, ledger: Any, readbacks: dict[str, str]) -> None:
        self.reads = {(e.tool_name, json.dumps(e.arguments, sort_keys=True, default=str)): e.result_text
                      for e in ledger.entries.values() if e.read_only}
        self.readbacks = {}
        for e in ledger.entries.values():
            if e.readback_tool and e.call_id in readbacks:
                key = (e.readback_tool, json.dumps(e.readback_arguments or {}, sort_keys=True, default=str))
                self.readbacks[key] = readbacks[e.call_id]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        key = (name, json.dumps(arguments, sort_keys=True, default=str))
        if key in self.reads:
            return self.reads[key]
        if key in self.readbacks:
            return self.readbacks[key]
        raise LookupError(f"no offline answer for {name}")


async def first_failing_level(claim: Claim, ledger: Any, readbacks: dict[str, str],
                              l3: L3Check | None = None) -> tuple[str | None, bool]:
    """The earliest level that rejects ``claim`` (None if all pass) and whether L2 ran offline."""
    results = run_l0_l1(claim, CheckContext(ledger=ledger))
    for level in ("L0", "L1"):
        if any(r.level == level and not r.ok for r in results):
            return level, True
    l2_ran = True
    try:
        replay = await replay_claim(claim, ledger, OfflineCaller(ledger, readbacks))
        if not replay.ok:
            return "L2", True
    except LookupError:
        l2_ran = False
    if l3 is not None and not await l3(claim, ledger):
        return "L3", l2_ran
    return None, l2_ran


def _events(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def load_items(run_dir: Path) -> list[Item]:
    """Every claim of finished evidence-arm runs in ``run_dir``.

    Full returns come from ``handoff`` ``subagent_output`` events (dev runs); traces without them (the
    pilot) give the claims their judge calls logged, which are the sampled claims only.
    """
    items: list[Item] = []
    for path in sorted(run_dir.glob("*__evidence*.jsonl")):
        events = _events(path)
        start = next((e for e in events if e.get("type") == "run_start"), None)
        if start is None or not any(e.get("type") == "run_end" for e in events):
            continue
        ledger_path = run_dir / "ledgers" / f"{start['task']}__{start['arm']}__{start['seed']}.json"
        if not ledger_path.is_file():
            continue
        ledger = read_ledger(ledger_path)
        run = f"{run_dir.name}/{start['task']}"
        readbacks: dict[str, str] = {}
        for e in events:
            if e.get("type") == "judge_call":
                for call_id, _tool, text in _READBACK.findall(e.get("request") or ""):
                    readbacks.setdefault(call_id, text)
        outputs = [e for e in events if e.get("type") == "handoff" and e.get("kind") == "subagent_output"]
        found = False
        for index, e in enumerate(outputs):
            try:
                parsed, _ = parse_subagent_return(e.get("text") or "")
            except Exception:  # noqa: BLE001 - an unparseable return has no claims to inject into
                continue
            for claim in parsed.claims:
                items.append(Item(run, index, claim, ledger, readbacks))
                found = True
        if not found:
            seen = set()
            for e in events:
                if e.get("type") == "judge_call" and e.get("claim"):
                    key = (int(e.get("return_index", -1)), str(e.get("claim_id")))
                    if key not in seen:
                        seen.add(key)
                        items.append(Item(run, key[0], Claim.model_validate(e["claim"]), ledger, readbacks))
    return items


def _bump(number: str) -> str:
    value = float(number)
    new = value + max(1.0, abs(value) * 0.07)
    if "." in number:
        return f"{new:.{len(number.split('.')[1])}f}"
    return str(int(round(new)))


def _numeric_change(claim: Claim) -> Claim | None:
    value = claim.value
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        new_value: Any = type(value)(float(_bump(repr(value))))
        old, new = str(value), str(new_value)
    elif isinstance(value, str) and _NUMBER.search(value):
        old = _NUMBER.search(value).group(0)
        new = _bump(old)
        new_value = value.replace(old, new, 1)
    elif isinstance(value, list):
        for i, element in enumerate(value):
            changed = _numeric_change(claim.model_copy(update={"value": element}))
            if changed is not None:
                new_value = list(value)
                new_value[i] = changed.value
                old, new = str(element), str(changed.value)
                break
        else:
            return None
    else:
        return None
    return claim.model_copy(update={"value": new_value, "text": claim.text.replace(old, new, 1)})


def _fake_like(element: Any) -> Any:
    """A made-up row shaped like ``element``: a number, a one-token or multi-word name, or a dict."""
    if isinstance(element, bool):
        return not element
    if isinstance(element, (int, float)):
        return 987654.321
    if isinstance(element, dict):
        return {key: _fake_like(item) for key, item in element.items()}
    if isinstance(element, str) and " " in element.strip():
        return FAKE_ENTITY
    return FAKE_ENTITY.split()[1]


def _fabricated_row(claim: Claim) -> Claim | None:
    """A list gains one made-up row shaped like its first row; a scalar is replaced by one."""
    value = claim.value
    if isinstance(value, list) and value:
        new_value: Any = [*value, _fake_like(value[0])]
    elif isinstance(value, (str, int, float)) and not isinstance(value, bool):
        new_value = _fake_like(value)
    else:
        return None
    return claim.model_copy(update={"value": new_value, "text": f"{claim.text} Also: {FAKE_ENTITY}."})


def _text_of(entry: Any) -> str:
    return logged_text(entry)


def _span_from(text: str) -> str | None:
    text = text.strip()
    for line in text.splitlines():
        line = line.strip()
        if len(line) >= 12:
            return line[:60]
    return text[:60] or None


def _wrong_pointer(claim: Claim, ledger: Any, variant: str) -> Claim | None:
    pointers = [i for i, ev in enumerate(claim.evidence) if isinstance(ev, ToolCallEvidence)]
    if not pointers:
        return None
    cited = {claim.evidence[i].call_id for i in pointers}
    for entry in sorted(ledger.entries.values(), key=lambda e: e.call_id):
        text = _text_of(entry)
        if entry.call_id in cited or not text.strip() or (claim.value is not None and str(claim.value) in text):
            continue
        evidence = list(claim.evidence)
        old = evidence[pointers[0]]
        span = old.span if variant == "naive" else _span_from(entry.result_text)
        if variant == "consistent" and not span:
            continue
        evidence[pointers[0]] = old.model_copy(update={"call_id": entry.call_id, "span": span})
        evidence = [evidence[pointers[0]]] if variant == "consistent" else evidence
        return claim.model_copy(update={"evidence": evidence})
    return None


def _false_done(claim: Claim, ledger: Any, variant: str) -> Claim | None:
    text = f"Sent the summary email to {FAKE_RECIPIENT}."
    if variant == "naive":
        if not any(isinstance(ev, ToolCallEvidence) for ev in claim.evidence):
            return None
        return claim.model_copy(update={"kind": "action_done", "text": text, "value": FAKE_RECIPIENT})
    writes = [e for e in sorted(ledger.entries.values(), key=lambda e: e.call_id) if not e.read_only and e.result_text.strip()]
    for entry in writes:
        span = _span_from(entry.result_text)
        if span:
            evidence = [ToolCallEvidence(type="tool_call", call_id=entry.call_id, span=span)]
            return claim.model_copy(update={"kind": "action_done", "text": text, "value": FAKE_RECIPIENT,
                                            "evidence": evidence})
    return None


def inject(claim: Claim, ledger: Any, fault: str, variant: str) -> Claim | None:
    """``claim`` with one fault, or None when the fault does not apply to it."""
    if fault == "numeric_change":
        return _numeric_change(claim)
    if fault == "fabricated_row":
        return _fabricated_row(claim)
    if fault == "wrong_pointer":
        return _wrong_pointer(claim, ledger, variant)
    if fault == "false_done":
        return _false_done(claim, ledger, variant)
    raise ValueError(fault)


def _rates(counts: dict[str, int], n: int, levels: tuple[str, ...]) -> dict[str, Any]:
    row: dict[str, Any] = {"n": n}
    total = 0
    for level in levels:
        total += counts.get(level, 0)
        row[f"first_{level}"] = counts.get(level, 0)
        row[f"cum_{level}"] = round(total / n, 3) if n else None
    return row


async def study(items: list[Item], l3: L3Check | None = None) -> dict[str, Any]:
    """Rejection of unmodified claims and catch of each fault, per level (cumulative)."""
    levels = LEVELS if l3 is not None else LEVELS[:3]
    base_counts: dict[str, int] = defaultdict(int)
    by_kind: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    l2_offline_gaps = 0
    clean: list[Item] = []
    for item in items:
        level, l2_ran = await first_failing_level(item.claim, item.ledger, item.readbacks, l3)
        l2_offline_gaps += 0 if l2_ran else 1
        by_kind[item.claim.kind]["n"] += 1
        if level is None:
            clean.append(item)
        else:
            base_counts[level] += 1
            by_kind[item.claim.kind][level] += 1
    faults: dict[str, Any] = {}
    for fault, variant in FAULTS:
        counts: dict[str, int] = defaultdict(int)
        n = 0
        for item in clean:
            bad = inject(item.claim, item.ledger, fault, variant)
            if bad is None:
                continue
            n += 1
            level, _ = await first_failing_level(bad, item.ledger, item.readbacks, l3)
            counts[level or "missed"] += 1
            if level is None and (bad.kind == "derived" or not checkable_items(bad.value)):
                counts["missed_value_left_to_L3"] += 1
        row = _rates(counts, n, levels)
        row["missed"] = counts.get("missed", 0)
        row["missed_value_left_to_L3"] = counts.get("missed_value_left_to_L3", 0)
        faults[f"{fault}/{variant}"] = row
    return {
        "claims": len(items),
        "runs": sorted({item.run for item in items}),
        "unmodified": _rates(base_counts, len(items), levels),
        "unmodified_by_kind": {k: dict(v) for k, v in sorted(by_kind.items())},
        "clean_claims": len(clean),
        "l2_not_checkable_offline": l2_offline_gaps,
        "levels": list(levels),
        "faults": faults,
    }
