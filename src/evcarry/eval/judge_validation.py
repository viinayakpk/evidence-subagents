"""Validate the L3 judge against hand labels, and a second judge against the first.

1. ``collect`` reads every LLM ``judge_call`` in run traces: the claim, the request the
   judge saw (claim and evidence), its verdict, and whether the claim was an injected fault.
2. ``synthesize_faults`` makes faults from clean judged claims when the traces have too
   few injected ones (the headline injects none): the claim's value perturbed, a list
   item dropped, or the evidence swapped for another claim's. These have no verdict
   until ``judge_items`` runs a judge on them.
3. ``stratified_sample`` draws the hand-label set: faults, clean claims the judge
   called supported, clean claims it rejected (40/30/30 by default; a short stratum
   gives its places to the others). Clean rows carry a weight (stratum size / sample
   size) so clean-claim metrics estimate the whole population.
4. ``write_export`` writes ``items.csv`` (blind: no verdict, no origin, shuffled) for the
   labeller and ``key.csv`` (verdicts, origin, weights, the full request) for scoring.
5. ``judge_items`` asks a judge model for every row (or only rows without a verdict) and
   stores its verdict in a named column; at M6 the local model re-judges the same rows.
6. ``score`` compares each judge column with the hand labels: agreement, Cohen's kappa,
   and precision/recall/F1 for flagging an unsupported claim (the judge's job in the
   audit), on clean claims (weighted) and on faults; and judge-vs-judge agreement.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evcarry.trace.writer import read_events

LABELS = ("supported", "contradicted", "incomplete", "unverifiable")
EVIDENCE_MARK = "\nEVIDENCE\n"
DEFAULT_ALLOCATION = {"fault": 40, "clean_supported": 30, "clean_unsupported": 30}
ITEM_FIELDS = ["item_id", "claim_kind", "claim_text", "claim_value", "evidence", "human_label", "human_notes"]
KEY_FIELDS = [
    "item_id", "stratum", "weight", "origin", "fault_kind", "run", "task", "arm", "seed", "return_index",
    "claim_id", "claim_json", "judge_model", "judge_verdict", "request",
]


@dataclass
class Item:
    item_id: str
    origin: str  # "clean" | "injected" | "synthetic"
    claim: dict[str, Any]
    evidence: str
    request: str
    run: str = ""
    task: str = ""
    arm: str = ""
    seed: str = ""
    return_index: str = ""
    fault_kind: str = ""
    judge_model: str = ""
    judge_verdict: str = ""
    stratum: str = ""
    weight: float = 1.0
    extra: dict[str, str] = field(default_factory=dict)


def _item_id(*parts: object) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:12]


def split_request(request: str) -> tuple[str, str]:
    head, _, evidence = request.partition(EVIDENCE_MARK)
    return head, evidence.rstrip("\n")


def build_request(claim: dict[str, Any], evidence: str) -> str:
    """The judge's user message for ``claim`` and ``evidence`` (same layout as ``protocol.judge.judge_request``)."""
    value = json.dumps(claim.get("value"), ensure_ascii=False, sort_keys=True, default=str)
    return (
        "CLAIM\n"
        f"kind: {claim.get('kind')}\n"
        f"text: {claim.get('text')}\n"
        f"value: {value}\n"
        "\n"
        "EVIDENCE\n"
        f"{evidence if evidence.strip() else '(no evidence text)'}\n"
    )


def collect(run_dirs: list[Path]) -> list[Item]:
    """LLM-judged claims from finished and unfinished traces; one item per distinct request."""
    items: dict[str, Item] = {}
    for run_dir in run_dirs:
        for path in sorted(Path(run_dir).glob("*.jsonl")):
            start: dict[str, Any] = {}
            for event in read_events(path):
                if event.get("type") == "run_start":
                    start = event
                if event.get("type") != "judge_call" or event.get("source") != "llm" or not event.get("request"):
                    continue
                injected = event.get("injected") or {}
                is_fault = bool(injected) and injected.get("claim_id") == event.get("claim_id")
                key = event.get("request_sha256") or _item_id(event["request"])
                if key in items:
                    continue
                _head, evidence = split_request(event["request"])
                items[key] = Item(
                    item_id=_item_id("trace", key),
                    origin="injected" if is_fault else "clean",
                    claim=event.get("claim") or {},
                    evidence=evidence,
                    request=event["request"],
                    run=path.name,
                    task=str(start.get("task", "")),
                    arm=str(event.get("arm") or start.get("arm", "")),
                    seed=str(start.get("seed", "")),
                    return_index=str(event.get("return_index", "")),
                    fault_kind=str(injected.get("kind", "")) if is_fault else "",
                    judge_model=str(event.get("judge_model", "")),
                    judge_verdict=str(event.get("verdict") or ""),
                )
    return sorted(items.values(), key=lambda item: item.item_id)


def _perturb_value(claim: dict[str, Any]) -> dict[str, Any] | None:
    value = claim.get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, int):
        new: int | float = value + max(1, round(abs(value) * 0.1))
    else:
        decimals = len(repr(value).split(".")[1]) if "." in repr(value) else 0
        new = round(value * 1.1 if value else 1.0, max(decimals, 1))
    # Replace the number as a whole token, not inside an identifier such as "t13".
    text, replaced = re.subn(rf"(?<![\w.]){re.escape(str(value))}(?![\w]|\.\d)", str(new), str(claim.get("text", "")), count=1)
    changed = dict(claim, value=new)
    if replaced:
        changed["text"] = text
    return changed


def _drop_item(claim: dict[str, Any]) -> dict[str, Any] | None:
    value = claim.get("value")
    if not isinstance(value, list) or len(value) < 2:
        return None
    dropped = value[-1]
    text = str(claim.get("text", ""))
    for pattern in (f", and {dropped}", f" and {dropped}", f", {dropped}", str(dropped)):
        if pattern in text:
            text = text.replace(pattern, "", 1)
            break
    return dict(claim, value=value[:-1], text=text)


def synthesize_faults(items: list[Item], count: int, seed: int) -> list[Item]:
    """``count`` faults made from distinct clean items, cycling value perturbation, list omission, evidence swap."""
    clean = [item for item in items if item.origin == "clean"]
    if len(clean) < 2 or count <= 0:
        return []
    order = list(clean)
    random.Random(f"faults|{seed}").shuffle(order)
    kinds = ("value_perturbation", "list_omission", "evidence_swap")
    faults: list[Item] = []
    for position, source in enumerate(order):
        if len(faults) >= count:
            break
        for offset in range(len(kinds)):
            kind = kinds[(position + offset) % len(kinds)]
            claim, evidence = source.claim, source.evidence
            if kind == "value_perturbation":
                claim = _perturb_value(source.claim)
            elif kind == "list_omission":
                claim = _drop_item(source.claim)
            else:
                donor = order[(position + 1) % len(order)]
                if donor.evidence == source.evidence:
                    claim = None
                else:
                    evidence = donor.evidence
            if claim is None:
                continue
            request = build_request(claim, evidence)
            faults.append(
                Item(
                    item_id=_item_id("synthetic", source.item_id, kind),
                    origin="synthetic",
                    claim=claim,
                    evidence=evidence,
                    request=request,
                    run=source.run,
                    task=source.task,
                    arm=source.arm,
                    seed=source.seed,
                    return_index=source.return_index,
                    fault_kind=kind,
                )
            )
            break
    return faults


def _stratum(item: Item) -> str:
    if item.origin != "clean":
        return "fault"
    return "clean_supported" if item.judge_verdict == "supported" else "clean_unsupported"


def stratified_sample(items: list[Item], *, n: int = 100, seed: int = 0, allocation: dict[str, int] | None = None) -> list[Item]:
    """Seeded draw per stratum; a stratum with too few items gives its places to the next ones in order."""
    quota = dict(allocation or DEFAULT_ALLOCATION)
    scale = n / sum(quota.values())
    quota = {name: round(size * scale) for name, size in quota.items()}
    groups: dict[str, list[Item]] = {name: [] for name in quota}
    for item in items:
        groups.setdefault(_stratum(item), []).append(item)
    order = list(quota)
    spare = 0
    for name in order:
        if len(groups[name]) < quota[name]:
            spare += quota[name] - len(groups[name])
            quota[name] = len(groups[name])
    for name in order:
        room = len(groups[name]) - quota[name]
        take = min(room, spare)
        quota[name] += take
        spare -= take
    chosen: list[Item] = []
    for name in order:
        pool = sorted(groups[name], key=lambda item: item.item_id)
        picked = random.Random(f"sample|{seed}|{name}").sample(pool, quota[name]) if quota[name] else []
        for item in picked:
            item.stratum = name
            # Clean rows estimate the population of clean judged claims; faults are reported on their own.
            item.weight = len(pool) / quota[name] if name != "fault" else 1.0
        chosen.extend(picked)
    return chosen


def write_export(sample: list[Item], out_dir: Path, *, seed: int = 0) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    shuffled = sorted(sample, key=lambda item: item.item_id)
    random.Random(f"order|{seed}").shuffle(shuffled)
    items_path, key_path = out_dir / "items.csv", out_dir / "key.csv"
    with items_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ITEM_FIELDS)
        writer.writeheader()
        for item in shuffled:
            writer.writerow(
                {
                    "item_id": item.item_id,
                    "claim_kind": item.claim.get("kind", ""),
                    "claim_text": item.claim.get("text", ""),
                    "claim_value": json.dumps(item.claim.get("value"), ensure_ascii=False, default=str),
                    "evidence": item.evidence,
                    "human_label": "",
                    "human_notes": "",
                }
            )
    write_key(sample, key_path)
    return items_path, key_path


def write_key(sample: list[Item], path: Path) -> None:
    extra = sorted({name for item in sample for name in item.extra})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*KEY_FIELDS, *extra])
        writer.writeheader()
        for item in sorted(sample, key=lambda row: row.item_id):
            writer.writerow(
                {
                    "item_id": item.item_id, "stratum": item.stratum, "weight": f"{item.weight:.6g}",
                    "origin": item.origin, "fault_kind": item.fault_kind, "run": item.run, "task": item.task,
                    "arm": item.arm, "seed": item.seed, "return_index": item.return_index,
                    "claim_id": item.claim.get("claim_id", ""), "claim_json": json.dumps(item.claim, ensure_ascii=False),
                    "judge_model": item.judge_model, "judge_verdict": item.judge_verdict, "request": item.request,
                    **{name: item.extra.get(name, "") for name in extra},
                }
            )


def read_key(path: Path) -> list[Item]:
    known = set(KEY_FIELDS)
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            claim = json.loads(row["claim_json"]) if row.get("claim_json") else {}
            rows.append(
                Item(
                    item_id=row["item_id"], origin=row["origin"], claim=claim,
                    evidence=split_request(row["request"])[1], request=row["request"], run=row["run"],
                    task=row["task"], arm=row["arm"], seed=row["seed"], return_index=row["return_index"],
                    fault_kind=row["fault_kind"], judge_model=row["judge_model"], judge_verdict=row["judge_verdict"],
                    stratum=row["stratum"], weight=float(row["weight"] or 1),
                    extra={key: value for key, value in row.items() if key not in known},
                )
            )
    return rows


async def judge_items(items: list[Item], judge: Any, *, column: str, only_missing: bool = False) -> int:
    """Ask ``judge`` (an ``LlmJudge``) about each row; store the verdict as ``judge_verdict`` or ``<column>_verdict``."""
    from evcarry.protocol.schema import Claim

    judge.max_evidence_chars = 10**9  # the stored evidence is already what the run's judge saw
    asked = 0
    for item in items:
        target = "judge_verdict" if column == "judge" else f"{column}_verdict"
        current = item.judge_verdict if column == "judge" else item.extra.get(target, "")
        if only_missing and current:
            continue
        claim = Claim.model_validate({"evidence": [], **item.claim, "claim_id": item.claim.get("claim_id") or item.item_id})
        verdict = await judge.ajudge(claim, item.evidence)
        label = verdict.verdict or ("supported" if verdict.supports else "unverifiable")
        if verdict.source != "llm":
            label = f"error:{verdict.source}"
        if column == "judge":
            item.judge_verdict, item.judge_model = label, judge.model_name
        else:
            item.extra[target] = label
            item.extra[f"{column}_model"] = judge.model_name
        asked += 1
    return asked


def read_labels(path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            label = (row.get("human_label") or "").strip().lower()
            if label:
                if label not in LABELS:
                    raise ValueError(f"item {row['item_id']}: human_label {label!r} is not one of {LABELS}")
                labels[row["item_id"]] = label
    return labels


def _kappa(pairs: list[tuple[str, str]], weights: list[float]) -> float | None:
    total = sum(weights)
    if not total:
        return None
    categories = sorted({a for a, _ in pairs} | {b for _, b in pairs})
    observed = sum(w for (a, b), w in zip(pairs, weights) if a == b) / total
    expected = sum(
        (sum(w for (a, _), w in zip(pairs, weights) if a == c) / total) * (sum(w for (_, b), w in zip(pairs, weights) if b == c) / total)
        for c in categories
    )
    return None if expected == 1 else (observed - expected) / (1 - expected)


def _binary(label: str) -> str:
    return "supported" if label == "supported" else "unsupported"


def flag_metrics(pairs: list[tuple[str, str]], weights: list[float] | None = None) -> dict[str, float | int | None]:
    """``pairs`` are (reference, judge) labels. Positive = flagged unsupported, the audit's rejection."""
    weights = weights or [1.0] * len(pairs)
    binary = [(_binary(ref), _binary(judged)) for ref, judged in pairs]
    tp = sum(w for (r, j), w in zip(binary, weights) if r == "unsupported" and j == "unsupported")
    fp = sum(w for (r, j), w in zip(binary, weights) if r == "supported" and j == "unsupported")
    fn = sum(w for (r, j), w in zip(binary, weights) if r == "unsupported" and j == "supported")
    total = sum(weights)
    agree = sum(w for (r, j), w in zip(binary, weights) if r == j)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
    exact = sum(w for (r, j), w in zip(pairs, weights) if r == j)
    return {
        "n": len(pairs),
        "agreement": agree / total if total else None,
        "kappa": _kappa(binary, weights),
        "precision_unsupported": precision,
        "recall_unsupported": recall,
        "f1_unsupported": f1,
        "four_way_agreement": exact / total if total else None,
        "four_way_kappa": _kappa(pairs, weights),
    }


def score(items: list[Item], labels: dict[str, str]) -> dict[str, Any]:
    columns = ["judge", *sorted(name[: -len("_verdict")] for item in items for name in item.extra if name.endswith("_verdict"))]
    columns = list(dict.fromkeys(columns))

    def verdict(item: Item, column: str) -> str:
        value = item.judge_verdict if column == "judge" else item.extra.get(f"{column}_verdict", "")
        return value if value in LABELS else ""

    report: dict[str, Any] = {
        "labelled": sum(1 for item in items if item.item_id in labels),
        "unlabelled": sum(1 for item in items if item.item_id not in labels),
        "judges": {},
    }
    for column in columns:
        clean = [item for item in items if item.origin == "clean" and item.item_id in labels and verdict(item, column)]
        faults = [item for item in items if item.origin != "clean" and item.item_id in labels and verdict(item, column)]
        every = [item for item in items if item.item_id in labels and verdict(item, column)]
        report["judges"][column] = {
            "model": next((item.judge_model if column == "judge" else item.extra.get(f"{column}_model", "") for item in every), ""),
            "all_unweighted": flag_metrics([(labels[i.item_id], verdict(i, column)) for i in every]),
            "clean_weighted": flag_metrics([(labels[i.item_id], verdict(i, column)) for i in clean], [i.weight for i in clean]),
            "faults_vs_human": flag_metrics([(labels[i.item_id], verdict(i, column)) for i in faults]),
            # A fault is unsupported by construction; this is the judge's catch rate on them.
            "fault_catch_rate": (
                sum(1 for i in faults if verdict(i, column) != "supported") / len(faults) if faults else None
            ),
            "human_says_fault_is_supported": sum(1 for i in faults if labels[i.item_id] == "supported"),
            "errors_or_missing": sum(1 for item in items if item.item_id in labels and not verdict(item, column)),
        }
    if len(columns) > 1:
        first = columns[0]
        report["between_judges"] = {}
        for other in columns[1:]:
            both = [item for item in items if verdict(item, first) and verdict(item, other)]
            report["between_judges"][f"{first}~{other}"] = flag_metrics([(verdict(i, first), verdict(i, other)) for i in both])
    return report
