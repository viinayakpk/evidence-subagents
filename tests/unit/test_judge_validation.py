"""Judge validation: export a blind, stratified sample; re-judge it; score it against hand labels."""

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evcarry.eval import judge_validation as jv
from evcarry.protocol.judge import JudgeVerdict


def _judge_call(index: int, verdict: str, *, injected: bool = False, value=None) -> dict:
    claim = {
        "claim_id": f"c{index}",
        "text": f"Table t{index} has {value if value is not None else index} rows",
        "kind": "fact",
        "value": value if value is not None else index,
        "evidence": [],
    }
    request = jv.build_request(claim, f"--- call call_{index}: snowflake-read_query {{}}\nrows: {index}")
    return {
        "type": "judge_call",
        "claim_id": claim["claim_id"],
        "arm": "evidence_risk",
        "return_index": index,
        "injected": {"claim_id": claim["claim_id"], "kind": "numeric_perturbation", "variant": "naive"} if injected else None,
        "judge_model": "openrouter_ds_1017",
        "request_sha256": f"sha{index}",
        "request": request,
        "claim": claim,
        "verdict": verdict,
        "source": "llm",
    }


def _traces(tmp_path: Path) -> Path:
    run_dir = tmp_path / "runs"
    run_dir.mkdir()
    events = [{"type": "run_start", "task": "sf-x", "arm": "evidence_risk", "seed": 1}]
    events += [_judge_call(i, "supported") for i in range(80)]
    events += [_judge_call(100 + i, "contradicted") for i in range(12)]
    events += [_judge_call(200 + i, "contradicted", injected=True) for i in range(5)]
    events.append({**_judge_call(300, "supported"), "source": "lexical_fallback", "verdict": None})  # not an LLM verdict
    (run_dir / "sf-x__evidence_risk__m__1.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    return run_dir


def test_export_is_stratified_weighted_and_blind(tmp_path: Path) -> None:
    items = jv.collect([_traces(tmp_path)])
    assert len(items) == 97 and sum(item.origin == "injected" for item in items) == 5
    faults = jv.synthesize_faults(items, 35, seed=1)
    assert len(faults) == 35 and all(item.origin == "synthetic" and not item.judge_verdict for item in faults)
    perturbed = next(item for item in faults if item.fault_kind == "value_perturbation")
    source = next(item for item in items if item.claim["claim_id"] == perturbed.claim["claim_id"])
    assert perturbed.claim["value"] != source.claim["value"]
    # Only the number changes, not the identifier that contains it ("t13 has 13 rows" -> "t13 has 14 rows").
    assert perturbed.claim["text"] == source.claim["text"].replace(f"has {source.claim['value']} rows", f"has {perturbed.claim['value']} rows")
    swapped = next(item for item in faults if item.fault_kind == "evidence_swap")
    assert swapped.evidence in swapped.request
    sample = jv.stratified_sample([*items, *faults], n=100, seed=7)
    counts = {name: sum(item.stratum == name for item in sample) for name in jv.DEFAULT_ALLOCATION}
    # Only 12 clean claims were rejected: the stratum takes all of them and gives 18 places on.
    assert counts == {"fault": 40, "clean_supported": 48, "clean_unsupported": 12}
    supported = [item for item in sample if item.stratum == "clean_supported"]
    assert supported[0].weight == pytest.approx(80 / 48)
    assert jv.stratified_sample([*items, *faults], n=100, seed=7) == sample  # seeded
    items_path, key_path = jv.write_export(sample, tmp_path / "out", seed=7)
    header = next(csv.reader(items_path.open(encoding="utf-8")))
    assert header == jv.ITEM_FIELDS  # no verdict, origin, stratum or run in the labeller's file
    text = items_path.read_text(encoding="utf-8")
    assert "contradicted" not in text and "synthetic" not in text and "injected" not in text
    assert {item.item_id for item in jv.read_key(key_path)} == {item.item_id for item in sample}


async def test_judge_fills_missing_verdicts_and_adds_a_second_judge_column(tmp_path: Path) -> None:
    items = jv.collect([_traces(tmp_path)])
    sample = jv.stratified_sample([*items, *jv.synthesize_faults(items, 35, seed=1)], n=100, seed=7)
    seen: list[str] = []

    class FakeJudge:
        model_name = "fake"
        max_evidence_chars = 12000

        async def ajudge(self, claim, evidence):
            seen.append(evidence)
            return JudgeVerdict(False, "no", "llm", verdict="contradicted")

    missing = sum(1 for item in sample if not item.judge_verdict)
    assert await jv.judge_items(sample, FakeJudge(), column="judge", only_missing=True) == missing == 35
    assert all(item.judge_verdict for item in sample)
    assert await jv.judge_items(sample, FakeJudge(), column="local") == 100
    assert all(item.extra["local_verdict"] == "contradicted" for item in sample)
    # The second judge sees the same evidence text the first one did.
    assert all(evidence in item.request for evidence, item in zip(seen[-100:], sample))


def test_score_metrics_on_known_labels() -> None:
    def item(item_id, origin, verdict, weight=1.0, local=""):
        return jv.Item(item_id=item_id, origin=origin, claim={}, evidence="", request="", judge_verdict=verdict,
                       weight=weight, extra={"local_verdict": local} if local else {})

    rows = [
        item("a", "clean", "supported", 2.0, "supported"),
        item("b", "clean", "supported", 2.0, "contradicted"),
        item("c", "clean", "contradicted", 1.0, "contradicted"),
        item("d", "clean", "contradicted", 1.0, "supported"),
        item("e", "synthetic", "incomplete", 1.0, "incomplete"),
        item("f", "synthetic", "supported", 1.0, "supported"),
    ]
    labels = {"a": "supported", "b": "contradicted", "c": "contradicted", "d": "supported", "e": "incomplete", "f": "contradicted"}
    report = jv.score(rows, labels)
    judge = report["judges"]["judge"]
    # Clean, weighted: TP c (1), FP d (1), FN b (2), TN a (2).
    assert judge["clean_weighted"]["precision_unsupported"] == pytest.approx(0.5)
    assert judge["clean_weighted"]["recall_unsupported"] == pytest.approx(1 / 3)
    assert judge["clean_weighted"]["agreement"] == pytest.approx(0.5)
    assert judge["fault_catch_rate"] == pytest.approx(0.5)
    assert report["judges"]["local"]["all_unweighted"]["agreement"] == pytest.approx(5 / 6)
    assert report["between_judges"]["judge~local"]["n"] == 6


def test_labels_must_be_known_verdicts(tmp_path: Path) -> None:
    path = tmp_path / "items.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=jv.ITEM_FIELDS)
        writer.writeheader()
        writer.writerow({"item_id": "a", "human_label": "Supported"})
        writer.writerow({"item_id": "b", "human_label": ""})
    assert jv.read_labels(path) == {"a": "supported"}
    with path.open("a", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=jv.ITEM_FIELDS).writerow({"item_id": "c", "human_label": "maybe"})
    with pytest.raises(ValueError, match="maybe"):
        jv.read_labels(path)


async def test_a_real_mock_run_feeds_the_export(tmp_path: Path) -> None:
    """judge_call events now carry the claim, the request and the audit context."""
    from evcarry.runs.config import load_experiment
    from evcarry.runs.manager import run_experiment

    config = load_experiment("configs/experiments/mock_e2e.yaml")
    config.arms = ["evidence_risk"]
    config.tasks = "tests/fixtures/tasks/one_sections.txt"
    config.judge_model = "mock_judge"
    await run_experiment(config, out_dir=tmp_path)
    items = jv.collect([tmp_path])
    assert items and all(item.claim.get("text") and "EVIDENCE" in item.request for item in items)
    assert all(item.arm == "evidence_risk" and item.return_index != "" and item.origin == "clean" for item in items)
    assert all(item.judge_verdict in jv.LABELS for item in items)
