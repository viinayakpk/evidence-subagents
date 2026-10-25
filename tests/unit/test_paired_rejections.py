from evcarry.eval.paired import matched_blocks, paired_difference, pass_hat_k, success_per_dollar
from evcarry.eval.rejections import false_alarm_runs, split_rejections, with_false_alarms_as_failures


def _rows():
    rows = []
    for t in range(10):
        for seed in (1, 2):
            rows.append({"task": f"t{t}", "seed": seed, "arm": "a", "passed": t < 3, "reference_cost_usd": 0.05})
            rows.append({"task": f"t{t}", "seed": seed, "arm": "b", "passed": t < 6, "reference_cost_usd": 0.10})
    rows.append({"task": "t0", "seed": 3, "arm": "a", "passed": True})  # unmatched block
    return rows


def test_matched_blocks_drop_unmatched_and_excluded():
    blocks = matched_blocks(_rows(), "a", "b", exclude=frozenset({"t9"}))
    assert len(blocks) == 18 and ("t0", 3) not in blocks


def test_paired_difference_clusters_by_task():
    result = paired_difference(_rows(), "a", "b", n_boot=2000)
    assert result["blocks"] == 20 and result["tasks"] == 10
    assert abs(result["diff"] - 0.3) < 1e-9
    low, high = result["ci"]
    assert 0 <= low < 0.3 < high and high - low > 0.3  # clustered: only 10 independent units
    assert result["b_only"] == 6 and result["a_only"] == 0


def test_pass_hat_k_and_success_per_dollar():
    rows = [{"task": "x", "arm": "a", "seed": s, "passed": p} for s, p in ((1, True), (2, True), (3, False))]
    assert abs(pass_hat_k(rows, "a", 1) - 2 / 3) < 1e-9
    assert abs(pass_hat_k(rows, "a", 2) - 1 / 3) < 1e-9
    assert pass_hat_k(rows, "a", 4) is None
    assert abs(success_per_dollar(_rows(), "b") - 12 / 2.0) < 1e-9


def test_rejection_split_and_sensitivity():
    rej = [{"task": "t1", "seed": 1, "class": "A"}, {"task": "t2", "seed": 1, "class": "D"},
           {"task": "t4", "seed": 1, "class": "C"}, {"task": "t5", "seed": 1, "class": "?"}]
    split = split_rejections(r["class"] for r in rej)
    assert split == {"attributed": 1, "false_alarm": 2, "unlabelled": 1, "total": 4, "false_alarm_share": 0.5}
    flagged = false_alarm_runs(rej)
    lower = with_false_alarms_as_failures(_rows(), "b", flagged)
    changed = [r for r in lower if r.get("sensitivity")]
    assert {(r["task"], r["seed"]) for r in changed} == {("t2", 1), ("t4", 1)}
