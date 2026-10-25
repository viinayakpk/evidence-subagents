"""run_end ``status`` agrees with ``passed`` (dev run 2026-10-25, train-sf-sales-trip evidence_risk).

That run's grader passed 45/45, but four tool calls searched the filesystem for grader files, so the run
is contaminated and ``passed`` is false. ``status`` still said ``passed``.
"""

from evcarry.runs.manager import final_verdict

SUSPECTS = [{"call_id": "chatcmpl-tool-912f4fe76195e978", "markers": "grader_search", "tool": "python_execute"}]


def _run_end(outcome: dict, **kwargs) -> dict:
    return {**outcome, **final_verdict(outcome, **kwargs)}


def test_a_contaminated_grader_pass_has_status_contaminated() -> None:
    end = _run_end({"status": "passed", "passed": True, "eval_ran": True, "ground_truth_suspects": SUSPECTS})
    assert (end["status"], end["passed"], end["eval_passed"], end["eval_status"]) == ("contaminated", False, True, "passed")


def test_a_contaminated_grader_fail_has_status_contaminated() -> None:
    end = _run_end({"status": "failed", "passed": False, "ground_truth_obtained": [{"call_id": "x"}]})
    assert (end["status"], end["passed"], end["eval_status"]) == ("contaminated", False, "failed")


def test_clean_runs_keep_the_grader_status() -> None:
    assert _run_end({"status": "passed", "passed": True})["status"] == "passed"
    assert _run_end({"status": "failed", "passed": False})["status"] == "failed"


def test_ungraded_runs_keep_their_status() -> None:
    end = _run_end({"status": "not_claimed_done", "passed": False, "eval_ran": False, "ground_truth_suspects": []})
    assert end["status"] == "not_claimed_done"
    end = _run_end({"status": "no_eval_script", "passed": None, "ground_truth_suspects": SUSPECTS})
    assert (end["status"], end["passed"]) == ("no_eval_script", None)


def test_the_ceiling_wins_over_contamination() -> None:
    end = _run_end({"status": "passed", "passed": True, "ground_truth_suspects": SUSPECTS}, ceiling_hit=True)
    assert (end["status"], end["passed"]) == ("cost_ceiling", False)


def test_status_and_passed_never_disagree() -> None:
    for status, passed in (("passed", True), ("failed", False)):
        for suspects in ([], SUSPECTS):
            for ceiling in (False, True):
                end = _run_end({"status": status, "passed": passed, "ground_truth_suspects": suspects}, ceiling_hit=ceiling)
                assert (end["status"] == "passed") == (end["passed"] is True), end
