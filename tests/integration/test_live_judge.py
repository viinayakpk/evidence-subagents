"""Opt-in: run the synthetic fault table through a real judge model.

    EVCARRY_LIVE_JUDGE=local_primary uv run pytest tests/integration/test_live_judge.py -s

The value is a key in configs/models.yaml. This sends 12 short requests to that
endpoint; with a billable model it costs money. Skipped unless the variable is set.
"""

from __future__ import annotations

import json
import os

import pytest

from evcarry.analysis.fault_injection import catch_table
from evcarry.protocol.judge import build_judge


@pytest.mark.live_provider
def test_live_judge_on_synthetic_faults() -> None:
    name = os.environ.get("EVCARRY_LIVE_JUDGE")
    if not name:
        pytest.skip("set EVCARRY_LIVE_JUDGE=<models.yaml key> to call a real judge model")
    judge = build_judge(name)
    rows = catch_table(judge, include_clean=True)
    print(json.dumps(rows, indent=1))
    assert judge.calls >= 12
    clean = {row["kind"]: row["earliest"] for row in rows if row["variant"] == "none"}
    assert clean == {"clean": None, "clean_list": None}, "the judge rejected an uncorrupted claim"
