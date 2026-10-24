import hashlib
import os
from pathlib import Path

import pytest

from evcarry.env.isolation import DockerIsolation, EnvironmentUnavailable
from evcarry.env.toolathlon import gym_task_root, load_task, load_task_list
from evcarry.eval.flips import pairwise_flips
from evcarry.eval.label_failures import TAXONOMY, label_events, write_labels
from evcarry.eval.metrics import mcnemar_pvalue
from evcarry.harness.prompts import prompt_text
from evcarry.harness.tools import ALL_TOOLS, build_tools
from evcarry.paths import repo_root
from evcarry.trace.processor import JsonlTraceProcessor
from evcarry.trace.writer import JsonlWriter, current_writer


def test_task_list_hashes_and_loader() -> None:
    root = repo_root()
    sums = (root / "tasks" / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    assert len(sums) == 3
    for line in sums:
        digest, name = line.split("  ", 1)
        path = root / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        names = load_task_list(path)
        assert names
        assert all(not item.startswith("#") for item in names)
    frozen = (root / "tasks" / "FROZEN.txt").read_text(encoding="utf-8")
    assert "ed735ba0af71d82688a55f952a6ea11479ae7206" in frozen


def test_real_gym_task_metadata() -> None:
    root = gym_task_root()
    task_dir = root / "academic-literature-review"
    if not task_dir.is_dir():
        pytest.skip("toolathlon_gym submodule is not checked out")
    task = load_task("academic-literature-review", root)
    assert task.local_eval is False
    assert len(task.mcp_servers) == 7


def test_prompt_bytes_and_sorted_tools() -> None:
    root = repo_root()
    text = prompt_text("prompts/v1", "sa")
    assert text == (root / "prompts" / "v1" / "sa_system.md").read_text(encoding="utf-8")
    assert "verify" in prompt_text("prompts/v1", "self_verify")
    tools = build_tools(ALL_TOOLS, caller=object())
    names = [tool.name for tool in tools]
    assert names == sorted(names)


def test_docker_plan_and_refusal_without_docker(monkeypatch) -> None:
    plan = DockerIsolation().plan()
    assert any("run_parallel.sh" in part for part in plan)
    assert any("evcarry/toolathlon-gym:ed735ba" in part for part in plan)
    monkeypatch.setenv("EVCARRY_DOCKER", "/nonexistent/docker")
    with pytest.raises(EnvironmentUnavailable, match="not installed"):
        DockerIsolation().up()


def test_trace_processor_writes_span(tmp_path: Path) -> None:
    writer = JsonlWriter(tmp_path / "run.jsonl")
    token = current_writer.set(writer)

    class SpanData:
        def export(self) -> dict:
            return {"name": "generation"}

    class Span:
        trace_id = "trace_1"
        span_id = "span_1"
        span_data = SpanData()

    try:
        JsonlTraceProcessor().on_span_end(Span())
    finally:
        current_writer.reset(token)
        writer.close()
    text = (tmp_path / "run.jsonl").read_text(encoding="utf-8")
    assert "sdk_span" in text
    assert "trace_1" in text


def test_flips_labels_and_mcnemar(tmp_path: Path) -> None:
    rows = [
        {"task": "a", "seed": 1, "arm": "sa", "passed": True},
        {"task": "a", "seed": 1, "arm": "evidence_risk", "passed": False},
    ]
    for index in range(8):
        rows.append({"task": f"b{index}", "seed": 1, "arm": "sa", "passed": False})
        rows.append({"task": f"b{index}", "seed": 1, "arm": "evidence_risk", "passed": True})
    rows.append({"task": "c", "seed": 1, "arm": "sa", "passed": True})
    rows.append({"task": "c", "seed": 1, "arm": "evidence_risk", "passed": True})
    flips = pairwise_flips(rows, "sa", "evidence_risk")
    assert flips["a_only"] == 1
    assert flips["b_only"] == 8
    assert flips["both_pass"] == 1
    pvalue = mcnemar_pvalue(rows, "sa", "evidence_risk")
    assert 0 < pvalue < 1
    label = label_events(
        [
            {"type": "policy", "action": "fallback"},
            {"type": "run_end", "passed": False},
        ]
    )
    assert label["label"] in TAXONOMY
    trace = tmp_path / "run.jsonl"
    trace.write_text('{"type": "run_end", "passed": false}\n', encoding="utf-8")
    before = trace.read_bytes()
    sidecar = write_labels(trace)
    assert trace.read_bytes() == before
    assert sidecar.is_file()


@pytest.mark.live_provider
def test_live_provider_opt_in() -> None:
    if os.environ.get("EVCARRY_LIVE_PROVIDER") != "1":
        pytest.skip("set EVCARRY_LIVE_PROVIDER=1 to call a real endpoint")


@pytest.mark.parametrize(
    "module",
    ["evcarry.runs.budget", "evcarry.providers.openai_compat", "evcarry.providers.scripted", "evcarry.protocol.judge", "evcarry.runs.manager"],
)
def test_each_spend_module_imports_first_in_a_fresh_interpreter(module: str) -> None:
    """budget is imported by every model; importing providers from it at module level made `import evcarry.runs.budget` fail."""
    import os
    import subprocess
    import sys

    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    result = subprocess.run([sys.executable, "-c", f"import {module}"], capture_output=True, text=True, env=env, check=False)
    assert result.returncode == 0, result.stderr[-1500:]
