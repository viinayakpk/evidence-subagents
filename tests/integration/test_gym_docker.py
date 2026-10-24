"""Real Toolathlon-Gym tasks in Docker with the scripted model. $0, no API key.

Skipped unless Docker is reachable and the gym images exist (``make gym-images``).
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

import pytest

from evcarry.env.gym_docker import GymImages, GymTaskEnv, leftover_resources, preflight
from evcarry.env.isolation import DockerIsolation, EnvironmentUnavailable
from evcarry.env.toolathlon import gym_task_root, load_task
from evcarry.protocol.replayer import render_tool_result
from evcarry.runs.config import InjectConfig, load_experiment
from evcarry.runs.manager import _gym_images, run_experiment
from evcarry.trace.writer import read_events


def _gym_unavailable() -> str | None:
    try:
        preflight(GymImages.default())
    except EnvironmentUnavailable as exc:
        return str(exc)
    if not (gym_task_root() / "wc-shipping-analysis").is_dir():
        return "toolathlon_gym submodule is not checked out"
    return None


_REASON = _gym_unavailable()
pytestmark = [pytest.mark.docker_gym, pytest.mark.skipif(_REASON is not None, reason=f"gym Docker environment: {_REASON}")]


def _names(events: list[dict]) -> set[str]:
    env = next(event for event in events if event["type"] == "environment")
    return {env["network"], env["postgres"], env["agent"]}


def _assert_gone(names: set[str]) -> None:
    left = leftover_resources()
    assert not names & set(left["containers"] + left["networks"]), left


async def test_gym_mock_grid_runs_grades_and_tears_down(tmp_path: Path) -> None:
    config = load_experiment("configs/experiments/gym_mock.yaml")
    await run_experiment(config, out_dir=tmp_path)
    files = sorted(tmp_path.glob("*.jsonl"))
    assert len(files) == 9
    names: set[str] = set()
    spans = []
    for path in files:
        events = read_events(path)
        end = events[-1]
        assert end["type"] == "run_end"
        assert end["passed"] is True, (path.name, end.get("eval"))
        assert end["eval_ran"] is True and end["claimed_done"] is True
        assert end["sealed"] is True and end["ground_truth_suspects"] == []
        assert "PASS" in end["eval"]["stdout_tail"]
        run_names = _names(events)
        assert not names & run_names  # every run had its own network and containers
        names |= run_names
        start = datetime.fromisoformat(events[0]["ts"])
        spans.append((start, datetime.fromisoformat(end["ts"])))
        if "__evidence_risk__" in path.name:
            assert any(event["type"] == "judge_call" and event["source"] == "llm" for event in events)
            checks = [event for event in events if event["type"] == "check"]
            assert checks and all(check["ok"] for check in checks)
            assert any(replay["reason"] == "replay matches" for check in checks for replay in check["replays"])
            assert (tmp_path / "ledgers" / path.name.replace("__mock-scripted", "").replace(".jsonl", ".json")).is_file()
    # parallel_tasks: 3 means some runs overlapped in time
    spans.sort()
    assert any(spans[i + 1][0] < spans[i][1] for i in range(len(spans) - 1))
    _assert_gone(names)


async def test_parallel_environments_do_not_share_state(tmp_path: Path) -> None:
    task = load_task("wc-shipping-analysis", gym_task_root())
    subject = "evcarry isolation probe"
    first = GymTaskEnv(task, log_dir=tmp_path / "a")
    second = GymTaskEnv(task, log_dir=tmp_path / "b")
    async with first, second:
        sent = render_tool_result(await first.call_tool("emails-send_email", {"to": "x@example.com", "subject": subject, "body": "hi"}))
        assert "sent successfully" in sent
        seen = render_tool_result(await first.call_tool("emails-search_emails", {"query": subject, "folder": "Sent"}))
        unseen = render_tool_result(await second.call_tool("emails-search_emails", {"query": subject, "folder": "Sent"}))
        names = {first.info.network, first.info.postgres, first.info.agent, second.info.network, second.info.postgres, second.info.agent}
    assert "Total results: 1" in seen
    assert "No emails found" in unseen
    _assert_gone(names)


async def test_claiming_done_without_work_fails_the_gym_eval(tmp_path: Path) -> None:
    config = load_experiment("configs/experiments/gym_mock.yaml")
    config.arms = ["sa"]
    config.tasks = "tests/fixtures/tasks/gym_wc.txt"
    await run_experiment(config, out_dir=tmp_path, script_overrides={"sa": Path("tests/fixtures/scripts/gym/sa_lazy.json")})
    end = read_events(next(tmp_path.glob("*.jsonl")))[-1]
    assert end["claimed_done"] is True and end["eval_ran"] is True
    assert end["passed"] is False
    assert "Agent output not found" in end["eval"]["stdout_tail"]


async def test_injected_fault_on_real_output_is_sent_back_once(tmp_path: Path) -> None:
    config = load_experiment("configs/experiments/gym_mock.yaml")
    config.arms = ["evidence_risk"]
    config.tasks = "tests/fixtures/tasks/gym_hr.txt"
    config.inject = InjectConfig(probability=1, types=["numeric_perturbation"], variant="naive", max_per_run=1)
    await run_experiment(
        config, out_dir=tmp_path, script_overrides={"evidence_risk": Path("tests/fixtures/scripts/gym/evidence_reinject.json")}
    )
    events = read_events(next(tmp_path.glob("*.jsonl")))
    assert [event["kind"] for event in events if event["type"] == "injection"] == ["numeric_perturbation"]
    checks = [event for event in events if event["type"] == "check"]
    assert checks[0]["ok"] is False
    assert any(f["reason"].startswith("L1: numeric value 718") for f in checks[0]["failures"])
    assert [event["action"] for event in events if event["type"] == "policy"] == ["redelegate", "accept", "accept"]
    assert events[-1]["passed"] is True


def test_pilot_config_passes_the_docker_gate() -> None:
    config = load_experiment("configs/experiments/pilot_deepseek.yaml")
    assert config.environment == "docker"
    assert DockerIsolation(_gym_images(config)).up()


def test_unknown_server_and_bad_image_fail_cleanly(tmp_path: Path) -> None:
    task = load_task("wc-shipping-analysis", gym_task_root())
    env = GymTaskEnv(task, images=GymImages(agent="evcarry/does-not-exist:none", postgres=GymImages.default().postgres), log_dir=tmp_path)

    async def scenario() -> None:
        async with env:
            pass

    with pytest.raises(RuntimeError):
        asyncio.run(scenario())
    _assert_gone({env.info.network, env.info.postgres, env.info.agent})


async def test_a_task_that_cannot_start_is_recorded_and_the_grid_continues(tmp_path: Path) -> None:
    # Its email_config.json has no password, so the gym's emails server refuses to start.
    config = load_experiment("configs/experiments/gym_mock.yaml")
    config.arms = ["sa"]
    config.tasks = "tests/fixtures/tasks/gym_env_error.txt"
    config.parallel_tasks = 2
    await run_experiment(config, out_dir=tmp_path)
    ends = {read_events(path)[-1]["task"]: read_events(path)[-1] for path in tmp_path.glob("*.jsonl")}
    broken = ends["arxiv-reading-plan-excel-gcal-email"]
    assert broken["status"] == "env_error" and broken["passed"] is None
    assert "emails" in broken["error"]
    assert ends["wc-shipping-analysis"]["passed"] is True
    left = leftover_resources()
    assert not any(end["gym_run_id"] in name for end in ends.values() for name in left["containers"] + left["networks"])


async def test_cancelling_during_setup_leaves_nothing(tmp_path: Path) -> None:
    task = load_task("wc-shipping-analysis", gym_task_root())
    env = GymTaskEnv(task, log_dir=tmp_path)
    starting = asyncio.create_task(env.__aenter__())
    await asyncio.sleep(0.3)
    starting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await starting
    await env.down()
    _assert_gone({env.info.network, env.info.postgres, env.info.agent})


async def test_python_execute_timeout_kills_the_process_in_the_container(tmp_path: Path) -> None:
    task = load_task("yf-portfolio-risk-excel", gym_task_root())
    async with GymTaskEnv(task, log_dir=tmp_path) as env:
        out = await env.call_tool("python_execute", {"code": "import time\ntime.sleep(60)", "filename": "slow.py", "timeout": 2})
        assert out.startswith("=== TIMEOUT ===")
        ps = await env._docker("exec", env.info.agent, "ps", "-eo", "args", check=False)
        assert "slow.py" not in ps.stdout


async def test_the_agent_container_holds_no_answers_until_grading(tmp_path: Path) -> None:
    """The image ships every task's evaluation and ground truth, and the gym's .git; the agent runs as root."""
    task = load_task("sf-hr-rating-distribution", gym_task_root())
    async with GymTaskEnv(task, log_dir=tmp_path / "logs") as env:
        assert env.info.sealed
        look = await env._docker(
            "exec", env.info.agent, "bash", "-c",
            "ls /workspace/tasks/finalpool; ls -a /workspace/tasks/finalpool/*; test -e /workspace/.git && echo HAS_GIT; "
            "find / -xdev -iname '*groundtruth*' -not -path '/proc/*' 2>/dev/null | grep -v /agent_workspace/ || true",
        )
        listing = look.stdout
        assert listing.splitlines()[0] == task.name  # no other task is left
        assert "evaluation" not in listing and "groundtruth" not in listing.lower() and "HAS_GIT" not in listing
        attempt = await env.python_execute(
            {"code": "import os; print(os.path.exists('/workspace/tasks/finalpool/sf-hr-rating-distribution/groundtruth_workspace'))"}
        )
        assert "False" in attempt
        # The host copy is the task directory as it was after preprocess.
        assert (env.sealed_copy / "evaluation" / "main.py").is_file()
        env.claimed_done = True
        result = await env.evaluate()  # the agent did nothing, so it fails, but it runs with its ground truth
        assert result["eval_ran"] is True and result["passed"] is False
        stderr = result["eval"]["stderr_tail"] + result["eval"]["stdout_tail"]
        assert "No such file" not in stderr and "ModuleNotFoundError" not in stderr
        back = await env._docker("exec", env.info.agent, "ls", f"/workspace/tasks/finalpool/{task.name}")
        assert "groundtruth_workspace" in back.stdout and "evaluation" in back.stdout
