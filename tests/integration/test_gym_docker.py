"""Real Toolathlon-Gym tasks in Docker with the scripted model. $0, no API key.

Skipped unless Docker is reachable and the gym images exist (``make gym-images``).
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

import pytest

from evcarry.env.gym_docker import (
    HIDE_KEEP,
    GymImages,
    GymTaskEnv,
    leftover_resources,
    load_server_specs,
    preflight,
    require_offline_overlay,
)
from evcarry.env.isolation import DockerIsolation, EnvironmentUnavailable
from evcarry.env.toolathlon import gym_task_root, load_task, load_task_list
from evcarry.paths import repo_root
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


def _overlay_missing() -> str | None:
    """The offline tests need images built after the offline overlay; a pilot-era image has the same tag."""
    if _REASON is not None:
        return _REASON
    try:
        require_offline_overlay(GymImages.default())
    except EnvironmentUnavailable as exc:
        return str(exc)
    return None


_NO_OVERLAY = _overlay_missing()
needs_offline_overlay = pytest.mark.skipif(_NO_OVERLAY is not None, reason=f"offline overlay: {_NO_OVERLAY}")


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
    assert any(f["reason"].startswith("L1: value 718") for f in checks[0]["failures"])
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


# -- P2-2: network_internal and hide_task_dir (needs images built with overlay offline-v1) -----------


def _subset70() -> list[str]:
    try:
        return load_task_list(repo_root() / "tasks" / "subset70.txt")
    except OSError:
        return []


def _preprocess_text(name: str) -> str:
    try:
        return (gym_task_root() / name / "preprocess" / "main.py").read_text(encoding="utf-8")
    except OSError:
        return ""


def _mock_web_tasks() -> list[str]:
    """subset70 tasks whose preprocess starts a mock web server from the task directory (17 at gym ed735ba)."""
    return [name for name in _subset70() if "http.server" in _preprocess_text(name)]


def _server_cover() -> list[str]:
    """subset70 tasks that together start every gym MCP server: greedy, ties broken by name, so it is fixed."""
    try:
        needs = {name: set(load_task(name, gym_task_root()).mcp_servers) for name in _subset70()}
        missing = set(load_server_specs())
    except OSError:
        return []
    cover: list[str] = []
    while missing:
        best = max(sorted(needs), key=lambda name: len(needs[name] & missing))
        if not needs[best] & missing:
            break
        cover.append(best)
        missing -= needs[best]
    return cover


def test_the_offline_checks_cover_what_they_claim() -> None:
    assert len(_mock_web_tasks()) == 17
    covered = set().union(*(load_task(name, gym_task_root()).mcp_servers for name in _server_cover()))
    assert covered == set(load_server_specs())


@needs_offline_overlay
async def test_the_internal_network_has_no_route_out_but_reaches_postgres(tmp_path: Path) -> None:
    task = load_task("wc-shipping-analysis", gym_task_root())
    async with GymTaskEnv(task, log_dir=tmp_path, network_internal=True) as env:
        agent = env.info.agent
        # No DNS assertion: Docker engines before 26.0 still forward DNS from --internal networks.
        # Connections fail either way, by name and by address.
        by_name = await env._docker("exec", agent, "curl", "-sS", "-m", "5", "-o", "/dev/null", "https://github.com", check=False)
        by_address = await env._docker("exec", agent, "curl", "-sS", "-m", "5", "-o", "/dev/null", "http://1.1.1.1", check=False)
        assert by_name.returncode != 0 and by_address.returncode != 0
        postgres = await env._docker(
            "exec", agent, "pg_isready", "-h", env.info.postgres, "-U", "eigent", "-d", "toolathlon_gym", check=False
        )
        assert postgres.returncode == 0, postgres.stdout + postgres.stderr
        out = await env.python_execute({"code": "print('offline ok')"})
        assert "offline ok" in out and "Return code: 0" in out, out
    assert env.info.as_dict()["network_internal"] is True


@needs_offline_overlay
@pytest.mark.parametrize("task_name", _server_cover())
async def test_every_server_starts_offline_with_the_task_dir_hidden(tmp_path: Path, task_name: str) -> None:
    task = load_task(task_name, gym_task_root())
    async with GymTaskEnv(task, log_dir=tmp_path, network_internal=True, hide_task_dir=True) as env:
        assert set(env.info.servers) == set(task.mcp_servers)
        assert env.info.hidden is not None and env.info.hidden["complete"] is True
        out = await env.python_execute({"code": "import os; print(sorted(os.listdir('/workspace')))"})
        assert "Return code: 0" in out and str(sorted(HIDE_KEEP)) in out, out
    for log in sorted(tmp_path.glob("mcp_*.log")):
        text = log.read_text(encoding="utf-8", errors="replace")
        assert "Creating virtual environment" not in text and "Downloading" not in text, (log.name, text[-500:])


@needs_offline_overlay
@pytest.mark.parametrize("task_name", _mock_web_tasks())
async def test_hidden_task_dir_keeps_the_mock_sites_and_grading(tmp_path: Path, task_name: str) -> None:
    task = load_task(task_name, gym_task_root())
    async with GymTaskEnv(task, log_dir=tmp_path, network_internal=True, hide_task_dir=True) as env:
        agent = env.info.agent
        hidden = env.info.hidden
        assert hidden is not None and hidden["complete"] is True and hidden["mock_servers"], hidden
        look = await env._docker("exec", agent, "bash", "-c", "ls -A /workspace; ls /srv/evcarry_mock")
        assert "tasks" not in look.stdout.split() and "utils" not in look.stdout.split()
        for server in hidden["mock_servers"]:
            assert server["after"] == server["before"], server
            assert 200 <= server["before"][0][0] < 400, server  # "/" answered
            # Still answering after the `docker exec` that restarted it has ended.
            for path, (status, _digest) in zip(server["paths"], server["before"], strict=True):
                if status == 0:
                    continue
                code = await env._docker(
                    "exec", agent, "curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
                    f"http://localhost:{server['port']}{path}", check=False,
                )
                assert code.stdout.strip() == str(status), (server["port"], path, code.stdout)
        # Through the agent's own tools too (the browser must also launch with no route out).
        site = f"http://localhost:{hidden['mock_servers'][0]['port']}/"
        if "fetch" in task.mcp_servers:
            fetched = render_tool_result(await env.call_tool("fetch-fetch_html", {"url": site}))
            assert "Failed to fetch" not in fetched, fetched[:500]
        if "playwright_with_chunk" in task.mcp_servers:
            page = render_tool_result(await env.call_tool("playwright_with_chunk-browser_navigate", {"url": site}))
            assert not any(sign in page for sign in ("Executable doesn't exist", "browserType.launch", "net::ERR_")), page[:800]
        env.claimed_done = True
        result = await env.evaluate()  # the agent did nothing, so it fails, but it runs with its files back
        assert result["eval_ran"] is True
        output = result["eval"]["stdout_tail"] + result["eval"]["stderr_tail"]
        assert "ModuleNotFoundError" not in output and "No module named" not in output, output[-800:]
        back = await env._docker("exec", agent, "bash", "-c", f"ls {env.task_dir}; ls /workspace/utils")
        assert "evaluation" in back.stdout.split() and "__init__.py" in back.stdout.split(), back.stdout


@needs_offline_overlay
async def test_hidden_workspace_holds_only_what_python_execute_needs(tmp_path: Path) -> None:
    task = load_task("sf-hr-rating-distribution", gym_task_root())
    async with GymTaskEnv(task, log_dir=tmp_path / "logs", network_internal=True, hide_task_dir=True) as env:
        out = await env.python_execute(
            {"code": "import os; print(sorted(os.listdir('/workspace')), os.path.exists('/workspace/tasks'))"}
        )
        assert f"{sorted(HIDE_KEEP)} False" in out, out
        assert (env.sealed_copy / "evaluation" / "main.py").is_file() and (env.hidden_utils_copy / "__init__.py").is_file()
        env.claimed_done = True
        result = await env.evaluate()
        assert result["eval_ran"] is True and result["passed"] is False
        output = result["eval"]["stderr_tail"] + result["eval"]["stdout_tail"]
        assert "No such file" not in output and "ModuleNotFoundError" not in output, output[-800:]
