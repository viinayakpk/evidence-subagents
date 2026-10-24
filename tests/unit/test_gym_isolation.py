"""network_internal and hide_task_dir (P2-2) against a fake docker. No Docker needed.

Every docker command goes through the module-level ``_run``; the fake records each argv and answers
the few commands whose output the code reads. MCP sessions are faked at ``_ServerSession.start``, so
the record also shows when the servers came up. The last tests run the two in-container scripts on
this machine against a real ``python -m http.server``.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from evcarry.env import gym_docker as gd
from evcarry.env.gym_docker import (
    GYM_HOME,
    HIDE_KEEP,
    LABEL,
    MOCK_ROOT,
    PYTHON_BIN,
    WORKSPACE,
    CommandResult,
    GymImages,
    GymTaskEnv,
    McpServerSpec,
    plan_mock_relocations,
    server_exec_args,
    task_url_paths,
)
from evcarry.env.isolation import EnvironmentUnavailable
from evcarry.env.toolathlon import TaskSpec

TASK = "demo-task"
TASK_DIR = f"{GYM_HOME}/tasks/finalpool/{TASK}"
MOCK_SOURCE = f"{TASK_DIR}/tmp/mock_pages"
TASK_TEXT = (
    "Read the catalogue at http://localhost:30219/catalog.html. Prices are at http://localhost:30219/prices.json, "
    "the index at http://localhost:30219"
)
MOCK_PROCESS = {"pid": 4242, "cwd": GYM_HOME, "argv": ["python3", "-m", "http.server", "30219", "--directory", MOCK_SOURCE]}
IMAGES = GymImages(agent="evcarry/test-agent:t", postgres="evcarry/test-pg:t")
FILESYSTEM = McpServerSpec(
    name="filesystem",
    command="node",
    args=("${local_servers_paths}/filesystem/dist/index.js", "${agent_workspace}"),
    env={},
    cwd="${agent_workspace}",
    timeout=300.0,
)
# Top level of /workspace in the gym image after _seal (COPY . . of the submodule, plus the run's dirs).
IMAGE_TOP_LEVEL = (
    ".dockerignore", ".gitignore", ".venv", "Dockerfile", "LICENSE", "README.md", "Weekly_Meal_Plan.xlsx",
    "agent_workspace", "configs", "db", "docker-compose.yml", "evcarry_run", "local_servers", "main.py",
    "pyproject.toml", "run_parallel.sh", "scripts", "tasks", "test_mcp_servers.py", "toolathlon-gym.mdx",
    "utils", "uv.lock",
)


class FakeDocker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.stdin: dict[int, str] = {}
        self.mock_servers: list[dict[str, Any]] = []
        self.move_ok = True
        self.after_names: tuple[str, ...] = HIDE_KEEP
        self.prefix = f"{GYM_HOME}/.venv"

    async def run(self, argv: list[str], *, timeout: float, stdin: str | None = None, settle: bool = False) -> CommandResult:
        call = tuple(argv)
        self.calls.append(call)
        if stdin is not None:
            self.stdin[len(self.calls) - 1] = stdin
        return self.answer(call, stdin)

    def answer(self, call: tuple[str, ...], stdin: str | None) -> CommandResult:
        def ok(out: str = "") -> CommandResult:
            return CommandResult(0, out, "", 0.01)

        if call[1:3] == ("image", "inspect"):
            return ok("sha256:feed")
        if gd._FIND_HTTP_SERVERS in call:
            return ok(json.dumps(self.mock_servers))
        if gd._MOVE_HTTP_SERVERS in call:
            plan = json.loads(stdin or "[]")
            servers = [
                {
                    **{key: item[key] for key in ("pid", "port", "source", "target", "paths")},
                    "before": [[200, "h"] for _ in item["paths"]],
                    "after": [[200, "h"] for _ in item["paths"]],
                }
                for item in plan
            ]
            if not self.move_ok:
                servers[-1]["error"] = "the restarted server did not open the port"
                return CommandResult(1, json.dumps({"ok": False, "servers": servers}), "", 0.01)
            return ok(json.dumps({"ok": True, "servers": servers}))
        if call[-1].startswith(f"ls -A {GYM_HOME} && echo"):
            return ok("\n".join(IMAGE_TOP_LEVEL) + f"\n{gd._LISTING_MARK}\n" + "\n".join(self.after_names) + "\n")
        if call[-1] == "import sys; print(sys.prefix)":
            return ok(self.prefix + "\n")
        if call[-2:] == ("cat", f"{gd.RUN_DIR}/eval_result.json"):
            return ok('{"passed": true}')
        return ok()


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeDocker:
    docker = FakeDocker()
    monkeypatch.setattr(gd, "_run", docker.run)

    def start(session: Any) -> None:
        docker.calls.append(("server-start", session.spec.name, session.params.command, *session.params.args))
        session.tools = []
        session.client = object()
        session.ready.set()

    async def close(session: Any, timeout: float = 10.0) -> None:
        return None

    monkeypatch.setattr(gd._ServerSession, "start", start)
    monkeypatch.setattr(gd._ServerSession, "close", close)
    monkeypatch.setattr(gd, "remove_run_resources", lambda run_id, binary: docker.calls.append(("teardown", run_id)))
    return docker


def _env(tmp_path: Path, **flags: bool) -> GymTaskEnv:
    root = tmp_path / "finalpool" / TASK
    for part in ("preprocess", "evaluation"):
        (root / part).mkdir(parents=True, exist_ok=True)
        (root / part / "main.py").write_text("", encoding="utf-8")
    task = TaskSpec(
        name=TASK, root=root, mcp_servers=["filesystem"], local_eval=True, task_text=TASK_TEXT,
        agent_prompt_sha256=None, domain="demo", stratum="demo",
    )
    return GymTaskEnv(
        task, images=IMAGES, policy=SimpleNamespace(), log_dir=tmp_path / "logs", docker="docker",
        server_specs={"filesystem": FILESYSTEM}, **flags,
    )


async def _drive(env: GymTaskEnv, fake: FakeDocker) -> dict[str, Any]:
    async with env:
        fake.calls.append(("agent-runs",))
        env.claimed_done = True
        fake.calls.append(("grading",))
        result = await env.evaluate()
    return result


def _todays_commands(env: GymTaskEnv) -> tuple[list[tuple[str, ...]], list[tuple[str, ...]]]:
    """The setup and grading commands as the code issued them before network_internal and hide_task_dir existed."""
    info = env.info
    labels = ("--label", LABEL, "--label", f"evcarry.run={info.run_id}", "--label", f"evcarry.task={TASK}")
    _, server_args = server_exec_args(FILESYSTEM, container=info.agent, container_env=env.container_env, task_dir=TASK_DIR)
    seal = " && ".join(
        [
            f"find {GYM_HOME}/tasks/finalpool -mindepth 1 -maxdepth 1 ! -name {TASK} -exec rm -rf {{}} +",
            f"rm -rf {GYM_HOME}/.git {GYM_HOME}/explorer {TASK_DIR}/evaluation {TASK_DIR}/groundtruth_workspace",
            f"find {TASK_DIR} -iname '*groundtruth*' -prune -exec rm -rf {{}} +",
        ]
    )
    setup = [
        ("docker", "network", "create", *labels, info.network),
        ("docker", "run", "-d", "--name", info.postgres, "--network", info.network, *labels, IMAGES.postgres),
        (
            "docker", "run", "-d", "--name", info.agent, "--network", info.network, *labels,
            "-e", f"PGHOST={info.postgres}", "-e", f"PG_HOST={info.postgres}", "-e", "PGPORT=5432", "-e", "PGUSER=eigent",
            "-e", "PGPASSWORD=camel", "-e", "PGDATABASE=toolathlon_gym", "-e", "LOCAL_SERVERS_PATH=/opt/local_servers",
            "-e", "PYTHON_BIN=/opt/venv/bin/python3", "-w", "/workspace", IMAGES.agent, "sleep", "infinity",
        ),
        ("docker", "image", "inspect", "--format", "{{.Id}}", IMAGES.agent),
        ("docker", "image", "inspect", "--format", "{{.Id}}", IMAGES.postgres),
        ("docker", "exec", info.postgres, "pg_isready", "-h", "127.0.0.1", "-U", "eigent", "-d", "toolathlon_gym", "-q"),
        (
            "docker", "exec", info.agent, "bash", "-c",
            f"mkdir -p /workspace/agent_workspace /workspace/evcarry_run && if [ -d {TASK_DIR}/initial_workspace ]; "
            f"then cp -a {TASK_DIR}/initial_workspace/. /workspace/agent_workspace/; fi",
        ),
        (
            "docker", "exec", "-w", "/workspace", info.agent, "timeout", "-s", "KILL", "300", "/opt/venv/bin/python3",
            "-m", f"tasks.finalpool.{TASK}.preprocess.main", "--agent_workspace", "/workspace/agent_workspace",
            "--launch_time", " ".join(info.launch_time.split()[:2]),
        ),
        ("docker", "exec", info.agent, "mkdir", "-p", "/workspace/agent_workspace"),
        ("server-start", "filesystem", "docker", *server_args),
        ("docker", "cp", f"{info.agent}:{TASK_DIR}", str(env.sealed_copy)),
        ("docker", "exec", info.agent, "bash", "-c", seal),
    ]
    grading = [
        ("docker", "exec", info.agent, "rm", "-rf", TASK_DIR),
        ("docker", "cp", f"{env.sealed_copy}/.", f"{info.agent}:{TASK_DIR}"),
        (
            "docker", "exec", "-w", "/workspace", info.agent, "timeout", "-s", "KILL", "600", "/opt/venv/bin/python3",
            "-m", f"tasks.finalpool.{TASK}.evaluation.main", "--res_log_file", "/workspace/evcarry_run/eval_result.json",
            "--agent_workspace", "/workspace/agent_workspace",
            "--groundtruth_workspace", f"tasks/finalpool/{TASK}/groundtruth_workspace", "--launch_time", info.launch_time,
        ),
        ("docker", "exec", info.agent, "cat", "/workspace/evcarry_run/eval_result.json"),
    ]
    return setup, grading


# -- GymTaskEnv ------------------------------------------------------------------------------------


async def test_defaults_issue_exactly_todays_docker_commands(tmp_path: Path, fake: FakeDocker) -> None:
    env = _env(tmp_path)
    result = await _drive(env, fake)
    setup, grading = _todays_commands(env)
    assert fake.calls == [*setup, ("agent-runs",), ("grading",), *grading, ("teardown", env.info.run_id)]
    assert result["eval_ran"] is True and result["passed"] is True
    recorded = env.info.as_dict()
    assert recorded["network_internal"] is False and recorded["hidden"] is None


async def test_network_internal_adds_only_the_flag(tmp_path: Path, fake: FakeDocker) -> None:
    env = _env(tmp_path, network_internal=True)
    await _drive(env, fake)
    setup, grading = _todays_commands(env)
    create = setup[0]
    setup[0] = (*create[:3], "--internal", *create[3:])
    assert fake.calls == [*setup, ("agent-runs",), ("grading",), *grading, ("teardown", env.info.run_id)]
    assert env.info.as_dict()["network_internal"] is True


async def test_hide_runs_after_the_servers_before_the_agent_and_grading_restores_first(
    tmp_path: Path, fake: FakeDocker
) -> None:
    fake.mock_servers = [MOCK_PROCESS]
    env = _env(tmp_path, hide_task_dir=True)
    await _drive(env, fake)
    info = env.info
    setup, grading = _todays_commands(env)
    calls = fake.calls
    assert calls[: len(setup)] == setup  # servers up and sealed exactly as before
    agent = calls.index(("agent-runs",))
    hide = calls[len(setup) : agent]
    assert hide[0] == ("docker", "exec", info.agent, PYTHON_BIN, "-c", gd._FIND_HTTP_SERVERS)
    assert hide[1] == ("docker", "exec", "-i", info.agent, PYTHON_BIN, "-c", gd._MOVE_HTTP_SERVERS)
    assert hide[2] == ("docker", "cp", f"{info.agent}:{GYM_HOME}/utils", str(env.hidden_utils_copy))
    assert hide[3][:5] == ("docker", "exec", info.agent, "bash", "-c")
    assert f"find {GYM_HOME} -mindepth 1 -maxdepth 1 " in hide[3][-1]
    assert all(f"! -name {name}" in hide[3][-1] for name in HIDE_KEEP)
    assert hide[4] == (
        "docker", "exec", "-w", WORKSPACE, info.agent, "timeout", "-s", "KILL", "120",
        "uv", "run", "--directory", WORKSPACE, "python", "-c", "import sys; print(sys.prefix)",
    )
    assert len(hide) == 5
    assert json.loads(fake.stdin[len(setup) + 1]) == [
        {
            "pid": 4242,
            "port": 30219,
            "source": MOCK_SOURCE,
            "target": f"{MOCK_ROOT}/30219",
            "log": f"{MOCK_ROOT}/30219.log",
            "argv": ["python3", "-m", "http.server", "30219", "--directory", f"{MOCK_ROOT}/30219"],
            "paths": ["/", "/catalog.html", "/prices.json"],
        }
    ]
    restore = [
        ("docker", "exec", info.agent, "bash", "-c", f"rm -rf {TASK_DIR} && mkdir -p {GYM_HOME}/tasks/finalpool"),
        ("docker", "cp", f"{env.sealed_copy}/.", f"{info.agent}:{TASK_DIR}"),
        ("docker", "cp", f"{env.hidden_utils_copy}/.", f"{info.agent}:{GYM_HOME}/utils"),
    ]
    assert calls[agent:] == [("agent-runs",), ("grading",), *restore, *grading[2:], ("teardown", info.run_id)]
    hidden = info.as_dict()["hidden"]
    assert hidden["complete"] is True
    assert hidden["deleted"] == sorted(set(IMAGE_TOP_LEVEL) - set(HIDE_KEEP))
    assert "tasks" in hidden["deleted"] and "utils" in hidden["deleted"] and "Weekly_Meal_Plan.xlsx" in hidden["deleted"]
    assert hidden["kept"] == sorted(HIDE_KEEP)
    assert [server["port"] for server in hidden["mock_servers"]] == [30219]
    assert hidden["left_running"] == []
    assert hidden["python_execute_prefix"] == f"{GYM_HOME}/.venv"
    assert "hide_s" in info.timings


async def test_hide_without_a_mock_server_moves_nothing(tmp_path: Path, fake: FakeDocker) -> None:
    env = _env(tmp_path, hide_task_dir=True)
    async with env:
        pass
    assert not any(gd._MOVE_HTTP_SERVERS in call for call in fake.calls)
    assert env.info.hidden is not None
    assert env.info.hidden["mock_servers"] == [] and env.info.hidden["complete"] is True


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("move", "moving the mock web servers"),
        ("unreadable", "cannot read the http.server arguments"),
        ("leftover", "still holds"),
        ("python", "no longer finds"),
    ],
)
async def test_a_failed_hide_step_stops_the_run_before_the_agent(
    tmp_path: Path, fake: FakeDocker, failure: str, message: str
) -> None:
    fake.mock_servers = [MOCK_PROCESS]
    if failure == "move":
        fake.move_ok = False
    elif failure == "unreadable":
        fake.mock_servers = [{"pid": 7, "cwd": TASK_DIR, "argv": ["python3", "-m", "http.server", "30219", "huh"]}]
    elif failure == "leftover":
        fake.after_names = (*HIDE_KEEP, "tasks")
    else:
        fake.prefix = "/opt/venv"
    env = _env(tmp_path, hide_task_dir=True)
    with pytest.raises(EnvironmentUnavailable, match=message):
        async with env:
            fake.calls.append(("agent-runs",))
    assert ("agent-runs",) not in fake.calls
    assert fake.calls[-1] == ("teardown", env.info.run_id)
    assert env.info.hidden is not None and env.info.hidden["complete"] is False
    deleted = any(isinstance(call[-1], str) and call[-1].startswith(f"ls -A {GYM_HOME}") for call in fake.calls)
    assert deleted is (failure in ("leftover", "python"))  # nothing is deleted after a failed move


def test_hide_needs_the_sealed_copy(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="seal"):
        _env(tmp_path, hide_task_dir=True, seal=False)


# -- planning the moves ---------------------------------------------------------------------------


def test_task_urls_are_read_per_port() -> None:
    text = (
        "See http://localhost:30219/a.html, then http://127.0.0.1:30219/b/c.json; also "
        "[x](http://localhost:30220/p?q=1#frag). Again http://localhost:30219/a.html and https://example.com:30219/z"
    )
    assert task_url_paths(text) == {30219: ["/a.html", "/b/c.json"], 30220: ["/p?q=1"]}


def test_plan_moves_servers_under_the_gym_tree_and_leaves_the_rest() -> None:
    processes = [
        MOCK_PROCESS,
        # wc-inventory-gold-repricing starts it with cwd= and no --directory
        {"pid": 11, "cwd": f"{TASK_DIR}/tmp", "argv": ["/opt/venv/bin/python3", "-m", "http.server", "30145"]},
        {"pid": 12, "cwd": TASK_DIR, "argv": ["python3", "-m", "http.server", "--bind", "0.0.0.0", "--directory=files/mock_pages", "30300"]},
        {"pid": 13, "cwd": "/", "argv": ["python3", "-m", "http.server", "-d", f"{TASK_DIR}/files/site"]},
        {"pid": 14, "cwd": "/", "argv": ["python3", "-m", "http.server", "8080", "--directory", f"{WORKSPACE}/site"]},
        {"pid": 15, "cwd": "/tmp", "argv": ["python3", "-m", "http.server", "8081"]},
        {"pid": 16, "cwd": "/", "argv": ["python3", "tool.py", "http.server"]},
    ]
    plan, left = plan_mock_relocations(processes, task_text=TASK_TEXT)
    assert [(item["pid"], item["port"], item["source"], item["argv"]) for item in plan] == [
        (4242, 30219, MOCK_SOURCE, ["python3", "-m", "http.server", "30219", "--directory", f"{MOCK_ROOT}/30219"]),
        (11, 30145, f"{TASK_DIR}/tmp", ["/opt/venv/bin/python3", "-m", "http.server", "30145"]),
        (
            12, 30300, f"{TASK_DIR}/files/mock_pages",
            ["python3", "-m", "http.server", "--bind", "0.0.0.0", f"--directory={MOCK_ROOT}/30300", "30300"],
        ),
        (13, 8000, f"{TASK_DIR}/files/site", ["python3", "-m", "http.server", "-d", f"{MOCK_ROOT}/8000"]),
    ]
    assert plan[0]["paths"] == ["/", "/catalog.html", "/prices.json"]
    assert all(item["paths"] == ["/"] for item in plan[1:])
    assert [item["pid"] for item in left] == [14, 15, 16]


@pytest.mark.parametrize(
    "processes",
    [
        [{"pid": 1, "cwd": TASK_DIR, "argv": ["python3", "-m", "http.server", "30219", "--directory"]}],
        [{"pid": 1, "cwd": TASK_DIR, "argv": ["python3", "-m", "http.server", "30219", "oops"]}],
        [{"pid": 1, "cwd": GYM_HOME, "argv": ["python3", "-m", "http.server", "30219"]}],
        [MOCK_PROCESS, {**MOCK_PROCESS, "pid": 2}],
    ],
    ids=["no-directory-value", "stray-argument", "whole-gym-tree", "port-twice"],
)
def test_plan_refuses_what_it_cannot_move_safely(processes: list[dict[str, Any]]) -> None:
    with pytest.raises(ValueError):
        plan_mock_relocations(processes, task_text="")


def test_the_in_container_scripts_compile() -> None:
    compile(gd._FIND_HTTP_SERVERS, "<find-http-servers>", "exec")
    compile(gd._MOVE_HTTP_SERVERS, "<move-http-servers>", "exec")


# -- the in-container scripts, run here against a real http.server ----------------------------------


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for(port: int, seconds: float = 10.0) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.05)
    raise AssertionError(f"nothing listens on {port}")


def _find_servers() -> list[dict[str, Any]]:
    found = subprocess.run([sys.executable, "-c", gd._FIND_HTTP_SERVERS], capture_output=True, text=True, check=True, timeout=30)
    return json.loads(found.stdout)


@pytest.mark.skipif(not Path("/proc/self/cwd").exists(), reason="needs Linux /proc")
def test_the_scripts_move_a_live_server_and_it_outlives_them(tmp_path: Path) -> None:
    gym = tmp_path / "workspace"
    site = gym / "tasks" / "finalpool" / TASK / "tmp" / "mock_pages"
    site.mkdir(parents=True)
    (site / "prices.json").write_text('{"a": 1}', encoding="utf-8")
    (site / "about.html").write_text("<p>about</p>", encoding="utf-8")
    port = _free_port()
    original = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1", "--directory", str(site)],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
    )
    moved_pids: list[int] = []
    try:
        _wait_for(port)
        ours = [process for process in _find_servers() if process["pid"] == original.pid]
        assert len(ours) == 1 and ours[0]["argv"][-1] == str(site)
        plan, left = plan_mock_relocations(
            ours,
            task_text=f"Prices: http://localhost:{port}/prices.json.",
            gym_home=str(gym),
            keep_roots=(str(gym / "agent_workspace"),),
            target_root=str(tmp_path / "srv"),
        )
        assert left == [] and plan[0]["paths"] == ["/", "/prices.json"]
        moved = subprocess.run(
            [sys.executable, "-c", gd._MOVE_HTTP_SERVERS], input=json.dumps(plan), capture_output=True, text=True, timeout=120
        )
        report = json.loads(moved.stdout)
        assert moved.returncode == 0 and report["ok"] is True, (report, moved.stderr)
        entry = report["servers"][0]
        assert entry["before"] == entry["after"] and [status for status, _ in entry["before"]] == [200, 200]
        original.wait(timeout=10)  # the script stopped it
        shutil.rmtree(gym)  # the old root is gone; the moved server serves its copy
        no_proxy = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with no_proxy.open(f"http://127.0.0.1:{port}/prices.json", timeout=5) as response:
            assert response.read() == b'{"a": 1}'
        target = os.path.realpath(plan[0]["target"])  # /proc/<pid>/cwd is resolved; tmp_path may not be
        moved_pids = [process["pid"] for process in _find_servers() if os.path.realpath(process["cwd"]) == target]
        assert len(moved_pids) == 1
    finally:
        if original.poll() is None:
            original.kill()
            original.wait(timeout=10)
        mine = (str(tmp_path), os.path.realpath(tmp_path))
        leftovers = moved_pids or [
            p["pid"] for p in _find_servers() if any(part in p["cwd"] + " " + " ".join(p["argv"]) for part in mine)
        ]
        for pid in leftovers:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
