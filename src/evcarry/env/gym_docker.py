"""One Toolathlon-Gym task environment per run, in Docker.

Each run gets its own Docker network, its own Postgres container (started from
the pre-loaded snapshot image), and its own agent container (the gym image).
That is the isolation ``run_parallel.sh`` uses, so parallel runs share no
state. Two differences from that script: the database is not re-imported per
task, and the gym source is not bind-mounted (preprocess scripts write into
their task directory, which a shared mount would let parallel runs of one
task overwrite).

The gym's MCP servers run inside the agent container with the commands, env
and cwd from ``configs/mcp_servers/*.yaml``. This process talks to each over
``docker exec -i`` stdio. Preprocess and evaluation are the task's own
scripts, run inside the agent container the way the gym's TaskAgent and
TaskEvaluator run them. Containers and the network carry the label
``evcarry.project=evidence-subagents`` and are removed on exit;
``evcarry gym clean`` removes leftovers after a crash.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import shutil
import subprocess
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from mcp import Client
from mcp.client.stdio import StdioServerParameters, stdio_client

from evcarry.env.gym_tools import CLAIM_DONE, CLAIM_DONE_META, PYTHON_EXECUTE, PYTHON_EXECUTE_META, GymToolPolicy
from evcarry.env.isolation import EnvironmentUnavailable
from evcarry.env.toolathlon import TaskSpec
from evcarry.harness.tools import ToolMeta
from evcarry.paths import repo_root

LABEL_KEY = "evcarry.project"
LABEL_VALUE = "evidence-subagents"
LABEL = f"{LABEL_KEY}={LABEL_VALUE}"

GYM_HOME = "/workspace"  # gym source inside the image (COPY . . in its Dockerfile)
WORKSPACE = "/workspace/agent_workspace"  # fixed so the system prompt is byte-identical across runs
RUN_DIR = "/workspace/evcarry_run"
PYTHON_BIN = "/opt/venv/bin/python3"
LOCAL_SERVERS = "/opt/local_servers"
PG_ENV = {"PGPORT": "5432", "PGUSER": "eigent", "PGPASSWORD": "camel", "PGDATABASE": "toolathlon_gym"}
# The gym's tool_servers.py copies these libpq names onto the PG_* names its servers read.
_PG_BRIDGE = (
    ("PG_HOST", "PGHOST"),
    ("PG_PORT", "PGPORT"),
    ("PG_DATABASE", "PGDATABASE"),
    ("PG_USER", "PGUSER"),
    ("PG_PASSWORD", "PGPASSWORD"),
)


def gym_commit() -> str:
    for line in (repo_root() / "tasks" / "FROZEN.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("gym_commit="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("tasks/FROZEN.txt has no gym_commit")


@dataclass(frozen=True)
class GymImages:
    agent: str
    postgres: str

    @classmethod
    def default(cls) -> GymImages:
        short = gym_commit()[:7]
        return cls(agent=f"evcarry/toolathlon-gym:{short}", postgres=f"evcarry/toolathlon-gym-pg:{short}")


def docker_bin() -> str | None:
    return shutil.which(os.environ.get("EVCARRY_DOCKER", "docker"))


def preflight(images: GymImages) -> str:
    """Return the docker binary, or raise ``EnvironmentUnavailable`` saying what is missing."""
    docker = docker_bin()
    if docker is None:
        raise EnvironmentUnavailable("Docker is not installed or not on PATH; the gym environment needs it.")
    info = subprocess.run([docker, "info", "--format", "{{.ServerVersion}}"], capture_output=True, text=True, check=False)
    if info.returncode != 0:
        raise EnvironmentUnavailable(f"The Docker daemon is not reachable: {info.stderr.strip()[:300]}")
    missing = [
        image
        for image in (images.agent, images.postgres)
        if subprocess.run([docker, "image", "inspect", image], capture_output=True, check=False).returncode != 0
    ]
    if missing:
        raise EnvironmentUnavailable(
            f"Gym images are missing: {missing}. Build them with `make gym-images` "
            "(scripts/build_gym_images.sh, about 5 minutes)."
        )
    return docker


@dataclass(frozen=True)
class McpServerSpec:
    name: str
    command: str
    args: tuple[str, ...]
    env: dict[str, str]
    cwd: str | None
    timeout: float


def load_server_specs(config_dir: Path | None = None) -> dict[str, McpServerSpec]:
    """Gym MCP server YAMLs keyed by server ``name`` (not file name)."""
    directory = config_dir or repo_root() / "third_party" / "toolathlon_gym" / "configs" / "mcp_servers"
    specs: dict[str, McpServerSpec] = {}
    for path in sorted(directory.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        params = data.get("params") or {}
        name = str(data.get("name") or path.stem)
        specs[name] = McpServerSpec(
            name=name,
            command=str(params.get("command", "")),
            args=tuple(str(item) for item in params.get("args") or ()),
            env={str(k): str(v) for k, v in (params.get("env") or {}).items()},
            cwd=params.get("cwd"),
            timeout=float(data.get("client_session_timeout_seconds", 60)),
        )
    return specs


def resolve_placeholders(value: str, *, task_dir: str) -> str:
    return (
        value.replace("${local_servers_paths}", LOCAL_SERVERS)
        .replace("${agent_workspace}", WORKSPACE)
        .replace("${task_dir}", task_dir)
    )


def server_exec_args(
    spec: McpServerSpec, *, container: str, container_env: dict[str, str], task_dir: str
) -> tuple[str, list[str]]:
    """``(cwd, docker exec args)`` that start one gym MCP server in ``container``.

    Same environment rule as the gym's ``build_mcp_clients``: YAML env first,
    the container's own env wins, then PG* is copied onto PG_*. Only values the
    container does not already have are passed with ``-e``.
    """
    resolve = lambda value: resolve_placeholders(value, task_dir=task_dir)  # noqa: E731
    merged = {key: resolve(value) for key, value in spec.env.items()}
    merged.update(container_env)
    for target, source in _PG_BRIDGE:
        if source in merged:
            merged[target] = merged[source]
    cwd = resolve(spec.cwd) if spec.cwd else WORKSPACE
    args = ["exec", "-i", "-w", cwd]
    for key in sorted(merged):
        if container_env.get(key) != merged[key]:
            args += ["-e", f"{key}={merged[key]}"]
    args += [container, resolve(spec.command), *(resolve(item) for item in spec.args)]
    return cwd, args


@dataclass
class CommandResult:
    returncode: int
    stdout: str
    stderr: str
    seconds: float
    timed_out: bool = False

    def as_dict(self, limit: int = 4000) -> dict[str, Any]:
        return {
            "returncode": self.returncode,
            "stdout_tail": self.stdout[-limit:],
            "stderr_tail": self.stderr[-limit:],
            "seconds": round(self.seconds, 3),
            "timed_out": self.timed_out,
        }


async def _run(argv: list[str], *, timeout: float, stdin: str | None = None) -> CommandResult:
    started = time.perf_counter()
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(
            process.communicate(stdin.encode("utf-8") if stdin is not None else None), timeout
        )
    except TimeoutError:
        with suppress(ProcessLookupError):
            process.kill()
        out, err = await process.communicate()
        return CommandResult(-1, out.decode(errors="replace"), err.decode(errors="replace"), time.perf_counter() - started, True)
    return CommandResult(
        process.returncode or 0,
        out.decode("utf-8", errors="replace"),
        err.decode("utf-8", errors="replace"),
        time.perf_counter() - started,
    )


class _ServerSession:
    """Holds one MCP stdio session open in its own task.

    The SDK's transports must be closed by the task that opened them, so each
    server lives in a holder task; ``call_tool`` is safe from any task.
    """

    def __init__(self, spec: McpServerSpec, params: StdioServerParameters, errlog: Path) -> None:
        self.spec = spec
        self.params = params
        self.errlog = errlog
        self.client: Client | None = None
        self.tools: list[Any] = []
        self.error: BaseException | None = None
        self.ready = asyncio.Event()
        self.stop = asyncio.Event()
        self.task: asyncio.Task[None] | None = None

    async def _hold(self) -> None:
        try:
            with self.errlog.open("w", encoding="utf-8") as errlog:
                transport = stdio_client(self.params, errlog=errlog)
                # Every gym server predates server/discover; two crash on it.
                async with Client(transport, read_timeout_seconds=self.spec.timeout, mode="legacy") as client:
                    tools: list[Any] = []
                    cursor = None
                    while True:
                        page = await client.list_tools(cursor=cursor)
                        tools.extend(page.tools)
                        cursor = page.next_cursor
                        if not cursor:
                            break
                    self.tools = tools
                    self.client = client
                    self.ready.set()
                    await self.stop.wait()
        except BaseException as exc:  # noqa: BLE001 - surfaced through self.error
            self.error = exc
            if isinstance(exc, asyncio.CancelledError):
                raise
        finally:
            self.client = None
            self.ready.set()

    def start(self) -> None:
        self.task = asyncio.create_task(self._hold(), name=f"mcp-{self.spec.name}")

    async def close(self, timeout: float = 10.0) -> None:
        self.stop.set()
        if self.task is None:
            return
        try:
            await asyncio.wait_for(asyncio.shield(self.task), timeout)
        except (TimeoutError, asyncio.CancelledError, Exception):
            self.task.cancel()
            with suppress(BaseException):
                await self.task


@dataclass
class GymRunInfo:
    run_id: str
    network: str
    postgres: str
    agent: str
    launch_time: str
    images: GymImages
    image_ids: dict[str, str] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    preprocess: dict[str, Any] | None = None
    servers: dict[str, dict[str, Any]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "network": self.network,
            "postgres": self.postgres,
            "agent": self.agent,
            "launch_time": self.launch_time,
            "images": {"agent": self.images.agent, "postgres": self.images.postgres},
            "image_ids": dict(self.image_ids),
            "timings": {key: round(value, 3) for key, value in self.timings.items()},
            "preprocess": self.preprocess,
            "servers": self.servers,
        }


class GymTaskEnv:
    """Async context manager: bring up one task's containers, serve its tools, grade, tear down."""

    def __init__(
        self,
        task: TaskSpec,
        *,
        images: GymImages | None = None,
        policy: GymToolPolicy | None = None,
        log_dir: Path,
        docker: str | None = None,
        server_specs: dict[str, McpServerSpec] | None = None,
        setup_timeout: float = 300.0,
        eval_timeout: float = 600.0,
    ) -> None:
        self.task = task
        self.images = images or GymImages.default()
        self.policy = policy or GymToolPolicy.load()
        self.log_dir = log_dir
        self.docker = docker or docker_bin() or "docker"
        self.server_specs = server_specs or load_server_specs()
        self.setup_timeout = setup_timeout
        self.eval_timeout = eval_timeout
        run_id = uuid.uuid4().hex[:10]
        self.info = GymRunInfo(
            run_id=run_id,
            network=f"evcarry-{run_id}-net",
            postgres=f"evcarry-{run_id}-pg",
            agent=f"evcarry-{run_id}-agent",
            launch_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S %A"),
            images=self.images,
        )
        self.task_dir = f"{GYM_HOME}/tasks/finalpool/{task.name}"
        self.container_env = {
            "PGHOST": self.info.postgres,
            "PG_HOST": self.info.postgres,
            **PG_ENV,
            "LOCAL_SERVERS_PATH": LOCAL_SERVERS,
            "PYTHON_BIN": PYTHON_BIN,
        }
        self.sessions: dict[str, _ServerSession] = {}
        self.routes: dict[str, tuple[str, str]] = {}
        self.catalog: dict[str, ToolMeta] = {}
        self.claimed_done = False
        self._started: list[str] = []

    # -- lifecycle -----------------------------------------------------------------

    async def __aenter__(self) -> GymTaskEnv:
        try:
            await self.up()
        except BaseException:
            await self.down()
            raise
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.down()

    async def _docker(self, *args: str, timeout: float = 120.0, stdin: str | None = None, check: bool = True) -> CommandResult:
        result = await _run([self.docker, *args], timeout=timeout, stdin=stdin)
        if check and result.returncode != 0:
            raise RuntimeError(f"docker {' '.join(args[:3])} failed ({result.returncode}): {result.stderr.strip()[:500]}")
        return result

    def _labels(self) -> list[str]:
        return ["--label", LABEL, "--label", f"evcarry.run={self.info.run_id}", "--label", f"evcarry.task={self.task.name}"]

    async def up(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        info = self.info
        await self._docker("network", "create", *self._labels(), info.network)
        self._started.append("network")
        await self._docker("run", "-d", "--name", info.postgres, "--network", info.network, *self._labels(), self.images.postgres)
        self._started.append("postgres")
        env_args = [item for key, value in self.container_env.items() for item in ("-e", f"{key}={value}")]
        await self._docker(
            "run", "-d", "--name", info.agent, "--network", info.network, *self._labels(),
            *env_args, "-w", GYM_HOME, self.images.agent, "sleep", "infinity",
        )
        self._started.append("agent")
        for name, image in (("agent", self.images.agent), ("postgres", self.images.postgres)):
            inspected = await self._docker("image", "inspect", "--format", "{{.Id}}", image, check=False)
            info.image_ids[name] = inspected.stdout.strip()
        await self._wait_for_postgres()
        info.timings["containers_s"] = time.perf_counter() - started
        await self._prepare_workspace()
        mark = time.perf_counter()
        info.preprocess = await self._preprocess()
        info.timings["preprocess_s"] = time.perf_counter() - mark
        mark = time.perf_counter()
        await self._connect_servers()
        info.timings["mcp_connect_s"] = time.perf_counter() - mark
        info.timings["setup_s"] = time.perf_counter() - started

    async def _wait_for_postgres(self) -> None:
        deadline = time.monotonic() + self.setup_timeout
        while time.monotonic() < deadline:
            probe = await self._docker(
                "exec", self.info.postgres, "pg_isready", "-h", "127.0.0.1", "-U", "eigent", "-d", "toolathlon_gym", "-q",
                check=False, timeout=30,
            )
            if probe.returncode == 0:
                return
            await asyncio.sleep(0.5)
        raise EnvironmentUnavailable(f"Postgres container {self.info.postgres} did not become ready")

    async def _prepare_workspace(self) -> None:
        # Mirrors TaskAgent._setup_workspace: copy initial_workspace, make per-server dirs.
        extra = {"arxiv_local": "arxiv_local_storage", "memory": "memory", "playwright_with_chunk": ".playwright_output"}
        dirs = [WORKSPACE, RUN_DIR] + [f"{WORKSPACE}/{d}" for server, d in extra.items() if server in self.task.mcp_servers]
        script = f"mkdir -p {' '.join(dirs)} && if [ -d {self.task_dir}/initial_workspace ]; then cp -a {self.task_dir}/initial_workspace/. {WORKSPACE}/; fi"
        await self._docker("exec", self.info.agent, "bash", "-c", script, timeout=120)

    def _module(self, part: str) -> str:
        return f"tasks.finalpool.{self.task.name}.{part}.main"

    async def _preprocess(self) -> dict[str, Any] | None:
        if not (self.task.root / "preprocess" / "main.py").is_file():
            return None
        launch = " ".join(self.info.launch_time.split()[:2])  # the gym drops the weekday here
        result = await self._docker(
            "exec", "-w", GYM_HOME, self.info.agent, PYTHON_BIN, "-m", self._module("preprocess"),
            "--agent_workspace", WORKSPACE, "--launch_time", launch,
            timeout=self.setup_timeout, check=False,
        )
        (self.log_dir / "preprocess.log").write_text(result.stdout + "\n--- stderr\n" + result.stderr, encoding="utf-8")
        return result.as_dict(limit=2000)

    async def _connect_servers(self) -> None:
        unknown = [name for name in self.task.mcp_servers if name not in self.server_specs]
        if unknown:
            raise EnvironmentUnavailable(f"No gym MCP config for servers {unknown}")
        for name in self.task.mcp_servers:
            spec = self.server_specs[name]
            cwd, args = server_exec_args(spec, container=self.info.agent, container_env=self.container_env, task_dir=self.task_dir)
            await self._docker("exec", self.info.agent, "mkdir", "-p", cwd, timeout=60)
            env = {key: value for key, value in os.environ.items() if key.startswith("DOCKER_")}
            params = StdioServerParameters(command=self.docker, args=args, env=env or None)
            session = _ServerSession(spec, params, self.log_dir / f"mcp_{name}.log")
            self.sessions[name] = session
            session.start()
        failed = []
        for name, session in self.sessions.items():
            try:
                await asyncio.wait_for(session.ready.wait(), max(session.spec.timeout, 60.0) + 30.0)
            except TimeoutError:
                session.error = TimeoutError(f"{name} did not finish initialize/list_tools")
            if session.client is None:
                failed.append(f"{name}: {session.error!r}"[:300])
                continue
            self.info.servers[name] = {"tools": len(session.tools)}
            for tool in sorted(session.tools, key=lambda item: item.name):
                meta = self.policy.meta(name, tool)
                self.catalog[meta.name] = meta
                self.routes[meta.name] = (name, tool.name)
        if failed:
            raise EnvironmentUnavailable("MCP servers failed to start: " + "; ".join(failed))
        self.catalog[CLAIM_DONE] = CLAIM_DONE_META
        self.catalog[PYTHON_EXECUTE] = PYTHON_EXECUTE_META

    async def down(self) -> None:
        mark = time.perf_counter()
        for session in self.sessions.values():
            await session.close()
        self.sessions.clear()
        # Synchronous so teardown still happens while the event loop is being cancelled.
        if "agent" in self._started or "postgres" in self._started:
            subprocess.run(
                [self.docker, "rm", "-fv", self.info.agent, self.info.postgres],
                capture_output=True, check=False, timeout=120,
            )
        if "network" in self._started:
            subprocess.run([self.docker, "network", "rm", self.info.network], capture_output=True, check=False, timeout=60)
        self._started.clear()
        self.info.timings["teardown_s"] = time.perf_counter() - mark

    # -- tools ------------------------------------------------------------------------

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        if name == CLAIM_DONE:
            self.claimed_done = True
            return "Task marked as done."
        if name == PYTHON_EXECUTE:
            return await self.python_execute(arguments)
        route = self.routes.get(name)
        if route is None:
            return f"Error: unknown tool {name}"
        server, tool = route
        session = self.sessions.get(server)
        if session is None or session.client is None:
            return f"Error: MCP server {server} is not connected"
        try:
            return await session.client.call_tool(tool, arguments)
        except Exception as exc:  # noqa: BLE001 - the model sees the error as the tool result
            return f"Error calling {name}: {exc}"

    async def python_execute(self, arguments: dict[str, Any]) -> str:
        """Same contract and output format as the gym's python_execute local tool."""
        code = str(arguments.get("code", ""))
        timeout = min(int(arguments.get("timeout") or 30), 120)
        filename = str(arguments.get("filename") or f"{uuid.uuid4()}.py")
        filename = os.path.basename(filename)
        if not filename.endswith(".py"):
            filename += ".py"
        target = f"{WORKSPACE}/.python_tmp/{filename}"
        await self._docker(
            "exec", "-i", self.info.agent, "bash", "-c", f"mkdir -p {WORKSPACE}/.python_tmp && cat > {shlex.quote(target)}",
            stdin=code, timeout=60,
        )
        result = await self._docker(
            "exec", "-w", WORKSPACE, self.info.agent, "uv", "run", "--directory", WORKSPACE, f"./.python_tmp/{filename}",
            timeout=timeout + 15, check=False,
        )
        if result.timed_out:
            return f"=== TIMEOUT ===\nExceeded {timeout}s limit."
        parts: list[str] = []
        if result.stdout:
            parts += ["=== STDOUT ===", result.stdout.rstrip()]
        if result.stderr:
            parts += ["=== STDERR ===", result.stderr.rstrip()]
        parts += ["=== INFO ===", f"Return code: {result.returncode}", f"Time: {result.seconds:.2f}s / {timeout}s limit"]
        return "\n".join(parts)

    def tool_names(self) -> list[str]:
        return sorted(self.catalog)

    # -- grading ----------------------------------------------------------------------

    async def evaluate(self) -> dict[str, Any]:
        """Run the task's evaluation script the way the gym's TaskEvaluator does.

        The gym grades only runs that called claim_done; otherwise the pass flag
        is null there and False here, and the script is not run.
        """
        if not self.claimed_done:
            return {"status": "not_claimed_done", "passed": False, "eval_ran": False}
        if not (self.task.root / "evaluation" / "main.py").is_file():
            return {"status": "no_eval_script", "passed": None, "eval_ran": False}
        result_file = f"{RUN_DIR}/eval_result.json"
        mark = time.perf_counter()
        result = await self._docker(
            "exec", "-w", GYM_HOME, self.info.agent, PYTHON_BIN, "-m", self._module("evaluation"),
            "--res_log_file", result_file,
            "--agent_workspace", WORKSPACE,
            "--groundtruth_workspace", f"tasks/finalpool/{self.task.name}/groundtruth_workspace",
            "--launch_time", self.info.launch_time,
            timeout=self.eval_timeout, check=False,
        )
        self.info.timings["eval_s"] = time.perf_counter() - mark
        (self.log_dir / "evaluation.log").write_text(result.stdout + "\n--- stderr\n" + result.stderr, encoding="utf-8")
        written = await self._docker("exec", self.info.agent, "cat", result_file, check=False, timeout=30)
        report: Any = None
        if written.returncode == 0 and written.stdout.strip():
            try:
                report = json.loads(written.stdout)
            except json.JSONDecodeError:
                report = written.stdout[-2000:]
        passed = result.returncode == 0 and not result.timed_out
        return {
            "status": "passed" if passed else "failed",
            "passed": passed,
            "eval_ran": True,
            "eval": result.as_dict(),
            "eval_report": report,
        }

    async def export_workspace(self, dest: Path) -> None:
        """Copy the agent workspace and eval output to the host (files owned by the caller)."""
        dest.mkdir(parents=True, exist_ok=True)
        for source, name in ((WORKSPACE, "workspace"), (RUN_DIR, "run")):
            await self._docker("cp", f"{self.info.agent}:{source}/.", str(dest / name), check=False, timeout=300)


def leftover_resources(docker: str | None = None) -> dict[str, list[str]]:
    """Containers and networks with this project's label."""
    binary = docker or docker_bin() or "docker"
    containers = subprocess.run(
        [binary, "ps", "-a", "--filter", f"label={LABEL}", "--format", "{{.Names}}"],
        capture_output=True, text=True, check=False,
    ).stdout.split()
    networks = subprocess.run(
        [binary, "network", "ls", "--filter", f"label={LABEL}", "--format", "{{.Name}}"],
        capture_output=True, text=True, check=False,
    ).stdout.split()
    return {"containers": containers, "networks": networks}


def clean_leftovers(docker: str | None = None) -> dict[str, list[str]]:
    """Remove labelled containers (with their anonymous volumes) and networks. Images stay."""
    binary = docker or docker_bin() or "docker"
    found = leftover_resources(binary)
    if found["containers"]:
        subprocess.run([binary, "rm", "-fv", *found["containers"]], capture_output=True, check=False)
    for network in found["networks"]:
        subprocess.run([binary, "network", "rm", network], capture_output=True, check=False)
    return found


async def smoke_task(task: TaskSpec, *, log_dir: Path, images: GymImages | None = None) -> dict[str, Any]:
    """Bring one task's environment up and down without a model: preprocess, MCP servers, tool count."""
    env = GymTaskEnv(task, images=images, log_dir=log_dir)
    row: dict[str, Any] = {"task": task.name, "servers": list(task.mcp_servers)}
    try:
        async with env:
            row["ok"] = True
            row["preprocess_rc"] = env.info.preprocess["returncode"] if env.info.preprocess else None
            row["tools"] = len(env.catalog)
            row["read_only_tools"] = sum(1 for meta in env.catalog.values() if meta.read_only)
    except Exception as exc:  # noqa: BLE001 - reported per task
        row["ok"] = False
        row["error"] = f"{type(exc).__name__}: {exc}"[:400]
    row["timings"] = {key: round(value, 2) for key, value in env.info.timings.items()}
    return row
