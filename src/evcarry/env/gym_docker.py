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

Two opt-in settings (P2-2 in docs/PILOT_DIAGNOSIS.md), both off by default; with
both off a run issues exactly the docker commands it did before they existed.

- ``network_internal``: the run's network is created ``--internal``, so the agent
  container reaches its own Postgres and nothing else. The MCP servers still talk
  to this process over ``docker exec`` stdio, and the tasks' mock web servers
  listen on the container's own loopback. Needs images built from the current
  ``docker/gym/overlay.Dockerfile`` (label ``evcarry.overlay=offline-v1``): older
  images rebuild six Python servers' environments from PyPI at every start.
- ``hide_task_dir``: after the MCP servers start, every ``python -m http.server``
  that preprocess started from the gym tree (the task's mock web sites, served from
  ``<task>/tmp`` or ``<task>/files``) is copied to ``/srv/evcarry_mock/<port>`` and
  restarted on the same port, checked before and after; then ``/workspace`` keeps
  only ``HIDE_KEEP``. Grading puts the task directory and ``utils`` back first.
- ``agent_db_readonly`` (plan A20, 2026-10-25): the agent's own code cannot write the
  run's database. See the notes above ``AGENT_PG`` and ``lock_sql``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import posixpath
import re
import secrets
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

class _SkipBannerLines(logging.Filter):
    """Several gym servers print a start-up banner on stdout. The MCP client skips the line
    but logs a full traceback for it; drop just that record."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage() != "Failed to parse JSONRPC message from server"


logging.getLogger("mcp.client.stdio").addFilter(_SkipBannerLines())

LABEL_KEY = "evcarry.project"
LABEL_VALUE = "evidence-subagents"
LABEL = f"{LABEL_KEY}={LABEL_VALUE}"

GYM_HOME = "/workspace"  # gym source inside the image (COPY . . in its Dockerfile)
WORKSPACE = "/workspace/agent_workspace"  # fixed so the system prompt is byte-identical across runs
RUN_DIR = "/workspace/evcarry_run"
PYTHON_BIN = "/opt/venv/bin/python3"
LOCAL_SERVERS = "/opt/local_servers"
PG_ENV = {"PGPORT": "5432", "PGUSER": "eigent", "PGPASSWORD": "camel", "PGDATABASE": "toolathlon_gym"}
# hide_task_dir: what stays under /workspace while the agent runs. python_execute runs
# `uv run --directory /workspace/agent_workspace`, which finds the gym's project file (a virtual
# project: no build system, no path sources) and its environment one level up; RUN_DIR holds the
# eval result. No running server reads anything else there: their YAMLs resolve to
# /opt/local_servers and the workspace, and the emails server reads ${task_dir}/email_config.json
# only at start.
HIDE_KEEP = ("agent_workspace", "evcarry_run", "pyproject.toml", "uv.lock", ".venv")
MOCK_ROOT = "/srv/evcarry_mock"  # moved mock web roots: <MOCK_ROOT>/<port>, log in <MOCK_ROOT>/<port>.log
# -- agent_db_readonly (plan A20) -------------------------------------------------------------------
# The gym's own preprocess and grader code connects as eigent/camel (the password is written into 966 task
# files), and eigent is the cluster superuser. With the guard on:
#   - the container env (what python_execute, the terminal server and every shell they start inherit) holds
#     AGENT_PG: a role with SELECT on every schema and no write, DDL or ownership anywhere;
#   - after preprocess, eigent's password becomes a per-run secret that only the MCP servers get, with
#     ``docker exec -e``. The snowflake server, which runs any SQL the agent sends (write_query, and
#     two-part names pass through untranslated), gets SF_ROLE instead: it writes only its own schemas;
#   - the agent's tools run as uid AGENT_UID (``setpriv``) with only the file capabilities root's tools used
#     (dac_override, fowner, chown): they read and write every file as before, but cannot read another uid's
#     /proc/<pid>/environ (that needs CAP_SYS_PTRACE, which Docker never grants), so the MCP servers'
#     secret stays out of reach, and they cannot signal those processes;
#   - grading: eigent's password goes back to camel and the grader gets eigent's env, as before.
AGENT_PG = {"PGUSER": "evcarry_agent", "PGPASSWORD": "evcarry_agent"}  # read-only role; its password is not a secret
GYM_PG_WRITE = {"PGUSER": "eigent", "PGPASSWORD": "camel"}  # what preprocess and graders use (hard-coded there too)
SF_ROLE = "evcarry_sf"
SF_SCHEMAS = ("sf_data", "sf")  # the snowflake mock's tables: sf_data."DB__SCHEMA__TABLE"
SF_SERVERS = frozenset({"snowflake"})
AGENT_SERVERS = frozenset({"terminal"})  # MCP servers that run commands the agent writes: run as the agent
AGENT_UID = 4242
AGENT_CAPS = "+dac_override,+fowner,+chown"
AGENT_EXEC = (
    "setpriv", f"--reuid={AGENT_UID}", f"--regid={AGENT_UID}", "--clear-groups",
    f"--inh-caps={AGENT_CAPS}", f"--ambient-caps={AGENT_CAPS}", "--",
)
AGENT_HOME = "/root"  # uv's cache and interpreters stay where root's tools found them


def _pg_env(creds: dict[str, str]) -> dict[str, str]:
    """libpq names plus the PG_* names the gym's servers read."""
    return {**creds, "PG_USER": creds["PGUSER"], "PG_PASSWORD": creds["PGPASSWORD"]}


def lock_sql(write_secret: str, sf_secret: str) -> str:
    """SQL run as eigent over the Postgres container's local socket after preprocess (``agent_db_readonly``).

    Read access is granted on every non-system schema, including tables the MCP servers create later
    (default privileges for eigent and SF_ROLE). The agent role owns nothing, has no CREATE on the database
    or any schema, and is not a member of any role. Temporary tables stay possible (they are private to the
    session and no grader reads them).
    """
    quote = lambda value: "'" + value.replace("'", "''") + "'"  # noqa: E731
    sf_list = ", ".join(quote(name) for name in SF_SCHEMAS)
    return f"""
CREATE ROLE {AGENT_PG['PGUSER']} LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS PASSWORD {quote(AGENT_PG['PGPASSWORD'])};
CREATE ROLE {SF_ROLE} LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS PASSWORD {quote(sf_secret)};
REVOKE CREATE ON DATABASE toolathlon_gym FROM PUBLIC;
DO $guard$
DECLARE s text;
BEGIN
  FOR s IN SELECT nspname FROM pg_namespace WHERE nspname !~ '^pg_' AND nspname <> 'information_schema' LOOP
    EXECUTE format('REVOKE CREATE ON SCHEMA %I FROM PUBLIC', s);
    EXECUTE format('GRANT USAGE ON SCHEMA %I TO {AGENT_PG['PGUSER']}, {SF_ROLE}', s);
    EXECUTE format('GRANT SELECT ON ALL TABLES IN SCHEMA %I TO {AGENT_PG['PGUSER']}, {SF_ROLE}', s);
    EXECUTE format('GRANT SELECT ON ALL SEQUENCES IN SCHEMA %I TO {AGENT_PG['PGUSER']}, {SF_ROLE}', s);
    EXECUTE format('ALTER DEFAULT PRIVILEGES FOR ROLE eigent IN SCHEMA %I GRANT SELECT ON TABLES TO {AGENT_PG['PGUSER']}, {SF_ROLE}', s);
    EXECUTE format('ALTER DEFAULT PRIVILEGES FOR ROLE eigent IN SCHEMA %I GRANT SELECT ON SEQUENCES TO {AGENT_PG['PGUSER']}, {SF_ROLE}', s);
    IF s IN ({sf_list}) THEN
      EXECUTE format('GRANT CREATE ON SCHEMA %I TO {SF_ROLE}', s);
      EXECUTE format('GRANT ALL ON ALL TABLES IN SCHEMA %I TO {SF_ROLE}', s);
      EXECUTE format('GRANT ALL ON ALL SEQUENCES IN SCHEMA %I TO {SF_ROLE}', s);
      EXECUTE format('ALTER DEFAULT PRIVILEGES FOR ROLE {SF_ROLE} IN SCHEMA %I GRANT SELECT ON TABLES TO {AGENT_PG['PGUSER']}', s);
    END IF;
  END LOOP;
END
$guard$;
ALTER ROLE eigent PASSWORD {quote(write_secret)};
"""


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


OFFLINE_OVERLAY = "offline-v1"


def require_offline_overlay(images: GymImages) -> None:
    """Refuse an agent image built before the offline overlay (label ``evcarry.overlay``).

    The image tag is keyed on the gym commit, so a pilot-era image has the same tag. On it
    six uv servers rebuild their environment from PyPI at start, which ``network_internal``
    blocks, so every run would end ``env_error``. Run ``make gym-images`` first.
    """
    docker = preflight(images)
    found = subprocess.run(
        [docker, "image", "inspect", "--format", '{{index .Config.Labels "evcarry.overlay"}}', images.agent],
        capture_output=True, text=True, check=False,
    )
    label = found.stdout.strip() if found.returncode == 0 else ""
    if label != OFFLINE_OVERLAY:
        raise EnvironmentUnavailable(
            f"{images.agent} has evcarry.overlay={label or '<none>'}, not {OFFLINE_OVERLAY}: it predates the offline overlay. "
            "network_internal and hide_task_dir need the rebuilt images (make gym-images)."
        )


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
    spec: McpServerSpec,
    *,
    container: str,
    container_env: dict[str, str],
    task_dir: str,
    env_override: dict[str, str] | None = None,
    prefix: tuple[str, ...] = (),
) -> tuple[str, list[str]]:
    """``(cwd, docker exec args)`` that start one gym MCP server in ``container``.

    Same environment rule as the gym's ``build_mcp_clients``: YAML env first,
    the container's own env wins, then PG* is copied onto PG_*. Only values the
    container does not already have are passed with ``-e``. ``env_override``
    (``agent_db_readonly``: this server's database role) wins over both, and
    ``prefix`` goes before the server command (the agent's ``setpriv``).
    """
    resolve = lambda value: resolve_placeholders(value, task_dir=task_dir)  # noqa: E731
    merged = {key: resolve(value) for key, value in spec.env.items()}
    merged.update(container_env)
    merged.update(env_override or {})
    for target, source in _PG_BRIDGE:
        if source in merged:
            merged[target] = merged[source]
    cwd = resolve(spec.cwd) if spec.cwd else WORKSPACE
    args = ["exec", "-i", "-w", cwd]
    for key in sorted(merged):
        if container_env.get(key) != merged[key]:
            args += ["-e", f"{key}={merged[key]}"]
    args += [container, *prefix, resolve(spec.command), *(resolve(item) for item in spec.args)]
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


async def _run(argv: list[str], *, timeout: float, stdin: str | None = None, settle: bool = False) -> CommandResult:
    """Run a command. ``settle``: if cancelled, let it finish before re-raising.

    Killing a ``docker run`` / ``network create`` client does not stop the daemon
    from creating the object, so creates settle and teardown then sees what they made.
    """
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
    except asyncio.CancelledError:
        if settle:
            with suppress(BaseException):
                await asyncio.wait_for(asyncio.shield(process.wait()), 60)
        with suppress(ProcessLookupError):
            process.kill()
        raise
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
    sealed: bool = False
    network_internal: bool = False
    # hide_task_dir: mock servers moved or left, /workspace entries deleted and kept, host copies.
    hidden: dict[str, Any] | None = None
    # agent_db_readonly: when the database was locked and unlocked, the agent's role and uid.
    db_guard: dict[str, Any] | None = None

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
            "sealed": self.sealed,
            "network_internal": self.network_internal,
            "hidden": self.hidden,
            "db_guard": self.db_guard,
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
        seal: bool = True,
        network_internal: bool = False,
        hide_task_dir: bool = False,
        agent_db_readonly: bool = False,
    ) -> None:
        if hide_task_dir and not seal:
            raise ValueError("hide_task_dir needs seal=True: grading restores the task directory from the sealed copy")
        self.task = task
        self.seal = seal
        self.network_internal = network_internal
        self.hide_task_dir = hide_task_dir
        self.agent_db_readonly = agent_db_readonly
        # Per-run secrets: eigent's password while the agent runs (MCP servers only) and the snowflake role's.
        self._write_secret = secrets.token_hex(16) if agent_db_readonly else ""
        self._sf_secret = secrets.token_hex(16) if agent_db_readonly else ""
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
            network_internal=network_internal,
        )
        self.task_dir = f"{GYM_HOME}/tasks/finalpool/{task.name}"
        self.container_env = {
            "PGHOST": self.info.postgres,
            "PG_HOST": self.info.postgres,
            **PG_ENV,
            **(AGENT_PG if agent_db_readonly else {}),
            "LOCAL_SERVERS_PATH": LOCAL_SERVERS,
            "PYTHON_BIN": PYTHON_BIN,
        }
        self.sessions: dict[str, _ServerSession] = {}
        self.routes: dict[str, tuple[str, str]] = {}
        self.catalog: dict[str, ToolMeta] = {}
        self.claimed_done = False

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
        settle = args[0] in ("run", "network")
        result = await _run([self.docker, *args], timeout=timeout, stdin=stdin, settle=settle)
        if check and result.returncode != 0:
            raise RuntimeError(f"docker {' '.join(args[:3])} failed ({result.returncode}): {result.stderr.strip()[:500]}")
        return result

    def _labels(self) -> list[str]:
        return ["--label", LABEL, "--label", f"evcarry.run={self.info.run_id}", "--label", f"evcarry.task={self.task.name}"]

    async def up(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        info = self.info
        internal = ["--internal"] if self.network_internal else []
        await self._docker("network", "create", *internal, *self._labels(), info.network)
        await self._docker("run", "-d", "--name", info.postgres, "--network", info.network, *self._labels(), self.images.postgres)
        env_args = [item for key, value in self.container_env.items() for item in ("-e", f"{key}={value}")]
        await self._docker(
            "run", "-d", "--name", info.agent, "--network", info.network, *self._labels(),
            *env_args, "-w", GYM_HOME, self.images.agent, "sleep", "infinity",
        )
        for name, image in (("agent", self.images.agent), ("postgres", self.images.postgres)):
            inspected = await self._docker("image", "inspect", "--format", "{{.Id}}", image, check=False)
            info.image_ids[name] = inspected.stdout.strip()
        await self._wait_for_postgres()
        info.timings["containers_s"] = time.perf_counter() - started
        await self._prepare_workspace()
        mark = time.perf_counter()
        info.preprocess = await self._preprocess()
        info.timings["preprocess_s"] = time.perf_counter() - mark
        if self.agent_db_readonly:
            await self._lock_database()
        mark = time.perf_counter()
        await self._connect_servers()
        info.timings["mcp_connect_s"] = time.perf_counter() - mark
        if self.seal:
            await self._seal()
        if self.hide_task_dir:
            await self._hide()
        info.timings["setup_s"] = time.perf_counter() - started

    @property
    def sealed_copy(self) -> Path:
        return self.log_dir / "sealed_task"

    @property
    def hidden_utils_copy(self) -> Path:
        """Host copy of the gym's ``utils``, kept like ``sealed_task`` and put back before grading.

        35 graders put ``/workspace`` on ``sys.path`` and try ``utils.verify_v2`` (absent at gym ed735ba;
        they fall back), and no grader imports anything else from the gym tree. Restoring ``utils``
        keeps grading exactly as in the sealed-only setup.
        """
        return self.log_dir / "hidden_utils"

    async def _seal(self) -> None:
        """Remove the answers from the container before the agent starts.

        The image holds all 503 tasks with their evaluation code and ground truth
        (``groundtruth_workspace/``, but also ``files/groundtruth_data.json``,
        ``generate_groundtruth.py`` and the like) plus the gym's ``.git``. The agent's
        ``python_execute`` and terminal tools run as root here, so the files are deleted,
        not hidden. This task's directory is first copied to the host as it is after
        preprocess (``sealed_task`` in the log dir); ``_unseal`` puts it back for grading.
        MCP servers are already running; the emails server read its config at start.
        """
        mark = time.perf_counter()
        shutil.rmtree(self.sealed_copy, ignore_errors=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        await self._docker("cp", f"{self.info.agent}:{self.task_dir}", str(self.sealed_copy), timeout=300)
        tasks = f"{GYM_HOME}/tasks/finalpool"
        script = " && ".join(
            [
                f"find {tasks} -mindepth 1 -maxdepth 1 ! -name {shlex.quote(self.task.name)} -exec rm -rf {{}} +",
                f"rm -rf {GYM_HOME}/.git {GYM_HOME}/explorer {self.task_dir}/evaluation {self.task_dir}/groundtruth_workspace",
                f"find {self.task_dir} -iname '*groundtruth*' -prune -exec rm -rf {{}} +",
            ]
        )
        await self._docker("exec", self.info.agent, "bash", "-c", script, timeout=300)
        self.info.sealed = True
        self.info.timings["seal_s"] = time.perf_counter() - mark

    async def _unseal(self) -> None:
        """Put this task's directory back exactly as it was after preprocess; agent edits there are discarded.

        After ``hide_task_dir`` the parent directories are gone too, and ``utils`` is put back from its host copy.
        """
        if not self.info.sealed:
            return
        if self.info.hidden is not None:
            parent = posixpath.dirname(self.task_dir)
            await self._docker(
                "exec", self.info.agent, "bash", "-c", f"rm -rf {self.task_dir} && mkdir -p {parent}", timeout=120
            )
            await self._docker("cp", f"{self.sealed_copy}/.", f"{self.info.agent}:{self.task_dir}", timeout=300)
            await self._docker("cp", f"{self.hidden_utils_copy}/.", f"{self.info.agent}:{GYM_HOME}/utils", timeout=300)
            return
        await self._docker("exec", self.info.agent, "rm", "-rf", self.task_dir, timeout=120)
        await self._docker("cp", f"{self.sealed_copy}/.", f"{self.info.agent}:{self.task_dir}", timeout=300)

    async def _hide(self) -> None:
        """Leave the agent nothing under ``/workspace`` but ``HIDE_KEEP`` (``hide_task_dir``).

        Runs after ``_seal``, with the MCP servers up and before the agent starts. Three steps:

        1. Move the mock web sites. Preprocess starts ``python -m http.server`` serving
           ``<task>/tmp/...`` or ``<task>/files/mock_pages`` (128 tasks; 39 of pool160), and that
           server opens files per request. Each one under the gym tree is copied to
           ``MOCK_ROOT/<port>`` and restarted on its port with the same arguments; ``/`` and every
           localhost path the task text names on that port must answer, and answer the same
           (status and body hash) before and after.
        2. Copy ``utils`` to the host, then delete every other entry of ``/workspace``
           (the task directory, ``utils``, ``configs``, ``db``, ``local_servers``, the gym's
           ``README.md`` and a stray ``Weekly_Meal_Plan.xlsx`` among them).
        3. Check that python_execute's ``uv run`` still finds its environment.

        The agent's tools run as root, so this deletes rather than hides. Any failure raises
        ``EnvironmentUnavailable``: the run is not started. ``info.hidden`` records each step.
        """
        mark = time.perf_counter()
        record: dict[str, Any] = {"mock_root": MOCK_ROOT, "keep": list(HIDE_KEEP), "complete": False}
        self.info.hidden = record
        try:
            await self._move_mock_servers(record)
            await self._delete_gym_tree(record)
            await self._check_python_execute(record)
        except EnvironmentUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - every setup failure means: do not start the run
            raise EnvironmentUnavailable(f"hide_task_dir for {self.task.name} failed: {exc}"[:1000]) from exc
        record["complete"] = True
        self.info.timings["hide_s"] = time.perf_counter() - mark

    async def _move_mock_servers(self, record: dict[str, Any]) -> None:
        found = await self._docker("exec", self.info.agent, PYTHON_BIN, "-c", _FIND_HTTP_SERVERS, timeout=60)
        plan, left = plan_mock_relocations(json.loads(found.stdout), task_text=self.task.task_text)
        record["mock_servers"] = []
        record["left_running"] = left
        if not plan:
            return
        limit = 60 + sum(40 + 10 * len(item["paths"]) for item in plan)
        moved = await self._docker(
            "exec", "-i", self.info.agent, PYTHON_BIN, "-c", _MOVE_HTTP_SERVERS,
            stdin=json.dumps(plan), timeout=limit, check=False,
        )
        try:
            report = json.loads(moved.stdout)
        except json.JSONDecodeError:
            report = None
        if isinstance(report, dict):
            record["mock_servers"] = report.get("servers", [])
        if moved.returncode != 0 or moved.timed_out or not isinstance(report, dict) or report.get("ok") is not True:
            detail = json.dumps(record["mock_servers"])[-600:] if record["mock_servers"] else (moved.stdout + moved.stderr)[-600:]
            raise EnvironmentUnavailable(
                f"moving the mock web servers of {self.task.name} failed (exit {moved.returncode}): {detail}"
            )

    async def _delete_gym_tree(self, record: dict[str, Any]) -> None:
        shutil.rmtree(self.hidden_utils_copy, ignore_errors=True)
        await self._docker("cp", f"{self.info.agent}:{GYM_HOME}/utils", str(self.hidden_utils_copy), timeout=300)
        record["saved_to_host"] = {"utils": str(self.hidden_utils_copy), "task_dir": str(self.sealed_copy)}
        keep = " ".join(f"! -name {shlex.quote(name)}" for name in HIDE_KEEP)
        script = (
            f"ls -A {GYM_HOME} && echo {shlex.quote(_LISTING_MARK)} && "
            f"find {GYM_HOME} -mindepth 1 -maxdepth 1 {keep} -exec rm -rf {{}} + && ls -A {GYM_HOME}"
        )
        listed = await self._docker("exec", self.info.agent, "bash", "-c", script, timeout=300)
        before, marker, after = listed.stdout.partition(_LISTING_MARK)
        if not marker:
            raise EnvironmentUnavailable(f"could not list {GYM_HOME} around the deletion: {listed.stdout[-300:]}")
        before_names, after_names = before.split(), after.split()
        record["deleted"] = sorted(set(before_names) - set(after_names))
        record["kept"] = sorted(after_names)
        unexpected = sorted(set(after_names) - set(HIDE_KEEP))
        if unexpected:
            raise EnvironmentUnavailable(f"{GYM_HOME} still holds {unexpected} after the deletion")

    async def _check_python_execute(self, record: dict[str, Any]) -> None:
        probe = await self._docker(
            "exec", "-w", WORKSPACE, *self._agent_exec(), self.info.agent, *_kill_after(120), *self._agent_prefix(),
            "uv", "run", "--directory", WORKSPACE, "python", "-c", "import sys; print(sys.prefix)",
            timeout=150, check=False,
        )
        lines = probe.stdout.strip().splitlines()
        record["python_execute_prefix"] = lines[-1] if lines else ""
        if probe.returncode != 0 or record["python_execute_prefix"] != f"{GYM_HOME}/.venv":
            raise EnvironmentUnavailable(
                f"python_execute's `uv run` no longer finds {GYM_HOME}/.venv (exit {probe.returncode}): "
                f"{(probe.stdout + probe.stderr)[-400:]}"
            )

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

    def _gym_env_args(self) -> list[str]:
        """``-e`` flags that give the gym's own preprocess and grader eigent's credentials (``agent_db_readonly``)."""
        if not self.agent_db_readonly:
            return []
        return [item for key, value in _pg_env(GYM_PG_WRITE).items() for item in ("-e", f"{key}={value}")]

    def _agent_exec(self) -> list[str]:
        """``docker exec`` options and command prefix for a process that runs the agent's code."""
        if not self.agent_db_readonly:
            return []
        return ["-e", f"HOME={AGENT_HOME}"]

    def _agent_prefix(self) -> list[str]:
        return list(AGENT_EXEC) if self.agent_db_readonly else []

    async def _psql(self, sql: str, *, timeout: float = 120.0) -> CommandResult:
        """Run SQL as eigent over the Postgres container's local socket (trust there; the agent cannot reach it)."""
        return await self._docker(
            "exec", "-i", self.info.postgres, "psql", "-q", "-U", "eigent", "-d", "toolathlon_gym", "-v", "ON_ERROR_STOP=1",
            stdin=sql, timeout=timeout,
        )

    async def _lock_database(self) -> None:
        """``agent_db_readonly``: create the agent and snowflake roles, then give eigent this run's secret password.

        Also creates the agent's uid in the agent container. Any failure raises ``EnvironmentUnavailable``.
        """
        mark = time.perf_counter()
        try:
            await self._psql(lock_sql(self._write_secret, self._sf_secret))
            await self._docker(
                "exec", self.info.agent, "bash", "-c",
                f"getent passwd {AGENT_UID} >/dev/null || useradd -M -o -u {AGENT_UID} -d {AGENT_HOME} -s /bin/bash evcarry_agent",
                timeout=60,
            )
        except Exception as exc:  # noqa: BLE001 - the run must not start without the guard it asked for
            raise EnvironmentUnavailable(f"agent_db_readonly setup for {self.task.name} failed: {exc}"[:1000]) from exc
        self.info.db_guard = {
            "agent_role": AGENT_PG["PGUSER"], "agent_uid": AGENT_UID, "agent_caps": AGENT_CAPS,
            "sf_role": SF_ROLE, "sf_servers": sorted(SF_SERVERS), "agent_servers": sorted(AGENT_SERVERS),
            "locked": True, "unlocked": False,
        }
        self.info.timings["db_lock_s"] = time.perf_counter() - mark

    async def _unlock_database(self) -> None:
        """Before grading: eigent's password back to the one the gym's graders have written into them."""
        if not self.agent_db_readonly or not (self.info.db_guard or {}).get("locked"):
            return
        await self._psql(f"ALTER ROLE eigent PASSWORD '{GYM_PG_WRITE['PGPASSWORD']}';")
        self.info.db_guard["unlocked"] = True

    def _server_env(self, name: str) -> tuple[dict[str, str] | None, tuple[str, ...]]:
        """``(env_override, prefix)`` for one MCP server under ``agent_db_readonly``."""
        if not self.agent_db_readonly:
            return None, ()
        if name in AGENT_SERVERS:
            return {"HOME": AGENT_HOME}, AGENT_EXEC
        if name in SF_SERVERS:
            return _pg_env({"PGUSER": SF_ROLE, "PGPASSWORD": self._sf_secret}), ()
        return _pg_env({"PGUSER": GYM_PG_WRITE["PGUSER"], "PGPASSWORD": self._write_secret}), ()

    def _module(self, part: str) -> str:
        return f"tasks.finalpool.{self.task.name}.{part}.main"

    async def _preprocess(self) -> dict[str, Any] | None:
        """Run the task's preprocess. A failure is an environment error: the task state is unknown.

        (The gym's own runner logs the failure and carries on.)
        """
        if not (self.task.root / "preprocess" / "main.py").is_file():
            return None
        launch = " ".join(self.info.launch_time.split()[:2])  # the gym drops the weekday here
        limit = int(self.setup_timeout)
        result = await self._docker(
            "exec", "-w", GYM_HOME, *self._gym_env_args(), self.info.agent, *_kill_after(limit), PYTHON_BIN, "-m", self._module("preprocess"),
            "--agent_workspace", WORKSPACE, "--launch_time", launch,
            timeout=limit + 30, check=False,
        )
        (self.log_dir / "preprocess.log").write_text(result.stdout + "\n--- stderr\n" + result.stderr, encoding="utf-8")
        if result.returncode != 0 or result.timed_out:
            raise EnvironmentUnavailable(
                f"preprocess for {self.task.name} failed (exit {result.returncode}): {result.stderr.strip()[-300:]}"
            )
        return result.as_dict(limit=2000)

    async def _connect_servers(self) -> None:
        unknown = [name for name in self.task.mcp_servers if name not in self.server_specs]
        if unknown:
            raise EnvironmentUnavailable(f"No gym MCP config for servers {unknown}")
        for name in self.task.mcp_servers:
            spec = self.server_specs[name]
            override, prefix = self._server_env(name)
            cwd, args = server_exec_args(
                spec, container=self.info.agent, container_env=self.container_env, task_dir=self.task_dir,
                env_override=override, prefix=prefix,
            )
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
        """Close the MCP sessions, then remove everything labelled with this run's id.

        Removal runs in a worker thread (it does not stall parallel runs) and is
        shielded: if this task is cancelled again, the removal still completes.
        """
        mark = time.perf_counter()
        for session in self.sessions.values():
            await session.close()
        self.sessions.clear()
        await asyncio.shield(asyncio.to_thread(remove_run_resources, self.info.run_id, self.docker))
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
        """Same contract and output format as the gym's python_execute local tool.

        It runs as root in the agent container, as the gym's tool does; so does the gym's terminal
        server, whose ``ALLOWED_DIR`` check is a string filter that ``..``, ``find`` or ``python -c``
        get past. Running them as a non-root user is an owner decision and is not done here.
        With ``hide_task_dir`` and ``network_internal`` both on, these root tools find nothing outside
        the workspace but the image's own code and runtime (``/opt/local_servers``, ``/opt/venv``, uv's
        interpreters, the kept ``/workspace/{pyproject.toml,uv.lock,.venv}``) and the moved mock sites
        under ``MOCK_ROOT``, which the task's localhost URLs serve anyway; and they have no route out.
        Without ``agent_db_readonly`` the container environment carries eigent's credentials, so direct
        SQL writes to the run's own database (the one graders read) are possible, in every arm alike.
        With it (plan A20) this runs as ``AGENT_UID`` with the read-only role: reads work, writes are denied.
        """
        code = str(arguments.get("code", ""))
        try:
            timeout = min(max(int(arguments.get("timeout") or 30), 1), 120)
        except (TypeError, ValueError):
            timeout = 30
        filename = str(arguments.get("filename") or f"{uuid.uuid4()}.py")
        filename = os.path.basename(filename)
        if not filename.endswith(".py"):
            filename += ".py"
        target = f"{WORKSPACE}/.python_tmp/{filename}"
        await self._docker(
            "exec", "-i", *self._agent_exec(), self.info.agent, *self._agent_prefix(),
            "bash", "-c", f"mkdir -p {WORKSPACE}/.python_tmp && cat > {shlex.quote(target)}",
            stdin=code, timeout=60,
        )
        result = await self._docker(
            "exec", "-w", WORKSPACE, *self._agent_exec(), self.info.agent, *_kill_after(timeout), *self._agent_prefix(),
            "uv", "run", "--directory", WORKSPACE, f"./.python_tmp/{filename}",
            timeout=timeout + 15, check=False,
        )
        if result.timed_out or result.returncode == _KILLED:
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
        await self._unseal()
        await self._unlock_database()
        limit = int(self.eval_timeout)
        result = await self._docker(
            "exec", "-w", GYM_HOME, *self._gym_env_args(), self.info.agent, *_kill_after(limit), PYTHON_BIN, "-m", self._module("evaluation"),
            "--res_log_file", result_file,
            "--agent_workspace", WORKSPACE,
            "--groundtruth_workspace", f"tasks/finalpool/{self.task.name}/groundtruth_workspace",
            "--launch_time", self.info.launch_time,
            timeout=limit + 30, check=False,
        )
        result.timed_out = result.timed_out or result.returncode == _KILLED
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


_KILLED = 137  # exit status of a command that `timeout -s KILL` stopped

# Tool arguments that reach for answers: removed task files, the gym's history, or its public repo
# (github.com/eigent-ai/...). Not "toolathlon": the gym's own Postgres database is named toolathlon_gym.
GROUND_TRUTH_MARKERS = (
    "groundtruth", "ground_truth", "/workspace/tasks", "tasks/finalpool", "/workspace/.git",
    "eigent-ai", "evaluation/main",
)


def ground_truth_suspects(ledger: Any) -> list[dict[str, str]]:
    """Ledger calls whose arguments mention a ground-truth marker. Recorded in run_end for review."""
    found = []
    for entry in getattr(ledger, "entries", {}).values():
        text = json.dumps(entry.arguments, ensure_ascii=False, default=str).lower()
        hits = [marker for marker in GROUND_TRUTH_MARKERS if marker in text]
        if hits:
            found.append({"call_id": entry.call_id, "tool": entry.tool_name, "markers": ",".join(hits)})
    return found


def _kill_after(seconds: int) -> list[str]:
    """Prefix that kills the command inside the container at ``seconds``.

    Killing the host's ``docker exec`` client would leave the process running in the container.
    """
    return ["timeout", "-s", "KILL", str(max(1, seconds))]


# -- hide_task_dir: the tasks' mock web servers -------------------------------------------------

_LISTING_MARK = "==evcarry-after-deletion=="
MAX_PROBE_PATHS = 20  # per moved server: "/" plus the first localhost paths the task text names on its port
_TASK_URL = re.compile(r"https?://(?:localhost|127\.0\.0\.1):(\d+)(/[^\s\"'<>`)\]]*)?")


def task_url_paths(task_text: str) -> dict[int, list[str]]:
    """Localhost paths per port named in the task text, in first-seen order (fragment and trailing punctuation dropped)."""
    found: dict[int, list[str]] = {}
    for match in _TASK_URL.finditer(task_text or ""):
        path = (match.group(2) or "/").split("#", 1)[0].rstrip(".,;:!?") or "/"
        paths = found.setdefault(int(match.group(1)), [])
        if path not in paths:
            paths.append(path)
    return found


def _http_server_args(argv: list[str]) -> tuple[int, int | None] | None:
    """``(port, index of the --directory value or None)`` for a ``python -m http.server`` argv, else None.

    Raises ValueError for an argv this cannot read, so the run stops instead of guessing.
    """
    try:
        at = argv.index("http.server")
    except ValueError:
        return None
    if at == 0 or argv[at - 1] != "-m":
        return None
    port, directory = 8000, None
    index = at + 1
    while index < len(argv):
        token = argv[index]
        if token in ("-d", "--directory"):
            if index + 1 >= len(argv):
                raise ValueError(f"{token} without a value in {argv}")
            directory = index + 1
            index += 2
        elif token.startswith("--directory="):
            directory = index
            index += 1
        elif token in ("-b", "--bind", "-p", "--protocol"):
            index += 2
        elif token.startswith("-"):  # --cgi, --bind=..., --protocol=...
            index += 1
        elif token.isdigit():
            port = int(token)
            index += 1
        else:
            raise ValueError(f"cannot read the http.server arguments {argv}")
    return port, directory


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def plan_mock_relocations(
    processes: list[dict[str, Any]],
    *,
    task_text: str,
    gym_home: str = GYM_HOME,
    keep_roots: tuple[str, ...] = (WORKSPACE, RUN_DIR),
    target_root: str = MOCK_ROOT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Which running ``http.server`` processes serve a directory that ``hide_task_dir`` deletes, and how to move each.

    ``processes`` is ``_FIND_HTTP_SERVERS``'s output (``pid``, ``cwd``, ``argv`` of every process with an
    ``http.server`` argument). The served root is ``--directory`` (relative to the process's cwd) or the
    cwd itself (one task starts it with ``cwd=``). Returns ``(plan, left)``: per server to move, its pid,
    port, source, target, log file, the argv to restart it with (``--directory`` pointed at the target;
    the restart's cwd is the target too) and the paths to compare before and after; and the processes
    left where they are. Raises ValueError for an argv it cannot read or two servers on one port.
    """
    urls = task_url_paths(task_text)
    plan: list[dict[str, Any]] = []
    left: list[dict[str, Any]] = []
    for process in processes:
        argv = [str(item) for item in process["argv"]]
        pid, cwd = int(process["pid"]), str(process["cwd"])
        parsed = _http_server_args(argv)
        if parsed is None:
            left.append({"pid": pid, "argv": argv, "reason": "not `python -m http.server`"})
            continue
        port, directory = parsed
        if directory is None:
            root = cwd
        else:
            value = argv[directory].split("=", 1)[1] if argv[directory].startswith("--directory=") else argv[directory]
            root = value if posixpath.isabs(value) else posixpath.join(cwd, value)
        root = posixpath.normpath(root)
        if root == posixpath.normpath(gym_home):
            raise ValueError(f"the http.server on port {port} serves all of {gym_home}")
        if not _under(root, gym_home) or any(_under(root, keep) for keep in keep_roots):
            left.append({"pid": pid, "port": port, "root": root, "reason": "its root is not deleted"})
            continue
        if any(item["port"] == port for item in plan):
            raise ValueError(f"two http.server processes on port {port}")
        target = f"{target_root}/{port}"
        restart = list(argv)
        if directory is not None:
            restart[directory] = f"--directory={target}" if argv[directory].startswith("--directory=") else target
        paths = ["/", *(path for path in urls.get(port, []) if path != "/")][:MAX_PROBE_PATHS]
        plan.append(
            {
                "pid": pid,
                "port": port,
                "source": root,
                "target": target,
                "log": f"{target_root}/{port}.log",
                "argv": restart,
                "paths": paths,
            }
        )
    return plan, left


# Run in the agent container with PYTHON_BIN (Python 3.10). Prints a JSON list of the processes whose
# argv has an "http.server" element; this script's own argv holds that text only inside one element.
# Zombies (killed servers whose parent is PID 1 `sleep`) have an empty cmdline and no cwd.
_FIND_HTTP_SERVERS = r"""
import json, os
found = []
for pid in os.listdir("/proc"):
    if not pid.isdigit():
        continue
    try:
        with open("/proc/%s/cmdline" % pid, "rb") as handle:
            raw = handle.read()
        cwd = os.readlink("/proc/%s/cwd" % pid)
    except OSError:
        continue
    argv = [part.decode("utf-8", "replace") for part in raw.split(b"\0")]
    if argv and argv[-1] == "":
        argv.pop()
    if "http.server" in argv:
        found.append({"pid": int(pid), "cwd": cwd, "argv": argv})
print(json.dumps(found))
"""

# Run in the agent container with PYTHON_BIN; reads plan_mock_relocations' plan on stdin. Per server:
# probe every path (status, sha256 of the body), copy the root, stop the old server, start the new one
# detached (own session, output to its log, so it outlives this `docker exec` the way preprocess's
# `nohup ... &` servers do), wait for the port, probe again. Stops at the first failure; exit 1 then.
_MOVE_HTTP_SERVERS = r"""
import hashlib, json, os, shutil, signal, socket, subprocess, sys, time, urllib.error, urllib.request

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never a proxy, even if one is set

def probe(port, path):
    url = "http://127.0.0.1:%d%s" % (port, path)
    try:
        with OPENER.open(url, timeout=5) as response:
            return [response.status, hashlib.sha256(response.read()).hexdigest()]
    except urllib.error.HTTPError as error:
        return [error.code, hashlib.sha256(error.read()).hexdigest()]
    except Exception as error:
        return [0, type(error).__name__]

def listening(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0

def wait(port, up, seconds):
    deadline = time.monotonic() + seconds
    while listening(port) != up:
        if time.monotonic() > deadline:
            return False
        time.sleep(0.1)
    return True

def move(item):
    port, paths = item["port"], item["paths"]
    entry = {key: item[key] for key in ("pid", "port", "source", "target", "paths")}
    entry["before"] = [probe(port, path) for path in paths]
    if entry["before"][0][0] == 0:
        entry["error"] = "the server did not answer before the move"
        return entry
    os.makedirs(os.path.dirname(item["target"]), exist_ok=True)
    shutil.copytree(item["source"], item["target"], ignore_dangling_symlinks=True)
    os.kill(item["pid"], signal.SIGTERM)
    if not wait(port, False, 10):
        try:
            os.kill(item["pid"], signal.SIGKILL)
        except ProcessLookupError:
            pass
        if not wait(port, False, 5):
            entry["error"] = "the old server kept the port"
            return entry
    with open(item["log"], "ab") as log:
        subprocess.Popen(
            item["argv"], cwd=item["target"], stdin=subprocess.DEVNULL, stdout=log, stderr=log,
            start_new_session=True,
        )
    if not wait(port, True, 15):
        entry["error"] = "the restarted server did not open the port"
        return entry
    entry["after"] = [probe(port, path) for path in paths]
    if entry["after"] != entry["before"]:
        entry["error"] = "the answers differ after the move"
    return entry

report = []
for item in json.load(sys.stdin):
    try:
        entry = move(item)
    except Exception as error:
        entry = {"port": item.get("port"), "source": item.get("source"), "error": "%s: %s" % (type(error).__name__, error)}
    report.append(entry)
    if "error" in entry:
        break
ok = bool(report) and all("error" not in entry for entry in report)
print(json.dumps({"ok": ok, "servers": report}))
sys.exit(0 if ok else 1)
"""


def remove_run_resources(run_id: str, docker: str) -> None:
    """Remove the containers (with their anonymous volumes) and the network of one run."""
    selector = f"label=evcarry.run={run_id}"
    listed = subprocess.run(
        [docker, "ps", "-a", "--filter", selector, "--format", "{{.Names}}"], capture_output=True, text=True, check=False
    )
    containers = listed.stdout.split()
    if containers:
        subprocess.run([docker, "rm", "-fv", *containers], capture_output=True, check=False, timeout=180)
    networks = subprocess.run(
        [docker, "network", "ls", "--filter", selector, "--format", "{{.Name}}"], capture_output=True, text=True, check=False
    ).stdout.split()
    for network in networks:
        subprocess.run([docker, "network", "rm", network], capture_output=True, check=False, timeout=60)


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
