"""Load Toolathlon-Gym task directories without copying gym agent code."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from evcarry.paths import repo_root


@dataclass
class TaskSpec:
    name: str
    root: Path
    mcp_servers: list[str]
    local_eval: bool
    task_text: str
    agent_prompt_sha256: str | None
    domain: str
    stratum: str


def load_task_list(path: Path) -> list[str]:
    """Task names, skipping blank lines and ``#`` comments. Hash the file bytes separately."""
    names: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        names.append(stripped)
    return names


def gym_task_root() -> Path:
    return repo_root() / "third_party" / "toolathlon_gym" / "tasks" / "finalpool"


def load_task(name: str, root: Path) -> TaskSpec:
    task_dir = root / name
    config_path = task_dir / "task_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    docs = task_dir / "docs" / "task.md"
    task_text = docs.read_text(encoding="utf-8") if docs.is_file() else ""
    agent_prompt = task_dir / "docs" / "agent_system_prompt.md"
    agent_hash = None
    if agent_prompt.is_file():
        agent_hash = hashlib.sha256(agent_prompt.read_bytes()).hexdigest()
    servers = [str(item) for item in config.get("needed_mcp_servers") or []]
    domain = name.split("-", 1)[0]
    return TaskSpec(
        name=name,
        root=task_dir,
        mcp_servers=servers,
        local_eval=bool(config.get("evcarry_local_eval")),
        task_text=task_text,
        agent_prompt_sha256=agent_hash,
        domain=domain,
        stratum=f"{domain}|{len(servers)}",
    )


def prepare_workspace(task: TaskSpec, dest: Path) -> Path:
    """Copy ``initial_workspace`` only. Gym preprocess needs Postgres and is not run."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    initial = task.root / "initial_workspace"
    if initial.is_dir():
        for item in initial.iterdir():
            target = dest / item.name
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)
    return dest
