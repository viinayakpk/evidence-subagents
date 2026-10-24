"""Regenerate the frozen Toolathlon-Gym task lists.

The committed files are the sample. Do not re-run this after the lists are
hashed unless the gym commit changes. Python's ``random`` is stable for a
given seed; the SHA256 of the list files is what later runs check.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

SEED = 20261024
GYM_COMMIT = "ed735ba0af71d82688a55f952a6ea11479ae7206"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def discover(root: Path) -> list[Path]:
    found = []
    for path in sorted(root.iterdir()):
        if path.is_dir() and (path / "task_config.json").is_file():
            found.append(path)
    return found


def stratum(task_dir: Path) -> str:
    config = json.loads((task_dir / "task_config.json").read_text(encoding="utf-8"))
    domain = task_dir.name.split("-", 1)[0]
    servers = config.get("needed_mcp_servers") or []
    return f"{domain}|{len(servers)}"


def allocate(groups: dict[str, list[str]], target: int) -> dict[str, int]:
    total = sum(len(names) for names in groups.values())
    floors: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for key, names in groups.items():
        quota = len(names) * target / total
        floors[key] = int(quota)
        remainders.append((quota - floors[key], key))
    left = target - sum(floors.values())
    remainders.sort(key=lambda item: (-item[0], item[1]))
    for index in range(left):
        floors[remainders[index][1]] += 1
    return floors


def sample(names_by_stratum: dict[str, list[str]], target: int, seed: int) -> list[str]:
    counts = allocate(names_by_stratum, target)
    rng = random.Random(seed)
    chosen: list[str] = []
    for key in sorted(names_by_stratum):
        names = sorted(names_by_stratum[key])
        rng.shuffle(names)
        chosen.extend(names[: counts[key]])
    return sorted(chosen)


def group_tasks(task_dirs: list[Path]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for task_dir in task_dirs:
        groups.setdefault(stratum(task_dir), []).append(task_dir.name)
    return groups


def write_list(path: Path, names: list[str], label: str) -> None:
    lines = [
        f"# frozen stratified sample seed {SEED}",
        f"# {label}",
        f"# gym {GYM_COMMIT}",
        *names,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    root = repo_root()
    gym = root / "third_party" / "toolathlon_gym" / "tasks" / "finalpool"
    groups = group_tasks(discover(gym))
    pool = sample(groups, 160, SEED)
    pool_groups: dict[str, list[str]] = {}
    by_name = {path.name: path for path in discover(gym)}
    for name in pool:
        pool_groups.setdefault(stratum(by_name[name]), []).append(name)
    subset = sample(pool_groups, 70, SEED)
    subset_groups: dict[str, list[str]] = {}
    for name in subset:
        subset_groups.setdefault(stratum(by_name[name]), []).append(name)
    pilot = sample(subset_groups, 10, SEED)
    tasks = root / "tasks"
    tasks.mkdir(exist_ok=True)
    write_list(tasks / "pool160.txt", pool, "pool 160 of all tasks")
    write_list(tasks / "subset70.txt", subset, "subset 70 restratified from the pool")
    write_list(tasks / "pilot10.txt", pilot, "pilot 10 restratified from the subset")
    frozen = tasks / "FROZEN.txt"
    frozen.write_text(
        "\n".join(
            [
                "gym_repo=https://github.com/eigent-ai/toolathlon_gym.git",
                f"gym_commit={GYM_COMMIT}",
                f"seed={SEED}",
                "pool=160",
                "subset=70",
                "pilot=10",
                "method=largest-remainder then Random(seed).shuffle within sorted stratum names",
                "",
            ]
        ),
        encoding="utf-8",
    )
    sums = []
    for name in ("pool160.txt", "subset70.txt", "pilot10.txt"):
        digest = hashlib.sha256((tasks / name).read_bytes()).hexdigest()
        sums.append(f"{digest}  tasks/{name}")
    (tasks / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    print(f"pool {len(pool)} subset {len(subset)} pilot {len(pilot)}")


if __name__ == "__main__":
    main()
