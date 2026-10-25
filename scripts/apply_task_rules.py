"""Apply the pre-registered task rules (``evcarry.env.task_rules``) to the frozen task lists.

Starts from the original stratified sample, regenerated in memory with
``scripts/freeze_task_lists.py`` (seed 20261024), so the result does not depend
on whatever the list files hold now. Then:

1. swaps every subset70 task with an email defect for a pool160 task (rule in
   ``task_rules``), and gives pilot10 the same replacements;
2. writes ``tasks/subset70.txt``, ``tasks/pilot10.txt``, ``tasks/SHA256SUMS``,
   the swap table ``tasks/swaps.csv``, the email audit ``tasks/email_audit.csv``
   and the delegation labels ``tasks/delegation_labels.csv``;
3. writes the pre-registered split (2026-10-25): ``tasks/headline60.txt`` (subset70
   minus pilot10, the primary headline) and ``tasks/dev.txt`` (the fix-verification
   tasks, ``task_rules.dev_tasks``);
4. records the rule versions and the pre-swap hashes in ``tasks/FROZEN.txt``.

``--check`` writes nothing and exits 1 if any output would change.
pool160 is not modified: it is the source of replacements.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import sys
from pathlib import Path

from evcarry.env.task_rules import (
    DELEGATION_MIN_SERVERS,
    DEV_COUNT,
    DEV_SALT,
    DEV_TOOL_LIMIT,
    RULES_VERSION,
    SPLIT_VERSION,
    TOOL_LIMIT,
    TaskFacts,
    apply_swaps,
    dev_tasks,
    headline_tasks,
    needs_delegation,
    plan_swaps,
    swap_key,
    task_facts,
)
from evcarry.env.toolathlon import gym_task_root

ROOT = Path(__file__).resolve().parents[1]


def _freeze_module():
    spec = importlib.util.spec_from_file_location("freeze_task_lists", ROOT / "scripts" / "freeze_task_lists.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def original_lists() -> dict[str, list[str]]:
    """pool160, subset70, pilot10 exactly as freeze_task_lists.py drew them."""
    freeze = _freeze_module()
    gym = gym_task_root()
    dirs = freeze.discover(gym)
    by_name = {path.name: path for path in dirs}
    pool = freeze.sample(freeze.group_tasks(dirs), 160, freeze.SEED)
    subset = freeze.sample(freeze.group_tasks([by_name[name] for name in pool]), 70, freeze.SEED)
    pilot = freeze.sample(freeze.group_tasks([by_name[name] for name in subset]), 10, freeze.SEED)
    return {"pool160": pool, "subset70": subset, "pilot10": pilot}


def original_bytes(name: str, names: list[str], label: str) -> bytes:
    freeze = _freeze_module()
    lines = [f"# frozen stratified sample seed {freeze.SEED}", f"# {label}", f"# gym {freeze.GYM_COMMIT}", *names, ""]
    return "\n".join(lines).encode("utf-8")


_LABELS = {
    "pool160": "pool 160 of all tasks",
    "subset70": "subset 70 restratified from the pool",
    "pilot10": "pilot 10 restratified from the subset",
}


def list_bytes(name: str, names: list[str], swapped: int) -> bytes:
    freeze = _freeze_module()
    lines = [f"# frozen stratified sample seed {freeze.SEED}", f"# {_LABELS[name]}", f"# gym {freeze.GYM_COMMIT}"]
    if swapped:
        lines.append(f"# {swapped} email-defect tasks swapped by {RULES_VERSION}; see docs/TASK_RULES.md")
    return "\n".join([*lines, *names, ""]).encode("utf-8")


def headline_bytes(names: list[str]) -> bytes:
    freeze = _freeze_module()
    lines = [
        f"# frozen stratified sample seed {freeze.SEED}",
        f"# headline 60: subset70 minus pilot10 ({SPLIT_VERSION}; the primary analysis)",
        f"# gym {freeze.GYM_COMMIT}",
    ]
    return "\n".join([*lines, *names, ""]).encode("utf-8")


def dev_bytes(names: list[str]) -> bytes:
    freeze = _freeze_module()
    lines = [
        f"# dev tasks: {DEV_SALT} over pool160 minus subset70 (no email defect, at most {DEV_TOOL_LIMIT} SA tools, "
        "grader runs in the grading Python [A13, 2026-10-25]), lowest sha256 first",
        f"# gym {freeze.GYM_COMMIT}",
    ]
    return "\n".join([*lines, *names, ""]).encode("utf-8")


def _csv(header: list[str], rows: list[list[object]]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def outputs() -> tuple[dict[str, bytes], list]:
    original = original_lists()
    names = sorted(path.name for path in gym_task_root().iterdir() if (path / "task_config.json").is_file())
    facts: dict[str, TaskFacts] = {name: task_facts(name) for name in names}
    swaps = plan_swaps(original["subset70"], original["pool160"], facts)
    subset = apply_swaps(original["subset70"], swaps)
    pilot = apply_swaps(original["pilot10"], swaps)
    pilot_swapped = sum(1 for swap in swaps if swap.old in original["pilot10"])
    files: dict[str, bytes] = {
        "tasks/subset70.txt": list_bytes("subset70", subset, len(swaps)),
        "tasks/pilot10.txt": list_bytes("pilot10", pilot, pilot_swapped),
        "tasks/headline60.txt": headline_bytes(headline_tasks(subset, pilot)),
        "tasks/dev.txt": dev_bytes(dev_tasks(original["pool160"], subset, pilot, facts)),
    }
    pool_bytes = (ROOT / "tasks" / "pool160.txt").read_bytes()
    assert pool_bytes == original_bytes("pool160", original["pool160"], _LABELS["pool160"]), "pool160 is not the frozen sample"
    sums = [
        f"{hashlib.sha256(pool_bytes).hexdigest()}  tasks/pool160.txt",
        f"{hashlib.sha256(files['tasks/subset70.txt']).hexdigest()}  tasks/subset70.txt",
        f"{hashlib.sha256(files['tasks/pilot10.txt']).hexdigest()}  tasks/pilot10.txt",
        f"{hashlib.sha256(files['tasks/headline60.txt']).hexdigest()}  tasks/headline60.txt",
        f"{hashlib.sha256(files['tasks/dev.txt']).hexdigest()}  tasks/dev.txt",
    ]
    files["tasks/SHA256SUMS"] = ("\n".join(sums) + "\n").encode("utf-8")
    files["tasks/swaps.csv"] = _csv(
        ["list", "old", "new", "reason", "tier", "old_stratum", "new_stratum", "new_sa_tools", "new_swap_key"],
        [
            [
                "subset70+pilot10" if swap.old in original["pilot10"] else "subset70",
                swap.old,
                swap.new,
                swap.reason,
                swap.tier,
                facts[swap.old].stratum,
                facts[swap.new].stratum,
                facts[swap.new].sa_tools,
                swap_key(swap.new)[:16],
            ]
            for swap in swaps
        ],
    )
    membership = {name: [name in original["pool160"], name in subset, name in pilot] for name in names}
    files["tasks/email_audit.csv"] = _csv(
        ["task", "in_pool160", "in_subset70", "in_pilot10", "sender", "required_senders", "email_defect"],
        [
            [name, *membership[name], fact.email_sender or "", " ".join(fact.required_senders), fact.email_defect or ""]
            for name, fact in facts.items()
            if fact.uses_email
        ],
    )
    files["tasks/delegation_labels.csv"] = _csv(
        [
            "task", "in_pool160", "in_subset70", "in_pilot10", "domain", "n_servers", "sa_tools",
            f"servers_ge_{DELEGATION_MIN_SERVERS}", f"sa_tools_gt_{TOOL_LIMIT}", "needs_delegation",
        ],
        [
            [
                name, *membership[name], fact.domain, fact.n_servers, fact.sa_tools,
                fact.n_servers >= DELEGATION_MIN_SERVERS, fact.sa_tools > TOOL_LIMIT, needs_delegation(fact),
            ]
            for name, fact in facts.items()
        ],
    )
    frozen = (ROOT / "tasks" / "FROZEN.txt").read_text(encoding="utf-8")
    generated = ("rules=", "pre_swap_sha256_", "swapped_", "split=", "dev_rule=")
    kept = [line for line in frozen.splitlines() if line and not line.startswith(generated)]
    kept += [
        f"rules={RULES_VERSION} (src/evcarry/env/task_rules.py, scripts/apply_task_rules.py, docs/TASK_RULES.md)",
        f"swapped_subset70={len(swaps)}",
        f"swapped_pilot10={pilot_swapped}",
        "pre_swap_sha256_subset70="
        + hashlib.sha256(original_bytes("subset70", original["subset70"], _LABELS["subset70"])).hexdigest(),
        "pre_swap_sha256_pilot10="
        + hashlib.sha256(original_bytes("pilot10", original["pilot10"], _LABELS["pilot10"])).hexdigest(),
        f"split={SPLIT_VERSION} (primary headline = subset70 minus pilot10, tasks/headline60.txt; "
        "pilot10 reported separately; fixes verified on tasks/dev.txt)",
        f"dev_rule={DEV_SALT}: pool160 minus subset70, no email defect, at most {DEV_TOOL_LIMIT} SA tools, "
        "a grader whose imports the grading Python has (A13, 2026-10-25), "
        f'the first {DEV_COUNT} by sha256("{DEV_SALT}|" + task)',
    ]
    files["tasks/FROZEN.txt"] = ("\n".join(kept) + "\n").encode("utf-8")
    return files, swaps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="write nothing; exit 1 if an output would change")
    args = parser.parse_args(argv)
    files, swaps = outputs()
    changed = [name for name, data in files.items() if not (ROOT / name).is_file() or (ROOT / name).read_bytes() != data]
    for swap in swaps:
        print(f"{swap.old} -> {swap.new} ({swap.reason}; {swap.tier})")
    if args.check:
        print("up to date" if not changed else f"would change: {changed}")
        return 1 if changed else 0
    for name in changed:
        (ROOT / name).write_bytes(files[name])
    print(f"wrote {changed or 'nothing (up to date)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
