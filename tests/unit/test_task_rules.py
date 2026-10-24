"""Pre-registered task rules: email defects, swaps, delegation labels. No Docker, no model."""

import csv
import importlib.util
import json
from pathlib import Path

import pytest

from evcarry.env.task_rules import (
    EMAIL_CONFIG_WITHOUT_PASSWORD,
    UNPASSABLE_SENDER_CHECK,
    TaskFacts,
    dev_key,
    dev_tasks,
    email_sender,
    headline_tasks,
    needs_delegation,
    plan_swaps,
    required_senders,
    swap_key,
    task_facts,
)
from evcarry.env.toolathlon import load_task_list
from evcarry.paths import repo_root

ROOT = repo_root()


def _evaluation(tmp_path: Path, code: str) -> Path:
    task = tmp_path / "task"
    (task / "evaluation").mkdir(parents=True)
    (task / "evaluation" / "main.py").write_text(code, encoding="utf-8")
    return task


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ('ok = "a@x.com" in from_str\n', {"a@x.com"}),
        ('ok = "a@x.com" in (from_addr or "").lower()\n', {"a@x.com"}),
        ('check("Email from a@x.com",\n      "a@x.com" in (e[1] or "").lower())\n', {"a@x.com"}),
        ('if "a@x.com" not in from_str:\n    errors.append(1)\n', {"a@x.com"}),
        ('ok = str(frm or "").strip().lower() == "a@x.com"\n', {"a@x.com"}),
        ('target_from = "a@x.com"\nok = target_from in (from_addr or "")\n', {"a@x.com"}),
        ('ok = any("a@x.com" in f for f in all_from)\n', {"a@x.com"}),
        ('q = "SELECT * FROM email.messages WHERE from_addr = \'a@x.com\'"\n', {"a@x.com"}),
        # Not a sender requirement:
        ('ok = "b@x.com" in to_str and "a@x.com" in from_str\n', {"a@x.com"}),
        ("q = \"SELECT 1 FROM email.messages WHERE from_addr NOT ILIKE '%a@x.com%'\"\n", set()),
        ("q = \"SELECT 1 FROM email.messages WHERE from_addr != 'a@x.com'\"\n", set()),
        ('# Find emails sent from a@x.com\nok = "b@x.com" in to_str\n', set()),
        ('target_from = "a@x.com"\nok = "b@x.com" in to_str\n', set()),
    ],
)
def test_required_senders_detector(tmp_path: Path, code: str, expected: set) -> None:
    assert required_senders(_evaluation(tmp_path, code)) == expected


def test_email_sender_from_config(tmp_path: Path) -> None:
    assert email_sender(tmp_path) == ("user@example.com", None)  # no config: the server's default
    (tmp_path / "email_config.json").write_text(json.dumps({"email": "A@x.com", "password": "p"}), encoding="utf-8")
    assert email_sender(tmp_path) == ("a@x.com", None)
    (tmp_path / "email_config.json").write_text(json.dumps([{"email": "a@x.com"}]), encoding="utf-8")
    assert email_sender(tmp_path) == ("a@x.com", EMAIL_CONFIG_WITHOUT_PASSWORD)


# Verdicts checked by hand on every pool160 task that needs the emails server (docs/TASK_RULES.md):
# each listed evaluation fails the whole task when its sender check fails.
HAND_CHECKED_SENDER = {
    "canvas-assignment-calendar",
    "canvas-quiz-analysis-email",
    "playwright-canvas-curriculum-word-notion",
    "scholarly-arxiv-cross-reference",
    "sf-hr-mentorship-excel-gcal-email",
    "sf-sla-breach-audit",
    "sf-ticket-resolution-notion",
    "terminal-canvas-gform-gcal-excel-email",
    "yf-dividend-tracker-gsheet",
}
HAND_CHECKED_CONFIG = {
    "arxiv-reading-plan-excel-gcal-email",
    "yf-portfolio-analysis-excel-word-email",
    "yf-sector-comparison-gform-excel-email",
}


def test_detector_matches_the_hand_review_on_pool160() -> None:
    pool = load_task_list(ROOT / "tasks" / "pool160.txt")
    facts = [task_facts(name) for name in pool]
    assert {f.name for f in facts if f.email_defect == UNPASSABLE_SENDER_CHECK} == HAND_CHECKED_SENDER
    assert {f.name for f in facts if f.email_defect == EMAIL_CONFIG_WITHOUT_PASSWORD} == HAND_CHECKED_CONFIG
    sla = task_facts("sf-sla-breach-audit")
    assert sla.email_sender == "user@example.com" and "sla-audit@company.com" in sla.required_senders
    # Sender required and configured: passable.
    assert task_facts("canvas-module-completion-notion-gcal").email_defect is None


def _apply_module():
    spec = importlib.util.spec_from_file_location("apply_task_rules", ROOT / "scripts" / "apply_task_rules.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_lists_are_the_seeded_sample_plus_the_rules() -> None:
    """Regenerates the original sample from the seed, applies the rules, and compares every output byte."""
    apply = _apply_module()
    assert apply.main(["--check"]) == 0
    frozen = dict(line.split("=", 1) for line in (ROOT / "tasks/FROZEN.txt").read_text(encoding="utf-8").splitlines())
    # The pre-swap hashes are the ones the lists were first frozen with.
    assert frozen["pre_swap_sha256_subset70"].startswith("0571b5ee")
    assert frozen["pre_swap_sha256_pilot10"].startswith("24721d50")


def test_lists_have_no_email_defect_and_stay_nested() -> None:
    pool = load_task_list(ROOT / "tasks" / "pool160.txt")
    subset = load_task_list(ROOT / "tasks" / "subset70.txt")
    pilot = load_task_list(ROOT / "tasks" / "pilot10.txt")
    assert len(subset) == 70 and len(pilot) == 10
    assert set(pilot) <= set(subset) <= set(pool)
    assert not [name for name in subset if task_facts(name).email_defect]
    swaps = list(csv.DictReader((ROOT / "tasks/swaps.csv").open(encoding="utf-8")))
    assert len(swaps) == 7
    for row in swaps:
        assert row["old"] not in subset and row["new"] in subset
        assert int(row["new_sa_tools"]) <= 128
    assert [row["old"] for row in swaps if row["list"] == "subset70+pilot10"] == ["sf-sla-breach-audit"]


def test_the_pre_registered_split_and_the_dev_rule() -> None:
    """P0-1 (2026-10-25): primary headline = subset70 minus pilot10; dev tasks from pool160 minus subset70, never pilot10."""
    pool = load_task_list(ROOT / "tasks" / "pool160.txt")
    subset = load_task_list(ROOT / "tasks" / "subset70.txt")
    pilot = load_task_list(ROOT / "tasks" / "pilot10.txt")
    headline = load_task_list(ROOT / "tasks" / "headline60.txt")
    assert headline == headline_tasks(subset, pilot)
    assert len(headline) == 60 and not set(headline) & set(pilot) and set(headline) | set(pilot) == set(subset)
    dev = load_task_list(ROOT / "tasks" / "dev.txt")
    facts = {name: task_facts(name) for name in pool}
    assert dev == dev_tasks(pool, subset, pilot, facts) == ["train-sf-sales-trip-excel-word-email", "sf-support-metrics-pdf-email"]
    assert not set(dev) & set(subset)
    assert all(facts[name].email_defect is None and facts[name].sa_tools <= 150 for name in dev)
    assert dev_key(dev[0]) < dev_key(dev[1])
    # The rule on synthetic facts: excluded lists, defects and tool counts are skipped; the lowest keys win.
    synthetic = {
        name: _facts(name, "d", 4, tools=tools, defect=defect)
        for name, tools, defect in (
            ("in-subset", 50, None), ("in-pilot", 50, None), ("broken", 50, EMAIL_CONFIG_WITHOUT_PASSWORD),
            ("too-many-tools", 151, None), ("a", 150, None), ("b", 10, None), ("c", 10, None),
        )
    }
    chosen = dev_tasks(list(synthetic), ["in-subset"], ["in-pilot"], synthetic, count=2)
    assert chosen == sorted(["a", "b", "c"], key=dev_key)[:2]


def test_the_docx_table_rule_flags_only_the_known_defect_in_subset70(tmp_path: Path) -> None:
    from evcarry.env.task_rules import docx_table_defect
    from evcarry.env.toolathlon import gym_task_root

    root = gym_task_root()
    if not (root / "sf-sales-region-analysis-gsheet-email").is_dir():
        pytest.skip("toolathlon_gym submodule is not checked out")
    subset = load_task_list(ROOT / "tasks" / "subset70.txt")
    assert [name for name in subset if docx_table_defect(root / name)] == ["sf-sales-region-analysis-gsheet-email"]
    # "vegetable" is not a table, and a grader that reads tables is fine.
    task = _evaluation(tmp_path, "doc = Document(p)\ntext = ' '.join(x.text for x in doc.paragraphs)\n")
    (task / "docs").mkdir()
    (task / "docs" / "task.md").write_text("Write Menu.docx listing each vegetable dish.\n", encoding="utf-8")
    assert not docx_table_defect(task)
    (task / "docs" / "task.md").write_text("Write Menu.docx with a table of dishes.\n", encoding="utf-8")
    assert docx_table_defect(task)
    (task / "evaluation" / "main.py").write_text("doc = Document(p)\nrows = doc.tables[0].rows\n", encoding="utf-8")
    assert not docx_table_defect(task)


def _facts(name: str, domain: str, n: int, tools: int = 50, defect: str | None = None, servers=()) -> TaskFacts:
    servers = tuple(servers) or tuple(f"s{i}" for i in range(n))
    return TaskFacts(name, domain, servers, tools, True, "u@x.com", (), defect, primary_servers=())


def test_swap_tiers_eligibility_and_hash_order() -> None:
    facts = {
        "aa-old": _facts("aa-old", "aa", 4, defect=UNPASSABLE_SENDER_CHECK),
        "bb-same-count-1": _facts("bb-same-count-1", "bb", 4),
        "bb-same-count-2": _facts("bb-same-count-2", "bb", 4),
        "aa-other-count": _facts("aa-other-count", "aa", 5),
        "aa-too-many-tools": _facts("aa-too-many-tools", "aa", 4, tools=129),
        "aa-broken": _facts("aa-broken", "aa", 4, defect=EMAIL_CONFIG_WITHOUT_PASSWORD),
    }
    pool = list(facts)
    (swap,) = plan_swaps(["aa-old"], pool, facts)
    # Same stratum is ineligible (tool count, defect), so the same domain wins over the same server count.
    assert (swap.new, swap.tier) == ("aa-other-count", "same domain")
    del facts["aa-other-count"]
    (swap,) = plan_swaps(["aa-old"], [name for name in pool if name in facts], facts)
    assert swap.tier == "same server count"
    assert swap.new == min(["bb-same-count-1", "bb-same-count-2"], key=swap_key)


def test_delegation_labels_file_matches_the_rule() -> None:
    rows = list(csv.DictReader((ROOT / "tasks/delegation_labels.csv").open(encoding="utf-8")))
    assert len(rows) == 503
    for row in rows[::25]:
        facts = task_facts(row["task"])
        assert row["needs_delegation"] == str(needs_delegation(facts))
        assert int(row["sa_tools"]) == facts.sa_tools
    assert needs_delegation(_facts("x", "x", 6)) and needs_delegation(_facts("x", "x", 4, tools=129))
    assert not needs_delegation(_facts("x", "x", 5, tools=128))
    pool = [row for row in rows if row["in_pool160"] == "True"]
    assert sum(row["needs_delegation"] == "True" for row in pool) == 103
