"""Static facts about Toolathlon-Gym tasks and the pre-registered rules built on them.

Everything here reads only the gym's task files and the MCP tool inventory
(``tests/fixtures/gym/tool_inventory.json``, gym commit ed735ba). Nothing runs
a model or reads a run result, so the rules can be fixed before any result is
seen. ``scripts/apply_task_rules.py`` applies them to the frozen lists and
writes ``docs/TASK_RULES.md``'s tables.

Rules (version ``RULES_VERSION``):

- **Email defect.** A task that needs the ``emails`` server is broken when
  (a) its ``email_config.json`` lacks an ``email`` or ``password`` (the server
  refuses to start), or (b) its evaluation requires the sent message's
  from-address to be a fixed address other than the one the server sends from
  (the config's ``email``, else ``user@example.com``). ``send_email`` takes no
  sender, so (b) is passable only by writing the row into Postgres directly.
- **Swap.** Each subset70 task with an email defect is replaced by a pool160
  task that is not in subset70, has no email defect, and gives the single
  agent at most 128 tools. Candidates are tried in tiers (same stratum = same
  name-prefix domain and same number of MCP servers; then same domain; then
  needs the old task's primary servers; then same number of servers; then
  any) and, within a tier, in ascending order of
  ``sha256("evcarry-swap-v1|" + task)``. A domain's primary servers are the
  servers other than ``filesystem`` that every gym task with that name prefix
  needs (``canvas`` -> canvas, ``sf`` -> snowflake; none for ``arxiv``, whose
  tasks share no server). Defective tasks are handled in sorted
  name order; a chosen replacement is not reused. A pilot10 task that is
  swapped out of subset70 gets the same replacement, so pilot10 stays inside
  subset70.
- **Needs delegation.** A task needs delegation when it needs at least 6 MCP
  servers or when the single agent would get more than 128 tools.

Split (``SPLIT_VERSION``, pre-registered 2026-10-25, before the headline): every
pilot10 task is in subset70, and the fixes after the pilot were found on pilot10's
traces. The primary headline is therefore subset70 minus pilot10 (60 tasks,
``tasks/headline60.txt``); pilot10 is reported separately and never pooled into the
primary analysis. Fixes are verified on dev tasks drawn by a seeded rule from
pool160 minus subset70 (never pilot10): no email defect, at most 150 SA tools (the
verified tool limit), the first ``DEV_COUNT`` in ascending order of
``sha256("evcarry-dev-v1|" + task)`` (``tasks/dev.txt``).
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from evcarry.env.toolathlon import gym_task_root
from evcarry.paths import resolve_repo_path

RULES_VERSION = "evcarry-task-rules-v1"
TOOL_INVENTORY = "tests/fixtures/gym/tool_inventory.json"
# claim_done, python_execute and view_overlong_output come on top of the task's MCP tools.
SA_LOCAL_TOOLS = 3
TOOL_LIMIT = 128
DELEGATION_MIN_SERVERS = 6
DEFAULT_SENDER = "user@example.com"
SWAP_SALT = "evcarry-swap-v1"
SPLIT_VERSION = "evcarry-split-v1"
DEV_SALT = "evcarry-dev-v1"
# DeepInfra took 150 tools in one request (docs/TASK_RULES.md section 1); more is unverified.
DEV_TOOL_LIMIT = 150
DEV_COUNT = 2

EMAIL_CONFIG_WITHOUT_PASSWORD = "email_config_without_password"
EMAIL_CONFIG_INVALID = "email_config_invalid"
UNPASSABLE_SENDER_CHECK = "unpassable_sender_check"

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# Names an evaluation uses for a message's from-address.
_FROM_NAME = re.compile(r"^(from\w*|\w*_from|frm|fr|sender\w*|\w*_sender)$", re.IGNORECASE)
# A check label that says what it checks: check("Email from x@y", <condition>).
_FROM_LABEL = re.compile(r"\b(from|sender)\b", re.IGNORECASE)
# SQL that selects on the from-address (NOT ILIKE and != exclude seeded mail and do not match).
_SQL_FROM = re.compile(r"from_addr\s*(?:=|I?LIKE)\s*'%?([^'%]+@[^'%]+)%?'", re.IGNORECASE)


@dataclass(frozen=True)
class TaskFacts:
    name: str
    domain: str
    servers: tuple[str, ...]
    sa_tools: int
    uses_email: bool
    email_sender: str | None
    required_senders: tuple[str, ...]
    email_defect: str | None
    primary_servers: tuple[str, ...] = ()

    @property
    def n_servers(self) -> int:
        return len(self.servers)

    @property
    def stratum(self) -> str:
        return f"{self.domain}|{self.n_servers}"


@cache
def tool_inventory(path: str = TOOL_INVENTORY) -> dict[str, int]:
    """MCP tools per gym server, as listed by the built image."""
    data = json.loads(resolve_repo_path(path).read_text(encoding="utf-8"))
    return {server: len(tools) for server, tools in data["servers"].items()}


def _mentions_from(node: ast.AST) -> bool:
    for child in ast.walk(node):
        name = child.id if isinstance(child, ast.Name) else child.attr if isinstance(child, ast.Attribute) else None
        if name and _FROM_NAME.match(name):
            return True
    return False


def _label_text(node: ast.AST) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(part.value for part in node.values if isinstance(part, ast.Constant) and isinstance(part.value, str))
    return ""


def _address(node: ast.AST, bound: dict[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str) and _EMAIL.fullmatch(node.value.strip("% ")):
        return node.value.strip("% ").lower()
    if isinstance(node, ast.Name):
        return bound.get(node.id)
    return None


def _from_context(node: ast.AST, other: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    """The compared value is a from-address: by its name, by the check's label, or by the list it iterates."""
    if _mentions_from(other):
        return True
    current = parents.get(node)
    while current is not None and not isinstance(current, ast.stmt):
        if isinstance(current, ast.Call) and current.args and _FROM_LABEL.search(_label_text(current.args[0])):
            return True
        if isinstance(current, (ast.GeneratorExp, ast.ListComp, ast.SetComp)) and any(
            _mentions_from(generator.iter) for generator in current.generators
        ):
            return True
        current = parents.get(current)
    return False


def required_senders(task_dir: Path) -> set[str]:
    """Fixed addresses the evaluation requires a message's from-address to contain or equal.

    Parsed with ``ast`` from ``evaluation/*.py``. A comparison counts when one side is a
    fixed address (a literal, or a name bound to one) and the comparison is ``in``,
    ``not in`` or ``==``, and either the other side names a from-address (``from_addr``,
    ``from_str``, ``frm``, ``sender``...), or the comparison sits in a call whose label
    says "from"/"sender" (``check("Email from x@y", "x@y" in (e[1] or ""))``), or in a
    comprehension over a from-address list. SQL strings count when they select
    ``from_addr = 'x@y'`` or ``from_addr ILIKE '%x@y%'``; ``NOT ILIKE`` and ``!=``
    (which exclude seeded mail) do not.
    """
    found: set[str] = set()
    for path in sorted((task_dir / "evaluation").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        bound = {
            node.targets[0].id: address
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and (address := _address(node.value, {})) is not None
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                found.update(address.lower() for address in _SQL_FROM.findall(node.value))
            if not (isinstance(node, ast.Compare) and len(node.ops) == 1):
                continue
            if not isinstance(node.ops[0], (ast.In, ast.NotIn, ast.Eq)):
                continue
            left, right = node.left, node.comparators[0]
            for literal, other in ((left, right), (right, left)):
                address = _address(literal, bound)
                if address is not None and _from_context(node, other, parents):
                    found.add(address)
    return found


def email_sender(task_dir: Path) -> tuple[str | None, str | None]:
    """(sender address, config defect). The emails server uses the config's first account."""
    config = task_dir / "email_config.json"
    if not config.is_file():
        return DEFAULT_SENDER, None
    try:
        data = json.loads(config.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None, EMAIL_CONFIG_INVALID
    account: Any = data[0] if isinstance(data, list) and data else data
    if not isinstance(account, dict):
        return None, EMAIL_CONFIG_INVALID
    address = str(account.get("email") or "").lower() or None
    if not address or not account.get("password"):
        return address, EMAIL_CONFIG_WITHOUT_PASSWORD
    return address, None


def _servers(task_dir: Path) -> tuple[str, ...]:
    config = json.loads((task_dir / "task_config.json").read_text(encoding="utf-8"))
    return tuple(str(item) for item in config.get("needed_mcp_servers") or [])


@cache
def primary_servers(domain: str, root: Path | None = None) -> tuple[str, ...]:
    """Servers (not filesystem) that every gym task named ``<domain>-...`` needs."""
    base = root or gym_task_root()
    sets = [
        set(_servers(path))
        for path in sorted(base.iterdir())
        if path.name.split("-", 1)[0] == domain and (path / "task_config.json").is_file()
    ]
    if not sets:
        return ()
    return tuple(sorted(set.intersection(*sets) - {"filesystem"}))


def task_facts(name: str, root: Path | None = None) -> TaskFacts:
    task_dir = (root or gym_task_root()) / name
    config = json.loads((task_dir / "task_config.json").read_text(encoding="utf-8"))
    servers = tuple(str(item) for item in config.get("needed_mcp_servers") or [])
    inventory = tool_inventory()
    sa_tools = sum(inventory[server] for server in servers) + SA_LOCAL_TOOLS
    uses_email = "emails" in servers
    sender: str | None = None
    required: set[str] = set()
    defect: str | None = None
    if uses_email:
        sender, defect = email_sender(task_dir)
        required = required_senders(task_dir)
        if defect is None and required and sender not in required:
            defect = UNPASSABLE_SENDER_CHECK
    domain = name.split("-", 1)[0]
    return TaskFacts(
        name=name,
        domain=domain,
        servers=servers,
        sa_tools=sa_tools,
        uses_email=uses_email,
        email_sender=sender,
        required_senders=tuple(sorted(required)),
        email_defect=defect,
        primary_servers=primary_servers(domain, root),
    )


def needs_delegation(facts: TaskFacts) -> bool:
    return facts.n_servers >= DELEGATION_MIN_SERVERS or facts.sa_tools > TOOL_LIMIT


def swap_key(name: str) -> str:
    return hashlib.sha256(f"{SWAP_SALT}|{name}".encode("utf-8")).hexdigest()


def eligible_replacement(facts: TaskFacts) -> bool:
    return facts.email_defect is None and facts.sa_tools <= TOOL_LIMIT


_TIERS = (
    ("same stratum", lambda old, new: old.stratum == new.stratum),
    ("same domain", lambda old, new: old.domain == new.domain),
    ("same primary servers", lambda old, new: bool(old.primary_servers) and set(old.primary_servers) <= set(new.servers)),
    ("same server count", lambda old, new: old.n_servers == new.n_servers),
    ("any", lambda old, new: True),
)


@dataclass(frozen=True)
class Swap:
    old: str
    new: str
    reason: str
    tier: str


def plan_swaps(subset: list[str], pool: list[str], facts: dict[str, TaskFacts]) -> list[Swap]:
    used = set(subset)
    swaps: list[Swap] = []
    for old in sorted(name for name in subset if facts[name].email_defect):
        candidates = [name for name in pool if name not in used and eligible_replacement(facts[name])]
        for tier, same in _TIERS:
            matches = sorted((name for name in candidates if same(facts[old], facts[name])), key=swap_key)
            if matches:
                swaps.append(Swap(old, matches[0], str(facts[old].email_defect), tier))
                used.add(matches[0])
                break
        else:
            raise RuntimeError(f"no eligible replacement for {old} in the pool")
    return swaps


def apply_swaps(names: list[str], swaps: list[Swap]) -> list[str]:
    mapping = {swap.old: swap.new for swap in swaps}
    return sorted(mapping.get(name, name) for name in names)


_WORD_DOCUMENT = re.compile(r"\bdocx\b|\bword (?:document|report|file)\b", re.IGNORECASE)
_TABLE_WORD = re.compile(r"\btables?\b", re.IGNORECASE)


def docx_table_defect(task_dir: Path) -> bool:
    """Grader rule (P2-4, 2026-10-25): the spec asks for a table in a Word document, the grader cannot see one.

    True when a line of ``docs/task.md`` names a Word document (``.docx``, "Word document")
    and a table, and the task's evaluation opens a docx (``Document(``) and reads its
    ``.paragraphs`` but never its ``.tables``: correct output then fails. In subset70 this
    flags only ``sf-sales-region-analysis-gsheet-email`` (a pilot10 task, outside the
    primary headline). The Notion property-shape defect is handled by the Notion mock
    patch (``docker/gym/wire_pg_routers.py``), not by this rule.
    """
    spec = task_dir / "docs" / "task.md"
    if not spec.is_file():
        return False
    lines = spec.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not any(_WORD_DOCUMENT.search(line) and _TABLE_WORD.search(line) for line in lines):
        return False
    code = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in sorted((task_dir / "evaluation").rglob("*.py")))
    return "Document(" in code and ".paragraphs" in code and ".tables" not in code


def headline_tasks(subset: list[str], pilot: list[str]) -> list[str]:
    """The primary headline set: subset70 minus pilot10, in subset order."""
    excluded = set(pilot)
    return [name for name in subset if name not in excluded]


def dev_key(name: str) -> str:
    return hashlib.sha256(f"{DEV_SALT}|{name}".encode("utf-8")).hexdigest()


def dev_tasks(
    pool: list[str], subset: list[str], pilot: list[str], facts: dict[str, TaskFacts], count: int = DEV_COUNT
) -> list[str]:
    """Fix-verification tasks: pool160 minus subset70 and pilot10, no email defect, at most 150 SA tools, by ``dev_key``."""
    excluded = set(subset) | set(pilot)
    candidates = [
        name
        for name in pool
        if name not in excluded and facts[name].email_defect is None and facts[name].sa_tools <= DEV_TOOL_LIMIT
    ]
    return sorted(candidates, key=dev_key)[:count]
