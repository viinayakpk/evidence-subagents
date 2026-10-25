"""Ground-truth reach in two tiers (P2-3, 2026-10-25, before the headline).

The pre-registered rule (plan section 5) names "the removed task files and
``groundtruth`` paths, the gym's git history, or its public repo". The pilot's
detector also matched the bare ``/workspace/tasks`` and ``tasks/finalpool``
markers, so every read of the task's own kept files (``docs/task.md``,
``task_config.json``, ``preprocess/``) was flagged: 117 of 169 flagged calls, and 3
grader passes voided. This module implements the registered definition:

- **attempted**: a tool call whose arguments name ground truth (``groundtruth``,
  ``ground_truth``), the task's removed grader (``.../evaluation/main``, the task's
  ``evaluation`` directory), a filesystem search (walk, glob, find, recursive ls or
  grep) for grader-like names (``eval``, ``evaluation``, ``grader``), another task's
  directory (removed by the seal), the gym's git history (``/workspace/.git``), or
  its public repo (``eigent-ai``). From 2026-10-25 (plan A14, after the first dev
  run) also: a database catalog search (``information_schema``, ``pg_tables``, ...)
  for grader-like table or column names (``db_grader_search``), a scan of installed
  packages for benchmark or grader names (``package_scan``), and a web or package-index
  lookup of the benchmark by name (``benchmark_lookup``). Reading the task's own kept files is not an
  attempt, and neither is prose or code that merely says "evaluation" or "ground
  truth" (an e-mail heading, a column name). Arguments are searched value by value.
- **obtained**: a tool call whose *result* contains a line of at least 40 characters
  of the task's removed grader code (``evaluation/``) that appears in no file the
  seal kept. Ground-truth data files are not fingerprinted: a correct agent that reads
  back its own output reproduces their lines, and reaching for them is already an
  attempt by path.

Both tiers apply to every arm. A run with any attempted or obtained call is
``contaminated`` and counts as a failure (the registered rule: arguments that
reach for ground truth); reports give both tiers, so success with only obtained
runs counted as failures can be shown as a sensitivity analysis. self_verify's
grader-seeking (35 of 37 pilot searches) falls under the same rule as every arm's.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_ATTEMPT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # Path forms only: the phrase "ground truth" in an ML summary is not a reach for answers.
    ("groundtruth", re.compile(r"ground_?truth", re.IGNORECASE)),
    ("grader", re.compile(r"evaluation[/\\.]main", re.IGNORECASE)),
    ("grader", re.compile(r"finalpool/[^/\s'\"]+/evaluation", re.IGNORECASE)),
    # find -name '*evaluation*' (other searches by name are caught by _FS_SEARCH with _GRADER_NAME below).
    ("grader_search", re.compile(r"-i?(?:name|path)\s+['\"]?\*?evaluation", re.IGNORECASE)),
    ("gym_git", re.compile(r"/workspace/\.git\b")),
    ("gym_repo", re.compile(r"eigent-ai", re.IGNORECASE)),
)
_OTHER_TASK = re.compile(r"tasks/finalpool/([A-Za-z0-9][A-Za-z0-9_.-]*)")
# A filesystem search (walk, glob, find, recursive ls or grep) for grader-like names: 'eval', '*evaluation*',
# 'grader'. Not 'grade', which Canvas tasks search for legitimately (grades.csv, grade columns).
_FS_SEARCH = re.compile(r"os\.walk|\bglob\b|\brglob\b|\bfind\s|os\.listdir|os\.scandir|\bls\s+-\w*R|\bgrep\s+-\w*r", re.IGNORECASE)
_GRADER_NAME = re.compile(r"""['"*/](?:eval|evaluat\w*|grader\w*|grading\w*)['"*/]""", re.IGNORECASE)

# A14 (2026-10-25). Names searched for are quoted literals or LIKE/regex patterns, never bare words in task SQL.
# Not toolathlon: the database every task connects to is toolathlon_gym (an older host name was toolathlon_pg).
# Not grade, grading, grader, answer(s) or rubric: canvas has grades, grading_type, grader_id, quiz_questions.answers,
# gform responses.answers and a rubrics table, which a task may legitimately look up in the catalog.
_DB_CATALOG = re.compile(
    r"information_schema|\bpg_(?:tables|class|catalog|namespace|attribute|stat_user_tables)\b|sqlite_master|"
    r"\bshow\s+(?:tables|schemas|databases)\b|\\dt\b",
    re.IGNORECASE,
)
_DB_GRADER_TERM = re.compile(
    r"""['"%(|]%?(?:eval|evals|evaluat\w*|expected\w*|solutions?|ground_?truth\w*|groundtruth\w*|answer_?keys?|"""
    r"""bench|benchmarks?|mcp_?bench)%?['"%)|]""",
    re.IGNORECASE,
)
_PACKAGE_SCAN = re.compile(
    r"site-packages|dist-packages|importlib\.metadata|importlib_metadata|pkg_resources|pkgutil|\bdistributions\(|"
    r"\b(?:uv\s+)?pip3?\s+(?:list|show|freeze)\b",
    re.IGNORECASE,
)
_PACKAGE_TERM = re.compile(
    r"""['"](?:eval|evaluation|grader|grading|bench|benchmark|toolathlon|mcp_?bench\w*|gym|rubric|groundtruth|ground_truth)['"]""",
    re.IGNORECASE,
)
_BENCHMARK_NAME = r"(?:toolathlon|mcp[-_]?bench|eigent)"
# Outside a URL, toolathlon_gym / toolathlon_pg is the task database or its host, and eigent its user, not the benchmark.
_BENCHMARK_WORD = r"(?:toolathlon(?![-_]?(?:gym|pg)\b)|mcp[-_]?bench|eigent[-_]ai)"
_BENCHMARK_LOOKUP = re.compile(
    r"https?://[^\s'\"]*" + _BENCHMARK_NAME
    + r"|\b(?:pip3?|uv|npm|pipx)\s+(?:install|download|index|search|view|add|pip\s+install)\b[^\n]{0,80}" + _BENCHMARK_WORD
    + r"|\bgit\s+clone\b[^\n]{0,120}" + _BENCHMARK_WORD
    + r"|(?:pypi\.org|github\.com|githubusercontent\.com|huggingface\.co|gitlab\.com)[^\n]{0,200}" + _BENCHMARK_WORD
    + r"|" + _BENCHMARK_WORD + r"[^\n]{0,200}(?:pypi\.org|github\.com|huggingface\.co|gitlab\.com)",
    re.IGNORECASE,
)

FINGERPRINT_MIN_CHARS = 40
_TEXT_SUFFIXES = {".py", ".txt", ".csv", ".tsv", ".json", ".jsonl", ".md", ".yaml", ".yml", ".html", ".xml", ".sql"}


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for item in value.values() for text in _strings(item)]
    if isinstance(value, (list, tuple)):
        return [text for item in value for text in _strings(item)]
    return [] if value is None else [json.dumps(value, default=str)]


def attempt_markers(arguments: dict[str, Any], task: str | None) -> list[str]:
    """The kinds of ground-truth reach a call's arguments show (empty: none)."""
    found: list[str] = []
    for text in _strings(arguments):
        for label, pattern in _ATTEMPT_PATTERNS:
            if pattern.search(text):
                found.append(label)
        if _FS_SEARCH.search(text) and _GRADER_NAME.search(text):
            found.append("grader_search")
        if _DB_CATALOG.search(text) and _DB_GRADER_TERM.search(text):
            found.append("db_grader_search")
        if _PACKAGE_SCAN.search(text) and _PACKAGE_TERM.search(text):
            found.append("package_scan")
        if _BENCHMARK_LOOKUP.search(text):
            found.append("benchmark_lookup")
        for match in _OTHER_TASK.finditer(text):
            if task is None or match.group(1) != task:
                found.append("other_task")
    return list(dict.fromkeys(found))


_GYM_ROOT = Path(__file__).resolve().parents[3] / "third_party" / "toolathlon_gym"
_SHARED_DIRS = ("local_servers", "utils", "scripts")
_shared_cache: dict[Path, str] = {}


def shared_gym_text(gym_root: Path | None = None) -> str:
    """The gym's server, utils and script sources, which every agent can read.

    A grader line that also appears there (for example a generic ``SELECT id, title FROM ...`` that
    the Sheets mock runs too) is not evidence of reading the removed grader, so it is no fingerprint.
    """
    root = (gym_root or _GYM_ROOT).resolve()
    if root not in _shared_cache:
        lines: set[str] = set()
        for name in _SHARED_DIRS:
            base = root / name
            if not base.is_dir():
                continue
            for path in base.rglob("*"):
                if path.is_file() and path.suffix.lower() in _TEXT_SUFFIXES | {".ts", ".js"} and "node_modules" not in path.parts:
                    try:
                        text = path.read_text(encoding="utf-8", errors="ignore")
                    except OSError:
                        continue
                    lines.update(raw.strip() for raw in text.splitlines())
        _shared_cache[root] = "\n".join(sorted(lines))
    return _shared_cache[root]


def removed_fingerprints(sealed_task: Path | None, *, gym_root: Path | None = None) -> list[str]:
    """Distinctive lines of the task's removed grader code (``evaluation/``), from the host copy of the task."""
    if sealed_task is None or not sealed_task.is_dir():
        return []
    removed: list[Path] = []
    kept: list[Path] = []
    for path in sorted(sealed_task.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _TEXT_SUFFIXES:
            continue
        top = path.relative_to(sealed_task).parts[0]
        if top == "evaluation":
            removed.append(path)
        elif top != "groundtruth_workspace" and "groundtruth" not in path.name.lower():
            kept.append(path)  # every file the seal keeps (docs, preprocess, files/, initial_workspace) is readable
    visible = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in kept)
    shared = shared_gym_text(gym_root)
    lines: list[str] = []
    for path in removed:
        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if len(line) >= FINGERPRINT_MIN_CHARS and re.search(r"[A-Za-z]", line) and line not in visible and line not in shared:
                lines.append(line)
    return list(dict.fromkeys(lines))


def contamination(ledger: Any, *, task: str | None, fingerprints: list[str] | None = None) -> dict[str, list[dict[str, str]]]:
    """``{"attempted": [...], "obtained": [...]}``: one row per flagged ledger call."""
    attempted: list[dict[str, str]] = []
    obtained: list[dict[str, str]] = []
    for entry in getattr(ledger, "entries", {}).values():
        markers = attempt_markers(entry.arguments, task)
        if markers:
            attempted.append({"call_id": entry.call_id, "tool": entry.tool_name, "markers": ",".join(markers)})
        hit = next((line for line in fingerprints or () if line in entry.result_text), None)
        if hit is not None:
            obtained.append({"call_id": entry.call_id, "tool": entry.tool_name, "match": hit[:120]})
    return {"attempted": attempted, "obtained": obtained}


def tier(found: dict[str, list[dict[str, str]]]) -> str:
    if found.get("obtained"):
        return "obtained"
    if found.get("attempted"):
        return "attempted"
    return "none"


# -- A20 (2026-10-25): direct writes to the mock services' data ---------------------------------------------
# A write statement (INSERT, UPDATE, DELETE, TRUNCATE, ALTER/DROP/CREATE TABLE, MERGE, COPY ... FROM) whose
# target is a two-part name in one of the run database's mock-service schemas, in any tool call's arguments:
# python_execute or terminal code, and the snowflake mock's write_query/read_query, which passes two-part
# names through untranslated. Snowflake's own DB.SCHEMA.TABLE names are three-part and do not match, so
# legitimate snowflake writes (sf_data, through its MCP server) are not flagged. Logged and reported, not a
# contamination tier: a write is not a reach for ground truth (A9, A14), and with ``agent_db_readonly`` it is
# denied, so the graded state is what the tools produced. ``denied``: the call's result says permission denied.
MOCK_SCHEMAS = (
    "arxiv", "arxiv_latex", "canvas", "email", "gcal", "gform", "gsheet", "notion", "public", "scholarly",
    "sf", "sf_data", "train", "wc", "yf", "youtube",
)
_WRITE_VERB = r"(?:INSERT\s+INTO|UPDATE|DELETE\s+FROM|TRUNCATE(?:\s+TABLE)?|(?:ALTER|DROP|CREATE)\s+TABLE(?:\s+IF\s+(?:NOT\s+)?EXISTS)?|MERGE\s+INTO|COPY)"
_DB_WRITE = re.compile(
    rf"\b{_WRITE_VERB}\s+(?:ONLY\s+)?\"?(?P<schema>{'|'.join(MOCK_SCHEMAS)})\"?\s*\.\s*\"?(?P<table>[A-Za-z_]\w*)\"?(?!\s*\.\s*\"?[A-Za-z_])",
    re.IGNORECASE,
)
_PERMISSION_DENIED = re.compile(r"permission denied for (?:table|relation|schema|sequence|database)|must be owner of", re.IGNORECASE)


def db_write_targets(arguments: dict[str, Any]) -> list[str]:
    """``schema.table`` targets of write statements against mock-service schemas in a call's arguments."""
    found: list[str] = []
    for text in _strings(arguments):
        for match in _DB_WRITE.finditer(text):
            found.append(f"{match.group('schema').lower()}.{match.group('table').lower()}")
    return list(dict.fromkeys(found))


def db_writes(ledger: Any) -> list[dict[str, Any]]:
    """One row per ledger call that tried to write the mock services' data directly (A20)."""
    rows: list[dict[str, Any]] = []
    for entry in getattr(ledger, "entries", {}).values():
        targets = db_write_targets(entry.arguments)
        if targets:
            rows.append(
                {
                    "call_id": entry.call_id,
                    "tool": entry.tool_name,
                    "targets": ",".join(targets),
                    "denied": bool(_PERMISSION_DENIED.search(entry.result_text or "")),
                }
            )
    return rows
