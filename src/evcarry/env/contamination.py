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
  its public repo (``eigent-ai``). Reading the task's own kept files is not an
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
