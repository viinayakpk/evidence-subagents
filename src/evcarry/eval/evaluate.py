"""Run a fixture evaluation script, or record that a real gym task was not run.

``evcarry_local_eval`` must be true. Real Toolathlon evaluation scripts are
not executed from this harness.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any

from evcarry.env.toolathlon import TaskSpec


def evaluate_task(task: TaskSpec, agent_workspace: Any) -> dict[str, Any]:
    if not task.local_eval:
        return {"status": "not_run", "passed": None}
    script = task.root / "evaluation" / "main.py"
    if not script.is_file():
        return {"status": "not_run", "passed": None, "reason": "no local evaluation script"}
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--agent_workspace",
            str(agent_workspace),
            "--groundtruth_workspace",
            str(task.root / "groundtruth_workspace"),
            "--launch_time",
            "evcarry",
            "--res_log_file",
            str(agent_workspace / "eval.log"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    passed = completed.returncode == 0
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
