"""Docker isolation for Toolathlon-Gym. Not attached in M0–M3."""

from __future__ import annotations

import os
import shutil


class EnvironmentUnavailable(RuntimeError):
    """Docker is missing, or EVCARRY_TOOLATHLON_DOCKER is not set."""


class DockerIsolation:
    """Plans the gym's own runner. ``up`` refuses unless Docker is explicitly enabled.

    M0–M3 does not place this harness's agent inside gym containers. The command
    is the gym's ``run_parallel.sh``, kept so a later milestone can launch the
    environment on the workstation.
    """

    def plan(self) -> list[str]:
        return ["bash", "third_party/toolathlon_gym/run_parallel.sh"]

    def up(self) -> None:
        if shutil.which("docker") is None or os.environ.get("EVCARRY_TOOLATHLON_DOCKER") != "1":
            raise EnvironmentUnavailable(
                "Docker isolation is unavailable. On the workstation, install Docker and set "
                "EVCARRY_TOOLATHLON_DOCKER=1. This milestone does not attach the harness "
                "inside Toolathlon containers."
            )
