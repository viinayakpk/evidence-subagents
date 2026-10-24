"""Docker isolation for Toolathlon-Gym runs: preflight and the dry-run plan.

The per-run environment itself is ``evcarry.env.gym_docker.GymTaskEnv``.
"""

from __future__ import annotations


class EnvironmentUnavailable(RuntimeError):
    """Docker, its daemon, or the gym images are missing, or a task environment failed to start."""


class DockerIsolation:
    """Checks that a docker-environment experiment can start, and describes what each run does."""

    def __init__(self, images=None) -> None:
        from evcarry.env.gym_docker import GymImages

        self.images = images or GymImages.default()

    def plan(self) -> list[str]:
        return [
            "per run: docker network create evcarry-<id>-net",
            f"per run: docker run {self.images.postgres} (pre-loaded gym database)",
            f"per run: docker run {self.images.agent} sleep infinity; MCP servers over docker exec -i",
            "per run: task preprocess, agent, task evaluation, docker rm -fv, docker network rm",
            "pattern: third_party/toolathlon_gym/run_parallel.sh (one Postgres, agent container and network per task)",
            "images: scripts/build_gym_images.sh (make gym-images)",
        ]

    def up(self) -> str:
        """Raise ``EnvironmentUnavailable`` unless Docker and both gym images are present."""
        from evcarry.env.gym_docker import preflight

        return preflight(self.images)
