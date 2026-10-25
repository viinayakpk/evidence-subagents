"""Experiment YAML."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from evcarry.paths import resolve_repo_path


class _Strict(BaseModel):
    """Unknown keys are errors: a misspelt key (``judge_modle``) would otherwise fall back to its default."""

    model_config = ConfigDict(extra="forbid")


class AuditConfig(_Strict):
    sampler: Literal["risk", "random"] = "risk"
    rate: float = 0.3
    checks: list[Literal["L0", "L1", "L2", "L3"]] = Field(default_factory=lambda: ["L0", "L1", "L2", "L3"])


class BudgetConfig(_Strict):
    cap_usd: float | None = None
    # Per-run spend (agent and judge calls) at which the run's next model call is refused: status cost_ceiling.
    run_ceiling_usd: float | None = None
    # What the ceiling counts. "billed": the dollars billed. "reference" (P2-1, 2026-10-25): every call's tokens
    # priced at the price table's reference endpoint (DeepInfra fp8), whichever endpoint served it, so every arm
    # and every provider mix gets the same token budget. The experiment cap always counts billed dollars.
    ceiling_basis: Literal["billed", "reference"] = "billed"
    # Billed dollars a run may spend per reference dollar, for the cap's per-run reserve
    # (reserve = 1.2 x ceiling x this). 1.0 for a billed ceiling.
    reserve_billed_per_reference: float = 1.0
    # Anything but offpeak_only skips the peak-window refusal, so a typo must not pass.
    schedule: Literal["any", "offpeak_only"] = "any"


class LimitsConfig(_Strict):
    # Main-agent turns.
    max_steps: int = 24
    # Turns per sub-agent run (each main-agent call, repair or re-delegation); None means max_steps.
    subagent_max_turns: int | None = None
    max_seconds: int = 120
    # Context window in tokens, enforced on every model call of every agent (P3-2, 2026-10-25):
    # older tool outputs are replaced by a stub in blocks (harness.context).
    context: int = 32000


class InjectConfig(_Strict):
    probability: float = 0.0
    types: list[str] = Field(
        default_factory=lambda: [
            "numeric_perturbation",
            "fabricated_entity",
            "omission",
            "false_action_done",
            "mismatched_pointer",
        ]
    )
    variant: Literal["naive", "pointer_consistent", "either"] = "naive"
    max_per_run: int | None = None


class GymConfig(_Strict):
    """Docker environment settings. Images default to the tags scripts/build_gym_images.sh makes."""

    agent_image: str | None = None
    postgres_image: str | None = None
    tool_policy: str = "configs/gym_tools.yaml"
    # The gym's own agent truncates tool output at 8000 characters.
    tool_output_chars: int = 8000
    setup_timeout_s: int = 300
    eval_timeout_s: int = 600
    export_workspace: bool = True
    # P2-2 (2026-10-25). The run's Docker network is created --internal: the agent container reaches its own
    # Postgres and nothing else (no internet). Needs the gym images rebuilt after the overlay change that bakes
    # every uv server's environment (make gym-images); the old images rebuild some at start from PyPI.
    network_internal: bool = False
    # P2-2. After the MCP servers start, the task's local mock web roots move out of /workspace/tasks, and
    # /workspace keeps only what the agent and the servers need; the task directory is restored before grading.
    hide_task_dir: bool = False


class ExperimentConfig(_Strict):
    experiment: str
    tasks: str
    arms: list[str]
    seeds: list[int]
    model: str
    audit: AuditConfig = Field(default_factory=AuditConfig)
    judge_model: str = "lexical"
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    prompts: str = "prompts/v1"
    # True: refuse to run unless the prompt set is frozen (FROZEN.json) and unchanged.
    # A billable run always refuses a frozen set that has changed.
    require_frozen_prompts: bool = False
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    parallel_tasks: int = 1
    # A19 (2026-10-25): attempts per (task, seed) block before a block hit by infrastructure errors is dropped.
    max_block_attempts: int = Field(default=2, ge=1)
    out: str = "runs/out"
    environment: Literal["in_process", "docker"] = "in_process"
    task_root: str | None = None
    gym: GymConfig = Field(default_factory=GymConfig)
    # Scripted-model fixtures, one JSON file per arm (ArmSpec.script).
    script_dir: str = "tests/fixtures/scripts"
    inject: InjectConfig | None = None
    source_path: str | None = None
    source_sha256: str = ""


def load_experiment(path: str | Path) -> ExperimentConfig:
    target = resolve_repo_path(path)
    raw_bytes = target.read_bytes()
    data: dict[str, Any] = yaml.safe_load(raw_bytes) or {}
    config = ExperimentConfig.model_validate(data)
    config.source_path = str(target)
    config.source_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    return config
