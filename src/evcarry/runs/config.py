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
    # Anything but offpeak_only skips the peak-window refusal, so a typo must not pass.
    schedule: Literal["any", "offpeak_only"] = "any"


class LimitsConfig(_Strict):
    max_steps: int = 24
    max_seconds: int = 120
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
