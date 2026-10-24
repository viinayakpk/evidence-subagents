"""Experiment YAML."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from evcarry.paths import resolve_repo_path


class AuditConfig(BaseModel):
    sampler: str = "risk"
    rate: float = 0.3
    checks: list[str] = Field(default_factory=lambda: ["L0", "L1", "L2", "L3"])


class BudgetConfig(BaseModel):
    cap_usd: float | None = None
    schedule: str = "any"


class LimitsConfig(BaseModel):
    max_steps: int = 24
    max_seconds: int = 120
    context: int = 32000


class InjectConfig(BaseModel):
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
    variant: str = "naive"
    max_per_run: int | None = None


class ExperimentConfig(BaseModel):
    experiment: str
    tasks: str
    arms: list[str]
    seeds: list[int]
    model: str
    audit: AuditConfig = Field(default_factory=AuditConfig)
    judge_model: str = "lexical"
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    prompts: str = "prompts/v1"
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    parallel_tasks: int = 1
    out: str = "runs/out"
    environment: Literal["in_process", "docker"] = "in_process"
    task_root: str | None = None
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
