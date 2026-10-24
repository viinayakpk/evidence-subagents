"""Load ``configs/models.yaml`` and build a model instance."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from evcarry.paths import resolve_repo_path
from evcarry.providers.openai_compat import RecordingModel, async_openai_client, openrouter_headers
from evcarry.providers.scripted import FixtureModel, ScriptStep, usage_from_script
from evcarry.providers.usage import PriceTable


class MissingApiKey(RuntimeError):
    """A billable provider has no API key in the environment."""


@dataclass
class ModelConfig:
    name: str
    kind: str
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    api_key_default: str | None = None
    billable: bool = False
    prices: str | None = None
    http_referer: str | None = None
    x_title: str | None = None
    routing: dict[str, Any] | None = None
    base_url_env: str | None = None
    model_env: str | None = None
    # False: the provider may reject json_schema response_format, so structured
    # sub-agent returns are requested in the prompt and parsed from text.
    json_schema_outputs: bool = True
    # Scripted judge fixture (kind: scripted); see providers.scripted.KeyedFixtureModel.
    script: str | None = None

    def price_table(self) -> PriceTable:
        if not self.prices:
            return PriceTable(0, 0, 0, peak_multiplier=1)
        path = resolve_repo_path(self.prices)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return PriceTable.from_mapping(data)


def load_models(path: str | Path | None = None) -> dict[str, ModelConfig]:
    target = resolve_repo_path(path or "configs/models.yaml")
    raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    if "models" in raw and isinstance(raw["models"], dict):
        raw = raw["models"]
    loaded: dict[str, ModelConfig] = {}
    for name, body in raw.items():
        if not isinstance(body, dict):
            continue
        loaded[name] = ModelConfig(
            name=name,
            kind=str(body.get("kind") or body.get("provider") or "openai_compatible"),
            model=str(body.get("model") or ""),
            base_url=body.get("base_url"),
            api_key_env=body.get("api_key_env"),
            api_key_default=body.get("api_key_default"),
            billable=bool(body.get("billable", False)),
            prices=body.get("prices"),
            http_referer=body.get("http_referer"),
            x_title=body.get("x_title"),
            routing=body.get("routing"),
            base_url_env=body.get("base_url_env"),
            model_env=body.get("model_env"),
            json_schema_outputs=bool(body.get("json_schema_outputs", True)),
            script=body.get("script"),
        )
    return loaded


def resolve_api_key(config: ModelConfig) -> str:
    if config.kind == "scripted":
        return "scripted"
    env_name = config.api_key_env or ""
    if env_name in {"NONE", "EMPTY", ""}:
        return config.api_key_default or "EMPTY"
    value = os.environ.get(env_name, "")
    if value:
        return value
    if config.api_key_default:
        return config.api_key_default
    if config.billable:
        raise MissingApiKey(f"{config.name} requires {env_name} to be set. No key is committed.")
    return "EMPTY"


def resolved_model_id(config: ModelConfig) -> str:
    if config.model_env:
        return os.environ.get(config.model_env) or config.model
    return config.model


def resolved_base_url(config: ModelConfig) -> str:
    if config.base_url_env:
        return os.environ.get(config.base_url_env) or (config.base_url or "")
    return config.base_url or ""


def build_model(
    config: ModelConfig,
    *,
    steps: list[ScriptStep] | None = None,
    usage: dict[str, Any] | None = None,
    agent_name: str = "agent",
) -> FixtureModel | RecordingModel:
    if config.kind == "scripted":
        return FixtureModel(
            steps or [],
            served_model=config.model or "mock-scripted",
            usage=usage_from_script(usage),
            agent_name=agent_name,
        )
    model_id = resolved_model_id(config)
    headers = None
    if config.kind == "openrouter":
        headers = openrouter_headers(http_referer=config.http_referer, x_title=config.x_title)
    client = async_openai_client(
        base_url=resolved_base_url(config),
        api_key=resolve_api_key(config),
        default_headers=headers,
    )
    return RecordingModel(
        model_id,
        client,
        kind=config.kind,
        routing=config.routing,
        extra_headers=headers,
    )
