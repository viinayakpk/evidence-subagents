"""Model providers: DeepSeek, vLLM, OpenRouter, OpenAI-compatible debug, and a scripted mock."""

from evcarry.providers.registry import ModelConfig, build_model, load_models

__all__ = ["ModelConfig", "build_model", "load_models"]
