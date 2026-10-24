"""Keep the default OpenAI trace exporter from starting, and keep tests off paid endpoints."""

import pytest
from agents import set_trace_processors

set_trace_processors([])


@pytest.fixture(autouse=True)
def _no_network_models(request, monkeypatch):
    """Fail any test that builds an HTTP-backed model, unless it is an opt-in live test."""
    if request.node.get_closest_marker("live_provider"):
        return

    def refuse(**_kwargs):
        raise RuntimeError("a test tried to build a network model client; use the scripted provider")

    monkeypatch.setattr("evcarry.providers.registry.async_openai_client", refuse)
