"""Keep the default OpenAI trace exporter from starting during tests."""

from agents import set_trace_processors

set_trace_processors([])
