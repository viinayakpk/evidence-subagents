"""SDK span processor that writes into the current run's JSONL file.

``set_trace_processors`` replaces the default OpenAI exporter, so spans are
not uploaded. ``Runner`` is also configured with tracing disabled for the
harness runs; this processor is what a unit test and a future live trace use.
"""

from __future__ import annotations

from typing import Any

from agents.tracing import TracingProcessor

from evcarry.trace.writer import current_writer


class JsonlTraceProcessor(TracingProcessor):
    def on_trace_start(self, trace: Any) -> None:
        return None

    def on_trace_end(self, trace: Any) -> None:
        return None

    def on_span_start(self, span: Any) -> None:
        return None

    def on_span_end(self, span: Any) -> None:
        writer = current_writer.get()
        if writer is None:
            return
        data = getattr(span, "span_data", None)
        exported = getattr(data, "export", None)
        payload = exported() if callable(exported) else {"repr": repr(data)}
        writer.write(
            {
                "type": "sdk_span",
                "trace_id": getattr(span, "trace_id", None),
                "span_id": getattr(span, "span_id", None),
                "span_data": payload,
            }
        )

    def shutdown(self) -> None:
        return None

    def force_flush(self) -> None:
        return None
