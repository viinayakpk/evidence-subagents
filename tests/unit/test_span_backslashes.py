"""A span with one backslash matches a Python repr that doubled it (dev run 2026-10-25, sf-sales-trip return 9)."""

from evcarry.protocol.checks import span_in_text

RESULT = "  (3076, 'INBOX', '/', [], 0, 0)\n  (3077, 'Sent', '/', ['\\\\Sent'], 0, 0)\n  (3082, 'INBOX.Sent', '/', [], 0, 0)\n"


def test_the_dev_run_span_now_matches() -> None:
    assert span_in_text("(3077, 'Sent', '/', ['\\Sent'], 0, 0)", RESULT)


def test_exact_spans_still_match() -> None:
    assert span_in_text("(3077, 'Sent', '/', ['\\\\Sent'], 0, 0)", RESULT)


def test_backslashes_are_not_dropped() -> None:
    assert not span_in_text("(3077, 'Sent', '/', ['Sent'], 0, 0)", RESULT)
    assert not span_in_text("(3078, 'Sent', '/', ['\\Sent'], 0, 0)", RESULT)
    assert not span_in_text("a\\b", "ab")
