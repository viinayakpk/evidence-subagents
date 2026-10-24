"""Write the committed scripted-model fixtures. Run from the repo root."""

from __future__ import annotations

import json
from pathlib import Path


def _return(subagent_id: str, claim_id: str, text: str, kind: str, value: object, evidence: list[dict]) -> str:
    payload = {
        "claims": [
            {
                "claim_id": claim_id,
                "confidence": "high",
                "depends_on": [],
                "evidence": evidence,
                "kind": kind,
                "text": text,
                "value": value,
            }
        ],
        "open_issues": [],
        "status": "complete",
        "subagent_id": subagent_id,
    }
    return json.dumps(payload, sort_keys=True)


def _tool(name: str, arguments: dict, call_id: str) -> dict:
    return {"arguments": arguments, "call_id": call_id, "tool": name}


def _text(content: str) -> dict:
    return {"content": content}


def _fact(call_id: str, *, bad: bool = False) -> str:
    if bad:
        return _return(
            "researcher",
            "c1",
            "The record cites student-Z.",
            "fact",
            "student-Z",
            [{"call_id": "call_missing", "span": "student-Z", "type": "tool_call"}],
        )
    return _return(
        "researcher",
        "c1",
        "The record contains 14 sections.",
        "fact",
        14,
        [{"call_id": call_id, "span": "14 sections", "type": "tool_call"}],
    )


def _rows(call_id: str) -> str:
    return _return(
        "researcher",
        "c1",
        "The rows include paper-A.",
        "fact",
        "paper-A",
        [{"call_id": call_id, "span": "paper-A", "type": "tool_call"}],
    )


def _wrote(call_id: str, value: str) -> str:
    return _return(
        "operator",
        "c2",
        "Wrote the answer file.",
        "action_done",
        value,
        [{"call_id": call_id, "span": "wrote answer.txt", "type": "tool_call"}],
    )


def _event(call_id: str) -> str:
    return _return(
        "operator",
        "c2",
        "Created the standup event.",
        "action_done",
        "standup",
        [{"call_id": call_id, "span": "created standup", "type": "tool_call"}],
    )


def _empty(subagent_id: str) -> str:
    return _return(subagent_id, "c1", "The record contains 14 sections.", "fact", 14, [])


def _bundle(tasks: dict) -> dict:
    return {
        "served_model": "mock-scripted",
        "tasks": tasks,
        "usage": {"cached_tokens": 80, "input_tokens": 100, "output_tokens": 20},
    }


def sa() -> dict:
    return _bundle(
        {
            "fixture-calendar": {
                "agents": {
                    "main": [
                        _tool("create_event", {"title": "standup"}, "call_sa_event"),
                        _tool("write_answer", {"text": "standup"}, "call_sa_write"),
                        _text("done"),
                    ]
                }
            },
            "fixture-rows": {
                "agents": {
                    "main": [
                        _tool("query_rows", {"sql": "select papers"}, "call_sa_rows"),
                        _tool("write_answer", {"text": "paper-A"}, "call_sa_write"),
                        _text("done"),
                    ]
                }
            },
            "fixture-sections": {
                "agents": {
                    "main": [
                        _tool("search_records", {"query": "sections"}, "call_sa_search"),
                        _tool("write_answer", {"text": "14 sections"}, "call_sa_write"),
                        _text("done"),
                    ]
                }
            },
        }
    )


def vanilla() -> dict:
    def main() -> list[dict]:
        return [
            _tool("researcher", {"input": "collect"}, "call_sub_research"),
            _tool("operator", {"input": "finish"}, "call_sub_operate"),
            _text("done"),
        ]

    return _bundle(
        {
            "fixture-calendar": {
                "agents": {
                    "main": main(),
                    "operator": [
                        _tool("create_event", {"title": "standup"}, "call_event"),
                        _tool("write_answer", {"text": "standup"}, "call_write"),
                        _text("created standup and wrote the answer"),
                    ],
                    "researcher": [
                        _tool("search_records", {"query": "calendar"}, "call_search"),
                        _text("The record contains 14 sections."),
                    ],
                }
            },
            "fixture-rows": {
                "agents": {
                    "main": main(),
                    "operator": [
                        _tool("write_answer", {"text": "paper-A"}, "call_write"),
                        _text("wrote the answer"),
                    ],
                    "researcher": [
                        _tool("query_rows", {"sql": "select papers"}, "call_rows"),
                        _text("The rows include paper-A."),
                    ],
                }
            },
            "fixture-sections": {
                "agents": {
                    "main": main(),
                    "operator": [
                        _tool("write_answer", {"text": "14 sections"}, "call_write"),
                        _text("wrote the answer"),
                    ],
                    "researcher": [
                        _tool("search_records", {"query": "sections"}, "call_search"),
                        _text("The record contains 14 sections."),
                    ],
                }
            },
        }
    )


def evidence() -> dict:
    def main() -> list[dict]:
        return [
            _tool("researcher", {"input": "collect"}, "call_sub_research"),
            _tool("operator", {"input": "finish"}, "call_sub_operate"),
            _text("done"),
        ]

    return _bundle(
        {
            "fixture-calendar": {
                "agents": {
                    "main": main(),
                    "operator": [
                        _tool("create_event", {"title": "standup"}, "call_event"),
                        _tool("write_answer", {"text": "standup"}, "call_write"),
                        _text(_event("call_event")),
                    ],
                    "researcher": [
                        _tool("search_records", {"query": "calendar"}, "call_search"),
                        _text(_fact("call_search")),
                    ],
                }
            },
            "fixture-rows": {
                "agents": {
                    "main": main(),
                    "operator": [
                        _tool("write_answer", {"text": "paper-A"}, "call_write"),
                        _text(_wrote("call_write", "paper-A")),
                    ],
                    "researcher": [
                        _tool("query_rows", {"sql": "select papers"}, "call_rows"),
                        _text(_rows("call_rows")),
                    ],
                }
            },
            "fixture-sections": {
                "agents": {
                    "main": main(),
                    "operator": [
                        _tool("write_answer", {"text": "14 sections"}, "call_write"),
                        _text(_wrote("call_write", "14 sections")),
                    ],
                    "researcher": [
                        _tool("search_records", {"query": "sections"}, "call_search"),
                        _text(_fact("call_search")),
                    ],
                }
            },
        }
    )


def typed() -> dict:
    def main() -> list[dict]:
        return [
            _tool("researcher", {"input": "collect"}, "call_sub_research"),
            _tool("operator", {"input": "finish"}, "call_sub_operate"),
            _text("done"),
        ]

    return _bundle(
        {
            "fixture-calendar": {
                "agents": {
                    "main": main(),
                    "operator": [
                        _tool("write_answer", {"text": "standup"}, "call_write"),
                        _text(_empty("operator")),
                    ],
                    "researcher": [_text(_empty("researcher"))],
                }
            },
            "fixture-rows": {
                "agents": {
                    "main": main(),
                    "operator": [
                        _tool("write_answer", {"text": "paper-A"}, "call_write"),
                        _text(_empty("operator")),
                    ],
                    "researcher": [_text(_empty("researcher"))],
                }
            },
            "fixture-sections": {
                "agents": {
                    "main": main(),
                    "operator": [
                        _tool("write_answer", {"text": "14 sections"}, "call_write"),
                        _text(_empty("operator")),
                    ],
                    "researcher": [_text(_empty("researcher"))],
                }
            },
        }
    )


def evidence_corrupt() -> dict:
    return _bundle(
        {
            "fixture-sections": {
                "agents": {
                    "main": [
                        _tool("researcher", {"input": "collect"}, "call_sub_research"),
                        _tool("operator", {"input": "finish"}, "call_sub_operate"),
                        _text("done"),
                    ],
                    "operator": [
                        _tool("write_answer", {"text": "14 sections"}, "call_write"),
                        _text(_wrote("call_write", "14 sections")),
                    ],
                    "researcher": [
                        _tool("search_records", {"query": "sections"}, "call_search"),
                        _text(_fact("call_search", bad=True)),
                        _tool("search_records", {"query": "sections"}, "call_search_b"),
                        _text(_fact("call_search_b")),
                    ],
                }
            }
        }
    )


def evidence_reinject() -> dict:
    return _bundle(
        {
            "fixture-sections": {
                "agents": {
                    "main": [
                        _tool("researcher", {"input": "collect"}, "call_sub_research"),
                        _tool("operator", {"input": "finish"}, "call_sub_operate"),
                        _text("done"),
                    ],
                    "operator": [
                        _tool("write_answer", {"text": "14 sections"}, "call_write"),
                        _text(_wrote("call_write", "14 sections")),
                    ],
                    "researcher": [
                        _tool("search_records", {"query": "sections"}, "call_search"),
                        _text(_fact("call_search")),
                        _tool("search_records", {"query": "sections"}, "call_search_b"),
                        _text(_fact("call_search_b")),
                    ],
                }
            }
        }
    )


def main() -> None:
    destination = Path(__file__).resolve().parent
    files = {
        "sa.json": sa(),
        "vanilla_mad.json": vanilla(),
        "evidence_risk.json": evidence(),
        "typed.json": typed(),
        "evidence_corrupt.json": evidence_corrupt(),
        "evidence_reinject.json": evidence_reinject(),
    }
    for name, payload in files.items():
        (destination / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(name)


if __name__ == "__main__":
    main()
