"""Gym tool policy, docker exec arguments, tool display, and replay rules. No Docker needed."""

from __future__ import annotations

import json
import re
from fnmatch import fnmatchcase
from types import SimpleNamespace

import pytest

from evcarry.env.gym_docker import WORKSPACE, load_server_specs, server_exec_args
from evcarry.env.gym_tools import GymToolPolicy, exposed_name
from evcarry.harness.prompts import prompt_text, subagent_instructions
from evcarry.harness.tools import ToolRouter, display_text, overlong_view
from evcarry.ledger.ledger import Ledger
from evcarry.paths import repo_root
from evcarry.protocol.checks import CheckContext, run_l0_l1
from evcarry.protocol.replayer import mask_volatile, replay_claim
from evcarry.protocol.schema import Claim, ToolCallEvidence

INVENTORY = json.loads((repo_root() / "tests/fixtures/gym/tool_inventory.json").read_text(encoding="utf-8"))["servers"]


def _tool(server: str, name: str) -> SimpleNamespace:
    hint = next(t["read_only_hint"] for t in INVENTORY[server] if t["name"] == name)
    annotations = SimpleNamespace(read_only_hint=hint) if hint is not None else None
    return SimpleNamespace(name=name, description=f"{name} tool", input_schema={"type": "object"}, annotations=annotations)


def test_policy_patterns_match_the_inventory() -> None:
    policy = GymToolPolicy.load()
    for section in (policy.read_only, policy.nondeterministic, policy.volatile):
        assert set(section) <= set(INVENTORY)
    for section in (policy.read_only, policy.nondeterministic):
        for server, patterns in section.items():
            names = [tool["name"] for tool in INVENTORY[server]]
            for pattern in patterns:
                assert any(fnmatchcase(name, pattern) for name in names), (server, pattern)
    exposed = {exposed_name(server, tool["name"]) for server, tools in INVENTORY.items() for tool in tools}
    for write, rule in policy.readback.items():
        assert write in exposed and rule["tool"] in exposed, write
        server, _, tool = rule["tool"].partition("-")
        assert policy.is_read_only(server, tool, None) or _tool(server, tool).annotations.read_only_hint


@pytest.mark.parametrize(
    ("server", "tool", "read_only"),
    [
        ("snowflake", "read_query", True),
        ("snowflake", "write_query", False),
        ("filesystem", "read_text_file", True),
        ("filesystem", "write_file", False),
        ("excel", "read_data_from_excel", True),
        ("excel", "write_data_to_excel", False),
        ("emails", "search_emails", True),
        ("emails", "send_email", False),
        ("woocommerce", "woo_orders_list", True),
        ("woocommerce", "woo_orders_create", False),
        ("canvas", "canvas_get_course", True),
        ("canvas", "canvas_submit_grade", False),
        ("notion", "API-post-database-query", True),
        ("notion", "API-post-page", False),
        ("terminal", "run_command", False),
        ("playwright_with_chunk", "browser_snapshot", False),
    ],
)
def test_read_only_classification(server: str, tool: str, read_only: bool) -> None:
    meta = GymToolPolicy.load().meta(server, _tool(server, tool))
    assert meta.read_only is read_only
    assert meta.name == f"{server}-{tool}"
    assert not meta.strict


def test_exposed_names_are_valid_and_unique() -> None:
    names = [exposed_name(server, tool["name"]) for server, tools in INVENTORY.items() for tool in tools]
    assert len(names) == len(set(names)) == 441
    assert all(re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name) for name in names)
    assert len(exposed_name("s" * 40, "t" * 40)) <= 64


def test_readback_and_volatile_metadata() -> None:
    policy = GymToolPolicy.load()
    send = policy.meta("emails", _tool("emails", "send_email"))
    assert send.readback_for({"to": "a@b.c", "subject": "Hello", "body": "x"}) == (
        "emails-search_emails",
        {"query": "Hello", "folder": "Sent"},
    )
    query = policy.meta("snowflake", _tool("snowflake", "read_query"))
    text = "type: data\ndata_id: 9f5ab76e-93d3-4a7c-b046-057ba8d8f807\ndata:\n- N: 3"
    other = text.replace("9f5ab76e", "11111111")
    assert mask_volatile(text, list(query.volatile)) == mask_volatile(other, list(query.volatile))
    assert policy.meta("filesystem", _tool("filesystem", "get_file_info")).nondeterministic


def test_server_exec_args_follow_the_gym_env_rule() -> None:
    specs = load_server_specs()
    assert len(specs) == 25
    container_env = {"PGHOST": "pg-1", "PG_HOST": "pg-1", "PGPORT": "5432", "PGUSER": "eigent",
                     "PGPASSWORD": "camel", "PGDATABASE": "toolathlon_gym", "LOCAL_SERVERS_PATH": "/opt/local_servers"}
    cwd, args = server_exec_args(specs["rail_12306"], container="agent-1", container_env=container_env, task_dir="/workspace/tasks/finalpool/t")
    assert args[:4] == ["exec", "-i", "-w", WORKSPACE]
    env = dict(item.split("=", 1) for item in args[args.index("-w") + 2 : args.index("agent-1")] if item != "-e")
    # The YAML says database toolathlon / user postgres; the bridge from PG* wins.
    assert env["PG_DATABASE"] == "toolathlon_gym" and env["PG_USER"] == "eigent" and env["PG_PASSWORD"] == "camel"
    assert "PG_HOST" not in env  # already in the container env with the same value
    assert args[args.index("agent-1") + 1 :] == ["node", "/opt/local_servers/12306-mcp/build/index.js"]
    cwd, args = server_exec_args(specs["emails"], container="agent-1", container_env=container_env, task_dir="/workspace/tasks/finalpool/t")
    assert "/workspace/tasks/finalpool/t/email_config.json" in args
    assert f"{WORKSPACE}/emails_download" in args


async def test_display_truncation_and_overlong_paging() -> None:
    ledger = Ledger()
    text = "x" * 25 + "END"
    ledger.record(call_id="c1", tool_name="t", arguments={}, result_text=text, read_only=True)
    shown = display_text(text, "c1", show_call_id=True, max_chars=10)
    assert shown.startswith("[call_id: c1]\nxxxxxxxxxx\n...[truncated: showing 10 of 28 chars.")
    view = overlong_view(ledger, page_size=10)
    assert (await view({"call_id": "c1", "page": 2})).endswith("xxxxxEND")
    assert "out of range" in await view({"call_id": "c1", "page": 3})
    assert "No recorded" in await view({"call_id": "nope", "page": 0})

    class Env:
        async def call_tool(self, name, arguments):
            return f"env:{name}"

    router = ToolRouter(Env(), {"view_overlong_output": view})
    assert await router.call_tool("snowflake-read_query", {}) == "env:snowflake-read_query"
    assert (await router.call_tool("view_overlong_output", {"call_id": "c1", "page": 0})).endswith("xxxxxxxxxx")


def test_ledger_unique_ids() -> None:
    ledger = Ledger()
    ledger.record(call_id="call_0", tool_name="t", arguments={}, result_text="a", read_only=True)
    assert ledger.unique_id("call_0") == "call_0#2"
    ledger.record(call_id="call_0#2", tool_name="t", arguments={}, result_text="b", read_only=True)
    assert ledger.unique_id("call_0") == "call_0#3"
    assert ledger.unique_id("fresh") == "fresh"


class _Caller:
    def __init__(self, texts: dict[str, str]) -> None:
        self.texts = texts
        self.calls: list[str] = []

    async def call_tool(self, name: str, arguments: dict) -> str:
        self.calls.append(name)
        return self.texts[name]


async def test_replay_unverifiable_cases_and_derived_values() -> None:
    ledger = Ledger()
    ledger.record(call_id="w", tool_name="excel-format_range", arguments={}, result_text="formatted", read_only=False)
    ledger.record(call_id="n", tool_name="filesystem-get_file_info", arguments={}, result_text="atime 1", read_only=True, nondeterministic=True)
    ledger.record(call_id="q", tool_name="snowflake-read_query", arguments={"query": "x"}, result_text="data_id: aaa\nN: 3\nN: 4",
                  read_only=True, volatile=["^data_id: [0-9a-f-]+$"])
    caller = _Caller({"snowflake-read_query": "data_id: bbb\nN: 3\nN: 4"})

    def claim(kind: str, value: object, call_id: str, span: str | None = None) -> Claim:
        return Claim(claim_id="c", text="t", kind=kind, value=value,
                     evidence=[ToolCallEvidence(type="tool_call", call_id=call_id, span=span)])

    unverifiable = await replay_claim(claim("action_done", "x", "w"), ledger, caller)
    assert unverifiable.ok and unverifiable.unverifiable and "no read-back" in unverifiable.reason
    assert (await replay_claim(claim("fact", "1", "n"), ledger, caller)).unverifiable
    assert (await replay_claim(claim("fact", 3, "q"), ledger, caller)).reason == "replay matches"
    assert not (await replay_claim(claim("fact", 7, "q"), ledger, caller)).ok
    assert (await replay_claim(claim("derived", 7, "q"), ledger, caller)).ok
    ctx = CheckContext(ledger=ledger)
    assert any(not r.ok for r in run_l0_l1(claim("fact", 7, "q", "N: 3"), ctx))
    assert all(r.ok for r in run_l0_l1(claim("derived", 7, "q", "N: 3"), ctx))
    assert not all(r.ok for r in run_l0_l1(claim("derived", 7, "q", "N: 9"), ctx))


def test_gym_prompts_name_the_fixed_workspace() -> None:
    for name in ("sa", "vanilla_mad", "self_verify", "typed", "evidence_risk", "researcher", "operator"):
        text = prompt_text("prompts/gym_v1", name)
        assert WORKSPACE in text
    assert "claim_done" in prompt_text("prompts/gym_v1", "sa")
    assert '"call_id"' in subagent_instructions("prompts/gym_v1", "researcher", schema=True)
    assert "plain-text report" in subagent_instructions("prompts/gym_v1", "operator", schema=False)
    # The fixture prompt set has no return-format files, so its sub-agent prompts are unchanged.
    assert subagent_instructions("prompts/v1", "researcher", schema=True) == prompt_text("prompts/v1", "researcher")


def test_schema_normalization_matches_the_gym_agent() -> None:
    from evcarry.env.gym_tools import normalize_schema

    schema = {
        "type": "object",
        "properties": {
            "link": {"type": ["object", "null"]},
            "end": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
            "level": {"type": "integer", "enum": [1, 2]},
            "rows": {"type": "array", "items": {"type": "array", "items": {}}},
            "extra": {"type": "object", "additionalProperties": True},
        },
    }
    normalize_schema(schema)
    props = schema["properties"]
    assert props["link"]["type"] == "object"
    assert props["end"] == {"type": "string", "default": None}
    assert "enum" not in props["level"]
    assert props["rows"]["items"]["items"] == {"type": "string"}
    assert "additionalProperties" not in props["extra"]


async def test_tool_errors_return_to_the_model_and_pages_are_not_cut_again() -> None:
    from evcarry.harness.tools import OVERLONG_META, VIEW_OVERLONG, ToolMeta, Toolset, build_function_tool

    class Ctx:
        tool_call_id = "call_1"

    class Raising:
        async def call_tool(self, name, arguments):
            raise ConnectionError("server gone")

    meta = ToolMeta("s-read", "read", {"type": "object", "properties": {}}, True, strict=False)
    ledger = Ledger()
    toolset = Toolset(catalog={"s-read": meta, VIEW_OVERLONG: OVERLONG_META}, caller=Raising(), all_tools=(),
                      researcher=(), operator=(), show_call_id=True, max_chars=10, ledger=ledger)
    tool = build_function_tool("s-read", Raising(), toolset)
    assert await tool.on_invoke_tool(Ctx(), "{}") == "Error calling s-read: ConnectionError: server gone"
    assert (await tool.on_invoke_tool(Ctx(), "{not json")).startswith("Error calling s-read: JSONDecodeError")
    assert ledger.entries == {}

    ledger.record(call_id="big", tool_name="s-read", arguments={}, result_text="a" * 10 + "b" * 10 + "TAIL", read_only=True)
    router = ToolRouter(None, {VIEW_OVERLONG: overlong_view(ledger, 10)})
    view = build_function_tool(VIEW_OVERLONG, router, toolset)
    page = await view.on_invoke_tool(Ctx(), '{"call_id": "big", "page": 2}')
    assert page.endswith("TAIL") and "truncated" not in page
    assert "integer" in await view.on_invoke_tool(Ctx(), '{"call_id": "big", "page": "next"}')


async def test_judge_evidence_shows_the_call_and_reads_back_once() -> None:
    from evcarry.protocol.audit import audit_return
    from evcarry.protocol.judge import LexicalJudge
    from evcarry.protocol.schema import SubagentReturn

    ledger = Ledger()
    ledger.record(call_id="py", tool_name="python_execute", arguments={"code": "print('Total orders: 718')"},
                  result_text="=== STDOUT ===\nTotal orders: 718", read_only=False)
    ledger.record(call_id="mail", tool_name="emails-send_email", arguments={"subject": "Report"},
                  result_text="Email sent", read_only=False, readback_tool="emails-search_emails",
                  readback_arguments={"query": "Report", "folder": "Sent"})
    seen: list[str] = []

    class Judge(LexicalJudge):
        def judge(self, claim, evidence_text):
            seen.append(evidence_text)
            return super().judge(claim, evidence_text)

    caller = _Caller({"emails-search_emails": "1. Subject: Report"})
    claims = [
        Claim(claim_id="a", text="There are 718 orders.", kind="fact", value=718,
              evidence=[ToolCallEvidence(type="tool_call", call_id="py", span="Total orders: 718")]),
        Claim(claim_id="b", text="Sent the report.", kind="action_done", value="Report",
              evidence=[ToolCallEvidence(type="tool_call", call_id="mail", span="Email sent")]),
    ]
    report = await audit_return(SubagentReturn(subagent_id="operator", status="complete", claims=claims),
                                ledger=ledger, caller=caller, checks=("L0", "L1", "L2", "L3"), sampler="risk",
                                rate=1.0, seed=1, judge=Judge())
    assert caller.calls == ["emails-search_emails"]
    by_text = {text.split(":", 1)[0]: text for text in seen}
    assert "print('Total orders: 718')" in next(t for t in seen if "python_execute" in t)
    assert "--- read-back of mail now (emails-search_emails):\n1. Subject: Report" in next(t for t in seen if "send_email" in t)
    assert by_text and not report.failures
