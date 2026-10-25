"""A20: with ``agent_db_readonly`` the agent's own code reads the run's database but cannot write it,
the MCP servers still write, and the gym's grader still runs. Real gym tasks in Docker, no model. $0."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from evcarry.env.gym_docker import AGENT_PG, AGENT_UID, GymImages, GymTaskEnv, preflight, require_offline_overlay
from evcarry.env.contamination import db_write_targets
from evcarry.env.isolation import EnvironmentUnavailable
from evcarry.env.toolathlon import gym_task_root, load_task
from evcarry.ledger.store import read_ledger
from evcarry.paths import repo_root
from evcarry.protocol.replayer import render_tool_result


def _unavailable() -> str | None:
    try:
        preflight(GymImages.default())
        require_offline_overlay(GymImages.default())
    except EnvironmentUnavailable as exc:
        return str(exc)
    if not (gym_task_root() / "gcal-snowflake-deadline-tracker").is_dir():
        return "toolathlon_gym submodule is not checked out"
    return None


_REASON = _unavailable()
pytestmark = [pytest.mark.docker_gym, pytest.mark.skipif(_REASON is not None, reason=f"gym Docker environment: {_REASON}")]

_PG = """
import os, psycopg2
def connect(**over):
    args = dict(host=os.environ["PGHOST"], port=os.environ["PGPORT"], dbname=os.environ["PGDATABASE"],
                user=os.environ["PGUSER"], password=os.environ["PGPASSWORD"])
    args.update(over)
    return psycopg2.connect(**args)
def run(sql, **over):
    try:
        conn = connect(**over); conn.autocommit = True
        cur = conn.cursor(); cur.execute(sql)
        print("OK", cur.fetchone() if cur.description else cur.rowcount)
    except Exception as exc:
        print("ERR", type(exc).__name__, str(exc).splitlines()[0])
"""


def _env(task_name: str, tmp_path: Path) -> GymTaskEnv:
    return GymTaskEnv(
        load_task(task_name, gym_task_root()), log_dir=tmp_path, network_internal=True, hide_task_dir=True,
        agent_db_readonly=True,
    )


async def _py(env: GymTaskEnv, body: str) -> list[str]:
    out = await env.python_execute({"code": _PG + body, "timeout": 60})
    assert "Return code: 0" in out, out
    return [line for line in out.splitlines() if line.startswith(("OK", "ERR", "VAL"))]


async def test_agent_code_reads_but_cannot_write_and_mcp_servers_still_write(tmp_path: Path) -> None:
    async with _env("gcal-snowflake-deadline-tracker", tmp_path) as env:
        assert env.info.db_guard and env.info.db_guard["locked"] is True
        lines = await _py(env, f"""
print("VAL", os.getuid(), os.environ["PGUSER"])
run("SELECT count(*) FROM email.messages")
run("SELECT count(*) FROM sf_data.\\"SUPPORT_CENTER__PUBLIC__TICKETS\\"")
run("UPDATE email.messages SET from_addr='x@y.z'")
run("INSERT INTO notion.pages (id) VALUES ('evc-test')")
run("DELETE FROM gcal.events")
run("CREATE TABLE public.evc_t (a int)")
run("CREATE SCHEMA evc_s")
run("DROP TABLE email.messages")
run("UPDATE sf_data.\\"SUPPORT_CENTER__PUBLIC__TICKETS\\" SET \\"PRIORITY\\"='Low'")
run("SELECT 1", user="eigent", password="camel")
run("SELECT 1", user="postgres", password="camel")
""")
        assert lines[0] == f"VAL {AGENT_UID} {AGENT_PG['PGUSER']}", lines
        assert lines[1].startswith("OK (") and lines[2].startswith("OK ("), lines  # reads work
        for line in lines[3:10]:
            assert line.startswith("ERR") and ("permission denied" in line or "must be owner" in line), lines
        assert lines[10].startswith("ERR") and "password authentication failed" in lines[10], lines
        assert lines[11].startswith("ERR"), lines

        # The MCP servers' write password is in no process environment the agent can read.
        scan = await env.python_execute({"code": (
            "import os\nseen, denied = set(), 0\n"
            "for pid in filter(str.isdigit, os.listdir('/proc')):\n"
            "    try:\n        data = open(f'/proc/{pid}/environ','rb').read()\n"
            "    except OSError:\n        denied += 1; continue\n"
            "    for item in data.split(b'\\0'):\n"
            "        if item.startswith((b'PGPASSWORD=', b'PG_PASSWORD=')): seen.add(item.split(b'=',1)[1].decode())\n"
            "print('SEEN', sorted(seen), 'DENIED', denied)\n"
        )})
        assert f"SEEN ['{AGENT_PG['PGPASSWORD']}']" in scan and "DENIED 0" not in scan, scan
        assert env._write_secret not in scan and env._sf_secret not in scan

        # Through the snowflake mock: two-part names pass through untranslated; its role writes only sf schemas.
        forged = render_tool_result(await env.call_tool(
            "snowflake-write_query", {"query": "UPDATE email.messages SET from_addr = 'sla-monitor@company.com'"}))
        assert "permission denied" in forged, forged
        made = render_tool_result(await env.call_tool(
            "snowflake-write_query", {"query": "CREATE TABLE SUPPORT_CENTER.PUBLIC.EVC_NOTES (NOTE TEXT)"}))
        added = render_tool_result(await env.call_tool(
            "snowflake-write_query", {"query": "INSERT INTO SUPPORT_CENTER.PUBLIC.EVC_NOTES VALUES ('ok')"}))
        assert "rror" not in made + added, (made, added)
        read = render_tool_result(await env.call_tool("snowflake-read_query", {"query": "SELECT * FROM SUPPORT_CENTER.PUBLIC.EVC_NOTES"}))
        assert "ok" in read, read

        # MCP writes: e-mail, calendar event, Notion page.
        sent = render_tool_result(await env.call_tool(
            "emails-send_email", {"to": "support-lead@company.com", "subject": "EVC guard test", "body": "hello"}))
        assert "success" in sent.lower(), sent
        event = render_tool_result(await env.call_tool("google_calendar-create_event", {
            "summary": "EVC guard test", "start": {"dateTime": "2026-03-07T09:00:00Z"}, "end": {"dateTime": "2026-03-07T09:30:00Z"}}))
        assert "Event created" in event, event
        page = render_tool_result(await env.call_tool("notion-API-post-page", {
            "parent": {"page_id": "00000000-0000-0000-0000-000000000000"},
            "properties": {"title": [{"text": {"content": "EVC guard test"}}]}}))
        assert '"object":"page"' in page.replace(" ", ""), page[:300]
        lines = await _py(env, """
run("SELECT count(*) FROM email.messages WHERE subject='EVC guard test'")
run("SELECT count(*) FROM gcal.events WHERE summary='EVC guard test'")
run("SELECT count(*) FROM notion.pages WHERE properties::text LIKE '%EVC guard test%'")
run("SELECT count(*) FROM sf_data.\\"SUPPORT_CENTER__PUBLIC__EVC_NOTES\\"")
""")
        assert lines == ["OK (1,)"] * 4, lines

        env.claimed_done = True
        result = await env.evaluate()
        assert env.info.db_guard["unlocked"] is True
        output = result["eval"]["stdout_tail"] + result["eval"]["stderr_tail"]
        assert result["eval_ran"] is True and "password authentication failed" not in output, output[-800:]
        assert "[PASS] At least 1 email sent" in output and "[PASS] noise emails preserved" in output, output[-800:]


async def test_the_terminal_server_runs_as_the_agent(tmp_path: Path) -> None:
    async with _env("canvas-exam-prep-scheduler", tmp_path) as env:
        tool = next(name for name in env.catalog if name.startswith("terminal-") and "command" in name)
        out = render_tool_result(await env.call_tool(tool, {"command": "whoami && env"}))
        assert "evcarry_agent" in out.split(), out[:500]
        assert f"PGUSER={AGENT_PG['PGUSER']}" in out and env._write_secret not in out, out[:800]


_LEDGER = repo_root() / "runs" / "dev_verify_v2" / "ledgers" / "gcal-snowflake-deadline-tracker__sa__1.json"
_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


@pytest.mark.skipif(not _LEDGER.is_file(), reason="dev v2 ledger not present")
async def test_the_dev_v2_tool_calls_pass_the_gcal_grader_without_the_sql_writes(tmp_path: Path) -> None:
    """Replay the SA run's own calls in order, minus the direct SQL writes, with ids the mocks hand out mapped."""
    ledger = read_ledger(_LEDGER)
    skipped, ids = [], {}
    async with _env("gcal-snowflake-deadline-tracker", tmp_path) as env:
        for entry in ledger.entries.values():
            if db_write_targets(entry.arguments):
                skipped.append(entry.tool_name)
                continue
            text = json.dumps(entry.arguments)
            for old, new in ids.items():
                text = text.replace(old, new)
            result = await env.call_tool(entry.tool_name, json.loads(text))
            fresh = render_tool_result(result) if not isinstance(result, str) else result
            for old, new in zip(_ID.findall(entry.result_text or ""), _ID.findall(fresh), strict=False):
                ids.setdefault(old, new)
        assert skipped and set(skipped) == {"python_execute"}, skipped  # the UPDATEs of email.messages / account_config
        result = await env.evaluate()
    output = result["eval"]["stdout_tail"]
    assert result["passed"] is True, output[-1500:]
