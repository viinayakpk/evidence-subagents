"""A20 (2026-10-25): the agent's database guard and the direct-write marker, without Docker."""

from __future__ import annotations

from types import SimpleNamespace

from evcarry.env.contamination import db_write_targets, db_writes
from evcarry.env.gym_docker import (
    AGENT_EXEC,
    AGENT_PG,
    PG_ENV,
    SF_ROLE,
    GymTaskEnv,
    McpServerSpec,
    lock_sql,
    server_exec_args,
)
from evcarry.env.toolathlon import gym_task_root, load_task
from evcarry.runs.config import load_experiment


def _spec(name: str, env: dict[str, str] | None = None) -> McpServerSpec:
    return McpServerSpec(name=name, command="uv", args=("run", name), env=env or {}, cwd=None, timeout=60)


def test_the_flagged_writes_are_two_part_names_in_mock_schemas() -> None:
    hits = {
        "cur.execute(\"UPDATE email.messages SET from_addr=%s WHERE id=%s\")": ["email.messages"],
        "DELETE FROM notion.pages WHERE id = %s": ["notion.pages"],
        "insert into  gcal.events (id) values (1)": ["gcal.events"],
        'TRUNCATE TABLE "wc"."orders"': ["wc.orders"],
        "CREATE TABLE IF NOT EXISTS public.tmp (a int)": ["public.tmp"],
        "ALTER TABLE canvas.quizzes ADD COLUMN x int": ["canvas.quizzes"],
    }
    for code, expected in hits.items():
        assert db_write_targets({"code": code}) == expected, code
    misses = [
        "SELECT * FROM email.messages",
        "INSERT INTO SUPPORT_CENTER.PUBLIC.NOTES VALUES (1)",  # snowflake three-part name: its own schema
        "UPDATE HR_ANALYTICS.PUBLIC.EMPLOYEES SET x = 1",
        "os.environ.update({'PG_HOST': 'x'})",
        "df.update(other)",
        "INSERT INTO results (a) VALUES (1)",  # unqualified: not a mock schema by name
    ]
    for code in misses:
        assert db_write_targets({"code": code}) == [], code


def test_denied_writes_are_marked() -> None:
    entries = {
        "a": SimpleNamespace(call_id="a", tool_name="python_execute", arguments={"code": "UPDATE email.messages SET x=1"},
                             result_text="psycopg2.errors.InsufficientPrivilege: permission denied for table messages"),
        "b": SimpleNamespace(call_id="b", tool_name="snowflake-write_query", arguments={"query": "DELETE FROM notion.pages"},
                             result_text="1 row affected"),
        "c": SimpleNamespace(call_id="c", tool_name="python_execute", arguments={"code": "SELECT 1"}, result_text=""),
    }
    rows = db_writes(SimpleNamespace(entries=entries))
    assert rows == [
        {"call_id": "a", "tool": "python_execute", "targets": "email.messages", "denied": True},
        {"call_id": "b", "tool": "snowflake-write_query", "targets": "notion.pages", "denied": False},
    ]


def test_lock_sql_grants_reads_only_and_rotates_the_write_password() -> None:
    sql = lock_sql("w-secret", "s-secret")
    assert f"CREATE ROLE {AGENT_PG['PGUSER']} LOGIN NOSUPERUSER" in sql and f"CREATE ROLE {SF_ROLE} LOGIN NOSUPERUSER" in sql
    assert "GRANT SELECT ON ALL TABLES IN SCHEMA %I TO evcarry_agent" in sql
    agent_grants = [line for line in sql.splitlines() if "evcarry_agent" in line and "GRANT" in line]
    assert agent_grants and all("SELECT" in line or "USAGE" in line for line in agent_grants), agent_grants
    assert "REVOKE CREATE ON DATABASE toolathlon_gym FROM PUBLIC" in sql
    assert sql.rstrip().endswith("ALTER ROLE eigent PASSWORD 'w-secret';")
    assert "PASSWORD 's-secret'" in sql


def test_servers_get_their_own_role_and_the_terminal_runs_as_the_agent() -> None:
    env = GymTaskEnv(load_task("gcal-snowflake-deadline-tracker", gym_task_root()), log_dir=None, agent_db_readonly=True)  # type: ignore[arg-type]
    assert env.container_env["PGUSER"] == AGENT_PG["PGUSER"] and env.container_env["PGPASSWORD"] == AGENT_PG["PGPASSWORD"]
    flags = lambda args: {args[i + 1] for i, item in enumerate(args) if item == "-e"}  # noqa: E731

    override, prefix = env._server_env("emails")
    _, args = server_exec_args(_spec("emails"), container="c", container_env=env.container_env, task_dir="/t", env_override=override, prefix=prefix)
    assert f"PGPASSWORD={env._write_secret}" in flags(args) and f"PG_PASSWORD={env._write_secret}" in flags(args)
    assert "PGUSER=eigent" in flags(args) and args[-2:] == ["run", "emails"] and "setpriv" not in args

    override, prefix = env._server_env("snowflake")
    yaml_env = {"PG_USER": "eigent", "PG_PASSWORD": "camel"}
    _, args = server_exec_args(_spec("snowflake", yaml_env), container="c", container_env=env.container_env, task_dir="/t", env_override=override, prefix=prefix)
    assert f"PG_USER={SF_ROLE}" in flags(args) and f"PG_PASSWORD={env._sf_secret}" in flags(args)
    assert env._write_secret not in " ".join(args)

    override, prefix = env._server_env("terminal")
    _, args = server_exec_args(_spec("terminal"), container="c", container_env=env.container_env, task_dir="/t", env_override=override, prefix=prefix)
    assert tuple(args[args.index("c") + 1 : args.index("c") + 1 + len(AGENT_EXEC)]) == AGENT_EXEC
    assert env._write_secret not in " ".join(args) and not any(flag.startswith("PGPASSWORD") for flag in flags(args))


def test_without_the_guard_nothing_changes() -> None:
    env = GymTaskEnv(load_task("gcal-snowflake-deadline-tracker", gym_task_root()), log_dir=None)  # type: ignore[arg-type]
    assert env.container_env["PGUSER"] == PG_ENV["PGUSER"] and env._server_env("emails") == (None, ())
    assert env._gym_env_args() == [] and env._agent_prefix() == [] and env._agent_exec() == []
    _, before = server_exec_args(_spec("emails"), container="c", container_env=env.container_env, task_dir="/t")
    _, same = server_exec_args(_spec("emails"), container="c", container_env=env.container_env, task_dir="/t", env_override=None, prefix=())
    assert before == same and "setpriv" not in before and "PG_PASSWORD=camel" in before


def test_headline_and_dev_configs_turn_the_guard_on() -> None:
    for name in ("headline_R_seed1", "headline_deepseek", "dev_verify"):
        assert load_experiment(f"configs/experiments/{name}.yaml").gym.agent_db_readonly is True
