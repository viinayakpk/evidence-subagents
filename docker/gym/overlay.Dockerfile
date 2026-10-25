# Finish the MCP servers that the upstream Toolathlon-Gym Dockerfile leaves
# unbuilt at gym commit ed735ba. Their task YAMLs point at build outputs that
# are neither committed nor produced by the base image:
#   notion-mcp-server/bin/cli.mjs, mcp-canvas-lms/build/index.js,
#   mcp-npx-fetch/dist/index.js, woocommerce-mcp/dist/index.js,
#   playwright-mcp/lib/program.js, 12306-mcp/node_modules.
# canvas, notion and woocommerce import `pg` (the gym's Postgres-backed
# patches) without declaring it, so it is added at the version 12306-mcp uses.
# Those three servers also never construct their Postgres routers;
# wire_pg_routers.py connects them (see its docstring).
# Then every uv-launched Python server gets the environment `uv run` will ask
# for at start, so no server start touches the network (see below).
# The submodule itself is not modified.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# offline-v1: no MCP server start and no python_execute call downloads anything
# (gym_docker.py network_internal relies on it).
LABEL evcarry.overlay=offline-v1

COPY wire_pg_routers.py /tmp/wire_pg_routers.py
RUN python3 /tmp/wire_pg_routers.py /opt/local_servers

RUN set -e; for dir in 12306-mcp mcp-canvas-lms notion-mcp-server mcp-npx-fetch woocommerce-mcp playwright-mcp; do \
        echo "=== $dir"; cd "/opt/local_servers/$dir"; \
        npm install --ignore-scripts --no-audit --no-fund; \
        case "$dir" in mcp-canvas-lms|notion-mcp-server|woocommerce-mcp) \
            npm install --no-save --ignore-scripts --no-audit --no-fund pg@^8.20.0 @types/pg@^8.18.0;; \
        esac; \
        npm run build; \
    done

# Six Python servers pin an interpreter in .python-version that differs from the
# 3.12 the base image synced them with (excel-mcp-server 3.10, cli-mcp-server 3.13,
# mcp-google-sheets 3.13, yahoo-finance-mcp 3.11, arxiv-mcp-server 3.11,
# mcp-scholarly 3.11). `uv run` follows the pin, so at every server start it removed
# .venv, downloaded that CPython and the locked packages, and rebuilt it (pilot
# logs mcp_{excel,terminal,google_sheet,arxiv_local,yahoo-finance}.log: "Creating
# virtual environment"; mcp-google-sheets, synced here with --python 3.12 before,
# was rebuilt the same way). Syncing without --python follows the same pin and
# uv.lock with the same uv binary, so these are the interpreters and packages those
# runs downloaded.
RUN set -e; for dir in excel-mcp-server cli-mcp-server mcp-google-sheets yahoo-finance-mcp arxiv-mcp-server mcp-scholarly; do \
        echo "=== $dir"; cd "/opt/local_servers/$dir"; uv sync --frozen; \
    done

# The gym's python_execute runs `uv run --directory <agent workspace>`. The
# workspace sits under /workspace, the gym's own uv project, so without this the
# first call in every container downloads and builds that environment
# (Python 3.12.11, ~250 locked packages). Built once here from uv.lock instead.
RUN cd /workspace && uv sync --frozen

# Fail the build if starting a uv-launched server, or python_execute, would still
# need the network: the same `uv --directory <dir> run` shape as
# configs/mcp_servers/*.yaml (Word adds --python 3.12 there), offline.
# mcp-youtube-transcript starts with its venv's python directly, not through uv.
RUN set -e; for dir in arxiv-latex-mcp arxiv-mcp-server cli-mcp-server emails-mcp excel-mcp-server \
            mcp-google-sheets mcp-scholarly mcp-snowflake-server Office-PowerPoint-MCP-Server \
            pdf-tools-mcp yahoo-finance-mcp; do \
        echo "=== offline start check: $dir"; \
        uv --directory "/opt/local_servers/$dir" run --offline python -c "import sys; print(sys.version)"; \
    done; \
    echo "=== offline start check: Office-Word-MCP-Server"; \
    uv --directory /opt/local_servers/Office-Word-MCP-Server run --offline --python 3.12 python -c "import sys; print(sys.version)"; \
    echo "=== offline start check: /workspace (python_execute)"; \
    uv --directory /workspace run --offline python -c "import sys; print(sys.prefix)"

WORKDIR /workspace
