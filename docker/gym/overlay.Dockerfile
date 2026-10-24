# Finish the MCP servers that the upstream Toolathlon-Gym Dockerfile leaves
# unbuilt at gym commit ed735ba. Their task YAMLs point at build outputs that
# are neither committed nor produced by the base image:
#   notion-mcp-server/bin/cli.mjs, mcp-canvas-lms/build/index.js,
#   mcp-npx-fetch/dist/index.js, woocommerce-mcp/dist/index.js,
#   playwright-mcp/lib/program.js, 12306-mcp/node_modules,
#   mcp-google-sheets/.venv (otherwise synced from the network on every start).
# canvas, notion and woocommerce import `pg` (the gym's Postgres-backed
# patches) without declaring it, so it is added at the version 12306-mcp uses.
# Those three servers also never construct their Postgres routers;
# wire_pg_routers.py connects them (see its docstring).
# The submodule itself is not modified.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

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

RUN cd /opt/local_servers/mcp-google-sheets && uv sync --python 3.12

# The gym's python_execute runs `uv run --directory <agent workspace>`. The
# workspace sits under /workspace, the gym's own uv project, so without this the
# first call in every container downloads and builds that environment
# (Python 3.12.11, ~250 locked packages). Built once here from uv.lock instead.
RUN cd /workspace && uv sync --frozen

WORKDIR /workspace
