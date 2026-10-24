"""Connect the gym's Postgres routers to the woocommerce, canvas and notion MCP servers.

At gym commit ed735ba each of these servers ships a router that serves its REST
API from the local database (pg-rest-router.ts, pg-canvas-router.ts,
pg-client.ts), but nothing constructs it: the servers still call
http://localhost:8081 (WordPress), https://localhost:8080 (Canvas) and the
Notion API, so every call fails with ECONNREFUSED. The calendar, forms and
youtube servers already use their routers. This script makes the other three
use theirs whenever PG_HOST is set (the gym bridges PGHOST to PG_HOST for every
server). Each edit is an exact string replacement; a missing anchor fails the
image build. Run inside the image against /opt/local_servers before `npm run build`.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/opt/local_servers")

EDITS = [
    (
        "woocommerce-mcp/src/services/base.ts",
        "import axios, { AxiosInstance } from 'axios';\n",
        "import axios, { AxiosInstance } from 'axios';\nimport { PgRestRouter } from './pg-rest-router.js';\n",
    ),
    (
        "woocommerce-mcp/src/services/base.ts",
        "    constructor() {\n        this.siteUrl = process.env.WORDPRESS_SITE_URL!;\n",
        "    constructor() {\n"
        "        if (process.env.PG_HOST) {\n"
        "            this.siteUrl = '';\n"
        "            this.consumerKey = '';\n"
        "            this.consumerSecret = '';\n"
        "            this.client = new PgRestRouter() as unknown as AxiosInstance;\n"
        "            return;\n"
        "        }\n"
        "        this.siteUrl = process.env.WORDPRESS_SITE_URL!;\n",
    ),
    (
        "mcp-canvas-lms/src/client.ts",
        "import axios, { AxiosInstance, AxiosError } from 'axios';\n",
        "import axios, { AxiosInstance, AxiosError } from 'axios';\nimport { PgCanvasRouter } from './pg-canvas-router.js';\n",
    ),
    (
        "mcp-canvas-lms/src/client.ts",
        "    this.retryDelay = options?.retryDelay ?? 1000;\n\n    this.client = axios.create({",
        "    this.retryDelay = options?.retryDelay ?? 1000;\n\n"
        "    if (process.env.PG_HOST) {\n"
        "      this.client = new PgCanvasRouter() as unknown as AxiosInstance;\n"
        "      return;\n"
        "    }\n\n"
        "    this.client = axios.create({",
    ),
    (
        "notion-mcp-server/src/openapi-mcp-server/mcp/proxy.ts",
        "import { HttpClient, HttpClientError } from '../client/http-client.js'\n",
        "import { HttpClient, HttpClientError } from '../client/http-client.js'\nimport { PgHttpClient } from '../client/pg-client.js'\n",
    ),
    (
        "notion-mcp-server/src/openapi-mcp-server/mcp/proxy.ts",
        "    this.httpClient = new HttpClient(\n",
        "    this.httpClient = new (process.env.PG_HOST ? (PgHttpClient as unknown as typeof HttpClient) : HttpClient)(\n",
    ),
]


def main() -> None:
    for relative, anchor, replacement in EDITS:
        path = ROOT / relative
        text = path.read_text(encoding="utf-8")
        if text.count(anchor) != 1:
            raise SystemExit(f"{relative}: expected exactly one anchor, found {text.count(anchor)}: {anchor[:60]!r}")
        path.write_text(text.replace(anchor, replacement), encoding="utf-8")
        print(f"patched {relative}")


if __name__ == "__main__":
    main()
