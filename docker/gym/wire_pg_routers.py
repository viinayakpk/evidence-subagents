"""Connect the gym's Postgres routers to the woocommerce, canvas and notion MCP servers,
and make the Notion router store rows in the Notion API response shape.

At gym commit ed735ba each of these servers ships a router that serves its REST
API from the local database (pg-rest-router.ts, pg-canvas-router.ts,
pg-client.ts), but nothing constructs it: the servers still call
http://localhost:8081 (WordPress), https://localhost:8080 (Canvas) and the
Notion API, so every call fails with ECONNREFUSED. The calendar, forms and
youtube servers already use their routers. This script makes the other three
use theirs whenever PG_HOST is set (the gym bridges PGHOST to PG_HOST for every
server).

Notion response shape (NOTION_SHAPE_EDITS). The Notion router stores every page,
database and block exactly as the request sent it, and the gym's graders read
those rows straight from Postgres. Several graders parse properties the way the
Notion API returns them ("type" on every property value, "plain_text" on rich
text), so a correct page created through the API fails them. The edits to
pg-client.ts make writes store the response shape. They are additive: every key
the request sent is kept, including text.content.
  - Page property values get "id" ("title" for the title property, otherwise
    sha1(name)[:8]) and "type". A bare-array properties.title, the shape the
    advertised post-page schema asks for, becomes {"title": [...]} of type title.
  - Title and rich-text items get type, text {content, link}, annotations
    (defaults), plain_text and href.
  - Database schema properties get id, name and type; the database title and
    description are shaped as rich text.
  - patch-page merges the sent properties into the stored ones instead of
    replacing them all.
  - Block children sent as JSON strings (post-page's schema types them as
    strings) are parsed instead of being stored character by character, and
    the rich text of every inserted block is shaped.
The patched pg-client.ts carries NOTION_SHAPE_PATCH in a /*! */ comment so a
built image can be identified; a file that already carries it is refused.

Each edit is an exact string replacement; a missing or repeated anchor fails the
image build. Run inside the image against /opt/local_servers before `npm run build`.
"""

from __future__ import annotations

import sys
from pathlib import Path

NOTION_SHAPE_PATCH = "evcarry-notion-shape-v1"
PG_CLIENT = "notion-mcp-server/src/openapi-mcp-server/client/pg-client.ts"

_NOTION_MARKER = (
    f'/*! NOTION_SHAPE_PATCH = "{NOTION_SHAPE_PATCH}": notion rows are stored in the Notion API '
    "response shape (added by evcarry docker/gym/wire_pg_routers.py) */\n"
)

# Methods added to PgHttpClient, inserted before nowISO().
_NOTION_HELPERS = """\
  // evcarry-notion-shape-v1 helpers. The gym's graders read notion.* rows straight from
  // Postgres and expect the Notion API response shape, so writes are normalised here.
  // Every change is additive: keys the request sent are kept.
  private static readonly PROPERTY_TYPES: readonly string[] = [
    'title', 'rich_text', 'number', 'select', 'multi_select', 'status', 'date', 'people', 'files',
    'checkbox', 'url', 'email', 'phone_number', 'relation', 'rollup', 'formula', 'created_time',
    'created_by', 'last_edited_time', 'last_edited_by', 'unique_id',
  ]

  private isRecord(value: any): value is Record<string, any> {
    return value !== null && typeof value === 'object' && !Array.isArray(value)
  }

  // Property ids: the title property is always 'title'; the others are short and stable.
  private propertyId(name: string, type: string): string {
    if (type === 'title') return 'title'
    return crypto.createHash('sha1').update(name, 'utf8').digest('hex').slice(0, 8)
  }

  private propertyType(value: Record<string, any>): string | undefined {
    if (typeof value.type === 'string') return value.type
    return PgHttpClient.PROPERTY_TYPES.find((t) => t in value)
  }

  // Rich text items as the API returns them: type, text with content and link,
  // annotations, plain_text, href.
  private shapeRichText(items: any): any {
    if (!Array.isArray(items)) return items
    return items.map((item: any) => {
      if (!this.isRecord(item)) return item
      const type: string =
        typeof item.type === 'string'
          ? item.type
          : item.mention !== undefined
            ? 'mention'
            : item.equation !== undefined
              ? 'equation'
              : 'text'
      const out: Record<string, any> = { ...item, type }
      if (type === 'text') {
        const text: Record<string, any> = this.isRecord(item.text) ? item.text : {}
        let content = ''
        if (typeof text.content === 'string') content = text.content
        else if (text.content !== undefined && text.content !== null) content = String(text.content)
        else if (typeof item.text === 'string') content = item.text
        else if (typeof item.plain_text === 'string') content = item.plain_text
        out.text = { ...text, content, link: text.link ?? null }
      }
      out.annotations = {
        bold: false,
        italic: false,
        strikethrough: false,
        underline: false,
        code: false,
        color: 'default',
        ...(this.isRecord(item.annotations) ? item.annotations : {}),
      }
      if (typeof item.plain_text !== 'string') {
        if (type === 'text') out.plain_text = out.text.content
        else if (type === 'equation' && this.isRecord(item.equation)) out.plain_text = String(item.equation.expression ?? '')
        else out.plain_text = ''
      }
      if (item.href === undefined) {
        const link: any = type === 'text' ? out.text.link : null
        out.href = this.isRecord(link) && typeof link.url === 'string' ? link.url : null
      }
      return out
    })
  }

  // Page property values as the API returns them: id, type and the typed value.
  private shapePageProperties(properties: any): any {
    if (!this.isRecord(properties)) return properties
    const out: Record<string, any> = {}
    for (const [name, raw] of Object.entries(properties)) {
      const value: any = name === 'title' && Array.isArray(raw) ? { title: raw } : raw
      if (!this.isRecord(value)) {
        out[name] = value
        continue
      }
      const type = this.propertyType(value)
      if (type === undefined) {
        out[name] = value
        continue
      }
      const shaped: Record<string, any> = { ...value, id: value.id ?? this.propertyId(name, type), type }
      if (type === 'title' || type === 'rich_text') shaped[type] = this.shapeRichText(value[type] ?? [])
      out[name] = shaped
    }
    return out
  }

  // Database property schema as the API returns it: id, name, type and the type's config.
  private shapeDatabaseProperties(properties: any): any {
    if (!this.isRecord(properties)) return properties
    const out: Record<string, any> = {}
    for (const [name, raw] of Object.entries(properties)) {
      if (!this.isRecord(raw)) {
        out[name] = raw
        continue
      }
      const type = this.propertyType(raw)
      if (type === undefined) {
        out[name] = raw
        continue
      }
      out[name] = { ...raw, id: raw.id ?? this.propertyId(name, type), name: raw.name ?? name, type, [type]: raw[type] ?? {} }
    }
    return out
  }

  private shapeDatabaseField(field: string, value: any): any {
    if (field === 'title' || field === 'description') return this.shapeRichText(value)
    if (field === 'properties') return this.shapeDatabaseProperties(value)
    return value
  }

  // patch-page changes only the properties it names, as the API does.
  private async mergedPageProperties(pageId: string, properties: any): Promise<any> {
    const shaped = this.shapePageProperties(properties)
    if (!this.isRecord(shaped)) return shaped
    const { rows } = await this.pool.query(`SELECT properties FROM notion.pages WHERE id = $1`, [pageId])
    let current: any = rows[0]?.properties
    if (typeof current === 'string') {
      try {
        current = JSON.parse(current)
      } catch {
        current = {}
      }
    }
    return { ...(this.isRecord(current) ? current : {}), ...shaped }
  }

  // post-page's schema types children as strings; parse them into block objects.
  private parseBlockChild(child: any): any {
    if (typeof child !== 'string') return child
    try {
      const parsed = JSON.parse(child)
      if (this.isRecord(parsed)) return parsed
    } catch {
      // not JSON: keep the text as one paragraph rather than storing its characters
    }
    return { object: 'block', type: 'paragraph', paragraph: { rich_text: [{ type: 'text', text: { content: child } }] } }
  }

  private shapeBlockData(blockType: string, blockData: Record<string, any>): Record<string, any> {
    const content = blockData[blockType]
    if (!this.isRecord(content)) return blockData
    const shaped: Record<string, any> = { ...content }
    if (Array.isArray(content.rich_text)) shaped.rich_text = this.shapeRichText(content.rich_text)
    if (Array.isArray(content.caption)) shaped.caption = this.shapeRichText(content.caption)
    return { ...blockData, [blockType]: shaped }
  }

"""

NOTION_SHAPE_EDITS = [
    # Marker, as a legal comment so esbuild keeps it in bin/cli.mjs as well as in the source.
    (PG_CLIENT, "export class PgHttpClient {\n", _NOTION_MARKER + "export class PgHttpClient {\n"),
    (PG_CLIENT, "  private nowISO(): string {\n", _NOTION_HELPERS + "  private nowISO(): string {\n"),
    # create-a-database: shape the title, description and property schema.
    (
        PG_CLIENT,
        "JSON.stringify(title), JSON.stringify(description), JSON.stringify(icon), JSON.stringify(cover), "
        "JSON.stringify(properties), JSON.stringify(parent), is_inline],",
        "JSON.stringify(this.shapeRichText(title)), JSON.stringify(this.shapeRichText(description)), "
        "JSON.stringify(icon), JSON.stringify(cover), JSON.stringify(this.shapeDatabaseProperties(properties)), "
        "JSON.stringify(parent), is_inline],",
    ),
    # update-a-database: shape the same fields.
    (
        PG_CLIENT,
        "        const isJsonField = ['title', 'description', 'icon', 'cover', 'properties'].includes(field)\n"
        "        setClauses.push(`${field} = $${idx}`)\n"
        "        values.push(isJsonField ? JSON.stringify(body[field]) : body[field])\n",
        "        const isJsonField = ['title', 'description', 'icon', 'cover', 'properties'].includes(field)\n"
        "        setClauses.push(`${field} = $${idx}`)\n"
        "        values.push(isJsonField ? JSON.stringify(this.shapeDatabaseField(field, body[field])) : body[field])\n",
    ),
    # patch-page: merge shaped properties into the stored ones.
    (
        PG_CLIENT,
        "        const isJsonField = ['properties', 'icon', 'cover'].includes(field)\n"
        "        setClauses.push(`${field} = $${idx}`)\n"
        "        values.push(isJsonField ? JSON.stringify(body[field]) : body[field])\n",
        "        const isJsonField = ['properties', 'icon', 'cover'].includes(field)\n"
        "        setClauses.push(`${field} = $${idx}`)\n"
        "        values.push(field === 'properties' ? JSON.stringify(await this.mergedPageProperties(page_id, body[field])) "
        ": isJsonField ? JSON.stringify(body[field]) : body[field])\n",
    ),
    # post-page: shape the properties.
    (
        PG_CLIENT,
        "[id, now, now, JSON.stringify(parent), JSON.stringify(properties), JSON.stringify(icon), JSON.stringify(cover)],",
        "[id, now, now, JSON.stringify(parent), JSON.stringify(this.shapePageProperties(properties)), "
        "JSON.stringify(icon), JSON.stringify(cover)],",
    ),
    # insertBlocks (post-page children, patch-block-children): parse string children ...
    (
        PG_CLIENT,
        "    for (const child of children) {\n      const id = crypto.randomUUID()\n",
        "    for (const rawChild of children) {\n      const child = this.parseBlockChild(rawChild)\n"
        "      const id = crypto.randomUUID()\n",
    ),
    # ... and shape the rich text of each stored block.
    (
        PG_CLIENT,
        "blockType, hasChildren, JSON.stringify(blockData), position],",
        "blockType, hasChildren, JSON.stringify(this.shapeBlockData(blockType, blockData)), position],",
    ),
]

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
    *NOTION_SHAPE_EDITS,
]


def apply_edits(relative: str, text: str) -> str:
    """Apply, in order, every edit for one file to its source text and return the result.

    Fails (SystemExit, as the image build expects) when an anchor is missing or appears more
    than once, and when pg-client.ts already carries the Notion shape marker.
    """
    if relative == PG_CLIENT and NOTION_SHAPE_PATCH in text:
        raise SystemExit(f"{relative}: already carries {NOTION_SHAPE_PATCH}; refusing to patch it twice")
    for path, anchor, replacement in EDITS:
        if path != relative:
            continue
        if text.count(anchor) != 1:
            raise SystemExit(f"{relative}: expected exactly one anchor, found {text.count(anchor)}: {anchor[:60]!r}")
        text = text.replace(anchor, replacement)
    return text


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/opt/local_servers")
    for relative in dict.fromkeys(path for path, _, _ in EDITS):
        path = root / relative
        path.write_text(apply_edits(relative, path.read_text(encoding="utf-8")), encoding="utf-8")
        print(f"patched {relative}")


if __name__ == "__main__":
    main()
