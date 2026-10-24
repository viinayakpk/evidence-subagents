"""One real call: does the pinned OpenRouter endpoint take more than 128 tools?

Sends 150 dummy function tools through the harness's own provider path
(``configs/models.yaml`` entry, ``RecordingModel``, the pinned routing and
headers). Tools are sorted by name, as the harness sends them. One tool, at
sorted position 146, is the only one whose description mentions a zebra; the
model is asked to call the tool that returns the zebra code. The tool names
are hashes, so a call to that tool shows the model saw the definitions past
the 128th. An HTTP 4xx is a rejection.

Does nothing without ``--hit``. Refuses if the worst-case cost estimate is
not under ``--max-usd`` (default $0.01). Writes the request summary, the raw
response body (no headers, so no key), and the verdict to ``--out``.

    set -a; source .env; set +a
    uv run python scripts/probe_tool_limit.py --hit
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx2
from agents import FunctionTool
from agents.model_settings import ModelSettings
from agents.models.interface import ModelTracing

from evcarry.paths import resolve_repo_path
from evcarry.providers.openai_compat import RecordingModel, async_openai_client, http_client_with_capture, openrouter_headers
from evcarry.providers.registry import load_models, resolve_api_key, resolved_base_url, resolved_model_id
from evcarry.providers.usage import cost_usd

N_TOOLS = 150
TARGET_POSITION = 145  # 0-based position in sorted order: the 146th tool
MAX_OUTPUT_TOKENS = 512
PROMPT = "Call the tool that returns the zebra code. Do not reply with text."


def tool_names(n: int) -> list[str]:
    return sorted(f"probe_{hashlib.sha256(str(i).encode()).hexdigest()[:10]}" for i in range(n))


def build_tools(n: int, target: int) -> tuple[list[FunctionTool], str]:
    names = tool_names(n)

    async def _never(ctx: Any, args: str) -> str:
        return "unused"

    tools = []
    for position, name in enumerate(names):
        if position == target:
            description = "Returns the zebra code."
        else:
            description = f"Returns placeholder record {position}. Not related to animals."
        tools.append(
            FunctionTool(
                name=name,
                description=description,
                params_json_schema={
                    "type": "object",
                    "properties": {"note": {"type": "string"}},
                    "required": ["note"],
                    "additionalProperties": False,
                },
                on_invoke_tool=_never,
                strict_json_schema=True,
            )
        )
    return tools, names[target]


def worst_case_usd(request_chars: int, price: Any) -> float:
    # One token per 2 characters is a generous upper bound for JSON tool definitions.
    input_tokens = request_chars // 2
    return (input_tokens * price.input_cache_miss + MAX_OUTPUT_TOKENS * price.output) / 1_000_000


async def probe(model_key: str, out: Path, max_usd: float) -> int:
    config = load_models()[model_key]
    if config.kind != "openrouter":
        raise SystemExit(f"{model_key} is not an OpenRouter entry")
    price = config.price_table()
    tools, target_name = build_tools(N_TOOLS, TARGET_POSITION)
    estimate = worst_case_usd(len(json.dumps([tool.params_json_schema for tool in tools])) + 150 * N_TOOLS, price)
    if estimate >= max_usd:
        raise SystemExit(f"worst-case estimate ${estimate:.5f} is not under ${max_usd}")

    captured: dict[str, Any] = {}

    async def on_request(request: httpx2.Request) -> None:
        captured["request"] = json.loads(request.content)

    async def on_response(response: httpx2.Response) -> None:
        await response.aread()
        captured["status"] = response.status_code
        try:
            captured["body"] = json.loads(response.content)
        except json.JSONDecodeError:
            captured["body"] = response.text[:4000]

    headers = openrouter_headers(http_referer=config.http_referer, x_title=config.x_title)
    client = async_openai_client(
        base_url=resolved_base_url(config),
        api_key=resolve_api_key(config),
        default_headers=headers,
        http_client=http_client_with_capture(event_hooks={"request": [on_request], "response": [on_response]}),
    )
    model = RecordingModel(
        resolved_model_id(config), client, kind=config.kind, routing=config.routing, extra_headers=headers
    )
    error = None
    called: list[str] = []
    try:
        response = await model.get_response(
            system_instructions=None,
            input=PROMPT,
            model_settings=ModelSettings(max_tokens=MAX_OUTPUT_TOKENS),
            tools=tools,
            output_schema=None,
            handoffs=[],
            tracing=ModelTracing.DISABLED,
            previous_response_id=None,
            conversation_id=None,
            prompt=None,
        )
        called = [item.name for item in response.output if getattr(item, "type", None) == "function_call"]
    except Exception as exc:  # noqa: BLE001 - a rejection is the result we are looking for
        error = f"{type(exc).__name__}: {exc}"[:2000]
    record = model.history[-1] if model.history else None
    sent = captured.get("request") or {}
    status = captured.get("status")
    accepted = status is not None and 200 <= int(status) < 300
    report = {
        "probe": "tool_limit",
        "ts": datetime.now(timezone.utc).isoformat(),
        "model_key": model_key,
        "configured_model": config.model,
        "routing": config.routing,
        "tools_sent": len(sent.get("tools") or []),
        "target_tool": target_name,
        "target_sorted_position_1_based": TARGET_POSITION + 1,
        "prompt": PROMPT,
        "max_tokens": sent.get("max_tokens") or sent.get("max_completion_tokens"),
        "request_provider_object": sent.get("provider"),
        "http_status": status,
        "accepted": accepted,
        "called_tools": called,
        "called_target": target_name in called,
        "error": error,
        "served": record.identity.as_dict() if record is not None else None,
        "usage": record.usage.as_dict() if record is not None else None,
        "cost_usd_price_table": cost_usd(record.usage, price, datetime.now(timezone.utc)) if record else None,
        "worst_case_estimate_usd": estimate,
        "response_body": captured.get("body"),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    summary = {key: report[key] for key in ("tools_sent", "http_status", "accepted", "called_target", "called_tools", "error")}
    print(json.dumps(summary, indent=2))
    print(f"cost (price table): {report['cost_usd_price_table']}; provider-reported: "
          f"{(report['usage'] or {}).get('provider_reported_cost_usd')}")
    print(f"report: {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hit", action="store_true", help="actually make the paid call")
    parser.add_argument("--model", default="openrouter_ds_1017")
    parser.add_argument("--max-usd", type=float, default=0.01)
    parser.add_argument("--out", default=f"docs/evidence/tool_limit_probe_{datetime.now(timezone.utc):%Y-%m-%d}.json")
    args = parser.parse_args(argv)
    if not args.hit:
        tools, target = build_tools(N_TOOLS, TARGET_POSITION)
        print(f"probe_tool_limit: not asked to hit the API ({len(tools)} tools, target {target}).")
        return 0
    return asyncio.run(probe(args.model, resolve_repo_path(args.out), args.max_usd))


if __name__ == "__main__":
    sys.exit(main())
