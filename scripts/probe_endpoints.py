"""One real call per OpenRouter endpoint tag: does each endpoint serve this model with tools?

Sends one request per tag through the harness's own provider path
(``configs/models.yaml`` entry, ``RecordingModel``, headers, temperature 0 and
the entry's routing, with ``only`` and ``order`` narrowed to that one tag;
``--tags whole-set`` sends the routing unchanged, as runs do). The
request carries one function tool and asks the model to call it. A reply with
that tool call is "serves with tools"; an HTTP 4xx (for example "no endpoints
found matching your data policy") is a refusal.

Does nothing without ``--hit``. Refuses if the worst-case cost of all calls is
not under ``--max-usd`` (default $0.01). Writes each request's provider object,
the HTTP status, the served provider and tier, usage, OpenRouter's cost and the
raw response body (no headers, so no key) to ``--out``.

    set -a; source .env; set +a
    uv run python scripts/probe_endpoints.py --hit
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx2
from agents import FunctionTool
from agents.model_settings import ModelSettings
from agents.models.interface import ModelTracing

from evcarry.harness.run_settings import TEMPERATURE
from evcarry.paths import resolve_repo_path
from evcarry.providers.openai_compat import RecordingModel, async_openai_client, http_client_with_capture, openrouter_headers
from evcarry.providers.pin import check_pin
from evcarry.providers.registry import load_models, resolve_api_key, resolved_base_url, resolved_model_id

MAX_OUTPUT_TOKENS = 256
PROMPT = "Call get_probe_code with note set to 'probe'. Do not reply with text."
# Generous: the prompt, the tool definition and OpenRouter's wrapping are far below this.
INPUT_TOKENS_BOUND = 2000


def probe_tool() -> FunctionTool:
    async def _never(ctx: Any, args: str) -> str:
        return "unused"

    return FunctionTool(
        name="get_probe_code",
        description="Returns the probe code.",
        params_json_schema={
            "type": "object",
            "properties": {"note": {"type": "string"}},
            "required": ["note"],
            "additionalProperties": False,
        },
        on_invoke_tool=_never,
        strict_json_schema=True,
    )


def worst_case_usd(price: Any, calls: int) -> float:
    rates = price.worst_rates()
    return calls * (INPUT_TOKENS_BOUND * rates.input_cache_miss + MAX_OUTPUT_TOKENS * rates.output) / 1_000_000


WHOLE_SET = "whole-set"


async def probe_one(config: Any, tag: str) -> dict[str, Any]:
    # WHOLE_SET sends the entry's routing unchanged, as runs do.
    routing = dict(config.routing or {}) if tag == WHOLE_SET else {**(config.routing or {}), "only": [tag], "order": [tag]}
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
    client = client.with_options(max_retries=0)  # one request per tag: a 429 is reported, not retried
    model = RecordingModel(resolved_model_id(config), client, kind=config.kind, routing=routing, extra_headers=headers)
    model.retry_waits = ()
    error = None
    called: list[str] = []
    try:
        response = await model.get_response(
            system_instructions=None,
            input=PROMPT,
            model_settings=ModelSettings(max_tokens=MAX_OUTPUT_TOKENS, temperature=TEMPERATURE),
            tools=[probe_tool()],
            output_schema=None,
            handoffs=[],
            tracing=ModelTracing.DISABLED,
            previous_response_id=None,
            conversation_id=None,
            prompt=None,
        )
        called = [item.name for item in response.output if getattr(item, "type", None) == "function_call"]
    except Exception as exc:  # noqa: BLE001 - a refusal is a result
        error = f"{type(exc).__name__}: {exc}"[:2000]
    record = model.history[-1] if model.history else None
    sent = captured.get("request") or {}
    status = captured.get("status")
    return {
        "tag": tag,
        "ts": datetime.now(timezone.utc).isoformat(),
        "request_provider_object": sent.get("provider"),
        "request_service_tier": sent.get("service_tier"),
        "request_temperature": sent.get("temperature"),
        "tools_sent": len(sent.get("tools") or []),
        "http_status": status,
        "served_with_tools": status == 200 and "get_probe_code" in called,
        "called_tools": called,
        "error": error,
        "served": record.identity.as_dict() if record is not None else None,
        # Checked against the whole configured set, as runs check it.
        "pin_check_against_set": check_pin(record.identity, config.routing).as_dict() if record and status == 200 else None,
        "usage": record.usage.as_dict() if record is not None else None,
        "response_body": captured.get("body"),
    }


async def probe(model_key: str, tags: list[str], out: Path, max_usd: float) -> int:
    config = load_models()[model_key]
    if config.kind != "openrouter":
        raise SystemExit(f"{model_key} is not an OpenRouter entry")
    estimate = worst_case_usd(config.price_table(), len(tags))
    if estimate >= max_usd:
        raise SystemExit(f"worst-case estimate ${estimate:.5f} is not under ${max_usd}")
    results = [await probe_one(config, tag) for tag in tags]
    billed = sum((item["usage"] or {}).get("provider_reported_cost_usd") or 0 for item in results)
    report = {
        "probe": "endpoints",
        "model_key": model_key,
        "configured_model": config.model,
        "routing": config.routing,
        "prompt": PROMPT,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "worst_case_estimate_usd": estimate,
        "billed_usd": billed,
        "results": results,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    for item in results:
        served = item["served"] or {}
        print(
            f"{item['tag']:24} status={item['http_status']} tools_ok={item['served_with_tools']} "
            f"provider={served.get('response_provider')} tier={served.get('service_tier')} "
            f"pin={(item['pin_check_against_set'] or {}).get('ok')} "
            f"cost={(item['usage'] or {}).get('provider_reported_cost_usd')} error={(item['error'] or '')[:300]}"
        )
    print(f"billed total: ${billed:.6f}; report: {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hit", action="store_true", help="actually make the paid calls")
    parser.add_argument("--model", default="openrouter_ds_1017")
    parser.add_argument("--tags", nargs="*", help=f"endpoint tags (default: the entry's `only` list); {WHOLE_SET} sends the routing as configured")
    parser.add_argument("--max-usd", type=float, default=0.01)
    parser.add_argument("--out", default=f"docs/evidence/endpoint_probe_{datetime.now(timezone.utc):%Y-%m-%d}.json")
    args = parser.parse_args(argv)
    config = load_models()[args.model]
    tags = args.tags or list((config.routing or {}).get("only") or (config.routing or {}).get("order") or [])
    if not args.hit:
        print(f"probe_endpoints: not asked to hit the API (tags {tags}, worst case ${worst_case_usd(config.price_table(), len(tags)):.5f}).")
        return 0
    return asyncio.run(probe(args.model, tags, resolve_repo_path(args.out), args.max_usd))


if __name__ == "__main__":
    sys.exit(main())
