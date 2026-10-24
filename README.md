# evcarry

Research harness for evidence-carrying sub-agent returns and spot-check audits. Milestones M0–M3 run here for $0 on a scripted model. DeepSeek, OpenRouter, and a local vLLM server are config entries for the workstation later.

## Setup

```bash
uv sync --extra dev --extra analysis
cp .env.example .env
```

`.env.example` lists `DEEPSEEK_API_KEY`, `OPENROUTER_API_KEY`, the free-debug trio, and `VLLM_API_KEY=EMPTY`. Leave them empty in git. No key is committed.

Toolathlon-Gym is a submodule at `third_party/toolathlon_gym` (Apache-2.0, commit `ed735ba0af71d82688a55f952a6ea11479ae7206`). It is used to load task metadata and to freeze the task lists. This checkout does not copy the unlicensed Toolathlon agent repo.

```bash
git submodule update --init third_party/toolathlon_gym
```

## What a mock run does

```bash
make test
make mock
make summarize
```

`make mock` runs `configs/experiments/mock_e2e.yaml`: three fixture tasks, arms `sa`, `vanilla_mad`, and `evidence_risk`, one seed. Traces land in `runs/mock_e2e/*.jsonl` (gitignored). `evcarry summarize runs/mock_e2e` prints a table of `run_end` rows. `runs/mock_e2e/index.parquet` is the same rows as a table.

`evcarry check CONFIG` validates a file. `evcarry run CONFIG --dry-run` prints the grid, the budget, and the Docker plan. A dry run does not read an API key and does not start Docker.

Exit codes from `evcarry run`: spend cap `2`, weekday peak window `3`, missing API key `4`, Docker unavailable `5`.

## Config

`configs/models.yaml` is the provider list.

| name | kind | notes |
| --- | --- | --- |
| `deepseek_flash` | `deepseek` | `https://api.deepseek.com`, model id `deepseek-flash`, `DEEPSEEK_API_KEY`, billable |
| `local_primary` | `vllm` | `http://127.0.0.1:8000/v1`, `primitive-ai/Qwen3.8-Flash-Next-NVFP4`, key `EMPTY` |
| `mock` | `scripted` | no network |
| `openrouter_pinned` | `openrouter` | see below |
| `free_debug` | `openai_compatible` | `FREE_DEBUG_BASE_URL`, `FREE_DEBUG_MODEL`, `FREE_DEBUG_API_KEY` |

Prices are dated files in `configs/prices/`. DeepSeek off-peak is $0.15 cache-miss / $0.003 cache-hit / $0.60 output per million tokens. Peak is double, on weekdays 01:00–04:00 and 06:00–10:00 UTC, end exclusive. Weekends are off-peak. After the 2026-10-25 DST change the Prague clock labels move; the UTC windows stay put. `offpeak_only` refuses a billable run that starts inside a peak window. Mock and other non-billable models skip both the window and the cap.

Experiment files live in `configs/experiments/`. Paths are relative to the repo root. The frozen lists are `tasks/pool160.txt`, `tasks/subset70.txt`, and `tasks/pilot10.txt`, checked against `tasks/SHA256SUMS`.

## OpenRouter

OpenRouter is a first-class provider, not a comment. The example entry is `openrouter_pinned`:

- base URL `https://openrouter.ai/api/v1`
- key `OPENROUTER_API_KEY` (empty in `.env.example`)
- optional `HTTP-Referer` from `http_referer`
- `X-Title` and `X-OpenRouter-Title` both set from `x_title`
- `X-OpenRouter-Metadata: enabled` so the response can name the endpoint that served the call
- routing `order: [deepseek]`, `allow_fallbacks: false`, `require_parameters: true`

Each `llm_call` records `served_provider` and `served_model` from the raw HTTP body (`openrouter_metadata.endpoints.available[]` with `selected: true`, else the last HTTP 200 attempt, else a top-level `provider` string). The Agents SDK's `provider_data.model` is the model we requested, so it is not used. If the metadata is missing, `served_provider` is null and `provider_unobserved` is true. Cache tokens are read from the raw usage object (`prompt_cache_hit_tokens` / `prompt_cache_miss_tokens`, or `prompt_tokens_details.cached_tokens`). `usage.cost` is stored as `provider_reported_cost_usd` and is not added into the price table.

`configs/prices/openrouter_example_2026-10-24.yaml` copies the DeepSeek numbers so a debug budget has a figure. It is labeled example-only and is not a rate card for the paper. The headline model stays direct DeepSeek `deepseek-flash`.

`configs/experiments/openrouter_debug.yaml` is a small pinned grid (pilot tasks, four headline arms, cap $5, schedule `any`). It uses the Docker environment and will refuse on this machine.

Confirm the slug `deepseek/deepseek-chat` on OpenRouter before any paid call.

## Running M4 on the workstation

Two commands. Neither is run in CI.

The first one stays inside this harness and the fixture tasks. It needs a DeepSeek key and refuses to start during a peak window or after the cap:

```bash
export DEEPSEEK_API_KEY=...
uv run evcarry run configs/experiments/deepseek_smoke.yaml
```

`deepseek_smoke` is in-process, arm `sa`, the three fixture tasks, cap $1, `offpeak_only`. It does not start Docker. The served model string from the response is written on every `llm_call`, because the alias `deepseek-flash` can move.

The gym pilot is the real task environments. It is still a stub: Docker must exist and `EVCARRY_TOOLATHLON_DOCKER=1` must be set, and even then this milestone does not place our agent inside the gym containers. `evcarry run configs/experiments/pilot_deepseek.yaml` exits 5 until that attach work lands. On the workstation, `bash third_party/toolathlon_gym/run_parallel.sh` is the gym's own runner for checking that Postgres and the MCP servers come up. Our `DockerIsolation.plan()` returns that command and does not execute it here.

Headline, when the attach exists: `make headline` (`configs/experiments/headline_deepseek.yaml`, 70 tasks, four arms, two seeds, cap $80, `offpeak_only`, eight parallel tasks, judge `local_primary` so L3 is not billed to DeepSeek). Parallel billable runs can overshoot the cap by about one batch, because the cap is checked at the start of each run.

M6 hooks, not used yet: `serving/serve_primary.sh`, `serving/serve_fallback.sh`, `serving/gpu_check.sh`, `serving/smoke_test.py` (no-op unless `--hit`).

## Arms and audits

| arm | delegation | schema | checks |
| --- | --- | --- | --- |
| `sa` | no | no | none |
| `vanilla_mad` | yes | free text | none |
| `self_verify` | yes | free text, different system prompt | none |
| `typed` | yes | `SubagentReturn`, evidence may be empty | pointers skipped |
| `evidence` | yes | yes | L0/L1, sample rate 0 |
| `evidence_random` | yes | yes | L0–L3, random sample |
| `evidence_risk` | yes | yes | L0–L3, risk sample |

The arm name picks the sampler. YAML supplies the rate for the spot-check arms. L0/L1 failures reject a return even when that claim was not sampled. The policy accepts, or re-delegates once with the failed claims named, then falls back to the main agent. There is no third attempt.

The fault injector is off unless an experiment sets `inject`. The decision key is `task|seed|return_index` and does not include the arm. `make fault-mock` prints the synthetic catch table. Notebook `analysis/03_fault_injection.ipynb` calls the same function.

## Traces

One JSONL file per run: `{task}__{arm}__{model}__{seed}.jsonl`. A run is finished when a `run_end` event exists. A restart skips that file and does not rewrite it. Events include `run_start`, `llm_call`, `tool_call`, `subagent_return`, `injection`, `check`, `audit`, `policy`, `run_end`. JSONL is the source of truth.

The system prompt is the frozen file under `prompts/v1/` and is the cache prefix. The task's `docs/task.md` is the user message. The gym `agent_system_prompt.md` is hashed into `run_start` and is not sent. Tool names are sorted.

## Deviations

1. Docker and Toolathlon containers are stubbed. The gym submodule loads tasks. M0–M3 does not run our agent inside gym containers, and it does not run gym `preprocess` (that needs Postgres). Fixture tasks set `evcarry_local_eval` and use a small local checker. Real tasks return `not_run`.
2. MS-Eval appendix prompts are not copied. `prompts/v1/` is original text.
3. L3 is a lexical stand-in until a local model exists. It requires hyphenated tokens, a non-numeric string value, and list items to appear in the evidence. Ordinary words are ignored, so omissions of list items whose remaining words still appear are not caught. The synthetic omission catch rate is 0 on purpose. `ModelJudge` is the hook for a real model that returns `{"supports": bool, "reason": str}`.
4. Agents SDK 0.24 already runs function tools from one turn concurrently (`tool_execution.max_function_tool_concurrency`). No batch tool was added. The SDK 0.24 install also has no `agents.testing.ScriptedModel`; `FixtureModel` implements the public `Model` interface.
5. Peak windows stay in UTC. The Prague display shifts on 2026-10-25; the UTC windows do not.
6. `pool160`, `subset70`, and `pilot10` are an algorithmic stratified sample (seed `20261024`, largest remainder, then `random.Random(seed)` inside sorted stratum names). They are not hand-reviewed. Do not re-roll them; `scripts/freeze_task_lists.py` is only for a new gym commit.
7. OpenRouter requests send `X-OpenRouter-Metadata: enabled`, and both `X-Title` and `X-OpenRouter-Title`.
8. SDK `Usage` drops DeepSeek cache fields (`cached_tokens` stays 0). The raw HTTP body is the accounting source. `provider_data.model` is the requested model.
9. Strict `output_type=SubagentReturn` works for valid JSON. Schema-invalid JSON fails inside the SDK before the extractor. The one-repair guarantee is `parse_subagent_return` and the string-mode extractor path.
10. Task-specific gym system prompts are not the frozen system prefix.
11. The OpenRouter example price table is not a real rate card.
12. Parallel paid runs can slightly overshoot the spend cap.

## Open issues

- Attaching this harness to per-task gym MCP servers and Postgres is not implemented.
- Lexical L3 misses omissions. A real judge is an M6 local model.
- DeepSeek cache field names are the documented ones (`prompt_cache_hit_tokens`, `prompt_cache_miss_tokens`, plus the OpenAI details object). Confirm them on a real M4 response.
- OpenRouter omits router metadata on some cache hits, so `served_provider` can be unobserved. The trace says so instead of inventing a provider.
- The task subset is algorithmic, not manually stratified.
- MS-Eval prompts and the official failure taxonomy are not transcribed. Labels in `eval/label_failures.py` are provisional: `delegated_failure_propagation`, `specification_miss`, `tool_error`, `context_loss`, `other`.
- The example OpenRouter price table is not authoritative. Confirm `deepseek/deepseek-chat` before a paid OpenRouter run. The headline model remains direct DeepSeek.
- `lerobot` / the GPU / M6 serving are untouched. `serving/gpu_check.sh` exits 0 when `nvidia-smi` is absent so CI does not fail.
