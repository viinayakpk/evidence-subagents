# evcarry

Research harness for evidence-carrying sub-agent returns and spot-check audits. Milestones M0–M3 run here for $0 on a scripted model, on fixture tasks in-process and on real Toolathlon-Gym tasks in Docker. DeepSeek, OpenRouter, and a local vLLM server are config entries. The pilot and headline model is DeepSeek-V4.1-Flash-1017 through OpenRouter, pinned to a fixed set of three fp8 endpoints, DeepInfra first (`openrouter_ds_1017`); it is also the L3 judge there. Pre-pilot decisions: `docs/STATUS_pilot_ready.md`.

## Setup

```bash
uv sync --extra dev --extra analysis
cp .env.example .env
```

`uv run pytest` also works without the extras: pytest is in the `dev` dependency group, and pytest loads only `pytest_asyncio` (a sourced ROS setup puts its own pytest plugins on `PYTHONPATH`).

`.env.example` lists `DEEPSEEK_API_KEY`, `OPENROUTER_API_KEY`, the free-debug trio, and `VLLM_API_KEY=EMPTY`. Leave them empty in git. No key is committed.

Toolathlon-Gym is a submodule at `third_party/toolathlon_gym` (Apache-2.0, commit `ed735ba0af71d82688a55f952a6ea11479ae7206`). It provides the task metadata, the frozen task lists, and (in Docker) the task environments and their evaluation scripts. This checkout does not copy the unlicensed Toolathlon agent repo.

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

Exit codes from `evcarry run`: spend cap `2` (also when the cap is reached mid-run), weekday peak window `3`, missing API key `4`, Docker, its daemon, or the gym images unavailable `5`, a frozen task list that no longer matches `tasks/SHA256SUMS` or a gym submodule not at the commit in `tasks/FROZEN.txt` `6`, a pinned OpenRouter call served by another provider `7`. The frozen inputs are checked first, then the key, then Docker.

## Config

`configs/models.yaml` is the provider list.

| name | kind | notes |
| --- | --- | --- |
| `openrouter_ds_1017` | `openrouter` | pilot and headline model and judge: `deepseek/deepseek-v4.1-flash-1017`, `only`/`order: [deepinfra/fp8, inference-net/fp8, inference-net/us]`, `quantizations: [fp8]`, `allow_fallbacks: false`, `service_tier: default`, `OPENROUTER_API_KEY`, billable |
| `deepseek_flash` | `deepseek` | `https://api.deepseek.com`, model id `deepseek-flash`, `DEEPSEEK_API_KEY`, billable (DeepSeek's own endpoint; not used by the pilot or headline) |
| `local_primary` | `vllm` | `http://127.0.0.1:8000/v1`, `primitive-ai/Qwen3.8-Flash-Next-NVFP4`, key `EMPTY` |
| `mock` | `scripted` | no network |
| `openrouter_pinned` | `openrouter` | see below |
| `free_debug` | `openai_compatible` | `FREE_DEBUG_BASE_URL`, `FREE_DEBUG_MODEL`, `FREE_DEBUG_API_KEY` |

Prices are dated files in `configs/prices/`. `openrouter_ds_1017` has one row per endpoint (read 2026-10-24), no peak windows: `deepinfra/fp8` $0.098 cache-miss / $0.00294 cache-hit / $0.294 output per million tokens; `inference-net/fp8` and `inference-net/us` $0.08 / $0.008 / $0.50. For this cache-heavy workload InferenceNet is the dearer endpoint (about 1.8x per run). OpenRouter's billed `usage.cost` matched the served endpoint's row exactly on every probe and smoke call. A call is priced at the row of the provider that served it (both InferenceNet rows answer `InferenceNet`, so it gets the larger rate of each component); an unobserved or unknown provider gets the larger rate of each component over the table. `cost_usd` on `llm_call`, `judge_call` and `llm_error` is the larger of that and OpenRouter's `usage.cost`, so it is never the cheapest endpoint's price and never below the bill; `price_table_cost_usd` and `priced_as` say how it was priced. DeepSeek direct off-peak is $0.15 cache-miss / $0.003 cache-hit / $0.60 output per million tokens. Peak is double, on weekdays 01:00–04:00 and 06:00–10:00 UTC, end exclusive. Weekends are off-peak. After the 2026-10-25 DST change the Prague clock labels move; the UTC windows stay put. `offpeak_only` refuses a billable run that starts inside a peak window. Mock and other non-billable models skip both the window and the cap.

Experiment files live in `configs/experiments/`. Paths are relative to the repo root. The frozen lists are `tasks/pool160.txt`, `tasks/subset70.txt`, and `tasks/pilot10.txt`, checked against `tasks/SHA256SUMS`. Seven subset70 tasks (one of them in pilot10) whose email setup cannot pass were swapped for pool160 tasks by a fixed rule; `docs/TASK_RULES.md` has the rule, the swap list, the 128-tool probe result and the delegation label (`tasks/delegation_labels.csv`). `uv run python scripts/apply_task_rules.py --check` rebuilds the lists from the seed and the rules.

## OpenRouter

OpenRouter is a first-class provider, not a comment. The pilot and headline entry is `openrouter_ds_1017` (routing `only`/`order: [deepinfra/fp8, inference-net/fp8, inference-net/us]`, `quantizations: [fp8]`: OpenRouter tries DeepInfra first and moves a call that DeepInfra answers with 429 to InferenceNet inside the same request; DeepSeek's own OpenRouter endpoint is blocked by the account's training-data privacy setting; BYOK is not available). `scripts/probe_endpoints.py --hit` sends one tool call per endpoint (`docs/evidence/endpoint_probe_2026-10-24.json`, $0.0002 for all). The older example entry is `openrouter_pinned`:

- base URL `https://openrouter.ai/api/v1`
- key `OPENROUTER_API_KEY` (empty in `.env.example`)
- optional `HTTP-Referer` from `http_referer`
- `X-Title` and `X-OpenRouter-Title` both set from `x_title`
- `X-OpenRouter-Metadata: enabled` so the response can name the endpoint that served the call
- routing `order: [deepseek]`, `allow_fallbacks: false`, `require_parameters: true`

Each `llm_call`, `llm_error` and `judge_call` records, from the raw HTTP body: `served_provider` and `served_model` (`openrouter_metadata.endpoints.available[]` with `selected: true`, else the last HTTP 200 attempt, else the top-level `provider`), `response_provider` (the top-level `provider` field, e.g. `DeepInfra`), `generation_id` (the body's `id`, `gen-...`, which OpenRouter's `/api/v1/generation` looks up), `service_tier`, and `quantization` when the response has one (OpenRouter's responses and generation stats did not on 2026-10-24; its endpoint listing says every DeepInfra endpoint of DeepSeek-V4.1-Flash-1017 is fp8, see `docs/evidence/`). The Agents SDK's `provider_data.model` is the model we requested, so it is not used. If neither provider name is present, `provider_unobserved` is true. Cache tokens are read from the raw usage object (`prompt_cache_hit_tokens` / `prompt_cache_miss_tokens`, or `prompt_tokens_details.cached_tokens`). `usage.cost` is stored as `provider_reported_cost_usd`; the spend cap counts the larger of it and the price-table cost for every `llm_call`, `judge_call` and answered `llm_error`.

Provider pin (`src/evcarry/providers/pin.py`): a routing with `allow_fallbacks: false` and an `only` (else `order`) list pins the set of endpoints its tags name. A call passes when one tag matches its provider (the tag's first segment, `inference-net` → `inferencenet`, compared with the response's names lower-cased and stripped to letters and digits), its quantization when observed (the tag's or the routing's `quantizations`), and its service tier (a tag without `flex`/`priority` is the default tier; InferenceNet answers `service_tier: null`, which passes). Every call's `provider_check` is `{"ok": true | false | null, ...}`. A call served by another provider (or, when observed, another quantization or service tier) writes a `provider_mismatch` event and a priced `llm_error`, its reply is not used, every later model call of the run refuses without a request, and the run ends without `run_end` (it is redone after a restart, its spend still counted). The grid stops with exit code `7`. A call whose provider is not observed passes and says so (`ok: null`).

Retries (`RecordingModel`, every arm and the judge): a 429, a 5xx or a dropped connection is retried with equal-jitter exponential backoff, each wait drawn from [c/2, c] with c = min(60 s, 2 s x 2^n), for about 10 minutes before the call (and the run) fails. Each attempt is one OpenRouter request, inside which OpenRouter already tries the next endpoint of the set; unanswered requests are not billed. `attempts` and `statuses` on each call record the retries.

Served provider is a logged covariate, not a treatment: OpenRouter picks the endpoint per call, so `scripts/pilot_report.py` gives the provider mix per arm, flags an arm more than 10 percentage points from the pooled mix, and re-prices each arm's tokens at DeepInfra's rates so cost comparisons do not depend on the mix.

`configs/prices/openrouter_example_2026-10-24.yaml` copies the DeepSeek numbers so a debug budget has a figure. It is labeled example-only and is not a rate card for the paper. `configs/prices/openrouter_deepseek_1017_2026-10-24.yaml` holds the real listed rates of the three endpoints of `openrouter_ds_1017`.

`configs/experiments/openrouter_debug.yaml` is a small pinned grid on the example entry (pilot tasks, four headline arms, cap $5, schedule `any`). Confirm the slug `deepseek/deepseek-chat` before using it. `configs/experiments/openrouter_smoke.yaml` is the in-process smoke test of `openrouter_ds_1017` (three fixture tasks, arm `sa`, cap $1); it passed for $0.0019.

## Running M4 on the workstation

Neither command is run in CI. Load the key without printing it: `set -a; source .env; set +a`.

The smoke test stays inside this harness and the fixture tasks (in-process, arm `sa`, cap $1, no Docker):

```bash
uv run evcarry run configs/experiments/openrouter_smoke.yaml
```

`deepseek_smoke` is the same test against DeepSeek's own API (`DEEPSEEK_API_KEY`, `offpeak_only`).

The gym pilot runs real Toolathlon-Gym tasks; see the next section. Build the images once, check the pilot's task environments, then run it:

```bash
make gym-images                                            # ~5 min, once per gym commit
uv run evcarry gym check --tasks tasks/pilot10.txt         # starts and stops each task environment, no model
uv run evcarry run configs/experiments/pilot_deepseek.yaml  # openrouter_ds_1017, judge billed, $5 budget run with a $4.50 cap, 8 parallel
```

Headline: `make headline` (`configs/experiments/headline_deepseek.yaml`, 70 tasks, four arms, two seeds, cap $45, eight parallel tasks, `openrouter_ds_1017` as model and L3 judge, `require_frozen_prompts: true`: it refuses until `evcarry prompts freeze prompts/gym_v1` has been run after the pilot).

Spend and turn limits (owner decision, 2026-10-24; pilot and headline, every arm): `limits.max_steps: 40` (main agent) and `limits.subagent_max_turns: 40` (each sub-agent run: a main-agent call, a repair, a re-delegation). `budget.run_ceiling_usd: 0.25`: each run has a meter fed by its own trace events (agent calls, judge calls, answered-then-failed calls, the same `event_spend` the cap uses); once it reaches $0.25 the run's next model call is refused, the run ends with `agent_status` and `status` `cost_ceiling`, `cost_ceiling: true`, `passed: false`, and is not graded (`eval_passed: null`). The ceiling counts the judge too: every arm gets the same dollars per run, and the audited arm pays for its audits (and its repairs and re-delegations) out of them, which is conservative for the method. Its `run_end` is written, so a restart does not redo it, and its spend counts toward the cap. The call that crosses the ceiling completes, so a run can end a little above $0.25. A sub-agent or judge call refused at the ceiling also fails the run (the judge does not fall back to the lexical judge). With a ceiling the cap is hard: a billable run starts only if the spend so far, plus 1.2 x the ceiling for it and for each run in flight (less what each has spent), fits under `cap_usd`; a run refused only because the runs in flight hold their reserves waits for one of them to end and is checked again (in grid order), so the grid stops only when the cap itself leaves no room. Calls already in flight always complete, so the cap holds as long as no run ends above 1.2 x the ceiling; `scripts/pilot_report.py` shows the largest run. If the cap is still reached mid-run, every later model call is refused, that run ends without `run_end` and the grid stops (exit 2). Caps: pilot $5 (run with `cap_usd: 4.5`, the owner's stop rule), headline $45.

M6 hooks, not used yet: `serving/serve_primary.sh`, `serving/serve_fallback.sh`, `serving/gpu_check.sh`, `serving/smoke_test.py` (no-op unless `--hit`).

## Toolathlon-Gym in Docker

`environment: docker` runs each (task, arm, seed) in its own Docker network with its own Postgres container and agent container, the isolation scheme of the gym's `run_parallel.sh`. Per run (`src/evcarry/env/gym_docker.py`):

1. Start Postgres from `evcarry/toolathlon-gym-pg:<commit>` (the gym dump already loaded, ready in about a second) and the agent container from `evcarry/toolathlon-gym:<commit>`.
2. Copy the task's `initial_workspace` to `/workspace/agent_workspace` and run the task's `preprocess` module, as the gym's `TaskAgent` does.
3. Start each MCP server the task needs inside the agent container with the command, env and cwd from the gym YAML, and talk to it over `docker exec -i` stdio (MCP SDK client in `legacy` handshake mode; two gym servers crash on `server/discover`).
   Then seal the container: the image holds all 503 tasks with their evaluation code and ground truth (`groundtruth_workspace/`, `files/groundtruth_data.json`, `generate_groundtruth.py`, ...) and the gym's `.git`, and the agent's `python_execute` and terminal tools run as root. This task's directory is copied to the host (`logs/<run>/sealed_task`), then every other task, this task's `evaluation/` and every `*groundtruth*` path, `.git` and `explorer/` are deleted. The directory is put back just before grading. On all ten pilot tasks, grading an untouched environment gives identical output sealed and unsealed.
4. Run the arm. Tools are named `{server}-{tool}`, with input schemas rewritten as the gym's agent rewrites them. Local tools are `claim_done` (ends the run, as in the gym), `python_execute` (the gym's contract, run in the container) and `view_overlong_output` (pages a result cut at `gym.tool_output_chars`, default 8000 like the gym).
5. If the agent called `claim_done`, run the task's `evaluation` module in the container; exit code 0 is a pass. Without `claim_done` the gym does not grade, and `passed` is false.
6. Copy the workspace to `runs/<exp>/workspaces/`, write the ledger to `runs/<exp>/ledgers/`, then remove the run's containers (with their volumes) and network by label.

`run_end` records `sealed` and `ground_truth_suspects`: ledger calls whose arguments mention ground-truth paths, the gym's git history, or its public repo (`eigent-ai`). The container still has internet access, so an agent could fetch the public repo; the suspects list is how that would show. Pre-registered (owner decision, 2026-10-24): a run with any suspect is `contaminated: true` and counts as a failure (`passed: false`; the grader's result is kept as `eval_passed`). `index.parquet` has both columns, and `evcarry summarize` ends with a table of contaminated runs.

A task whose environment cannot start (a server, the preprocess, or a docker step fails) ends with `status: env_error` and `passed: null`; the other runs continue. Tool errors go back to the model as text. Preprocess, `python_execute` and evaluation are killed inside the container at their time limits.

Every container and network carries `evcarry.project=evidence-subagents`. `make gym-clean` removes leftovers after a crash (and the containers of any experiment still running); `make gym-check` is the preflight. `configs/gym_tools.yaml` says which gym tools are read-only (researcher allowlist, L2 replay), which reads are nondeterministic, which output fields are masked before replay (snowflake's per-call `data_id`), and which read shows each write.

The images (`scripts/build_gym_images.sh`): the gym's own Dockerfile unchanged, then `docker/gym/overlay.Dockerfile`, which builds MCP servers the upstream image leaves unbuilt (notion, canvas, fetch, woocommerce, playwright, google-sheets), connects the Postgres routers that woocommerce, canvas and notion ship but never construct (`docker/gym/wire_pg_routers.py`), and pre-builds the gym project's Python environment that `python_execute` uses. The submodule is not modified. From 2026-10-25 (`docs/STATUS_fixes.md`) the overlay also bakes every uv server's environment so no server downloads at start (needed for runs without internet, `gym.network_internal`), and the Notion mock stores and returns properties in the Notion API's response shape.

`make gym-mock` runs `configs/experiments/gym_mock.yaml`: the scripted model on three real tasks (`sf-hr-rating-distribution`, `wc-shipping-analysis`, `yf-portfolio-risk-excel`), arms `sa`, `vanilla_mad`, `evidence_risk`, three runs in parallel, $0. The scripts (`tests/fixtures/scripts/gym/generate.py`) solve the tasks, so the gym's evaluation passes.

## Arms and audits

| arm | delegation | schema | checks |
| --- | --- | --- | --- |
| `sa` | no | no | none |
| `vanilla_mad` | yes | free text | none |
| `self_verify` | yes | free text; sub-agents re-check their values and quote the tool output behind each; the main agent checks the quotes | none |
| `typed` | yes | `SubagentReturn` with empty evidence (same rules as evidence otherwise) | pointers skipped |
| `evidence` | yes | yes | L0/L1, sample rate 0 |
| `evidence_random` | yes | yes | L0–L3, random sample |
| `evidence_risk` | yes | yes | L0–L3, risk sample |

The L3 judge is `judge_model` in the experiment YAML: `lexical`, or any `configs/models.yaml` key (`openrouter_ds_1017` for the pilot and headline, `local_primary`, `mock_judge`, ...). An LLM judge gets the fixed prompt `prompts/v1/judge.md` and returns `{"verdict": "supported" | "contradicted" | "incomplete" | "unverifiable", "missing": [...], "reason": ...}`; only `supported` passes. Each call is a `judge_call` trace event (the claim, the full request, its hash, the raw reply, provider, tokens, cost, and the arm, return index and injected fault it belongs to); `run_end` has `judge_calls`, `judge_fallbacks` and `judge_cost_usd`, and a billable judge counts toward the spend cap. If a call or its parse fails, the lexical judge decides and the verdict says `lexical_fallback`. `scripts/judge_validation.py` exports a blind, stratified 100-claim sample for hand labels and scores the judge (agreement, kappa, precision, recall); at M6 the local model re-judges the same sample as a second judge.

The arm name picks the sampler. YAML supplies the rate for the spot-check arms. Each sub-agent return with at least one claim gets `max(1, round_half_up(rate × claims))` audited claims, capped at the number of claims (`protocol/sampler.py:budget_count`): the floor of one means a one- or two-claim return is audited too (owner decision, 2026-10-24; at r = 0.3 the rate alone audits nothing below two claims). The floor is part of the shared budget, so `evidence_risk` audits its top-risk claim and `evidence_random` a random one, the same number in both; the `evidence` arm (rate 0) still audits nothing. L0/L1 failures reject a return even when that claim was not sampled. The policy accepts, or re-delegates once with the failed claims named, then returns a `FALLBACK` message to the main agent, which re-plans the step and delegates it again (it has no workspace tools in any delegating arm). There is no third attempt on the same call.

Sub-agent sessions (owner decision, 2026-10-24): in every delegating arm each sub-agent (`researcher`, `operator`) keeps one conversation for the whole run (`harness/sessions.py`). Every call from the main agent continues it, and so do the harness's repair and re-delegation messages in the schema arms. A main agent that asks again (vanilla, self-verify, or any arm after a `FALLBACK`) therefore reaches a sub-agent that remembers its earlier instructions, tool calls and replies, as the evidence arms' harness re-delegation already did; retry context is the same in every arm. Calls to one sub-agent run one at a time in call order (parallel calls would interleave one history); the two sub-agents still run in parallel. The session grows with the run; DeepInfra's endpoint takes 1M tokens of context, and the per-run cost ceiling bounds the growth. The SDK merges tool items that share a `call_id`; if a provider ever repeats an id within a sub-agent's session, the stored copy of the new pair is renamed (`<id>r<n>`, `subagent_call_id_reused` in the trace) so later calls keep both results. DeepInfra's and InferenceNet's ids are random.

Prompts (`prompts/gym_v1`): every delegating arm's main prompt is the same preamble byte for byte, followed by one arm block that says what the sub-agents reply with (and, in the audited arms only, how citations are checked and what `FALLBACK` means). Sub-agents get the same role prompt in every arm plus the arm's return format: `return_text.md` (vanilla), `return_text_self_verify.md`, `return_typed.md`, `return_claims.md` (evidence arms). The two sub-agent tool descriptions are the same in every arm. A repair uses `repair.md`.

Prompt freeze: `evcarry prompts status [DIR]` prints the hash of every model-visible text the prompt set produces (main and sub-agent prompts per arm, sub-agent and local tool descriptions, repair, re-delegation and `FALLBACK` messages, the tool-result header, the judge prompt), rendered as runs render them, so a change in code counts too. `evcarry prompts freeze DIR` writes `DIR/FROZEN.json`; the owner runs it once the pilot is done. A billable run refuses (exit `6`) when a frozen set has changed, and a config with `require_frozen_prompts: true` (headline, local core) refuses unless the set is frozen and unchanged. Every `run_start` records `prompt_manifest_sha256` and `prompts_frozen`. `gym_v1` is not frozen yet.

The fault injector is off unless an experiment sets `inject`. The decision key is `task|seed|return_index` and does not include the arm. `make fault-mock` prints the synthetic catch table. Notebook `analysis/03_fault_injection.ipynb` calls the same function.

## Traces

One JSONL file per run: `{task}__{arm}__{model}__{seed}.jsonl`. A run is finished when a `run_end` event exists. A restart skips that file and does not rewrite it. An unfinished trace is kept as `{name}.jsonl.aborted-{n}` and the run starts again; its calls still count toward the spend cap, which starts from every `llm_call` and `judge_call` cost already in the output directory. Events include `run_start`, `llm_call`, `llm_error` (a model call that raised; priced when it was answered, with `role` agent or judge), `provider_mismatch`, `tool_call`, `subagent_return`, `subagent_error` (a sub-agent run or its audit raised), `injection` (`applied: false` when the corruption did not fit the claim), `check`, `audit`, `policy`, `judge_call`, `run_error` (the run raised; no `run_end` follows, so a restart runs it again), `run_end`. JSONL is the source of truth. In a parallel grid a failed run stops new runs from starting; runs already going finish, then the error is raised.

Every model call (main agent, sub-agents, repairs, re-delegations, L3 judge) in every arm and on every provider is sent with `temperature: 0` (`harness/run_settings.py:TEMPERATURE`, owner decision 2026-10-24), so no provider default applies. `run_start` records `temperature` and `judge_temperature`; each `llm_call`, `llm_error` and `judge_call` records the `temperature` its HTTP request body carried (`null` would mean it was not sent).

The system prompt is the arm's file under the experiment's prompt set (`prompts/v1/` for fixtures, `prompts/gym_v1/` for gym tasks) and is the cache prefix. The task's `docs/task.md` is the user message. The gym `agent_system_prompt.md` is hashed into `run_start` and is not sent. Tool names are sorted.

## Deviations

1. Fixture tasks (`environment: in_process`) set `evcarry_local_eval` and use a small local checker. Gym tasks (`environment: docker`) run in containers and are graded by their own evaluation scripts; see "Toolathlon-Gym in Docker" for what the overlay image changes and why.
2. MS-Eval appendix prompts are not copied. `prompts/v1/` is original text.
3. L3 is the lexical judge unless `judge_model` names a model. The lexical judge requires hyphenated tokens, a non-numeric string value, and list items to appear in the evidence, so it does not catch an omitted list item. In the synthetic fault table the omission rows now start from a claim listing all five papers and drop one that the cited evidence still shows; the lexical judge misses both rows and the scripted judge (`make fault-judge`) catches both at L3, with no rejects on the two clean claims.
4. Agents SDK 0.24 already runs function tools from one turn concurrently (`tool_execution.max_function_tool_concurrency`). No batch tool was added. The SDK 0.24 install also has no `agents.testing.ScriptedModel`; `FixtureModel` implements the public `Model` interface.
5. Peak windows stay in UTC. The Prague display shifts on 2026-10-25; the UTC windows do not.
6. `pool160`, `subset70`, and `pilot10` are an algorithmic stratified sample (seed `20261024`, largest remainder, then `random.Random(seed)` inside sorted stratum names), with the email-defect swaps of `docs/TASK_RULES.md` applied to subset70 and pilot10. They are not hand-reviewed. Do not re-roll them; `scripts/freeze_task_lists.py` is only for a new gym commit.
7. OpenRouter requests send `X-OpenRouter-Metadata: enabled`, and both `X-Title` and `X-OpenRouter-Title`.
8. SDK `Usage` drops DeepSeek cache fields (`cached_tokens` stays 0). The raw HTTP body is the accounting source. `provider_data.model` is the requested model.
9. Strict `output_type=SubagentReturn` works for valid JSON. Schema-invalid JSON fails inside the SDK before the extractor. The one-repair guarantee is `parse_subagent_return` and the string-mode extractor path.
10. Task-specific gym system prompts are not the frozen system prefix.
11. The OpenRouter example price table is not a real rate card.
12. Without `run_ceiling_usd`, parallel paid runs can overshoot the spend cap by about one batch. With it (pilot, headline) admission reserves the ceiling for runs in flight.
13. L1 and L2 do not look for a `derived` claim's value in the cited output (a count or rate is computed, not quoted); its spans are still checked. Only L3 checks the value, and only on sampled claims (see `docs/STATUS_gym_judge.md`).
14. L2 reports `unverifiable` (not a failure) for a write with no read-back in `configs/gym_tools.yaml` and for reads marked nondeterministic there. L3 still runs on those claims.
15. Gym runs use `prompts/gym_v1/` (the gym's workspace and `claim_done` instructions, a fixed workspace path so the system prompt is byte-identical across runs, and a return-format section per arm). Tool results carry a `[call_id: ...]` header so a real model can cite them.
18. After two rejections the plan has the main agent do the sub-task itself. Here the main agent has no workspace tools in any delegating arm, so `FALLBACK` tells it to re-plan the step and delegate it again, which is what the evidence prompt says.
16. DeepSeek and the OpenRouter DeepSeek route have `json_schema_outputs: false`: typed and evidence sub-agents get the claim schema in the prompt and their text is parsed (one repair), because DeepSeek documents `json_object` but not `json_schema`.
17. Sub-agent runs get `limits.subagent_max_turns` (default: `max_steps`) as their turn limit (the SDK default is 10). The pilot and headline set both to 40. The SA agent therefore has 40 turns for the whole task, while a delegating main agent has 40 turns and each sub-agent call 40 more.
19. Sub-agents keep one conversation per run in every delegating arm (see "Arms and audits"). The plan does not say whether a repeated delegation starts fresh; persistent sessions make the retry context the same across arms, so H1 compares evidence plus audit, not retry memory.

## Open issues

- More than 128 tools: the pinned DeepInfra route accepted 150 (`docs/TASK_RULES.md`); no task was trimmed. Twelve subset70 tasks give the SA agent more than 150 tools, which is unverified.
- The LLM judge has only been run against the scripted provider. Its accuracy on a real model is unmeasured until the pilot's judged claims are hand-labelled (`scripts/judge_validation.py`).
- On the OpenRouter/DeepInfra route, cache hits arrive as `prompt_tokens_details.cached_tokens` (the smoke run logged hits from the second call on). DeepSeek's own field names (`prompt_cache_hit_tokens`, `prompt_cache_miss_tokens`) are still unconfirmed on a real response.
- OpenRouter omits router metadata on some cache hits, so `served_provider` can be unobserved. The trace says so instead of inventing a provider.
- The task subset is algorithmic, not manually stratified.
- MS-Eval prompts and the official failure taxonomy are not transcribed. Labels in `eval/label_failures.py` are provisional: `delegated_failure_propagation`, `specification_miss`, `tool_error`, `context_loss`, `other`.
- The example OpenRouter price table is not authoritative; the `openrouter_ds_1017` table is DeepInfra's listed rate.
- `lerobot` / the GPU / M6 serving are untouched. `serving/gpu_check.sh` exits 0 when `nvidia-smi` is absent so CI does not fail.
