# Status: real gym containers and the L3 judge

Branch `feat/gym-containers-l3-judge`, based on `cursor/evidence-harness-m0-m3-3dd8` (PR #1). Written 2026-10-24.

No paid API call was made, no model was loaded on the GPU, and no process or container outside this project was touched. Every container and network this work created has been removed; the three images it built are kept (`evcarry/toolathlon-gym-base:ed735ba`, `evcarry/toolathlon-gym:ed735ba`, `evcarry/toolathlon-gym-pg:ed735ba`).

## What works

### Toolathlon-Gym tasks run in Docker and are graded by the gym

- `environment: docker` gives each run its own Docker network, Postgres container and agent container (the `run_parallel.sh` scheme). The harness stays on the host and talks to the gym's own MCP servers inside the agent container over `docker exec -i` stdio, with the commands, env and cwd from the gym YAMLs. The task's `preprocess` and `evaluation` modules run in the container exactly as the gym runs them. Code: `src/evcarry/env/gym_docker.py`.
- Postgres starts from a snapshot image with the gym dump already loaded and the gym's `email.sent_log` foreign-key fix applied, so it is ready in about a second instead of re-importing per task. Readiness is checked over TCP; the gym's socket-based health check can pass while the init server is still loading.
- All 25 gym MCP servers start and list their tools (441 tools in total). Two needed the MCP client's `legacy` handshake: the 2026 client's `server/discover` probe crashes them.
- With the scripted model, three real gym tasks (`sf-hr-rating-distribution`, `wc-shipping-analysis`, `yf-portfolio-risk-excel`) × three arms (`sa`, `vanilla_mad`, `evidence_risk`) all pass the gym's evaluation, three runs in parallel, 32 s wall clock for the nine runs. Median per run: setup 4.4 s (MCP connect 3.2 s), evaluation 0.24 s, teardown 2.8 s. No container or network is left afterwards.
- In the `evidence_risk` runs every claim was sampled (rate 1.0): L2 re-issued the snowflake, woocommerce and yahoo-finance reads against the live servers and matched the ledger; the excel and email writes were checked through their read-backs; `python_execute` was reported unverifiable rather than failed; every sampled claim got a logged `judge_call`.
- An injected numeric fault on real snowflake output is rejected at L1, the researcher is sent back once, and the second return is accepted (`test_injected_fault_on_real_output_is_sent_back_once`).
- Two environments of the same task running at once do not share state: an email sent in one is invisible in the other.
- `uv run evcarry gym check --tasks tasks/pilot10.txt` starts and stops every pilot task's environment without a model: 10/10 start, all preprocess scripts exit 0. On `tasks/subset70.txt`: 69/70 start (see finding 3).
- `configs/experiments/pilot_deepseek.yaml` no longer exits 5 when Docker and the images are present. Without `DEEPSEEK_API_KEY` it exits 4; the key is now checked before Docker.

What it took beyond wiring, all outside the submodule (which is unmodified):

| Problem at gym commit ed735ba | Effect | Fix here |
| --- | --- | --- |
| notion, canvas, fetch, woocommerce, playwright servers not built by the gym Dockerfile; google-sheets has no venv | those servers cannot start | `docker/gym/overlay.Dockerfile` builds them |
| canvas, notion, woocommerce import `pg` without declaring it | build fails | overlay installs `pg@^8.20` |
| woocommerce, canvas, notion ship Postgres routers that nothing constructs | every call goes to `localhost:8081`, `:8080` or api.notion.com and fails | `docker/gym/wire_pg_routers.py` (six anchored edits) |
| `python_execute` runs `uv run` inside `/workspace`, the gym's own uv project | first call per container downloads ~250 packages | overlay pre-builds that environment from `uv.lock` |
| snowflake `read_query` output has a fresh `data_id` uuid per call | every L2 replay would drift | `volatile` masks in `configs/gym_tools.yaml` |
| raw MCP input schemas contain type lists, `anyOf` with null, untyped array items | some OpenAI-compatible APIs reject such tool schemas | schemas are rewritten as the gym's own agent rewrites them (`normalize_schema`) |

Without the router wiring, 6 of the 10 pilot tasks would have failed on dead servers.

### L3 support judge

- `LlmJudge` (`src/evcarry/protocol/judge.py`) calls any `configs/models.yaml` entry through the existing provider layer: `judge_model: deepseek_flash`, `openrouter_pinned`, `local_primary` (vLLM), or `mock_judge` (scripted). `lexical` keeps the old judge.
- Fixed prompt `prompts/v1/judge.md` (it replaced a one-line placeholder that no run had read). Structured verdict `{"verdict": supported | contradicted | incomplete | unverifiable, "missing": [...], "reason": ...}`; only `supported` passes. `incomplete` is the omission case.
- Every call writes a `judge_call` trace event: judge model, served model, prompt and request sha256, raw reply, verdict, missing items, tokens, cost, latency, error. Verdicts also appear in each `check` event. `run_end` records `judge_calls` and `judge_cost_usd`; a billable judge counts toward the spend cap.
- A failed call or an unparseable reply falls back to the lexical judge and is marked `lexical_fallback` with the error.
- The synthetic fault set now contains a real omission: the cited search result lists papers A to E, the claim starts from all five, and the corruption drops E. The lexical judge misses both omission rows; the scripted judge catches both at L3, and neither judge rejects the two clean claims. The other eight rows are caught at the same levels as before (`test_scripted_judge_catches_omissions_on_the_synthetic_set`). This shows the wiring, not a real model's catch rate; that needs the opt-in live test below.

## Commands

DeepSeek smoke test (fixture tasks, in-process, arm `sa`, cap $1, refuses during weekday peak windows):

```bash
export DEEPSEEK_API_KEY=...
uv run evcarry run configs/experiments/deepseek_smoke.yaml
uv run evcarry summarize runs/deepseek_smoke
```

Gym pilot (10 tasks × 4 arms, cap $5, off-peak only, 4 runs in parallel):

```bash
make gym-images                                      # once; ~5 min, rebuilds are cached
uv run evcarry gym check --tasks tasks/pilot10.txt   # no model, ~25 s
export DEEPSEEK_API_KEY=...
uv run evcarry run configs/experiments/pilot_deepseek.yaml
uv run evcarry summarize runs/pilot_deepseek
uv run evcarry gym clean                             # only if a run was killed
```

$0 checks:

```bash
uv run pytest                 # includes the Docker gym tests when Docker and the images exist
make gym-mock                 # 3 real tasks × 3 arms with the scripted model
make fault-judge              # synthetic fault table with the scripted LLM judge
EVCARRY_LIVE_JUDGE=local_primary uv run pytest tests/integration/test_live_judge.py -s   # once vLLM serves
```

## Test results

`uv run pytest` on the workstation: 71 passed, 2 skipped (the two opt-in live-provider tests), about 70 s. The nine Docker gym tests skip with the reason when Docker, its daemon, or the images are missing (`EVCARRY_DOCKER=/nonexistent uv run pytest tests/integration/test_gym_docker.py` shows them all skipped). A test-wide fixture refuses to build any HTTP model client, so no test can reach a paid endpoint.

`uv run pytest` failed on this machine before this branch: the shell sources ROS Jazzy, whose pytest plugins load from `PYTHONPATH` into the Python 3.14 venv and crash, and pytest was only installed through the `dev` extra. `pyproject.toml` now disables plugin autoload (loading only `pytest_asyncio`) and adds a `dev` dependency group.

## Findings that need a decision before paid runs

1. **More than 128 tools.** The SA agent gets every MCP tool of the task plus three local tools; the operator sub-agent gets every MCP tool. 3 of `pilot10` and 28 of `subset70` exceed 128 for the SA agent (largest: `terminal-canvas-wc-excel-ppt-email`, 230). DeepSeek's chat API, like OpenAI's, documents a limit of 128 functions; the gym trims to 128 only for the OpenAI platform. I did not trim, because any rule changes task difficulty per arm. Options: confirm the limit with one cheap call; adopt a fixed trimming rule (for example drop each server's admin and delete tools); or replace those tasks. Pilot tasks affected: `canvas-module-completion-notion-gcal` (135), `canvas-quiz-analysis-ppt` (148), `sf-sales-region-analysis-gsheet-email` (140). `subset70`: academic-conference-planning-coordination (175), arxiv-research-workflow-pipeline (130), canvas-assignment-effectiveness-ppt-notion-email (178), canvas-course-comparison (130), canvas-exam-prep-scheduler (133), canvas-module-completion-notion-gcal (135), canvas-quiz-analysis-email (136), canvas-quiz-analysis-ppt (148), canvas-quiz-performance-gcal-gform (132), fetch-sf-kpi-dashboard-excel-word-email (141), fetch-wc-supplier-restock-excel-word-gcal (157), howtocook-wellness-tracker (139), pw-sf-benchmark-excel-word-email (168), pw-wc-customer-segment-gsheet-email (149), sf-sales-region-analysis-gsheet-email (140), sf-wc-order-reconciliation-excel-word-email (192), terminal-canvas-gform-gcal-excel-email (148), terminal-canvas-wc-excel-ppt-email (230), terminal-wc-pw-pricing-excel-word-gcal (189), terminal-wc-sf-revenue-reconcile-excel-ppt (150), terminal-wc-yf-ppt-notion-email (165), terminal-yf-canvas-excel-gcal-email (153), wc-coupon-campaign-gcal-gform (132), wc-customer-loyalty-word-gsheet (145), wc-product-bundle-excel-ppt-gcal (150), wc-shipping-tax-excel-word (176), wc-shipping-zone-ppt (153), wc-yf-commodity-pricing-impact (167).
2. **Unpassable sender check.** The emails server sends from the address in the task's `email_config.json`, or `user@example.com` when there is none. At least 11 tasks (a regex lower bound) have no config but an evaluation that requires a specific sender. In `pilot10` that is `sf-sla-breach-audit` (requires `sla-audit@company.com`): an agent can pass only by writing the row into `email.messages` directly. 2 such tasks are in `subset70`, 4 in `pool160`. Options: swap the task, or ship an `email_config.json` for these tasks in the overlay (that edits the benchmark).
3. **Email config without a password.** 10 tasks have an `email_config.json` without `password`, and the emails server refuses to start (`arxiv-reading-plan-excel-gcal-email` is in `subset70`; `yf-portfolio-analysis-excel-word-email` and `yf-sector-comparison-gform-excel-email` are also in `pool160`; none in `pilot10`). The run fails before the agent starts. A placeholder password would let it start; that also edits the benchmark.
4. **Judge model for the headline.** `headline_deepseek.yaml` uses `judge_model: local_primary`, but CLAUDE.md puts the local model last (M6), after the headline (M5). With vLLM down, every L3 call falls back to the lexical judge (logged as `lexical_fallback`). The pilot config still uses `judge_model: lexical`; `deepseek_flash` would bill the judge to the API.
5. **Gym prompts are drafts.** `prompts/gym_v1/` is new text (workspace path, `claim_done`, delegation, the claim format with `[call_id: ...]` citations). Review it before it is frozen by a paid run.

## Protocol changes to review

These change what the audit accepts; each is a one-line revert if unwanted.

- L1 and L2 no longer require a `derived` claim's value to appear verbatim in the cited output; spans are still checked. Without this, every count or percentage a sub-agent computes is rejected (the scripted `wc-shipping-analysis` run showed it: a method count labelled `fact` was rejected at L1 on real output). The cost: a derived value is now checked only by L3, and only when the claim is sampled. The `evidence` arm (L0/L1, rate 0) never checks one; at rate 0.3 a single-claim return samples nothing; a numeric fault injected into a derived claim passes L1. Derived claims already get +4 in the risk score. **Decision needed:** keep this, or always send derived claims to L3 (more judge calls, outside the audit budget).
- L2 reports `unverifiable` instead of failing for a write with no read-back and for nondeterministic reads.
- DeepSeek entries have `json_schema_outputs: false`: typed and evidence sub-agents get the claim schema in the prompt and their text is parsed with one repair, because DeepSeek documents `json_object` but not `json_schema` response formats. Unverified without a paid call.
- Sub-agents get the experiment's `max_steps` as their turn limit; the Agents SDK default was 10.
- Agent turn and time limits now end a run with `agent_status` `max_turns_reached` or `timeout` instead of stopping the whole grid. Other errors still stop it; with parallel runs the siblings are cancelled and torn down first.

## Independent review

A separate review of the branch found eight bugs, all fixed with a regression test each:

1. An exception inside a tool (bad JSON arguments, a dead MCP server, a non-integer page) ended the run and then cancelled the grid. Tool errors now go back to the model; a task whose environment cannot start ends as `status: env_error`, `passed: null` and the grid continues.
2. A spend-cap or peak-window refusal cancelled parallel runs already in progress (and paid for). Now no new run starts and the running ones finish before the refusal is raised.
3. Cancelling a run while its containers were being created could leak them. Creates now finish before the cancel takes effect, and teardown removes everything carrying the run's label, off the event loop.
4. The LLM judge read the model's reasoning text and took the first JSON object, so a drafted verdict could win. It now reads only the answer and takes the last verdict object.
5. `view_overlong_output` pages were cut again by the display limit, hiding the end of each page.
6. See the `derived` decision above.
7. A `python_execute` that printed a number could serve as its own evidence. The judge now sees each cited call's tool and arguments (the code) next to its output; read-backs are no longer called twice and evidence is no longer duplicated.
8. A timed-out preprocess, `python_execute` or evaluation kept running inside the container. Each now runs under `timeout -s KILL` in the container; a failed preprocess is an `env_error`.

## Open issues

- L2 replay compares a read's full text with the ledger. A legitimate later write (for example a new file in a listed directory) makes an earlier read drift and rejects a true claim. Expect some false rejects in the pilot; the trace names each one.
- The gym's evaluation counts only runs that called `claim_done`; a model that finishes with a text reply fails. `prompts/gym_v1` tells the model to call it.
- In the delegating arms the main agent has no workspace tools, so after a `FALLBACK` it can only re-delegate (this was already so in M0–M3). The plan says the main agent does the sub-task itself.
- The `yf-portfolio-risk-excel` fixture builds its workbook with `python_execute` reading Postgres directly; the other two fixtures use only MCP tools.
- The overlay image is 8.4 GB (the base is 4.8 GB).
- DeepSeek's usage field names and served-model string are still unconfirmed on a real response (unchanged from M0–M3).
- Each run builds its own API clients (agents and judge) and never closes them, as in M0–M3. Harmless for a few hundred runs; worth closing before long local grids.
- `evcarry gym clean` removes every evcarry container, including those of an experiment running in another process. Use it only when nothing is running.
