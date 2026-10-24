# Status: ready for the 10-task pilot

Branch `feat/pilot-ready`, from `feat/gym-containers-l3-judge` at `489dbc2`. Written 2026-10-24. PR: PR_LINK

Paid calls: one, the 150-tool probe through `openrouter_ds_1017`, $0.00104. Two free metadata reads went to OpenRouter (the probe's generation stats and the model's public endpoint list). No pilot run, no model on the GPU, no process outside this project touched. Docker was used for `$0` checks only; every container and network those checks created is gone.

`uv run pytest`: 132 passed, 2 skipped (the two opt-in live-provider tests), including the Docker gym tests.

## What changed

| Commit | Change |
| --- | --- |
| `549e2ed` | The OpenRouter files that were in the tree: `openrouter_ds_1017` (DeepSeek-V4.1-Flash-1017, `order: [deepinfra/fp8]`, `allow_fallbacks: false`), its price table, `openrouter_smoke.yaml`. |
| `061fd72` | `scripts/probe_tool_limit.py` and its result (section below). |
| `e84b801` | Task 1. Every `llm_call`, `llm_error` and `judge_call` records `generation_id`, the response's `provider` (`response_provider`), the metadata's selected provider, `service_tier`, `quantization` (when a response has one) and `provider_check`. A call served by another provider than the pinned one is refused: `provider_mismatch` event, priced `llm_error`, every later call of the run refuses without a request, no `run_end`, exit code 7. Also: an answered-then-failed call was billed but not counted toward the cap; it is now. The cap counts the larger of the price-table cost and OpenRouter's billed cost. |
| `6836d79` | Tasks 3–4. Seven subset70 tasks (one in pilot10) with an email defect swapped by a fixed rule; new hashes. `docs/TASK_RULES.md`. |
| `be22ff6` | Task 6. Delegation labels for all 503 tasks, `tasks/delegation_labels.csv`. |
| `645b864`, `73648cc` | Task 7. Prompt review fixes and the prompt freeze check (section below). |
| `a8e6684`, `fdc630e` | Task 5. Judge validation: `judge_call` keeps the claim and the request; `scripts/judge_validation.py` export / judge / score. |
| `fc97e9a` | Tasks 2 and 5. Pilot and headline configs use `openrouter_ds_1017` for the agents and the L3 judge (billed, counted toward the cap); `run_end` has `judge_fallbacks`. README, CLAUDE.md and the plan updated. |
| `f6620b5` | REVIEW A, the critical ground-truth leak (section below). |
| `ff96b2c` | Bug: `peak_windows_utc: []` was read as missing and replaced by DeepSeek's windows, so the OpenRouter pilot (`offpeak_only`) would have refused to start on weekday mornings. |
| `2bdbaa6` | REVIEW E: a failed run no longer cancels the paid runs in flight; they finish, nothing new starts, the error is raised after. The failed run writes `run_error`. |
| `36a7042` | A failed judge call's cost was counted as agent cost in `run_end`; it now counts as judge cost. |

## 128-tool limit: DeepInfra takes 150

One call with 150 dummy tools, sent through the harness's own provider path, pinned route, tools sorted by name as in runs. HTTP 200, and the model called the one tool whose description answered the prompt. That tool had a hashed name and sat at sorted position 146, so the definitions past the 128th reached the model. Raw request summary and response: `docs/evidence/tool_limit_probe_2026-10-24.json`.

Per the rule written before the call: no task is trimmed or swapped for its tool count. Every pilot10 task is within the tested 150 (largest 148). Twelve subset70 tasks give the single agent 151–230 tools; that range is untested (owner decision 3).

Other facts from the same call: the body has `id` (generation id), `provider: "DeepInfra"`, `service_tier: "default"`, cache hits in `prompt_tokens_details.cached_tokens`, and no quantization field; the generation-stats endpoint has none either. OpenRouter's endpoint list (`docs/evidence/openrouter_endpoints_…json`) shows three DeepInfra endpoints for this model (`deepinfra/fp8`, `/fp8/flex`, `/fp8/priority`), all fp8. The billed cost matched the `deepinfra/fp8` price exactly.

## Swap list

Rule (`docs/TASK_RULES.md`, `src/evcarry/env/task_rules.py`): a task that needs the emails server has an email defect when its `email_config.json` has no password (the server will not start) or when its evaluation requires a from-address the server cannot send from (`send_email` takes no sender). An `ast`-based detector finds them and matches my hand review of all 81 pool160 email tasks. Each defective subset70 task is replaced by the pool160 task outside subset70 with no defect and at most 128 single-agent tools, from the first matching tier (same stratum, same domain, same primary servers, same server count, any), lowest `sha256("evcarry-swap-v1|" + task)` first. pilot10 takes the same replacement. The benchmark is not edited.

| removed | reason | added | tier |
| --- | --- | --- | --- |
| arxiv-reading-plan-excel-gcal-email | config without password | arxiv-method-benchmark-tracker | same stratum |
| canvas-quiz-analysis-email | sender check | sf-hr-performance-ppt | same server count |
| sf-hr-mentorship-excel-gcal-email | sender check | sf-hr-department-budget | same stratum |
| **sf-sla-breach-audit** (also pilot10) | sender check | **sf-hr-compensation-pdf-notion** | same stratum |
| sf-ticket-resolution-notion | sender check | sf-sales-regional-summary | same stratum |
| terminal-canvas-gform-gcal-excel-email | sender check | terminal-howtocook-scholarly-excel-word-gform | same stratum |
| yf-dividend-tracker-gsheet | sender check | yf-financial-statements-overview | same stratum |

New hashes: subset70 `6c1960c1…`, pilot10 `828d4ea9…` (pool160 unchanged). `tasks/FROZEN.txt` records the rule version and the pre-swap hashes. `uv run python scripts/apply_task_rules.py --check` rebuilds the lists from the seed and the rule (it is a test). All 70 subset70 environments start (`docs/evidence/gym_check_subset70_2026-10-24.txt`; before the swaps 69 of 70 did).

## Delegation rule

> A task needs delegation when it needs at least 6 MCP servers (one above the pool160 median of 5) or when the single agent would get more than 128 tools (the OpenAI/DeepSeek function limit).

Every pool task needs at least 4 servers, so "2+ servers" would label all of them. Counts: pool160 103 of 160, subset70 42 of 70, pilot10 5 of 10. Both parts are separate columns in `tasks/delegation_labels.csv` (all 503 gym tasks).

## Prompt review (`prompts/gym_v1`)

Fixed:

- The delegating arms' main prompts are now one byte-identical preamble plus a short arm block. The FALLBACK sentence moved to the audited arms, the only ones that can receive it.
- The two sub-agent tool descriptions said "gather evidence … return claims" in every arm, the free-text baselines included. They are now neutral.
- Typed sub-agents got the evidence arm's citation format. They now get the same JSON with empty evidence (`return_typed.md`), as the plan defines typed.
- Self-verify sub-agents got vanilla's format, so the main agent was told to check tool output it never saw. They now re-check their values and quote the output behind each.
- The JSON formats asked for every list item and for what could not be done; the text formats asked for a "short" report. All four formats now have the same three rules.
- FALLBACK told the main agent to "do this sub-task yourself", which no delegating main agent can. It now says to re-plan and delegate again, as the prompt does. The repair sends `repair.md` instead of inline text.

Freeze check: `uv run evcarry prompts status prompts/gym_v1` hashes every model-visible text the set renders (prompt files and text built in code). `uv run evcarry prompts freeze prompts/gym_v1` writes `prompts/gym_v1/FROZEN.json`. After that a billable run refuses a changed set (exit 6). `require_frozen_prompts: true` (headline, local core) refuses an unfrozen set, so `make headline` refuses today. `gym_v1` is **not frozen**; its manifest is `153b44ac…` now, and every `run_start` records it (`prompt_manifest_sha256`).

## Judge validation

`scripts/judge_validation.py export runs/pilot_deepseek --out validation/judge_v1` draws 100 judged claims: 40 faults, 30 clean claims the judge accepted, 30 it rejected. A short stratum passes its places on. The pilot injects no faults, so faults are made from clean judged claims by changing the value, dropping a list item, or swapping in another claim's evidence. The sources of those faults are kept out of the clean sample. `items.csv` is blind (no verdict, origin or run; shuffled); `key.csv` holds the answers and weights. `judge --only-missing --hit` lets `openrouter_ds_1017` rate the synthetic faults (a few cents, capped by `--max-usd`). `score` reports agreement, Cohen's kappa, and precision/recall/F1 for flagging unsupported claims; clean claims are weighted back to their population, faults are reported on their own. At M6, `judge --model local_primary --column local` re-judges the same rows and `score` adds judge-vs-judge agreement (in the plan).

## Ground truth in the agent container (REVIEW A, fixed)

The image holds all 503 tasks with their evaluation code and ground truth (not only `groundtruth_workspace/`: also `files/groundtruth_data.json`, `generate_groundtruth.py` and similar) and the gym's `.git`. `python_execute` and the terminal server run as root. Every arm could read the answers, the SA arm directly and the delegating arms through the operator.

After preprocess, this task's directory is copied to the host. Then every other task, this task's `evaluation/` and `*groundtruth*` paths, `.git` and `explorer/` are deleted. The copy goes back just before grading. Checks:

- all nine gym-mock runs still pass;
- a Docker test finds no answers in the container and confirms grading still has them;
- on all ten pilot tasks, grading an untouched environment gives identical output sealed and unsealed.

`run_end.ground_truth_suspects` lists tool calls whose arguments reach for ground-truth paths, the gym's history, or its public repo.

## Self-review

My own pass found the ground-truth leak, the peak-window bug, the grid cancellation, and the uncounted answered-then-failed calls. An independent reviewer (fresh context, given only the diff) found the return-format asymmetry, the non-blind validation export, self-contradicting synthetic faults, and judge errors counted as agent cost. It also found the peak-window and cancellation problems independently. All of these are fixed, each with a test.

It confirmed:

- the delegating prompts share a byte-identical preamble;
- the audit context never reaches the judge;
- every billed event is counted exactly once;
- the pin check passes on the real response and rejects a priority-tier reply;
- the swap rule is deterministic and matches the doc.

## Owner decisions, before the pilot unless noted

1. **Audit budget (REVIEW B, not changed).** At r = 0.3, a return with one claim is never audited (`round(0.3 × 1) = 0`); 2–4 claims get one audit. If sub-agents return one or two claims, `evidence_risk` audits little and behaves like L0/L1 only. Options: at least one sampled claim per return when r > 0, or carry the fractional budget across a run's returns. Recommendation: decide before the pilot, because this defines the method arm.
2. **Sampling temperature (REVIEW L).** Agents send no temperature, so DeepInfra's default applies and is not logged; the judge sends 0. Recommendation: fix a temperature in the experiment YAML and log it in `run_start` before the pilot.
3. **More than 150 tools (headline only).** Twelve subset70 tasks give the SA agent 151–230 tools, and only 150 were tested. A second probe with 231 tools costs about $0.002. If it fails, the swap rule above would need a tool-count criterion.
4. **Canvas replacement.** No canvas task in pool160 fits "at most 128 tools" (all are 135–212 except one with its own sender defect), so `canvas-quiz-analysis-email` went to `sf-hr-performance-ppt`. With the bound at 150 (the tested one), the rule would pick a canvas task. The 128 bound is kept because it was your instruction.
5. **Same model as agent and judge.** The judge is DeepSeek judging DeepSeek's sub-agents, a possible self-preference bias. The hand labels and the M6 second judge measure it; say so in the paper.
6. **Retry asymmetry (design).** The evidence arms' harness re-delegation continues the sub-agent's conversation. A self-verify main agent that "asks again" starts a fresh sub-agent. The headline has no typed/evidence-only ablation, so H1 cannot separate "a retry with memory" from "evidence plus audit". Accept and state it, or give every arm's sub-agent calls persistent sessions.
7. **FALLBACK (REVIEW D).** The plan says the main agent does the failed sub-task itself, but no delegating main agent has workspace tools. It now re-plans and delegates again (README deviation 18). Accept, or give the main agent the operator's tools after a FALLBACK.
8. **Internet access in the agent container.** An agent could fetch the gym's public repo. That would show up in `ground_truth_suspects` but is not blocked. Pre-register what a flagged run counts as (recommendation: failed, and listed), or add an egress allowlist.
9. **Structured outputs.** DeepInfra lists `structured_outputs` for this model, but `json_schema_outputs` stays false (schema in the prompt, one repair), as for DeepSeek's own API. Switching changes failure handling and is untested.
10. **Freeze the prompts after the pilot:** `uv run evcarry prompts freeze prompts/gym_v1`, then commit `prompts/gym_v1/FROZEN.json`. The headline refuses to start until then.
11. **pool160** still contains 12 email-defect tasks. Any M6 grid drawn from pool160 must exclude them by the same rule.

Known open items from `docs/REVIEW.md` that the pilot will surface and that this branch did not change:

- C: the fault injector is fixture-specific; it matters at M6.
- G: there is no per-run cost ceiling; sub-agents get 100 turns.
- I: the risk features `hit_limits`, `had_errors` and `finished_very_fast` are never set.
- J: L2 compares a read's full text with its replay, so a later legitimate write causes false rejects in the audited arms.
- K: judge evidence is cut at 12,000 characters.

## Running the pilot

```bash
set -a; source .env; set +a                      # never print .env
uv run evcarry gym check --tasks tasks/pilot10.txt
uv run evcarry run configs/experiments/pilot_deepseek.yaml --dry-run
uv run evcarry run configs/experiments/pilot_deepseek.yaml    # 40 runs, cap $5, 4 in parallel
uv run evcarry summarize runs/pilot_deepseek
uv run python scripts/judge_validation.py export runs/pilot_deepseek --out validation/judge_v1
```

After the pilot, check in the traces:

- `provider_check.ok` is true on every call;
- `judge_fallbacks` is 0;
- `ground_truth_suspects` is empty;
- the cache-hit share and $ per run.

Then label `validation/judge_v1/items.csv`, freeze the prompts, and update the headline cap from the measured cost.
