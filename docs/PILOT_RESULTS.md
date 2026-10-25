# Pilot results (M4)

DeepSeek-V4.1-Flash-1017 on OpenRouter, 10 pilot tasks × 4 arms × 1 seed (`configs/experiments/pilot_deepseek.yaml`, `runs/pilot_deepseek`). Branch `feat/pilot-ready`, PR #3. The completing launch ran on commit `461d2b6` from 19:52 to 21:38 UTC on 2026-10-24. Prompts `gym_v1` are not frozen (manifest `153b44ac…`), and the headline was not started.

All tables can be regenerated with:

```bash
uv run python scripts/pilot_report.py runs/pilot_deepseek --config configs/experiments/pilot_deepseek.yaml --headline-config configs/experiments/headline_deepseek.yaml
```

## Summary

- Spend across all attempts was $4.2263, under the $4.50 stop rule and the $5 budget. That is $4.1941 for this launch plus $0.0323 logged before it (one finished run and 15 aborted traces). Probe calls outside the run directory add $0.0003.
- 35 of 40 runs finished. The grid stopped on the cap: "spent $4.226340 … plus $0.30 for this one would pass the cap $4.500000" (exit 2). Five runs never started: `yf-portfolio-risk-excel` × evidence_risk and all four arms of `yt-ml-repos-github-notion-excel-email`. A cap stop is not a rate-limit or provider error, so they were not relaunched.
- Runs passed as counted: SA 1/9, vanilla MAD 3/9, self-verify 2/9, evidence + risk 0/8. By the grader alone: SA 2/9, vanilla 4/9, self-verify 3/5, evidence 0/4. Contamination and cost-ceiling runs account for the difference.
- 19 of 35 runs are flagged contaminated and count as failures, which voids 3 grader passes. Fifteen of them read only their own task's remaining files. In four self-verify runs the operator searched the filesystem for the grader and ground truth. No search found a task's answers.
- 8 runs stopped at the $0.25 ceiling: 4 self-verify and 4 evidence + risk. The largest run cost $0.268, under the 1.2 × $0.25 = $0.30 that the hard cap assumes.
- InferenceNet served 92% of calls. OpenRouter moved 2,456 of 2,595 agent calls from DeepInfra to InferenceNet inside the request. No 429 reached the harness, no call needed a harness retry, and each arm had 6–10% DeepInfra calls.
- The headline does not fit $45. It projects to about $68 at the measured mix, $73 with every call at the endpoint set's upper bound, and $47 with every call at DeepInfra's rates, so no route choice closes the gap (see "Projected headline cost").

## How the pilot ran

| Attempt (all 2026-10-24, UTC) | Route | Outcome | Spend |
| --- | --- | --- | --- |
| 1, from 18:15 | `deepinfra/fp8` | `deepinfra/fp8/priority` served under the base slug; the pin check stopped all four runs (fixed in `8ae8b21`: `service_tier: default`) | in the $0.0323 below |
| 2, from 18:17 | same | the first 429 `engine_overloaded` ended all four runs (fixed in `b61648a`: retries) | |
| 3, 18:19 to about 18:32 | same, 4-minute retry window | 429 bursts outlasted the retry window; all four runs failed | |
| 4, 18:33 to about 18:54 | same, 13-minute retry window (`d03743c`) | the canvas SA run finished ($0.0082); the other three failed on 429s after waiting minutes; then stopped on purpose | |
| before this launch | | one finished run, 15 aborted traces | **$0.0323** |
| 5, 19:52–21:38 | fp8 set: `deepinfra/fp8`, `inference-net/fp8`, `inference-net/us` (`ffd33ff`), 8 in parallel, cap $4.50 | 34 runs finished, stopped by the cap | **$4.1941** |

Changes made before launch 5, each with tests (`uv run pytest`: 178 passed, 2 skipped):

- `6e140d9`: retries use equal-jitter exponential backoff from 2 s, capped at 60 s, with about a 10-minute window.
- `f1ffe04`: each call is priced at the endpoint that served it, and `cost_usd` is the larger of that and OpenRouter's `usage.cost`.
- `ffd33ff`: the route is the fixed fp8 set, and the pin check matches a call against one tag of the set. Probe evidence: `docs/evidence/endpoint_probe_2026-10-24.json`, $0.000195.
- `5f07397`: the pilot report adds the provider mix per arm, wall-clock time, and the worst-case projection.
- `c507d53`: 8 runs in parallel; `cap_usd: 4.5`.
- `461d2b6`: the served provider is documented as a logged covariate in the plan.

## Completion

| arm | planned | finished | not started | status | agent_status |
| --- | --- | --- | --- | --- | --- |
| sa | 10 | 9 | 1 | failed 3, not_claimed_done 4, passed 2 | completed 5, max_turns_reached 4 |
| vanilla_mad | 10 | 9 | 1 | failed 4, not_claimed_done 1, passed 4 | completed 9 |
| self_verify | 10 | 9 | 1 | cost_ceiling 4, failed 2, passed 3 | completed 5, cost_ceiling 4 |
| evidence_risk | 10 | 8 | 2 | cost_ceiling 4, failed 4 | completed 4, cost_ceiling 4 |

`status` is the grader's. A `passed` status in a contaminated run still counts as a failure.

## Task success and silent failures

| arm | passed / finished (as counted) | grader passed / graded | claimed done | silent failures (claimed done, not passed) | silent by the grader alone |
| --- | --- | --- | --- | --- | --- |
| sa | 1/9 (11%) | 2/9 | 5 | 4/9 (44%) | 3/9 |
| vanilla_mad | 3/9 (33%) | 4/9 | 8 | 5/9 (56%) | 4/9 |
| self_verify | 2/9 (22%) | 3/5 | 5 | 3/9 (33%) | 2/5 |
| evidence_risk | 0/8 (0%) | 0/4 | 4 | 4/8 (50%) | 4/4 |

Runs stopped at the cost ceiling are not graded, so the grader's denominator is smaller for self-verify and evidence + risk.

| task | sa | vanilla_mad | self_verify | evidence_risk |
| --- | --- | --- | --- | --- |
| canvas-module-completion-notion-gcal | fail | fail | ceiling | fail |
| canvas-quiz-analysis-ppt | fail (turns) | fail (grader pass, contaminated) | ceiling | ceiling |
| fetch-sf-hr-training-notion-gcal-excel | fail (turns) | fail | ceiling | fail |
| howtocook-weekly-gsheet-gcal | pass | pass | fail (grader pass, contaminated) | ceiling |
| sf-hr-compensation-pdf-notion | fail (turns) | pass | pass | ceiling |
| sf-sales-region-analysis-gsheet-email | fail | fail | fail | fail |
| terminal-arxiv-latex-excel-word-notion | fail (turns) | fail | ceiling | ceiling |
| wc-shipping-analysis | fail | fail | fail | fail |
| yf-portfolio-risk-excel | fail (grader pass, contaminated) | pass | pass | not started |
| yt-ml-repos-github-notion-excel-email | not started | not started | not started | not started |

With nine tasks per arm the pilot cannot support a flip analysis. It measures cost, failure modes and harness health, not arm effects.

## Faults, audits, and false rejects

The pilot config has no `inject` block, so no faults were injected and none could be caught. Fault catch rates by type and level come from M6 fault injection.

Evidence + risk, natural claims only:

| | count |
| --- | --- |
| sub-agent returns | 115 |
| claims (all checked at L0 and L1) | 1,185 |
| claims audited at L2 and L3 (risk sampler, r = 0.3, at least one per return) | 361 |
| claims failing L0 | 18 (1.5% of 1,185) |
| claims failing L1 | 30 (2.6% of 1,168 with L1 checks) |
| claims failing L2 | 48 (13.3% of 361) |
| claims failing L3 | 85 (23.5% of 361) |
| distinct claims rejected at any level | 141 (11.9% of 1,185) |
| policy actions | accept 51, re-delegate 41, fallback 23 |

With no injected faults, every rejected claim is a natural claim, so 11.9% is an upper bound on the false-reject rate. H4 targets at most 5%. Which rejections were right is only known after the claims are hand-labelled: `scripts/judge_validation.py export` draws the sample. Two known sources of false rejects should be checked first:

- REVIEW J: L2 compares a read with its replay, so a later legitimate write fails it.
- REVIEW K: the judge sees at most 12,000 characters of evidence, which may explain part of the 69 "unverifiable" verdicts among the 85 L3 failures.

## L3 judge

| arm | judge calls | lexical fallbacks | verdicts | judge $ |
| --- | --- | --- | --- | --- |
| evidence_risk | 361 | 0 | supported 276, unverifiable 69, contradicted 11, incomplete 5 | 0.1350 |

That is $0.00037 per judged claim, 9% of the evidence arm's spend. `judge_fallbacks` is 0 in every `run_end`.

## Tokens and dollars

Every attempt, finished or not:

| arm | agent calls | judge calls | input tokens | cache hits | output tokens | agent $ | judge $ | spend $ | mean $/finished run | max $/run | $ at DeepInfra rates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sa | 277 | 0 | 11,376,712 | 92% | 129,683 | 0.2178 | 0 | 0.2178 | 0.0239 | 0.0691 | 0.1607 |
| vanilla_mad | 712 | 0 | 40,408,582 | 90% | 678,144 | 0.9736 | 0 | 0.9736 | 0.1076 | 0.2459 | 0.7209 |
| self_verify | 883 | 0 | 78,767,102 | 93% | 1,034,284 | 1.5348 | 0 | 1.5348 | 0.1690 | 0.2679 | 1.0683 |
| evidence_risk | 802 | 361 | 59,140,732 | 93% | 1,499,226 | 1.3652 | 0.1350 | 1.5002 | 0.1872 | 0.2525 | 0.9924 |
| **total** | 2,674 | 361 | 189,693,128 | 92% | 3,341,337 | 4.0913 | 0.1350 | **4.2263** | | | 2.9422 |

- Per finished run, evidence + risk costs 74% more than vanilla MAD ($0.187 vs $0.108; at DeepInfra rates $0.124 vs $0.080, +56%). H2 expects at most +20%. The ceiling truncated 4 evidence runs and no vanilla run, so the uncapped gap is likely larger.
- In launch 5 the price-table cost of each of the 2,956 billed calls equals OpenRouter's `usage.cost`. `cost_usd` takes the larger of the two, so it is never below the bill.

## Served provider (a logged covariate)

| arm | answered calls | DeepInfra | InferenceNet | balance (more than 10 pp from pooled?) |
| --- | --- | --- | --- | --- |
| sa | 275 | 28 (10%) | 247 (90%) | ok |
| vanilla_mad | 708 | 40 (6%) | 668 (94%) | ok |
| self_verify | 879 | 81 (9%) | 798 (91%) | ok |
| evidence_risk | 1,158 | 80 (7%) | 1,078 (93%) | ok |
| pooled | 3,020 | 229 (8%) | 2,791 (92%) | |

- `provider_check.ok` is true on all 3,016 `llm_call` and `judge_call` events.
- The 4 `provider_mismatch` events are all from attempt 1 (the priority tier). Launch 5 had none.
- Temperature 0 was sent on every call.
- InferenceNet answers `service_tier: null`, DeepInfra `default`.
- The cache-hit share stayed at 90–93% per arm, even though calls moved between providers.

## Wall-clock time

Minutes per finished run, from `run_start` to `run_end`:

| arm | runs | median | mean | max |
| --- | --- | --- | --- | --- |
| sa | 9 | 3.0 | 4.2 | 15.7 |
| vanilla_mad | 9 | 11.5 | 12.1 | 21.8 |
| self_verify | 9 | 21.6 | 17.9 | 26.3 |
| evidence_risk | 8 | 27.2 | 26.2 | 38.2 |

The SA maximum (15.7 minutes) is the canvas run from attempt 4, which waited on 429s. Launch 5 took 1 h 46 min for 34 runs at 8 in parallel. Fewer runs were in flight once the cap reserve began to bind: at $3.72 only two were running. The 35 finished runs took 518 run-minutes, 14.8 minutes each on average.

## Contaminated runs

The `ground_truth_suspects` detector flags any tool call whose arguments touch `/workspace/tasks`, `tasks/finalpool`, a `groundtruth` path, `evaluation/main`, the gym's git history or its public repo. Pre-registered rule: a flagged run counts as a failure.

| kind | runs | arms | grader result |
| --- | --- | --- | --- |
| reads of the task's own remaining files only (`email_config.json`, `task_config.json`, `docs/`, `preprocess/`, listings of `/workspace/tasks/finalpool/<task>`) | 15 | sa 5, vanilla 4, self-verify 1, evidence 5 | 3 grader passes voided (yf-portfolio SA, canvas-quiz vanilla, howtocook self-verify) |
| searches for the grader or ground truth (`os.path.exists` and `os.walk` over `/workspace`, `/opt` and other roots, looking for `evaluation/`, `groundtruth_workspace/`, `evaluation/main.py`), all 20 flagged calls by the self-verify operator sub-agent | 4 | self-verify (canvas-module, canvas-quiz, fetch-sf-hr, terminal-arxiv) | all 4 stopped at the ceiling, ungraded |

No search found a task's answers. Every `evaluation/` and `groundtruth_workspace/` path of the task was missing: sealing removed them, and the grader still found them at the end.

What the searches did find was the gym's generic grader, `utils/evaluation/evaluator.py` (123 lines; it loads a task's `evaluation/` and `groundtruth_workspace`, both deleted), and `scripts/eval_config.json` (the gym's own run settings). Neither holds answers. Agents in six runs printed them: the self-verify operator in the four runs above, and the SA main agent in `sf-hr-compensation-pdf-notion` and `yf-portfolio-risk-excel`. All six runs are flagged, but only because of their other calls. A read of `utils/evaluation/` alone matches no marker, which is a gap in the detector.

The agent's `/workspace` is the gym's checkout (`README.md`, `toolathlon-gym.mdx`, `utils/`, `scripts/`, `tasks/finalpool/<task>/`), so any agent that lists its working directory finds the task's own folder and the grader code. The run list with every flagged call is in the generated report.

## Cost-ceiling aborts

| arm | runs stopped at $0.25 | spend of those runs |
| --- | --- | --- |
| sa | 0 | |
| vanilla_mad | 0 | |
| self_verify | 4 | $0.2503, $0.2524, $0.2555, $0.2679 |
| evidence_risk | 4 | $0.2507, $0.2516, $0.2516, $0.2525 |

The same $0.25 and 40 turns bind different arms. SA reached the 40-turn limit in 4 of 9 runs and never the dollar ceiling. The main agents of self-verify and evidence + risk never reached the turn limit, but those arms hit the ceiling in 8 of 17 runs. Sub-agents hit their own 40-turn limit 6 times: 4 in self-verify and 2 in vanilla MAD.

## Errors

| | launch 5 | attempts 1–4 |
| --- | --- | --- |
| `run_error` | none | RateLimitError 11, ProviderMismatch 4 |
| `llm_error` (agent) | none | RateLimitError 15, ProviderMismatch 4 |
| `subagent_error` | MaxTurnsExceeded 6 (self-verify 4, vanilla 2), CostCeilingExceeded 5 (a call refused at the run's ceiling) | RateLimitError 4, ProviderMismatch 4 |
| environment errors | none | none |

## Projected headline cost and time

The headline is 560 runs: 70 tasks × 4 arms × 2 seeds. The projection uses the per-arm means from this pilot; subset70 has other tasks, including 12 with more than 150 SA tools.

| assumption | $ |
| --- | --- |
| measured mix (92% InferenceNet) | **68.3** |
| every call at the set's upper bound ($0.098 / $0.008 / $0.50 per M) | **72.8** |
| every call at DeepInfra's rates (DeepInfra alone) | 47.3 |
| one seed, measured mix (280 runs) | 34.1 (DeepInfra rates 23.7) |
| 45 tasks × 2 seeds, measured mix | 43.9 |
| 40 tasks × 2 seeds, measured mix | 39.0 |
| run ceiling $0.15 (12 of 35 pilot runs would hit it) | 54.2 |
| run ceiling $0.10 (18 of 35 would hit it) | 42.0 |

The full headline does not fit $45 under any route. The owner's rule was to keep `deepinfra/fp8` plus whichever endpoint adds capacity within budget, if the worst case does not fit:

- DeepInfra alone would still be over budget ($47.3).
- DeepInfra alone would also be throttled: it answered only 139 of 2,595 agent calls on the first try, and OpenRouter moved the other 2,456.
- Among the fp8 endpoints OpenRouter lists, InferenceNet is the cheapest that adds capacity: 1.34× DeepInfra on the measured mix. GMICloud is next at 1.92×, then AtlasCloud at 2.08×.

So the route stays `deepinfra/fp8` + `inference-net/fp8` + `inference-net/us`, and the gap must be closed by the size of the grid, the ceiling, or the budget. That is an owner decision; see recommendation 1.

At 14.8 minutes per run, 560 runs at 8 in parallel take about 17 hours, about 8.7 hours per seed. Launch 5 needed no retries. If DeepInfra and InferenceNet were both overloaded, a call could wait up to about 10 minutes before its run fails.

## Recommendations (owner decisions)

1. Fit the headline to $45. The full 560 runs project to $68–73. The options:
   - one seed of the 70 tasks: $34 measured, $36 at the upper bound;
   - about 40 tasks × 2 seeds: $39;
   - a larger budget: about $75 covers the upper bound;
   - a lower ceiling, which truncates the delegating arms most and so biases H1 against the method.

   I recommend one full seed first, under the existing $45 cap. Decide on seed 2 from its measured cost, and write that rule into the plan before launching.
2. Decide the contamination rule before the headline. As pre-registered, it voids runs that read the task's own non-answer files: 15 of 35 runs, including 3 grader passes. The four self-verify ground-truth searches stay contaminated under either option:
   - narrow the detector to removed answer paths and grader code (`groundtruth*`, `evaluation/`, `utils/evaluation`, other tasks, git history, the public repo), and amend the pre-registration with a date before any headline run;
   - or give the agent a working directory without the gym checkout and the task folder (copy `initial_workspace` there, and keep the task folder where only the servers and the grader read it).

   I recommend the second option. It also hides the gym's `README.md`, `utils/evaluation/evaluator.py` and `scripts/`, which agents in six runs read to learn how grading works.
3. Report the self-verify searches for the grader as a finding. The operator sub-agent looked for `evaluation/main.py` and `groundtruth_workspace` in 4 of 9 runs. The seal held, but the searches are attempts to read the grader, and they cost turns and dollars. Changing the prompt would change the arm, so decide before freezing.
4. Hold off reading the evidence arm until the claims are labelled. It passed 0/8 here (0/4 graded), cost 1.7× vanilla, and hit the ceiling in half its runs. Its 115 returns led to 41 re-delegations and 23 fallbacks, which suggests rejections drive much of the extra spend. Hand-label the judge sample, and check L2 (REVIEW J) and the 12,000-character judge evidence cut (REVIEW K). The 11.9% upper bound on false rejects is above H4's 5%.
5. Settle the limits. 40 turns binds SA (4/9 runs) and $0.25 binds the verifying arms (8/17), the asymmetry the review flagged. Either keep it and state it, or rebalance before freezing. In both cases, report `max_turns_reached` and `cost_ceiling` per arm next to success.
6. Keep the route and the parallelism. On the fp8 set no call in launch 5 needed a harness retry, and the provider mix is balanced across arms. Keep the provider covariate analysis as pre-registered. Eight runs in parallel ran without Docker or rate-limit trouble.
7. Leave the 5 runs that did not start (yt-ml-repos × 4, yf-portfolio × evidence). The decisions above do not need them, and at the per-arm means they would take spend to about $4.90, past the $4.50 stop rule.
8. Next steps, in order: owner decisions 1–5; `scripts/judge_validation.py export runs/pilot_deepseek --out validation/judge_v1` and the hand labels; then freeze `prompts/gym_v1`; then the headline.
