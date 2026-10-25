# Pilot diagnosis: `runs/pilot_deepseek` (DeepSeek-V4.1-Flash-1017, 24 Oct 2026)

Read-only analysis of the pilot traces, ledgers, grader logs and sealed task copies. No API calls, no `evcarry` commands, no tracked files touched. Every count below comes from the scripts in `/tmp/pd/` (listed in Appendix C) run against the files in `runs/pilot_deepseek/`.

## 0. Findings in brief

1. The evidence arm's 0/8 is mostly caused by the harness, not by sub-agent errors. Of 64 rejected returns, 11 (17%) caught something genuinely wrong: 2 cited a python call that had crashed, 9 had a wrong side detail while the value the task needed was right. 36 (56%) were triggered by bugs in the L1/L2/L3 implementation, and 17 (27%) by claims that cannot carry evidence by construction (tool inventories, "no files were modified") or that pack unshown details into one claim. None of the 11 genuine catches touched a value a grader checks. The rejections pushed 4 of 8 evidence runs into the $0.25 ceiling, including `howtocook`, which all three other arms passed.
2. The contamination flag found no ground-truth access. No flagged call in any of the 19 flagged runs returned task ground truth. 117 of the 169 flagged calls read the task's own directory, which `_seal` keeps on purpose and which the `/workspace/tasks` marker matches. 37 calls did search for graders (35 of them from self_verify operators) and found only the gym's generic evaluator code. The flag turned 3 grader passes into failures.
3. Most failures are not arm effects. 4 of the 10 pilot tasks have grader or spec defects that fail correct output (0 of 14 graded runs on them passed, in any arm). The limits bind differently per arm: SA hit the 40-turn cap in 4 of 9 runs and wrote no deliverable in any of them; self_verify and evidence hit $0.25 in 8 of 17 runs.
4. Cost is a routing problem first. 94% of calls were served by InferenceNet, whose cached-input and output prices are 2.7x and 1.7x DeepInfra's. The same tokens at DeepInfra prices cost $2.92 instead of $4.20. Caching already works (93% hit rate, stable prefix). With DeepInfra capacity outside OpenRouter's shared pool, the evidence fixes and SA at 100 turns, the 560-run headline projects to about $39-42, and the 480-run headline without the pilot tasks to about $34-36. Without that capacity it projects to about $65 and does not fit the $45 cap.
5. The method has a structural limit. In this pilot sub-agents rarely returned wrong data; the grader failures came from task interpretation. In `sf-sales` the researcher reported both the all-status revenue (3,048,998.33, the grader's value) and the non-cancelled revenue (2,911,038.6); the judge correctly marked that claim supported, and the main agent chose the wrong total. Evidence-carrying claims check provenance, not whether the right question was asked. vanilla_mad had no natural success-to-failure flips against SA, so H1 had nothing to halve on these tasks.

## 1. Data

| | sa | vanilla_mad | self_verify | evidence_risk | total |
|---|---|---|---|---|---|
| finished runs | 9 | 9 | 9 | 8 | 35 |
| passed (after contamination rule) | 1 | 3 | 2 | 0 | 6 |
| grader passed (`eval_passed`) | 2 | 4 | 3 | 0 of 4 graded | 9 |
| billed $ per run | 0.0239 | 0.1076 | 0.1690 | 0.1872 | mean 0.1201 |

Spend in finished traces is $4.202, plus $0.023 in the 15 aborted attempts (429s and the priority-tier pin failure), which matches the runner's own $4.226. `runs/pilot_deepseek_run2.log` shows the runner refusing to start another run at that spend against the $4.50 stop, so `yf-portfolio-risk-excel/evidence_risk` and all four `yt-ml-repos-github-notion-excel-email` runs did not run. The numbers in the brief (45 accept, 39 redelegate, 23 fallback, 3 ceilings) were mid-run; the final evidence-arm counts are 51 accept, 41 redelegate, 23 fallback and 4 ceilings.

The traces do not contain message content. Tracing is disabled (`trace/processor.py` is never fed spans), so the main agent's instructions to sub-agents and the sub-agents' final outputs are not logged. Claims are visible only for the 361 claims that went to the judge (`judge_call.claim`); for the other 824 claims only the check reason survives. This limits what can be said about unsampled claims, and it blocks the planned MS-Eval propagation labels (section 7, fix P0-2).

## 2. Evidence arm: why returns were rejected

### 2.1 Totals

115 sub-agent returns carried 1,185 claims (median 8 per return, mean 10.3, max 28). 141 distinct claims failed at least one check (11.9%). Because one failing claim rejects the whole return, 64 of 115 returns (56%) were rejected. The 64 rejected returns hold 192 failing checks: L3 85, L2 48, L1 41, L0 18.

### 2.2 Every failing check by mechanism

| level | mechanism | count | verdict | where |
|---|---|---|---|---|
| L1 | numeric value must appear in every cited result, not in at least one | 13 | check bug | `checks.py:147-156`, `_l1` line 214 |
| L1 | span quotes the harness note `...[truncated: showing 8000 of N chars...]`, which the model sees but the ledger does not store | 3 | harness bug | `tools.py:139-149` |
| L1 | span formatting slip on correct content (collapsed table whitespace, added quotes, escaped unicode, span copied from the code argument, invented `[FILE] name` format, paraphrased error text) | 14 | format | |
| L1 | span cites a python call that crashed | 7 | genuine | |
| L1 | count claim typed `fact` instead of `derived` | 1 | format | |
| L1 | tool-inventory claim | 1 | meta | |
| L1 | unsampled claim, content not logged | 2 | unknown | |
| L0 | `claim has no evidence pointer`: tool-inventory claims (two returns, 8 and 9 claims) | 17 | meta | |
| L0/L2 | truncated call id `chatcmpl-tool-7e9` | 2 | format | |
| L2 | write claim checked against the read-back only; the logged write result is not in the comparison | 7 | check bug | `replayer.py:95-122` |
| L2 | claim value not verbatim in the replay: dicts with derived keys, summary strings ("read-only exploration only", "3 entries verified") | 12 | check bug | `value_in_text` |
| L2 | value appears only in a non-replayable cited result (python_execute, nondeterministic tool), which L2 skips and then ignores | 9 | check bug | `replayer.py:82-105` |
| L2 | full-text drift: `canvas_health_check` timestamp (2); email folders, email lists and a directory tree changed by the sub-agent's own later writes (4) | 6 | check bug | `replayer.py:87`; REVIEW item J |
| L2 | tool-inventory or negative claim | 13 | meta | |
| L3 | tool inventories, parameter schemas, "no files were modified", "the only data source is" | 35 | meta | |
| L3 | judge saw evidence cut in the middle at 12,000 chars and cited the omitted part | 19 | harness bug | `judge.py:154-158`; REVIEW item K |
| L3 | claim adds details the cited calls do not show | 18 | over-claim | |
| L3 | evidence contradicts a side detail | 9 | genuine | |
| L3 | fabricated support (cited call crashed) | 3 | genuine | |
| L3 | judge misread (email IDs vs Message-IDs) | 1 | judge error | |

Examples, with the trace files under `runs/pilot_deepseek/`:

- L1 per-pointer bug. `sf-sales-region-analysis-gsheet-email__evidence_risk` return 8: five claims, one per regional email, each with value such as `"617263.14"` and two pointers, the `send_email` result `Email sent successfully to europe.manager@company.com` and a `read_email` result containing `Total Revenue: 617263.14`. L1 failed every claim on the `send_email` pointer. The emails were correct (`[PASS] Email check passed` in the grader log). The operator then re-sent all five: the run has 10 `send_email` calls and a later `emails-delete_emails` on IDs 5949-5953.
- L2 write bug. `wc-shipping-analysis__evidence_risk` return 4, claim value `"Email sent successfully to logistics@shop.com"`, compared with the read-back `emails-search_emails(folder=Sent, query=subject)`, which can never contain that string. The grader log says `Checking email... PASS`. The operator had already sent the email once; the run sent it twice and later deleted one copy.
- L2 non-replayable bug. `sf-hr-compensation-pdf-notion__evidence_risk` return 12: value `11`, present in the cited python output `Blocks on final page: 11 | expected: 11`, absent from the replayed Notion read. L1 also failed it four times, once per pointer without "11".
- L3 cut evidence. `canvas-quiz-analysis-ppt__evidence_risk` return 7, judge: "The evidence explicitly shows the account 3 output truncated and the account 4 output omitted entirely ("[6452 characters omitted]"), contradicting the claim that all four outputs were returned in full." The harness cut was read as a contradiction. Across all 361 judge calls, 53 had cut evidence; the judge supported 24 of 53 (45%) of those and 252 of 308 (82%) of the rest.
- Meta claims. `canvas-module` return 5 and `sf-hr-compensation` return 13 list the sub-agent's own tools ("Canvas MCP server, 41 tools: ...", "notion-API-post-page: description 'Notion | Create a page'; parameter names ..."). No tool call shows a tool inventory, so L0 fails with "claim has no evidence pointer". These claims answer main-agent questions about capabilities: the main agent is told only that the researcher "looks things up with read-only tools (databases, files, mail, web pages, documents)" (`prompts/gym_v1/evidence_system.md`).
- Genuine catches. `terminal-arxiv` return 2 claims "Three independent extraction methods returned identical text" and quotes `/F2 18 Tf 22 TL` from a python call whose output is a `ValueError` traceback; return 9 claims `notion.databases count: 0` from a call that failed with `KeyError: 'PG_PORT'`. Both re-delegations were accepted. The 9 wrong side details include "the only visible Notion user is the bot" (the user has `type: person`), "each course has Fall and Spring variants" (contradicted by the course list), "no pre-existing HowToCook directory" (`/opt/local_servers/HowToCook-mcp` exists) and "7 temp files" (the read-back shows 4).

The minimum-one-audit rule is not a driver: only 7 returns had a single claim, and the 2 rejected ones were the same L2 non-replayable bug. Call-id citation works: one malformed id in 1,185 claims.

### 2.3 Rejected returns by class

A return is classed by its most serious failure: A if any failure is fabricated support, else B if any genuine wrong detail, else D if any check or harness bug, else C. Appendix A lists all 64.

| class | redelegate | fallback | total |
|---|---|---|---|
| A genuine catch: fabricated support | 2 | 0 | 2 |
| B genuine catch: wrong side detail, task value right | 7 | 2 | 9 |
| C only meta or over-broad claims, no error shown | 11 | 6 | 17 |
| D at least one check or harness bug, no genuine catch | 21 | 15 | 36 |
| total | 41 | 23 | 64 |

True catches: 11 (17%). False rejects: 53 (83%), of which 36 from check or harness bugs and 17 from claims that cannot be checked or say more than the evidence shows. The judge's own record matches: 85 of 361 judged claims were not supported (23.5%), and 12 of those 85 (14%) were genuine errors.

Fixing layers one at a time (a return stays rejected while any failure outside the fixed layers remains):

| after fixing | rejected returns left (of 64) |
|---|---|
| check and harness bugs | 42 |
| plus citation-format tolerance | 39 |
| plus no meta claims | 24 |
| plus one checkable fact per claim, judge misread | 11 (= the genuine catches) |

Fixing the code alone clears only 22 of 64. The claim scope (what sub-agents are asked to claim) matters as much as the checks.

### 2.4 Do redelegations succeed, and does fallback break the task?

| class of the rejected return | retry accepted | retry fell back |
|---|---|---|
| A | 2 | 0 |
| B | 3 | 4 |
| C | 6 | 5 |
| D | 7 | 14 |

Retries fix genuine errors (A 2/2) and mostly fail on bugs (D 7/21), because the sub-agent cannot satisfy a broken check: the read-back still lacks the success string, the health check still has a new timestamp. 18 of 41 redelegations were accepted.

Fallback discards the whole return. `delegate.py:179-180` sends only `FALLBACK: ... Plan this step again and delegate it with a more specific instruction.`, so the verified claims in that return (median 9 claims per fallback return, up to 28) never reach the main agent, and the main agent delegates the step again. Two side effects:

- Repeated writes. The feedback text is "Fix only the failed claims and return the full structured result again" (`policy.py:57`), and after a fallback the main agent delegates the step afresh; for an `action_done` claim the operator redoes the action. `sf-sales` sent 10 emails for 5 recipients: the second round (IDs 5954-5958) followed the fallback at return 4, and the operator later deleted 5949-5953. `wc-shipping` sent the review email twice (the second after the fallback at return 2) and deleted one copy. No other arm sent an email twice. Repeated Notion creations (`sf-hr-compensation` 7 pages titled "Compensation Analysis 2026" under different parents, `fetch-sf-hr` 10 databases) look like probing for a valid parent page, and self_verify does the same (`fetch-sf-hr` 7 databases, `canvas-module` 5), so those are not specific to the evidence arm.
- Budget exhaustion. Redelegation reruns alone cost $0.225 (60 sub-agent calls, 15% of evidence spend). The 4 evidence runs that hit $0.25 had 6, 4, 4 and 1 fallbacks; `howtocook` had 8 returns, 0 accepted, and wrote no deliverable. On the same tasks vanilla passed `sf-hr-compensation` and `canvas-quiz` (the latter flagged), and SA, vanilla and self_verify all passed `howtocook`. The protocol caused at least those losses.

### 2.5 Is the method weak?

As implemented, yes: on this pilot it lost tasks and caught nothing that mattered to a grader. After the fixes in section 7 most of the self-inflicted damage should go, but two limits remain.

First, the deterministic checks mostly test formatting. L1 and L2 match strings; many claims are derived values (147 of 361 judged claims were `derived`), which pass L1 by design and fall to L3. The pilot's only fabrications were caught by L1 because the cited call had crashed, which is cheap and worth keeping.

Second, the errors that failed tasks were task-interpretation errors that evidence cannot see. `sf-sales` claim c14 (return 2) reported both totals and was judged supported with the reason "The query results directly show all listed revenue values". `fetch-sf-hr` picked Support over R&D by a correct raw average. `wc-shipping` named a column "Zone name" as the spec says. Each claim was true and well cited. The planned fault-injection study (M6) is where catch rates can be shown; the DeepSeek headline may only support "no harm at modest cost" unless natural flips appear, and on these 10 tasks there were none for vanilla_mad.

## 3. Contamination flags

The detector (`gym_docker.py:656-670`) flags any ledger call whose arguments contain `groundtruth`, `ground_truth`, `/workspace/tasks`, `tasks/finalpool`, `/workspace/.git`, `eigent-ai` or `evaluation/main`. `_seal` deletes every other task, this task's `evaluation/` and `groundtruth_workspace/`, `*groundtruth*` files and the gym's `.git`, and keeps this task's `docs/task.md`, `docs/agent_system_prompt.md`, `task_config.json`, `preprocess/` and, where present, `email_config.json` and `initial_workspace/`. Any read of those files matches `/workspace/tasks`.

| | sa | vanilla_mad | self_verify | evidence_risk | total |
|---|---|---|---|---|---|
| runs flagged | 5/9 | 4/9 | 5/9 | 5/8 | 19/35 |

| flagged calls | count | what came back |
|---|---|---|
| reads or listings of the task's own sealed directory | 117 | the task text, `task_config.json`, `preprocess/main.py` (for `canvas-quiz` and `yf-portfolio` it states that no data is injected or cleared) |
| searches for `groundtruth`, `evaluation`, grader or expected-output names | 37 | only `/workspace/utils/evaluation/evaluator.py` (the gym's generic evaluator) and lines such as `groundtruth_workspace MISSING`, `EXISTS: False` |
| access denied or tool error | 15 | nothing |

35 of the 37 searches came from self_verify operators (`canvas-module` 7, `canvas-quiz` 10, `fetch-sf-hr` 7, `terminal-arxiv` 11), for example `find / ... -iname '*groundtruth*' -o -iname '*evaluation*'` and a grep for `evaluation_command`, `Quiz_Performance`. The self_verify prompt tells the main agent to check every reply and ask again; the operators looked for the grader to check against. All four of those runs ended at the cost ceiling, ungraded. The other two searches were SA `sf-hr-compensation` (failed) and SA `yf-portfolio-risk-excel` (a walk of `/workspace/tasks` for file names containing "eval", which found nothing).

Three grader passes were counted as failures:

| run | flagged calls | verdict |
|---|---|---|
| `canvas-quiz-analysis-ppt` vanilla_mad | 6 own-directory reads, 1 blocked | false positive |
| `howtocook-weekly-gsheet-gcal` self_verify | 3 own-directory reads | false positive |
| `yf-portfolio-risk-excel` sa | 3 own-directory reads, 1 search for "eval" files (empty) | attempted, nothing obtained |

The pre-registered rule (plan section 5) names "the removed task files and `groundtruth` paths, the gym's git history, or its public repo". The task's own directory is not removed, so the `/workspace/tasks` and `tasks/finalpool` markers are broader than the rule. Under the written rule, the first two runs pass and the third depends on whether a search that found nothing counts. Corrected pass counts: sa 1/9 (2/9 if attempts do not count), vanilla_mad 4/9, self_verify 3/9, evidence_risk 0/8.

Three things the detector does not see but the traces show:

- The container has internet access. The `howtocook` evidence operator ran `git ls-remote` and `git clone --depth 1 https://github.com/Anduin2017/HowToCook.git` into the workspace (not flagged: the marker is `eigent-ai` only).
- Sub-agents read `docs/task.md` from disk, so in the delegating arms they can learn the full task even though "a sub-agent does not see the task".
- `python_execute` reaches the shared Postgres directly with the container's PG variables. In `canvas-quiz` an operator counted rows in all 95 tables of every schema; in `sf-sales` the evidence operator renamed a duplicate spreadsheet with a direct SQL update. Graders read the same database.

## 4. Why runs fail

| arm | runs | passed | grader pass, flagged | grader fail | stopped by turn cap | stopped at $0.25 | other |
|---|---|---|---|---|---|---|---|
| sa | 9 | 1 | 1 | 3 | 4 | 0 | |
| vanilla_mad | 9 | 3 | 1 | 4 | 0 | 0 | 1 ended without `claim_done`, graded fail |
| self_verify | 9 | 2 | 1 | 2 | 0 | 4 | |
| evidence_risk | 8 | 0 | 0 | 4 | 0 | 4 | |

Of the 13 runs in the grader-fail column, 12 are on four tasks whose graders or specs fail correct output.

| task | defect | evidence |
|---|---|---|
| `canvas-module-completion-notion-gcal` | Notion grader reads each property's `"type"` key (`_val_from_property`), which the Notion mock never stores; it stores the request as sent | SA created the database with the exact title and 5 pages with correct values; grader: `Module 'Introduction' exists in tracker: Found modules: []` |
| `fetch-sf-hr-training-notion-gcal-excel` | same Notion parser (needs `"type"` and `plain_text`); grader hard-codes `PRIORITY_DEPTS = ["Operations", "Sales", "R&D"]` | raw averages from the warehouse: Support 3.1956, R&D 3.1968 (both 3.20 at 2 dp); the spec asks for the three lowest, which is Support |
| `sf-sales-region-analysis-gsheet-email` | docx check reads only `doc.paragraphs`, needs at least 30 characters and "europe", while the spec asks for "a heading Regional Sales Summary and a table"; the grader also expects all order statuses in "all-time sales" | every arm fails the docx check; the agents' total 2,911,038.6 excludes cancelled orders, the grader's 3,048,998.33 includes them; self_verify's Excel passed |
| `wc-shipping-analysis` | spec says the sheet "should have Zone name", grader requires the header `zone`; Summary expected as label rows | sa, self_verify and evidence all fail with the same three errors |

0 of 14 graded runs on these four tasks passed (this includes the vanilla `wc-shipping` run that ended without `claim_done`). The 13th run in the grader-fail column is vanilla `terminal-arxiv`, which included a noise paper. In subset70 the same Notion parser appears in five graders (`canvas-module`, `fetch-sf-hr`, `yt-ml-repos-github-notion-excel-email`, `wc-review-summary`, `yf-financial-health-excel-notion`), and 22 graders check docx text through `doc.paragraphs`; those 22 need a per-task look, since many may be fine.

Limits:

- 40 turns is too tight for SA. All 4 capped SA runs (`canvas-quiz`, `fetch-sf-hr`, `sf-hr-compensation`, `terminal-arxiv`) wrote no deliverable. By a rough heuristic, 13 of 66, 30 of 65, 16 of 87 and 26 of 78 of their tool calls were environment scans (`os.walk`, `find /`, task directory, `/opt`). Toolathlon-Gym's own default is `MAX_STEPS=100` (`third_party/toolathlon_gym/README.md`). SA costs little: the most expensive SA run was $0.069.
- $0.25 binds the delegating arms: 4/9 self_verify, 4/8 evidence, 0/9 vanilla (its maximum was $0.246). Sub-agents also hit their own 40-turn cap 6 times (`subagent_error: MaxTurnsExceeded`).
- So SA is turn-bound while the delegating arms are dollar-bound. SA is the weakest arm only because of that cap, which suppresses the success-to-failure flips H1 measures: the only SA-pass, arm-fail pair in the pilot is evidence on `howtocook`, caused by the protocol.
- Tool errors are not a main cause (0-12 errored or denied calls per SA run).

The repo has no Toolathlon-Gym pass rates for comparable models. The plan cites vendor Toolathlon-Verified scores (DeepSeek-V4-Flash-0731 70.3), a different, 108-task benchmark run under its own harness; it is not comparable to these numbers.

## 5. Cost

### 5.1 Where the money went

| component | calls | $ billed | input Mtok | cache hit | output Mtok |
|---|---|---|---|---|---|
| SA main | 273 | 0.215 | 11.3 | 92% | 0.13 |
| delegating main agents (3 arms) | 174 | 0.192 | 2.5 | 64-74% | 0.24 |
| researchers (3 arms) | 947 | 1.850 | 81.5 | 91-95% | 1.64 |
| operators (3 arms) | 1,224 | 1.809 | 93.1 | 88-94% | 1.13 |
| L3 judge | 361 | 0.135 | 0.7 | 20% | 0.18 |
| total | 2,979 | 4.201 | 189.1 | 93% | 3.31 |

Evidence-arm extras: judge $0.135 (9% of the arm), redelegation reruns $0.225 (15%), and 144 of 361 judge calls ($0.057) were spent on returns that L0-L2 had already rejected.

Per arm, at the price of the provider that served each call: cached input is 29-37% of cost, uncached input 23-37%, output 28-49% (evidence 49%: its researchers write about 2,370 output tokens per call against vanilla's 1,250). Reasoning is on: the judge was billed 182k output tokens for about 26k tokens of visible replies.

### 5.2 Providers

| provider | calls | $ billed | same tokens at DeepInfra prices |
|---|---|---|---|
| InferenceNet | 2,791 (94%) | 3.712 | 2.428 |
| DeepInfra | 188 (6%) | 0.491 | 0.491 |

OpenRouter tried DeepInfra first and moved 2,456 calls to InferenceNet (`router.attempt: 2`); the 429 body says `limit_source: upstream_provider_shared_pool` and suggests adding your own provider key. Arm shares of InferenceNet range from 90% (sa) to 96% (vanilla), inside the pre-registered 10-point band.

### 5.3 Caching, prefix and context

- Both providers report cached tokens (`prompt_tokens_details.cached_tokens`, logged as `cache_hit_tokens`). Overall hit rate 93%.
- The prefix is stable: every agent kept one system-prompt hash per run (87 of 87 agent-runs), and tools are sorted by name.
- Consecutive calls of one agent on the same provider hit 95%; after a provider switch, 61% (142 switches). Switches added about 4.0M miss tokens, 27% of all misses, about $0.38 at DeepInfra's miss price.
- Tool schemas are about 16% of input tokens overall and 38% for SA (first-call input: SA and operators 9.7k-24k tokens, researchers 5.4k-9.2k).
- Persistent sub-agent sessions grow without bound. Evidence researchers: median 78k input tokens per call, 90th percentile 210k, maximum 329k; 146 of 353 calls exceeded 100k. The configured `context: 131072` is never enforced.

### 5.4 The proposed savings

Percentages are of pilot spend unless stated.

| option | saving | keeps validity? |
|---|---|---|
| (a) caching, stable prefix | about 1% beyond (b). Already 93%; the remaining misses come from provider switches, which (b) removes. Main-agent and judge hit rates are low but they are 5% and 3% of spend | yes, neutral |
| (b) provider with the best cached price | 30.5% on the same tokens ($4.20 to $2.92), plus about 9% from no switches if every call stays on one provider. DeepInfra through OpenRouter's shared pool refused 94% of first attempts, so this needs your own DeepInfra capacity (OpenRouter BYOK or DeepInfra's API). If the plan's "BYOK is not possible" still holds, InferenceNet-only gives the switch saving (about 9%) at the higher price | yes, and it removes the provider covariate, if it is the same fp8 deployment. Re-express the $0.25 ceiling at reference prices, or the cheaper tokens let ceiling-bound runs go about 45% further (about +$10 over 560 runs) |
| (c) compact tool descriptions, all arms | 2-4% at DeepInfra prices (schemas are 16% of input and almost all cache hits) | changes the benchmark's tool interface in every arm; not worth it |
| (d) trim old tool results, all arms | a 64k block-trim removes 39% of input tokens but only about 7% of cost at DeepInfra prices (12% at InferenceNet), because the trimmed tokens are cache hits. Enforcing the configured 131k removes 14.5% of input tokens, about 3% of cost | yes if identical in all arms and frozen before the headline; trim in blocks so the prefix stays cacheable; evidence citations still resolve through the ledger and `view_overlong_output` |
| (e) cheaper judge, or judge only after L0-L2 pass | judge is 3% of all spend. Skipping L3 for claims that already failed saves up to 40% of judge calls; lowering judge reasoning could halve the rest | yes, if set before the hand-labelling (not done yet, so now) |
| (f) fewer redelegations after fixing false rejects | redelegation reruns are 15% of the evidence arm. The evidence arm costs 1.56x vanilla at DeepInfra prices while ceilings truncate half its runs; after the fixes I expect 1.3-1.6x vanilla, so dollar savings are modest, but ceiling hits and duplicate writes should mostly go | required; the fixes come from pilot traces (section 7, P0-1) |
| (g) bulk seeds on the local GPU, DeepSeek for one seed | halves DeepSeek spend | does not keep the seeds. Seed 2 would be a different model, and pooling seeds would mix models. At temperature 0 the seed is not sent to the provider and the risk sampler is deterministic, so DeepSeek seeds 1 and 2 are replicates that differ only by serving nondeterminism. CLAUDE.md also puts local runs last and requires your OK to stop the lerobot process. Use Qwen as the planned second model (M6), or for free fix-verification loops once you approve |

Better options than (a)-(g):

1. Get your own DeepInfra capacity and pin one provider. This is the largest lever (about 40% with the switch saving) and it improves validity.
2. Run the primary headline on the 60 subset70 tasks outside pilot10. That is 14% fewer runs, and it removes test-set tuning (section 8).
3. Fix or rule out the defective graders. They burn 4 of 10 pilot tasks for no signal.
4. Hide `/workspace/tasks` and the gym tree from the agent and add an egress allowlist. SA's capped runs spent their turns scanning the container, and self_verify operators spent theirs hunting graders.
5. Tell every delegating main agent which servers and tools each sub-agent has (identical text in all delegating arms). This removes the "list your tools" delegations, which produced 66 of the 192 failing checks.

## 6. Headline projection

Assumptions: the pilot's per-run token use per arm is representative of subset70; prices from `configs/prices/openrouter_deepseek_1017_2026-10-24.yaml`; the run ceiling is kept at the pilot's token budget (about $0.17 at DeepInfra prices) unless stated; 140 task-seeds for 70 tasks x 2 seeds, 120 for 60 x 2; each task-seed runs the 4 arms.

| scenario | $ per task-seed | 70 tasks x 2 (560 runs) | 60 tasks x 2 (480 runs) |
|---|---|---|---|
| A as piloted, mixed providers, billed | 0.488 | 68.3 | 58.5 |
| B all calls at DeepInfra prices, same tokens | 0.338 | 47.3 | 40.6 |
| C B plus one provider (no cache loss on switches) | 0.295 | 41.3 | 35.4 |
| D C plus evidence fixes, evidence at 1.3x / 1.45x / 1.6x vanilla | 0.274 / 0.286 / 0.298 | 38.4 / 40.1 / 41.7 | 32.9 / 34.3 / 35.8 |
| E D (1.45x) plus SA at 100 turns (+$0.017 per SA run) | 0.303 | 42.4 | 36.4 |
| F E plus the 131k context enforced | 0.294 | 41.2 | 35.3 |
| G E plus a 64k block-trim | 0.282 | 39.5 | 33.8 |
| E with InferenceNet only (no DeepInfra capacity) | 0.464 | 64.9 | 55.6 |

Add about $1 for the fix-verification runs and a few cents for the judge-validation re-judging. With DeepInfra capacity and that $1 counted, scenarios E-G leave $7-10 of the $45 for the 60-task headline and $2-5 for the 70-task headline; at the high evidence multiplier (1.6x), scenario E leaves about $6 for 60 tasks and reaches the cap for 70. Without it, neither fits unless output tokens fall (reasoning effort, a model-wide change that would need a new pilot) or the run count drops. The largest uncertainties: subset70's 12 tasks with 151-230 tools carry larger schemas; the evidence-fix multiplier is a judgement (1.3-1.6x); and if the $0.25 ceiling stays nominal after moving to DeepInfra, add about $10 at 70 tasks.

## 7. Fix list, in priority order

Verification costs assume DeepInfra prices; "dev tasks" means 2-3 tasks drawn by a seeded rule from pool160 minus subset70 (no email defect, at most 150 tools), never pilot10.

| # | change | why (evidence above) | cheap verification | validity risk |
|---|---|---|---|---|
| P0-1 | Pre-register a dev/test split now: primary headline = subset70 minus pilot10 (60 tasks); report pilot10 separately; verify every fix on dev tasks | all 10 pilot tasks are in subset70, and every fix below was found on their traces | $0 | none; also 14% cheaper |
| P0-2 | Log handoffs: main-to-sub instruction, sub-agent final output, repair, redelegation and fallback messages | the trace has no message content (section 1); MS-Eval propagation labels need it | $0, mock run | none |
| P1-1 | L1: numeric value must appear in at least one cited result | 13 failures; sole cause of `canvas-module` r1 and `sf-sales` r4, r8 | $0: unit test from `sf-sales` r8 claim and ledger | low; matches the plan's L1 definition |
| P1-2 | L2 writes: check the value against the logged write result plus the read-back; read-backs verify effect fields (recipient, subject, title), not the success string; add read-backs for Notion page and database creation, Sheets writes, calendar events, or report them unverifiable | 7 failures; both repeated email sends followed rejected email claims | $0: fixtures from `wc-shipping` r1-r4 | low |
| P1-3 | L2 reads: replace full-text equality with "cited span or value present in the replay"; mark `canvas_health_check` volatile; treat reads of state the run itself writes as stateful | 6 drift failures; REVIEW item J | $0 | low |
| P1-4 | L2 with non-replayable pointers: include their logged text in the value check, or defer to L3 | 9 failures | $0 | low |
| P1-5 | One value-matching rule for L1 and L2 (today L1 skips non-numeric values and L2 demands them verbatim) | 12 failures | $0 | low |
| P1-6 | Strip harness notes from spans and collapse whitespace before matching | 3 truncation-note and 2 whitespace failures | $0 | low |
| P1-7 | L3: no blind middle cut; send span-anchored excerpts plus the head of each result; tell the judge that omitted text is unknown, not absent | 19 rejects; 45% vs 82% support | about $0.01: re-judge the 19 claims with full evidence | low; freeze before hand-labelling |
| P1-8 | Per-claim acceptance: pass verified claims, re-delegate only failed ones, and keep verified claims on fallback | 10.3 claims per return at 11.9% failure; fallback dropped 23 returns' worth of verified work | dev tasks, evidence arm, about $0.40 | method change; pre-register on design grounds |
| P1-9 | Redelegation text: "do not repeat actions that already succeeded" | repeated email sends in 2 of 8 evidence runs, none in other arms | same runs; count repeated writes | method change, small |
| P1-10 | Claim scope (evidence return format only): claims about task data and actions; tool inventories, environment notes and negative statements go to `open_issues`; one checkable fact per claim | 66 meta and 18 over-claim failures | same runs; target at most one non-genuine reject per run | method change on the method arm only: tuning on the test set unless P0-1 holds |
| P1-11 | Skip L3 for a claim that already failed L0-L2 | 40% of judge calls were on already-rejected returns | $0 | changes the redelegation feedback slightly; pre-register |
| P2-1 | Limits: 100 turns for every agent (gym default) and one reference-priced $ ceiling for every arm | SA capped in 4/9 with no deliverables; delegating arms capped in 8/26 by dollars | SA on 2 dev tasks, about $0.10 | re-registers decision G; makes the comparison fairer and gives H1 flips to measure |
| P2-2 | Remove `/workspace/tasks` for the run (restore before grading, as `evaluation/` is); restrict `python_execute` and terminal to the workspace; egress allowlist | 117 own-directory reads, 37 grader searches, a GitHub clone, direct SQL writes | $0, Docker test | same for all arms; check which servers need network (the fetch API is on localhost) |
| P2-3 | Detector: implement the pre-registered definition with two tiers, "attempted" (arguments name grader or ground-truth paths, removed files or the gym repo) and "obtained" (result contains removed content); drop the bare `/workspace/tasks` marker | 3 false failures; 0 obtained | $0: rerun on pilot ledgers | changes a classification after seeing outcomes: justify as code-matches-registration, apply to all arms, report both tiers |
| P2-4 | Grader defects: a static rule like `docs/TASK_RULES.md` (Notion `type`/`plain_text` parser, docx-paragraph checks on table specs, hard-coded answers that contradict the data), or patch the Notion mock to store properties in the API's response shape | 0/14 graded runs passed on 4 defective tasks | $0: grade a hand-written correct solution | a task change: rule-based and decided before the headline, or 4-7 of 70 tasks give no signal |
| P2-5 | Tell every delegating main agent each sub-agent's servers and tool names (shared preamble) | "list your tools" delegations | inside the P1-8 runs | changes all delegating arms equally |
| P3-1 | Own DeepInfra capacity, one provider | 94% InferenceNet; 30-40% saving | 20-call probe, about $0.02; one SA run to read cache hits, about $0.02 | positive; confirm the same fp8 deployment |
| P3-2 | Enforce the 131k context (or a 64k block-trim), all arms | calls up to 329k | 2 dev tasks, vanilla, about $0.15 | uniform; freeze before the headline |
| P3-3 | Judge reasoning setting, fixed before hand-labelling | 86% of judge output is hidden reasoning | inside P1-7 | none if fixed first |

Total cheap verification: under $1.

## 8. Fairness and test-set flags

- Pilot tasks inside the headline. All of pilot10 is in subset70. P1-8 to P1-11 change only the method arm, and they were designed after watching it lose on pilot10. Reporting them on a set that includes pilot10 is tuning on the test set. Keep pilot10 out of the primary analysis (P0-1) and draw verification tasks outside subset70.
- Arm-specific limits. SA is turn-bound and the delegating arms are dollar-bound; one reference-priced budget and the gym's 100 turns for all removes the asymmetry.
- Contamination rule and self_verify. The self_verify prompt induces grader-seeking (35 of 37 searches). If "attempted" counts as contamination, self_verify is penalised for a behaviour its own prompt provokes. Decide and register that before the headline, and report both tiers.
- Retry asymmetry. The evidence arm gets harness redelegations (extra sub-agent calls) that other arms do not; with persistent sessions this is a stated design choice, but count those calls in H2's cost.
- H2 is unlikely to hold. The evidence arm cost 1.56x vanilla at DeepInfra prices in the pilot, and I expect 1.3-1.6x after the fixes, not the 1.2x H2 allows.
- Same model as judge. The judge was harsh, not lenient, here; the hand labels and the M6 second judge still need to measure it.
- Grader defects hit every arm equally, so they do not bias the comparison, but they cut power: 4 of 10 pilot tasks gave no signal.

## Appendix A. Every rejected evidence-arm return (64)

Classes: A = cited a failed call (genuine catch); B = a side detail was genuinely wrong, the task value was right; C = only meta, over-broad or unverifiable claims, no error shown; D = at least one check or harness bug, no genuine catch. Mechanism labels are per failing check; `meta` = tool inventories, environment or negative claims; `overclaim` = claim adds details the cited calls do not show.

| task | return | action | class | mechanisms |
|---|---|---|---|---|
| canvas-module | 0 | redelegate | D | L1-span-format, L2-drift, meta, overclaim |
| canvas-module | 1 | fallback | D | L1-per-pointer |
| canvas-module | 3 | redelegate | C | overclaim |
| canvas-module | 4 | fallback | D | L3-judge-saw-cut-evidence |
| canvas-module | 5 | redelegate | C | meta |
| canvas-module | 6 | fallback | C | meta |
| canvas-module | 10 | redelegate | B | wrong-detail |
| canvas-module | 13 | redelegate | C | meta |
| canvas-module | 16 | redelegate | D | L2-drift, L2-write-vs-readback, meta |
| canvas-module | 17 | fallback | D | L2-drift, L2-write-vs-readback |
| canvas-quiz | 0 | redelegate | D | L2-drift, meta |
| canvas-quiz | 1 | fallback | D | L3-judge-saw-cut-evidence |
| canvas-quiz | 2 | redelegate | B | L1-quotes-trunc-note, L3-judge-saw-cut-evidence, meta, overclaim, wrong-detail |
| canvas-quiz | 3 | fallback | D | L3-judge-saw-cut-evidence, meta |
| canvas-quiz | 4 | redelegate | D | L2-value-not-verbatim, meta |
| canvas-quiz | 5 | fallback | C | meta |
| canvas-quiz | 6 | redelegate | D | L1-quotes-trunc-note, L3-judge-saw-cut-evidence |
| canvas-quiz | 7 | fallback | D | L3-judge-saw-cut-evidence |
| canvas-quiz | 8 | redelegate | B | L1-span-format, overclaim, wrong-detail |
| canvas-quiz | 10 | redelegate | B | L1-span-format, meta, wrong-detail |
| canvas-quiz | 11 | fallback | C | overclaim |
| canvas-quiz | 13 | redelegate | D | L2-value-in-nonreplayable |
| canvas-quiz | 14 | fallback | D | L2-value-in-nonreplayable |
| fetch-sf-hr | 2 | redelegate | C | overclaim |
| fetch-sf-hr | 5 | redelegate | D | L1-span-format, L2-value-not-verbatim, overclaim |
| fetch-sf-hr | 6 | fallback | D | L3-judge-saw-cut-evidence |
| fetch-sf-hr | 8 | redelegate | D | L2-value-not-verbatim, L3-judge-saw-cut-evidence |
| howtocook | 0 | redelegate | D | L1-span-format, L2-value-not-verbatim, L3-judge-saw-cut-evidence |
| howtocook | 1 | fallback | D | L3-judge-saw-cut-evidence |
| howtocook | 2 | redelegate | D | L2-value-in-nonreplayable |
| howtocook | 3 | fallback | D | L2-value-in-nonreplayable |
| howtocook | 4 | redelegate | B | wrong-detail |
| howtocook | 5 | fallback | B | wrong-detail |
| howtocook | 6 | redelegate | D | L2-value-not-verbatim |
| howtocook | 7 | fallback | C | overclaim |
| sf-hr-comp | 0 | redelegate | B | meta, wrong-detail |
| sf-hr-comp | 1 | fallback | C | meta |
| sf-hr-comp | 2 | redelegate | D | L2-drift |
| sf-hr-comp | 4 | redelegate | D | L2-value-not-verbatim, overclaim |
| sf-hr-comp | 6 | redelegate | C | meta |
| sf-hr-comp | 7 | fallback | B | meta, wrong-detail |
| sf-hr-comp | 9 | redelegate | C | overclaim |
| sf-hr-comp | 11 | redelegate | D | L2-value-in-nonreplayable, unknown(unsampled) |
| sf-hr-comp | 12 | fallback | D | L1-per-pointer, L2-value-in-nonreplayable |
| sf-hr-comp | 13 | redelegate | C | meta |
| sf-hr-comp | 14 | fallback | C | meta |
| sf-sales | 0 | redelegate | D | L1-per-pointer, meta |
| sf-sales | 3 | redelegate | D | L0/L2-callid-typo, L2-write-vs-readback |
| sf-sales | 4 | fallback | D | L1-per-pointer |
| sf-sales | 6 | redelegate | B | L2-value-in-nonreplayable, wrong-detail |
| sf-sales | 8 | redelegate | D | L1-per-pointer |
| sf-sales | 10 | redelegate | C | meta, overclaim |
| sf-sales | 11 | fallback | D | L2-drift |
| sf-sales | 13 | redelegate | C | overclaim |
| terminal | 0 | redelegate | D | L2-value-in-nonreplayable |
| terminal | 2 | redelegate | A | fabricated, unknown(unsampled) |
| terminal | 4 | redelegate | D | L2-value-not-verbatim, meta, overclaim |
| terminal | 5 | fallback | D | L3-judge-saw-cut-evidence, overclaim |
| terminal | 9 | redelegate | A | L1-span-format, L2-value-in-nonreplayable, L3-judge-saw-cut-evidence, fabricated |
| terminal | 12 | redelegate | C | meta |
| wc-shipping | 1 | redelegate | D | L1-numeric-count, L2-value-not-verbatim, L2-write-vs-readback |
| wc-shipping | 2 | fallback | D | L2-write-vs-readback |
| wc-shipping | 4 | redelegate | D | L2-write-vs-readback |
| wc-shipping | 6 | redelegate | C | judge-misread |

## Appendix B. Every run the detector flagged (19 of 35)

Counts are flagged tool calls. `own-dir` = reads or listings of the task's own sealed directory (`docs/task.md`, `task_config.json`, `preprocess/main.py`, `email_config.json`, `initial_workspace/`), which the seal leaves in place. `seek` = searches for `groundtruth`, `evaluation`, grader or expected-output names. `blocked` = the call returned an access-denied or tool error. No flagged call in any run returned task ground truth.

| task | arm | grader | own-dir | seek | blocked | effect on result |
|---|---|---|---|---|---|---|
| canvas-module-completion-notion-gcal | sa | fail | 3 | 0 | 0 | none (already failed or ungraded) |
| canvas-module-completion-notion-gcal | self_verify | ungraded (ceiling) | 14 | 7 | 2 | none (already failed or ungraded) |
| canvas-module-completion-notion-gcal | vanilla_mad | fail | 5 | 0 | 0 | none (already failed or ungraded) |
| canvas-quiz-analysis-ppt | evidence_risk | ungraded (ceiling) | 5 | 0 | 1 | none (already failed or ungraded) |
| canvas-quiz-analysis-ppt | sa | fail | 3 | 0 | 2 | none (already failed or ungraded) |
| canvas-quiz-analysis-ppt | self_verify | ungraded (ceiling) | 5 | 10 | 9 | none (already failed or ungraded) |
| canvas-quiz-analysis-ppt | vanilla_mad | pass | 6 | 0 | 1 | grader pass counted as fail |
| fetch-sf-hr-training-notion-gcal-excel | evidence_risk | fail | 3 | 0 | 0 | none (already failed or ungraded) |
| fetch-sf-hr-training-notion-gcal-excel | sa | fail | 10 | 0 | 0 | none (already failed or ungraded) |
| fetch-sf-hr-training-notion-gcal-excel | self_verify | ungraded (ceiling) | 17 | 7 | 0 | none (already failed or ungraded) |
| fetch-sf-hr-training-notion-gcal-excel | vanilla_mad | fail | 4 | 0 | 0 | none (already failed or ungraded) |
| howtocook-weekly-gsheet-gcal | evidence_risk | ungraded (ceiling) | 2 | 0 | 0 | none (already failed or ungraded) |
| howtocook-weekly-gsheet-gcal | self_verify | pass | 3 | 0 | 0 | grader pass counted as fail |
| sf-hr-compensation-pdf-notion | evidence_risk | ungraded (ceiling) | 4 | 0 | 0 | none (already failed or ungraded) |
| sf-hr-compensation-pdf-notion | sa | fail | 3 | 1 | 0 | none (already failed or ungraded) |
| sf-sales-region-analysis-gsheet-email | evidence_risk | fail | 4 | 0 | 0 | none (already failed or ungraded) |
| sf-sales-region-analysis-gsheet-email | vanilla_mad | fail | 3 | 0 | 0 | none (already failed or ungraded) |
| terminal-arxiv-latex-excel-word-notion | self_verify | ungraded (ceiling) | 20 | 11 | 0 | none (already failed or ungraded) |
| yf-portfolio-risk-excel | sa | pass | 3 | 1 | 0 | grader pass counted as fail |

## Appendix C. Scripts (all in `/tmp/pd/`, read-only over `runs/pilot_deepseek/`)

| script | what it computes |
|---|---|
| `survey.py`, `sample.py` | event types per trace; sample events |
| `fails.py`, `dump_fail.py` (output `fails_all.txt`) | every policy decision and every failing check, with claim, span and ledger match |
| `subclass.py`, `l1span.py` | L1/L2/L3 failure mechanisms; each failing span against the logged result; judge evidence cuts |
| `classify.py` (output `reject_rows.json`), `retry.py`, `appendix.py` | per-failure labels (automatic for L0-L2, manual for L3 from reading all 85 judge reasons), return classes, counterfactual layers, redelegation outcomes, Appendix A |
| `contam.py`, `contam2.py`, `contam_app.py` (output `contam.txt`) | every ground-truth suspect with arguments and result; categories; Appendix B |
| `runs.py`, `evals.py`, `sa.py`, `dups.py` | run status, grader failure lines, SA turn use, duplicate writes |
| `cost.py`, `ctx.py`, `cache.py`, `trim.py`, `reason.py`, `model.py`, `project.py` | cost by arm, agent and provider; context sizes; cache continuity; schema share; trimming; judge reasoning; per-run costs; headline scenarios |

Manual judgement enters in one place: the 85 L3 labels in `classify.py` (dictionary `M`), which I assigned after reading each judge reason against the claim and its evidence. Everything else is computed.
