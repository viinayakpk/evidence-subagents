# Dev check review (runs/dev_verify_v1, 2026-10-25)

Offline review of the dev run (2 dev tasks x 4 arms, seed 1, finished 02:27 CET, $0.2621 billed, $0.2458 at reference rates). No paid call was made for this review.

## Results

| task | arm | grader | passed | billed $ | turns (main/researcher/operator) |
| --- | --- | --- | --- | --- | --- |
| train-sf-sales-trip | sa | 40/45 | no | 0.0133 | 28 |
| train-sf-sales-trip | vanilla_mad | 45/45 | yes | 0.0187 | 5/18/20 |
| train-sf-sales-trip | self_verify | 45/45 | yes | 0.0176 | 4/21/26 |
| train-sf-sales-trip | evidence_risk | 45/45 | no (contaminated, attempted) | 0.1332 | 14/25/72 |
| sf-support-metrics | sa | fail | no | 0.0346 | 53 |
| sf-support-metrics | vanilla_mad | fail | no | 0.0097 | 6/13/17 |
| sf-support-metrics | self_verify | fail | no | 0.0101 | 6/16/14 |
| sf-support-metrics | evidence_risk | fail | no | 0.0250 | 7/29/22 |

## Gates

- (a) env or provider errors: pass. 0 retries, every provider check ok, no env_error. DeepInfra served 428 of 513 agent and judge calls, InferenceNet 85.
- (b) offline, hidden, restored: pass. The evidence run's operator listed /workspace and saw only .venv, agent_workspace, evcarry_run, pyproject.toml, uv.lock; its requests to pypi.org and api.github.com failed name resolution; every grader ran after restore.
- (c) handoff events: pass, in all six delegating runs.
- (d) evidence arm: partly fails. sf-support-metrics had 2 FALLBACKs and 2 re-delegations caused only by cited ids missing the `chatcmpl-tool-` prefix (fixed, 4dcef0a; offline, all 47 pointers of those 4 returns now resolve and match). sf-sales-trip had 4 re-delegations: c13/c14 (value `SALES_DW.PUBLIC.CUSTOMERS` built from three cited spans; check strictness, not fixed), return 9 c9 (backslash escaping; fixed, b256aa7), the rest genuine (wrong call cited, unsupported counts). No duplicate writes (one email per run). skipped_l3 used. partial_accept never triggered: every re-delegation was then fully accepted.
- (e) SA turn cap: pass. 28 and 53 turns of 100. SA's sales-trip failure is genuine: Customer_Priority lacks Isabella and Mason.
- (f) context: pass. Largest call 123,001 input tokens; block trim fired 3 times.
- (g) contamination: hand-checked. One run (sales-trip evidence_risk) is attempted-tier, correctly. Nothing obtained. The detector caught 4 filesystem searches for grader names but missed the same run's SQL searches for tables named eval/grade/answer/expected, a package-name scan, and HTTP lookups of "mcpbench" on PyPI and GitHub (they failed offline). Widening the detector is an amendment and is not done here.
- (h) cost and harness failures: cost pass (every arm well under the per-run projection: sa 0.024 vs 0.047, vanilla 0.014 vs 0.110, self_verify 0.014 vs 0.169, evidence 0.079 vs 0.164). Harness failures: fail. sf-support-metrics fails in every arm because the grading Python has no PDF library (below).

## 1. Why the sales-trip evidence run cost $0.133 and took 54 minutes

Not a loop. The main agent made 11 delegations; 4 re-delegations each cleared in one round. 6 of the 11 delegations were not task work: the main agent, unsure whether totals should include cancelled orders, sent the operator to explore the filesystem, the database, installed packages, environment variables, and the internet for the benchmark's tasks. Judge calls (80) equal the sampled claims (82) minus 2 skipped; no claim was judged twice except the 4 re-checked ones. The judge cost $0.016. Most cost is 7.86M input tokens and 178k output tokens of sub-agents writing 20-35k-character claim returns. The 52.8 minutes are serial model latency. Reference spend was $0.129, under the $0.17 ceiling. Harness share: the return format makes returns long; the hunting is the model's.

## 2. status "passed" with passed false

`final_verdict` set `passed` false for a contaminated run but left the grader's `status`. Now a graded contaminated run has `status: contaminated` and `eval_status` keeps the grader's word (ddd3683, tests/unit/test_run_status.py, 6 tests).

## 3. sf-support-metrics-pdf-email

All four PDFs are correct (every count and average matches the ground-truth PDF). The grader tries fitz, pypdf, PyPDF2 and pdfplumber, none of which is installed in /opt/venv (the grading Python, also the upstream default), then falls back to raw PDF bytes, where the compressed text is invisible. So the grader fails every correct answer: a grader defect like the pilot's. Only this task imports a PDF library: the 62 headline and dev graders otherwise import only modules present in /opt/venv (checked in the image). The headline is not affected. Proposal (not applied): add to evcarry-dev-v1 the filter "grader imports only modules the grading Python has"; the next task by the rule is gcal-snowflake-deadline-tracker (imports argparse, json, openpyxl, os, psycopg2, sys, decimal).

## 4. SA on sales-trip

Genuine failure at 28 turns: its customer list omits Isabella and Mason.

## 5. reserve_billed_per_reference 1.6 vs worst ratio 2.72

No overrun risk: the cap counts billed dollars and refuses every model call once reached, so overrun is bounded by calls already in flight (cents). The risk is waste: if the cap is reached mid-run, those runs end without run_end. The dev run's billed/reference ratio was 1.07 overall (1.41 worst run), well under 1.6.

## 6. Seed 1 projection (60 tasks x 4 arms)

At the dev run's per-run means (sa 0.024, vanilla 0.014, self_verify 0.014, evidence 0.079): about $7.9. The registered projection from the pilot's mix: $28.2. The dev tasks were cheap and DeepInfra served most calls, so expect $8-28; a $30 cap holds at the registered projection.
