# Task rules: tool limit, email swaps, delegation labels

Rule version `evcarry-task-rules-v1`. Written 2026-10-24, before any pilot or headline run; no task outcome from any model existed when these rules were fixed. The rules read only the gym's task files (commit `ed735ba`) and the MCP tool inventory of the built image (`tests/fixtures/gym/tool_inventory.json`). Code: `src/evcarry/env/task_rules.py`. Apply or check:

```bash
uv run python scripts/apply_task_rules.py --check   # exits 0 when the committed files are what the rules give
```

The script regenerates the original stratified sample from seed `20261024` (`scripts/freeze_task_lists.py`), applies the rules, and writes `tasks/subset70.txt`, `tasks/pilot10.txt`, `tasks/SHA256SUMS`, `tasks/swaps.csv`, `tasks/email_audit.csv`, `tasks/delegation_labels.csv`, and the rule lines of `tasks/FROZEN.txt` (with the pre-swap hashes `0571b5ee…` for subset70 and `24721d50…` for pilot10). `tasks/pool160.txt` is unchanged; it is where replacements come from.

## 1. More than 128 tools: keep every task

The single agent (SA arm) gets every MCP tool of its task plus `claim_done`, `python_execute` and `view_overlong_output`. 69 of pool160, 26 of the new subset70 and 3 of pilot10 exceed 128 tools; OpenAI and DeepSeek's own API document a 128-function limit.

One paid call tested the headline route (`openrouter_ds_1017`: DeepSeek-V4.1-Flash-1017 on OpenRouter, pinned to `deepinfra/fp8`, no fallbacks) with 150 dummy tools (`scripts/probe_tool_limit.py`, result in `docs/evidence/tool_limit_probe_2026-10-24.json`, cost $0.00104). DeepInfra answered HTTP 200, and the model called the one tool whose description answered the prompt. That tool's name was a hash and it sat at sorted position 146, so the model saw the tool definitions past the 128th.

Decision, per the rule set before the call: DeepInfra accepts more than 128 tools, so no task is trimmed or swapped for its tool count.

Limit of this evidence: the probe sent 150 tools. Every pilot10 task is within that (largest: `canvas-quiz-analysis-ppt`, 148). Twelve subset70 tasks give the SA agent more than 150 tools (largest: `terminal-canvas-wc-excel-ppt-email`, 230): academic-conference-planning-coordination (175), canvas-assignment-effectiveness-ppt-notion-email (178), fetch-wc-supplier-restock-excel-word-gcal (157), pw-sf-benchmark-excel-word-email (168), sf-wc-order-reconciliation-excel-word-email (192), terminal-canvas-wc-excel-ppt-email (230), terminal-wc-pw-pricing-excel-word-gcal (189), terminal-wc-yf-ppt-notion-email (165), terminal-yf-canvas-excel-gcal-email (153), wc-shipping-tax-excel-word (176), wc-shipping-zone-ppt (153), wc-yf-commodity-pricing-impact (167). Whether DeepInfra takes 230 tools is unverified. A second probe with 231 tools (about $0.002) would settle it before the headline.

## 2. Email tasks that cannot pass: swapped

The gym's `emails` server sends from the first account in the task's `email_config.json`, or from `user@example.com` when the task has none; `send_email` takes no sender argument. A task that needs `emails` has an **email defect** when:

- `email_config_without_password`: its `email_config.json` lacks `email` or `password`. The server raises "Email and password are required" and the run cannot start.
- `unpassable_sender_check`: its evaluation requires the sent message's from-address to be a fixed address other than the one the server sends from. An agent passes only by inserting the row into Postgres itself.

Detector (`required_senders`, Python `ast` over `evaluation/*.py`): a comparison `in`, `not in` or `==` between a fixed address (a literal, or a name bound to one) and either a from-named value (`from_addr`, `from_str`, `frm`, `sender`, …), or anything inside a call labelled "…from…"/"…sender…" (`check("Email from x@y", "x@y" in (e[1] or ""))`), or a comprehension over a from-named list. SQL strings count for `from_addr = 'x'` and `from_addr ILIKE '%x%'`; `NOT ILIKE` and `!=` (which exclude seeded mail) do not.

Hand check: I read the evaluation of every pool160 task the detector flags and of every other pool160 task that compares anything with a from-address, and confirmed each flagged check fails the whole task (the evaluation exits non-zero when any check fails). The detector matches the hand review exactly on all 81 pool160 email tasks (`tests/unit/test_task_rules.py`). Outside pool160 it is a lower bound: it misses partial-address checks (`"nutrition" in from_addr`) and some comprehension forms; no list uses those tasks.

| pool160 task | sends from | evaluation requires | evaluation line |
| --- | --- | --- | --- |
| canvas-assignment-calendar | user@example.com | coordinator@openuniversity.ac.uk | main.py:241 |
| canvas-quiz-analysis-email | user@example.com | coordinator@university.edu | main.py:185 |
| playwright-canvas-curriculum-word-notion | assistant@company.example.com | accreditation@university.edu | main.py:298 |
| scholarly-arxiv-cross-reference | user@example.com | librarian@university.edu | main.py:216 |
| sf-hr-mentorship-excel-gcal-email | user@example.com | hr@company.example.com | main.py:258 |
| sf-sla-breach-audit | user@example.com | sla-audit@company.com | main.py:189 |
| sf-ticket-resolution-notion | assistant@company.example.com | escalations@support.example.com | main.py:281 |
| terminal-canvas-gform-gcal-excel-email | coordinator@ddd-program.example.com | coordinator@assessment.example.com | main.py:478 |
| yf-dividend-tracker-gsheet | assistant@company.example.com | portfolio-alerts@finance.example.com | main.py:182 |

Config without password in pool160: arxiv-reading-plan-excel-gcal-email, yf-portfolio-analysis-excel-word-email, yf-sector-comparison-gform-excel-email. Across all 503 gym tasks the detector finds 29 sender defects and 10 password-less configs (`tasks/email_audit.csv`).

The benchmark is not edited (no `email_config.json` is added or patched); defective tasks are replaced instead.

### Swap rule

For each subset70 task with an email defect, in sorted name order:

1. Candidates: pool160 tasks not already in subset70 (or chosen earlier), with no email defect and at most 128 SA tools.
2. Take the first tier that has a candidate: same stratum (name-prefix domain and number of MCP servers, the strata the sample was drawn from); same domain; needs the old task's primary servers (the servers other than `filesystem` that every gym task with that name prefix needs, e.g. `canvas` → canvas); same number of servers; any.
3. Within the tier, take the candidate with the smallest `sha256("evcarry-swap-v1|" + task)`.

A pilot10 task swapped out of subset70 gets the same replacement, so pilot10 stays inside subset70.

### Swap list

| list | removed | reason | added | tier |
| --- | --- | --- | --- | --- |
| subset70 | arxiv-reading-plan-excel-gcal-email | email_config_without_password | arxiv-method-benchmark-tracker | same stratum (arxiv, 5 servers) |
| subset70 | canvas-quiz-analysis-email | unpassable_sender_check | sf-hr-performance-ppt | same server count (4) |
| subset70 | sf-hr-mentorship-excel-gcal-email | unpassable_sender_check | sf-hr-department-budget | same stratum (sf, 5) |
| subset70, pilot10 | sf-sla-breach-audit | unpassable_sender_check | sf-hr-compensation-pdf-notion | same stratum (sf, 4) |
| subset70 | sf-ticket-resolution-notion | unpassable_sender_check | sf-sales-regional-summary | same stratum (sf, 4) |
| subset70 | terminal-canvas-gform-gcal-excel-email | unpassable_sender_check | terminal-howtocook-scholarly-excel-word-gform | same stratum (terminal, 7) |
| subset70 | yf-dividend-tracker-gsheet | unpassable_sender_check | yf-financial-statements-overview | same stratum (yf, 4) |

`canvas-quiz-analysis-email` has no canvas replacement: every pool160 task outside subset70 that needs the canvas server gives the SA agent more than 128 tools, except `canvas-assignment-calendar`, which has its own sender defect. With the bound at 150 tools (the verified one) instead of 128, the rule would pick a canvas task; the 128 bound was set by the owner's instruction before the probe and is kept.

New hashes (`tasks/SHA256SUMS`): subset70 `6c1960c1…`, pilot10 `828d4ea9…`; pool160 unchanged (`977eda0b…`).

## 3. Needs delegation

MS-Eval's delegation labels are not public. The label used here:

> A task **needs delegation** when it needs at least 6 MCP servers, or when the single agent would get more than 128 tools.

- 6 servers is one above the pool160 median (5). A task spanning that many systems has several separable sub-tasks (read from one system, write to others), which is the work a main agent hands to sub-agents.
- 128 tools is the documented per-request function limit of the OpenAI and DeepSeek chat APIs; past it a single agent cannot hold the full tool set on those APIs.

Both thresholds were chosen from task metadata only. The two parts are stored separately (`servers_ge_6`, `sa_tools_gt_128`) so an analysis can report each.

| list | tasks | needs delegation | ≥ 6 servers | > 128 SA tools |
| --- | --- | --- | --- | --- |
| all gym tasks | 503 | 316 | 248 | 203 |
| pool160 | 160 | 103 | 78 | 69 |
| subset70 (after swaps) | 70 | 42 | 34 | 26 |
| pilot10 (after swaps) | 10 | 5 | 2 | 3 |

File: `tasks/delegation_labels.csv` (every gym task, with list membership).

## Left as is

- pool160 still contains the 12 defective email tasks above; it is only the source of replacements. Any M6 grid drawn from pool160 must exclude them by the same rule.
- These rules do not touch tasks whose evaluation checks only part of an address or reads seeded mail; none of them is in the lists.
