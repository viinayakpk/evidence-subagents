# Failure codebook: is a failure catchable at the sub-agent boundary? (2026-10-25)

Purpose: estimate the share of baseline (`vanilla_mad`) failures that an evidence check at the sub-agent
boundary could catch at all. This bounds what the method can add on these tasks. Labels are by hand from
the trace, the grader log and the SA run on the same task. They live in
`validation/vanilla_failure_labels.json`. `scripts/label_vanilla_failures.py` lists every failed run, so
an unlabelled failure shows as `UNLABELLED`. This codebook replaces the provisional heuristic taxonomy in
`src/evcarry/eval/label_failures.py` for the paper.

## Labels

| label | definition | test | MAST (2503.13657) | MS-Eval (2610.22869) |
|---|---|---|---|---|
| `catchable_propagation` | A sub-agent return held a false or unsupported statement, the main agent used it, and that statement caused at least one failed grader check. The statement can be a wrong value, a made-up entity, a pointer to the wrong call, or a "done" for a write that did not happen. | The wrong statement is in a `subagent_output` (or a researcher/operator reply), and fixing only that statement would fix the failed check. | FC2: FM-2.4 information withholding, FM-2.6 reasoning-action mismatch; FC3: FM-3.2/3.3 at the main agent | delegated failure propagation (DFP) |
| `spec_misreading` | Every sub-agent statement was true for what it said, but some agent read the task differently from the grader: a filter, a definition, a format limit, or scope. | The SA run makes the same error, or the data needed for the right answer was reported to the main agent. | FC1: FM-1.1 disobey task specification (FM-2.2 if a clarifying question was possible) | not DFP (a "Spec"-style failure) |
| `other` | Grader defect, environment or tool error, contamination, a turn, cost or termination limit. Subtypes: `grader_defect`, `env_tool`, `contamination`, `no_claim_done`, `budget`. | A grader defect is listed in `docs/TASK_RULES.md` section 5 or fails a correct output. | FM-1.5 for `no_claim_done`; otherwise outside MAST | not DFP |
| `not_a_failure` | Graded fail only by a rule the registration does not contain (e.g. the bare `/workspace/tasks` contamination marker, A9). | | | |

Rules: a run gets one primary label. When the causes are mixed, `catchable_propagation` wins if any failed
check traces to a false sub-agent statement. `ms_eval_dfp` is `"unresolved"` when the trace does not
decide this. Only `catchable_propagation` counts as "catchable" in the paper's share.

## Result on the pilot and dev runs (8 vanilla_mad failures)

| run | task | label | why |
|---|---|---|---|
| pilot | canvas-module | other / grader_defect | Notion property shape; SA fails identically |
| pilot | canvas-quiz | not_a_failure | grader pass; only the bare contamination marker (A9 false positive) |
| pilot | fetch-sf-hr | other / grader_defect | Notion parser; grader hard-codes R&D over Support (3.1968 vs 3.1956) |
| pilot | sf-sales-region | spec_misreading | cancelled orders excluded; the researcher reported both totals |
| pilot | terminal-arxiv | spec_misreading (DFP unresolved) | sentences over 200 characters, a noise paper; one missing citation pair might come from a sub-agent omission |
| pilot | wc-shipping | other / no_claim_done | ended without claim_done; header grader defect hits all arms |
| dev v1 | sf-support-metrics | other / grader_defect | A13 PDF imports |
| dev v2 | sales-trip | spec_misreading | priority ranks 3-5 from the cancelled-orders reading; SA fails identically |

Counts: 0 confirmed `catchable_propagation` (at most 1, terminal-arxiv, if the unresolved check is
counted), 3 `spec_misreading`, 3 `other` (grader defect), 1 `other` (no claim_done), 1 `not_a_failure`.
Of 7 real failures, 0/7 are confirmed catchable (upper bound 1/7). This matches `PILOT_DIAGNOSIS.md`
section 5: on these tasks the sub-agents rarely returned wrong data. The fault-injection study
(`scripts/offline_injection.py`) measures catch rates instead. The headline run labels every baseline
failure with this codebook, and the paper reports the catchable share with a Wilson interval.
