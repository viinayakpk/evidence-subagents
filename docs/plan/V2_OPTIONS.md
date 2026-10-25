# v2 design options, costed (draft, 2026-10-25)

Status: draft for Vinayak's decision. Nothing here is implemented or registered yet. Choosing a design
needs a dated amendment (A19) before any headline run. Budget left: about $45 billed.

## 1. Question, arms, endpoint

**Question.** On stateful MCP tasks, does making sub-agents return evidence-carrying claims, with
spot-check audits at the boundary, raise task success over plain delegation? Is any gain more than the
extra compute alone buys? And how much of the delegated failure mass can such checks reach at all?

**Primary comparison.** evidence_risk vs vanilla_mad, paired task success on matched (task, seed)
blocks, with a bootstrap that resamples tasks (A15).

**Secondary results.**
- evidence_risk vs a budget-matched control.
- Success per reference dollar.
- Rejections split into attributed and false alarm (A16).
- The sensitivity with grader exclusions dropped (A17).
- The catchable share (A18).
- L0–L2 fault-injection catch rates. These are already measured offline at $0.

**Arms considered.**

| arm | role | in recommended design |
|---|---|---|
| sa | reference for MS-Eval-style flips and catchability labels | yes, 1 seed |
| vanilla_mad | baseline (primary comparator) | yes |
| evidence_risk | method | yes |
| budget_matched | vanilla_mad plus one blind redelegation of each sub-agent task ("check your work and return again"), no evidence or audit; same spend as evidence_risk within ±20% | yes (new; needs implementing and a $0 mock test) |
| self_verify | free-text self-check baseline | dropped (descriptive pilot and dev data only) |

## 2. Unit costs ($ per run)

| arm | measured, dev v2 (2 tasks) | registered projection | pilot billed |
|---|---|---|---|
| sa | 0.033 | 0.047 | 0.024 |
| vanilla_mad | 0.059 | 0.110 | 0.108 |
| self_verify | 0.073 | 0.169 | 0.169 |
| evidence_risk (same-model L3) | 0.083 | 0.164 | 0.187 |
| evidence_risk without L3 | 0.076 | 0.149 | — |
| evidence_risk, cross-family judge | 0.077 | 0.152 | — |
| budget_matched (estimate: vanilla × 1.4) | 0.083 | 0.154 | — |

Notes:
- The L3 judge was 9% of evidence-arm spend in the pilot ($0.00037 per judged claim).
- The cheap different-family judge is `qwen/qwen3.7-flash` (Alibaba): $0.03 input and $0.13 output per million tokens, 1M context, structured output and a reasoning setting. That is about 1/5 of DeepSeek-V4.1-Flash-1017 at $0.15 / $0.60 (OpenRouter public model catalog, read 2026-10-25). It works out to about $0.00008 per claim if token use is similar.
- Measured costs come from only 2 dev tasks, so treat them as a lower bound. The projection is the upper bound.

## 3. Options (60 headline tasks unless stated)

Power: the chance that the task-clustered paired test (two-sided α = 0.05) detects the true gain of
evidence_risk over vanilla_mad. I simulated it with a task-level logistic model calibrated to 30%
discordance and a 33% baseline, 1,500 runs per cell (`/tmp/power2.py` on the box). At T = 0, seeds were
modelled as copies with probability 0.8; at T > 0, with probability 0.3.

| # | design | runs | $ measured | $ projected | fits $45? | power +10 / +15 / +20 pp |
|---|---|---|---|---|---|---|
| full | 5 arms (with budget_matched) × 3 seeds, T > 0 | 900 | 59.6 | 116 | no | 0.49 / 0.77 / 0.94 |
| (i) | 4 arms × 2 seeds, T = 0 | 480 | 29.8 | 58.8 | measured only | 0.34 / 0.59 / 0.82 |
| (ii) | (i) + budget_matched | 600 | 39.7 | 77.3 | measured only | 0.34 / 0.59 / 0.82 |
| (iii) | (i), evidence without L3 | 480 | 28.9 | 57.0 | measured only | same as (i) |
| (iv) | 4 arms × 3 seeds, T > 0 | 720 | 44.6 | 88.2 | barely, measured only | 0.49 / 0.77 / 0.94 |
| (iv′) | 4 arms × 2 seeds, T > 0 | 480 | 29.8 | 58.8 | measured only | 0.42 / 0.69 / 0.90 |
| (v) | (i) with the qwen judge | 480 | 29.0 | 57.4 | measured only | same as (i) |
| (vi-a) | drop self_verify: sa, vanilla, evidence, budget_matched × 2 seeds, T > 0 | 480 | 31.0 | 57.0 | measured only | 0.42 / 0.69 / 0.90 |
| (vi-b) | fewer tasks: 4 arms × 40 tasks × 3 seeds, T > 0 | 480 | 29.8 | 58.8 | measured only | 0.36 / 0.59 / 0.84 |
| **R** | **sa × 1 seed; vanilla, evidence, budget_matched × 2 seeds, T > 0** | 420 | **29.0** | **54.2** | yes at measured; seed 1 alone always fits ($15.5 / $28.5) | **0.42 / 0.69 / 0.90** (seed 1 only: 0.29 / 0.53 / 0.77) |

What the table shows:
- Dropping self_verify does not raise the power of the primary test, because power depends only on the
  pair being compared. It frees $4.4 per seed at measured cost ($10 at projected cost) for the budget-matched control.
- Fewer tasks with more seeds loses power: 40 × 3 gives 0.59 at +15 pp, against 0.69 for 60 × 2.
  Tasks are the independent unit, so keep all 60.
- At T = 0 a second seed adds little (0.53 → 0.59 at +15 pp). At T > 0 it adds more (0.69).
  T > 0 needs an amendment and makes runs less reproducible.
- Dropping L3 saves less than $1 and loses the only check on derived and multi-word values. Offline,
  93–100% of the injected faults that L0–L2 missed had a value that only L3 can judge.
- Flips (H1) stay underpowered in every design (≤ 0.56). They are descriptive under A15.

## 4. Recommendation: design R

- **Arms.** sa (seed 1), vanilla_mad, evidence_risk (registered same-model judge) and budget_matched,
  all on 60 headline tasks.
- **Seeds and temperature.** 2 seeds at T = 0.6 for the three delegating arms (an amendment before the
  headline).
- **Run order.** Run seed 1 for every arm first, in a hash-shuffled task order. Run seed 2 only while
  spend plus the reserve fits the cap, and stop only at a complete block of all arms. If the projection
  is right, the study ends at seed 1 with power 0.53 at +15 pp. Report whichever applies.
- **Judge.** Keep the registered same-model L3 judge (A12). Add `qwen/qwen3.7-flash` as a cross-family
  second judge on the logged L3 requests (about 1,500 claims, roughly $0.15) and 100 hand labels, to
  measure self-preference.
- **Before spending.** Implement budget_matched and test it on the mock at $0. Run one dev task with it
  (about $0.10).

**What is lost compared with the full design.**
1. There is no free-text self-verification arm. The claim "better than self-verification" can only rest
   on the pilot and dev self_verify runs, which are descriptive. Section 1's falsification criterion
   changes accordingly.
2. Power at +15 pp is 0.69 rather than 0.77, or 0.53 if only seed 1 fits.
3. SA has one seed, so flip and catchability labels are per task, not per seed.
4. H3 (risk vs random sampling) and the typed-contract and evidence-only arms stay with the $0 local
   M6 grid.
5. If the dev costs turn out optimistic, seed 2 is lost.
