Source Page: https://cursor.com/canvas/store-ecb7e97a-5043-457c-9a0d-1e8b3fc2887e/0859781d-ee9c-407f-a088-dbf62586c4b4 — exported 2026-10-24

# Agent research paper ideas – Oct 2026

> Six open research gaps in agent orchestration, context, memory and multi-agent systems, ranked for a hobby researcher with one 96 GB GPU and frontier API keys. Each comes from papers published May to Oct 2026 and has a concrete experiment you can run.

## Contents

- [Recommendation](#rec)
- [Ranked comparison](#table)
- [Novelty versus feasibility](#matrix)
- [1. Epistemic ledger: compaction and handoffs that keep tentative claims tentative](#idea1)
- [2. Evidence-carrying sub-agent returns with spot-check audits](#idea2)
- [3. Calibrated continuation critic against premature termination](#idea3)
- [4. Sim-to-real orchestration: train a planner against a cheap simulator](#idea4)
- [5. A small trained evidence-gatherer for experiential agent memory](#idea5)
- [6. Profile the model first, then pick the harness](#idea6)
- [Areas checked and judged crowded](#crowded)
- [Later option: VLA](#vla)
- [How this was put together](#method)

<a id="rec"></a>

## Recommendation

> **Start with idea 1, run idea 2 next.**
>
> The epistemic ledger can produce a workshop-ready result in about three weeks with a prompt-only version on BrowseComp-Plus, almost all on your own GPU, then grow into a trained 8B compactor. Evidence-carrying sub-agent returns builds on the public Toolathlon-Gym tasks behind MS-Eval and a cheap API model, and attacks the main documented reason sub-agents make results worse.

Both ideas share one theme: state gets corrupted when it moves (across a compaction boundary or between agents). That gives you a coherent line of two or three papers rather than unrelated one-offs.

<a id="table"></a>

## Ranked comparison

*Scores are my judgment out of 10; costs are rough estimates for a minimum publishable experiment.*

| Idea | Novelty | Feasibility | Scoop risk | API cost | GPU time | Score |
| --- | --- | --- | --- | --- | --- | --- |
| 1. Epistemic ledger for compaction and handoffs | High | High | Medium | $100–400 | Days to 2 weeks | 8.5 |
| 2. Evidence-carrying sub-agent returns | High | High | Medium-low | $150–500 | Light | 8.0 |
| 3. Continuation critic vs early stopping | Medium-high | High | Medium | $50–200 | Days | 7.0 |
| 4. Sim-to-real planner training | High | Medium | Medium | $200–500 | 1–2 weeks | 7.0 |
| 5. Small trained memory searcher | Medium | Medium | Medium-high | $200–600 | Days | 6.5 |
| 6. Behavior-probed harness selection | Medium | Medium | Medium-high | $300–1,000 | Light | 6.0 |

<a id="matrix"></a>

## Novelty versus feasibility

2x2 matrix. X axis: Harder to run → Easier to run. Y axis: Less novel → More novel. Quadrants: Ambitious, Start here, Skip, Safe filler.

| Point | x | y |
| --- | --- | --- |
| 1 Ledger | 0.8 | 0.82 |
| 2 Evidence returns | 0.75 | 0.78 |
| 3 Continuation critic | 0.82 | 0.6 |
| 4 Sim-to-real planner | 0.35 | 0.85 |
| 5 Memory searcher | 0.5 | 0.45 |
| 6 Harness selection | 0.4 | 0.4 |

<a id="idea1"></a>

## 1. Epistemic ledger: compaction and handoffs that keep tentative claims tentative

- **Gap:** When an agent's history is compacted, or handed to another agent, the summary quietly changes what is known. CompAct-Bench (Oct 2026) cuts tasks the executor always solves, compacts the history to 10% of its length, and resumes: average success falls to 56.7% across 12 frontier compactors (best 68.5%). Its two failure paths are omissions (repeated work) and distortion, where tentative evidence becomes a conclusion and the agent stops early. The handoff study 'When Must Becomes Maybe' (Aug 2026) finds the same: multihop compression deactivates 97.2% of correctly flagged blockers, while a schema-preserving handoff deactivates 27.8% with zero forbidden actions. Recent learned-compaction work (SelfCompact, CompactionRL, AutoCompact) is about when to compact and trains the whole agent. SLEUTH keeps a typed fact/hypothesis/question memory by prompting, but for multi-hop QA, not compaction or handoffs. No one I found tests a typed, disk-backed working state as the compaction target with a resume-and-finish metric, or trains a small portable compactor for it.
- **Why now:** Every major harness (Claude Code, Codex, Kimi Code) compacts and delegates by default, and long-horizon runs now cross the context boundary routinely. CompAct-Bench also shows search state lives only in the context while coding state lives on disk, which suggests a cheap fix: give the agent a second tape.
- **Hypothesis:** H1: Compacting into a typed ledger (verified facts with evidence pointers, tentative hypotheses, binding constraints, refuted items, done/next), stored as a file the agent can grep, closes at least half the gap between free-text compaction and no compaction, mostly by cutting distortion-driven early stops on search tasks. H2: An 8B compactor trained with SFT on frontier-written ledgers and then RL on resume success matches frontier free-text compactors locally and transfers to executors it was not trained with.
- **Compute and cost:** GPU: local executor (an open ~30B MoE coder model fits in bf16 on 96 GB) and the Qwen3-32B judge BrowseComp-Plus uses. RL of an 8B LoRA compactor needs a resumed rollout per reward, so plan on 1 to 2 weeks of GPU time. API: about 600 frontier compaction calls for baselines, estimated $100 to $400. The prompt-only H1 needs only a few days of GPU time and under $100.
- **Scoop risk:** Medium. Compaction is very active (AutoCompact Oct 2026, CompactionRL Jul, SelfCompact Jun), and SLEUTH and 'Compaction as Epistemic Failure' are adjacent. The specific angle (epistemic status preservation, measured by resume success, across compaction and handoffs) was not found. Ship H1 fast as a workshop paper.
- **Fit:** Best fit. Runs mostly on your GPU, uses public benchmarks, has a clean metric, and a quick first result.

1. **Experiment step 1:** Rebuild the CompAct-Bench protocol (its CompanyBench split is internal; I could not confirm a code release) on BrowseComp-Plus (fixed 100K-document corpus, local index) and SWE-bench Verified: keep tasks a local executor solves, cut at 25/50/75%, compact to 10%, resume 3 times.
2. **Experiment step 2:** Arms: no compaction; observation masking; free-text summary (frontier and local); SLEUTH-style prompt; typed ledger in context; typed ledger plus greppable file; trained 8B ledger compactor.
3. **Experiment step 3:** Metrics: post-compaction success (Avg@3), repeated-work rate, premature-termination rate, status-flip rate (claims tentative in the history but 'confirmed' in the summary, LLM judge plus manual spot checks), tokens.
4. **Experiment step 4:** Second setting: reuse the 'Must becomes Maybe' blocker tasks inside sub-agent returns to show the same ledger fixes handoffs.

**Links:**

- [CompAct-Bench: working-context compaction in long-horizon agent tasks](https://compaction-eval.github.io/CompAct-Bench/) — compaction-eval.github.io, Oct 2026
- [When Must Becomes Maybe: erosion of workflow constraints in agent handoffs](https://arxiv.org/abs/2608.24569) — arxiv.org/abs/2608.24569, Aug 2026
- [AutoCompact: learning when to compact context in coding agents](https://arxiv.org/abs/2610.02163) — arxiv.org/abs/2610.02163, Oct 2026
- [SLEUTH: epistemic working memory for multi-hop reasoning](https://arxiv.org/html/2607.12267v1) — arxiv.org/abs/2607.12267, Jul 2026
- [Compaction as Epistemic Failure](https://arxiv.org/html/2607.13071) — arxiv.org/abs/2607.13071, Jul 2026
- [BrowseComp-Plus (fixed-corpus deep research benchmark)](https://github.com/texttron/BrowseComp-Plus) — github.com/texttron, ACL 2026

<a id="idea2"></a>

## 2. Evidence-carrying sub-agent returns with spot-check audits

- **Gap:** Sub-agents return prose summaries and the main agent trusts them. In MS-Eval (Oct 2026), Delegated Failure Propagation drives most success-to-failure flips versus a single agent: 14 of 22 for the Claude model and 33 of 45 for the GPT model it tested. In 'Same Context, Different Agents' (Oct 2026) the orchestration verifier approves 87% of the wrong answers from the 'investigator' group. The dynamic-concurrency study of Codex, Claude Code and Kimi Code (Oct 2026, 2,124 runs, over $20K) finds concurrency cost Claude Code about 24 points on SWE-bench Verified, with shared state and merge errors the largest failure class (33.2%). A trace-based assurance framework (Mar 2026) defines unsupported-claim propagation metrics but does not test a fix on agent benchmarks. Open question: what if every sub-agent claim had to cite a replayable pointer (tool-call id, file and line, query and row), and the main agent re-ran a few of them before relying on the result?
- **Why now:** Sub-agents ship natively in all major coding harnesses, and their returns are still free-form text. A protocol-level fix borrowed from audit sampling and proof-carrying code would be new and directly usable.
- **Hypothesis:** Evidence-carrying returns plus an audit of k claims per return cut delegated failure propagation by at least half and recover most of MS-Eval's success-to-failure losses, at under 20% extra tokens, and beat an LLM verifier that reads only the summary.
- **Compute and cost:** With DeepSeek-V4-Flash ($0.45 input / $1.34 output per million tokens) the roughly 3,500 runs are estimated at $150 to $500. The local open-model arm runs on your GPU.
- **Scoop risk:** Medium-low. No controlled study of evidence-carrying returns on agent benchmarks found; the closest item is a conceptual assurance framework.
- **Fit:** Strong. Mostly harness engineering plus careful evaluation; good second paper or a parallel track.

1. **Experiment step 1:** Benchmark: Toolathlon-Gym tasks (public), the source of MS-Eval's 166 tasks (89 annotated as needing delegation), with MS-Eval's Spec setup rebuilt from the paper. MS-Eval's own code link (github.com/LINs-lab/MS-Eval) returned 404 on 24 Oct 2026, so it is not public yet.
2. **Experiment step 2:** Models: DeepSeek-V4-Flash (the cheap model MS-Eval uses), one open model served locally, one frontier model.
3. **Experiment step 3:** Arms: prose return; typed contract (status, artifacts, open issues); evidence-carrying return without audit; with random audit k=1 and k=3; risk-weighted audit; LLM-verifier baseline. 3 seeds.
4. **Experiment step 4:** Metrics: success rate, success-to-failure and failure-to-success flips versus single agent, propagation rate (MS-Eval taxonomy, LLM-labelled with manual checks), critical steps, dollars per task.

**Links:**

- [MS-Eval: sub-agent delegation and orchestration in MCP-intensive scenarios](https://arxiv.org/abs/2610.22869) — arxiv.org/abs/2610.22869, Oct 2026
- [Toolathlon-Gym (task source for MS-Eval; MS-Eval's own code link returned 404 on 24 Oct 2026)](https://github.com/eigent-ai/toolathlon_gym) — github.com/eigent-ai, 2026
- [When Sub-Agents Work in Parallel: dynamic concurrency in long-horizon coding](https://arxiv.org/abs/2610.10263) — arxiv.org/abs/2610.10263, Oct 2026
- [Same Context, Different Agents](https://arxiv.org/abs/2610.20959) — arxiv.org/abs/2610.20959, Oct 2026
- [A trace-based assurance framework for agentic orchestration](http://arxiv.org/abs/2603.18096) — arxiv.org/abs/2603.18096, Mar 2026

<a id="idea3"></a>

## 3. Calibrated continuation critic against premature termination

- **Gap:** 'Diagnosing and Mitigating Context Rot in Long-horizon Search' (Jun 2026) finds that with long context, models give up or answer uncertainly well before the window is full, and that this premature-termination rate rises with context length. Context management helps mainly by reducing it. 'Same Context, Different Agents' identifies a group of 'early stoppers' that quit before finishing the investigation. CompAct-Bench links distorted summaries to early commitment. Existing fixes change the context (compaction, isolation) or add a rubric. A separate small critic that estimates whether continuing will pay off, trained on forked rollouts, did not show up in my searches.
- **Why now:** Premature stopping is a cheap-to-measure, under-addressed failure that sits under deep-research and data agents, and a critic is portable across models.
- **Hypothesis:** A 1 to 4B critic trained on (continue vs stop) forks vetoes at least 40% of premature stops and raises BrowseComp-Plus accuracy at matched tokens, beating 'never give up' prompting, budget forcing, and the behavior-aware filtering from the Context Rot paper.
- **Compute and cost:** Mostly GPU time for local rollouts and forks (days, not weeks). API $50 to $200 for one frontier check. Estimates.
- **Scoop risk:** Medium. No direct match found, but test-time scaling and verifier work is dense, so check again before starting.
- **Fit:** Good. Almost fully local and a clear single-figure result; pairs naturally with idea 1.

1. **Experiment step 1:** Benchmarks: BrowseComp-Plus plus the settings in the GAIR-NLP ContextRot repo.
2. **Experiment step 2:** Labels: at each stop point, fork N continuations with a local model and record whether any succeeds.
3. **Experiment step 3:** Arms: base agent; prompt-only persistence; budget forcing; parallel sampling with behavior-aware filtering; critic veto; critic veto plus compaction.
4. **Experiment step 4:** Metrics: accuracy, premature-termination rate, false-veto rate, tokens per solved task.

**Links:**

- [Diagnosing and Mitigating Context Rot in Long-horizon Search](https://arxiv.org/abs/2606.29718) — arxiv.org/abs/2606.29718, Jun 2026
- [Self-Compacting Language Model Agents (SelfCompact)](https://arxiv.org/html/2606.23525v2) — arxiv.org/abs/2606.23525, Jun 2026
- [Same Context, Different Agents](https://arxiv.org/abs/2610.20959) — arxiv.org/abs/2610.20959, Oct 2026

<a id="idea4"></a>

## 4. Sim-to-real orchestration: train a planner against a cheap simulator

- **Gap:** OrchBench (Jul 2026) scores multi-agent orchestration plans with a deterministic simulator, without running workers. Its scores correlate with Claude Code executions at r = 0.816 while using 1.3% of the tokens and 10.3% of the wall-clock time. Recent routing work (Planner-as-Router, Sep 2026; ProgRouter, Aug 2026) assigns models within plans, but I found none that trains the planner on a cheap simulated reward. Using the simulator as an RL reward and checking transfer to real execution is untested. I found no official OrchBench code, so the simulator would need reimplementing from the paper.
- **Why now:** Planner/worker orchestration is where frontier harnesses are heading, and training planners on real executions is too expensive for anyone without a lab budget. A cheap proxy reward would change that.
- **Hypothesis:** An 8 to 14B planner trained with GRPO on simulator reward improves real-execution quality per dollar over its base and over a prompted frontier planner; where the simulator is miscalibrated, reward hacking is measurable and bounded.
- **Compute and cost:** GPU-heavy: RL for 1 to 2 weeks. API $200 to $500 for real-execution validation. Estimates; reimplementation is the main time cost.
- **Scoop risk:** Medium. It is the obvious next step for the OrchBench authors.
- **Fit:** High upside, higher risk. Better as a second or third project.

1. **Experiment step 1:** Reimplement the simulator (task DAGs, per-agent context limits, information transfers with retention ratios); ask the authors for code first.
2. **Experiment step 2:** Train on synthetic DAGs; evaluate on held-out simulated DAGs and 50 to 100 real tasks executed with cheap workers (for example MS-Eval tasks).
3. **Experiment step 3:** Baselines: base planner, frontier planner, Planner-as-Router-style tier assignment.
4. **Experiment step 4:** Metrics: simulated score, real success, makespan, tokens, sim-to-real correlation before and after training, hacking cases.

**Links:**

- [OrchBench: evaluating orchestration plans via deterministic simulation](https://arxiv.org/abs/2607.25656) — arxiv.org/abs/2607.25656, Jul 2026
- [Planner-as-Router: plan-time model routing](https://arxiv.org/html/2609.32917v1) — arxiv.org/abs/2609.32917, Sep 2026
- [ProgRouter: progress-guided orchestration](https://arxiv.org/html/2608.25992v1) — arxiv.org/abs/2608.25992, Aug 2026

<a id="idea5"></a>

## 5. A small trained evidence-gatherer for experiential agent memory

- **Gap:** LongMemEval-V2 (May 2026) asks whether memory turns long web-agent histories (up to 500 trajectories, 115M tokens) into usable experience. Storing trajectories as files and letting a coding agent search them (AgentRunbook-C) reaches 72.5%, against 48.5% for the best RAG baseline, but with high latency; the authors say much room remains. LazyMem (Jul 2026) shows a 4B model trained to build memory at query time reaches 0.85 on the original LongMemEval with about 213 tokens of constructed memory. Open: distil agentic file search into a small local model trained with RL, aiming for AgentRunbook-C accuracy at a fraction of the latency. The benchmark has public leaderboard tiers.
- **Why now:** Agent memory is shifting from user chat history to environment experience, and a leaderboard gives a hobby paper a clear, comparable number.
- **Hypothesis:** A 4 to 8B searcher with grep/read tools, SFT on teacher traces then RL on answer correctness, gets within 5 points of AgentRunbook-C at one fifth of its latency or less.
- **Compute and cost:** API for teacher traces estimated $200 to $600 (histories are large); SFT and RL take days on your GPU.
- **Scoop risk:** Medium-high. Memory is crowded and the benchmark authors are likely iterating on their own AgentRunbook baselines.
- **Fit:** Solid but less novel; pick it if you want a leaderboard result.

1. **Experiment step 1:** LongMemEval-V2 small and medium tiers with the repo's baselines (AgentRunbook-R and -C, vanilla coding agent).
2. **Experiment step 2:** Teacher: frontier coding agent; student: open 4B or 8B model.
3. **Experiment step 3:** Training questions generated from held-out environments to avoid test leakage.
4. **Experiment step 4:** Metrics: accuracy per memory ability (5), latency, tokens.

**Links:**

- [LongMemEval-V2](https://arxiv.org/abs/2605.12493) — arxiv.org/abs/2605.12493, May 2026
- [LongMemEval-V2 code and leaderboard](https://github.com/xiaowu0162/LongMemEval-V2) — github.com/xiaowu0162, 2026
- [LazyMem: retrieve broadly, construct selectively](https://arxiv.org/html/2607.22690v2) — arxiv.org/abs/2607.22690, Jul 2026

<a id="idea6"></a>

## 6. Profile the model first, then pick the harness

- **Gap:** 'Same Context, Different Agents' shows the same technique can add 23 points for models that answer before looking and cost 5 points for models that already investigate; a model's behavior group can be identified from about 10 runs. The authors list adaptive selection as future work. MS-Eval finds sub-agents help most open models and hurt the closed ones it tested. Adjacent adaptive work selects per task or per step: AdaCoM (an RL-trained context manager), Dats (topology by problem difficulty), AdaptOrch. Selecting the harness configuration from a cheap behavioral probe of the model, on agentic tasks, appears open.
- **Why now:** New models ship monthly and nobody re-tunes their harness for each; a 20-task probe that tells you whether to turn on sub-agents or compaction would be practical.
- **Hypothesis:** Behavior features from 10 to 20 probe tasks predict which of single agent, sub-agents, compaction, or a semantic layer helps a new model, better than task-difficulty routers at equal cost.
- **Compute and cost:** Highest API spend of the list because it needs a full model-by-technique grid: estimated $300 to $1,000.
- **Scoop risk:** Medium-high, given AdaCoM, Dats and the authors' own stated next step.
- **Fit:** Useful but expensive and near crowded work; keep as a backup.

1. **Experiment step 1:** 6 to 8 models (local open ones plus API ones) on MS-Eval and a KramaBench/DataClaw subset.
2. **Experiment step 2:** Leave-one-model-out prediction of each technique's effect from probe behavior features.
3. **Experiment step 3:** Baselines: always-on configurations, difficulty router in the style of Dats.

**Links:**

- [Same Context, Different Agents](https://arxiv.org/abs/2610.20959) — arxiv.org/abs/2610.20959, Oct 2026
- [AdaCoM: agent-compatible context management](https://arxiv.org/html/2605.30785) — arxiv.org/abs/2605.30785, May 2026
- [Dats: difficulty-aware topology selection](http://arxiv.org/abs/2609.13890v1) — arxiv.org/abs/2609.13890, Sep 2026

<a id="crowded"></a>

## Areas checked and judged crowded

- Skill routing and injection: SkillRouter (2603.22455), SkillApt (2609.26863), SkillBrowser (2610.23554), Lost in Implicit Routing (2610.18522), plus SkillsBench, SWE-Skills-Bench, WebDev-Skills-Bench.
- When to compact: SelfCompact, CompactionRL (2607.05378), AutoCompact (2610.02163), SWE-Compressor.
- Concurrency control for parallel coding agents: STORM (2605.20563), CoAgent (2606.15376), CAID (2603.21489), PASC (ESEM 2026).
- Memory staleness and environment drift: GLOVE (2601.19249), STALE (2605.06527), SkillShift-Bench (EMNLP 2026).
- Tool retrieval over large MCP catalogs: HumanMCP (2602.23367), ComplexMCP (2605.10787).
- Failure attribution: Who&When (2505.00212), Who&When Pro (2607.09996), TraceElephant (2604.22708).
- Single vs multi-agent at equal budget: 2604.02460 and the information-bottleneck study 2607.16133.

<a id="vla"></a>

## Later option: VLA

Failure recovery for vision-language-action policies is open and fits your GPU (OpenVLA-OFT LoRA fine-tuning on LIBERO needs about 25 GB at batch 1 and 62 GB at batch 8). LIBERO-Recover (Sep 2026, arxiv.org/abs/2609.05178) builds recovery scenarios from real VLA failures, but says the project "will be available", so check the data release before planning around it.

<a id="method"></a>

## How this was put together

Sources: arXiv abstracts and full texts read on 24 Oct 2026, project pages, GitHub repos, and X posts from mid-September to October 2026. Every linked paper was opened or returned by search with its abstract; numbers quoted come from those pages. Not verified: whether the CompAct-Bench and OrchBench code is public (I found no official repo for either), and whether the concurrency study's trajectory artifact is fully downloadable (its GitHub repo exists but describes itself as a transfer corpus). For free publishing: arXiv may ask a first-time submitter for an endorsement; TMLR and OpenReview workshops accept submissions directly.
