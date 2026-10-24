# CLAUDE.md — evidence-subagents

## What this is
Research harness for a paper: sub-agents fail silently, and making every sub-agent
claim carry replayable evidence (tool-call ID, file+lines, search, DB row) plus a
risk-based spot-check audit (L0 existence, L1 content match, L2 replay, L3 judge)
removes most delegated failure propagation cheaply. Python package `evcarry`, built
on the OpenAI Agents SDK, with Toolathlon-Gym as the task environment.

## Read first
- docs/plan/EXPERIMENT_PLAN.md — full plan: architecture, schema, configs, tests,
  milestones M0–M6, run matrix, metrics, costs, timeline.
- docs/plan/RESEARCH_IDEAS.md — the ranked idea list this project came from.

## Current state
- M0–M3 in PR #1 (`cursor/evidence-harness-m0-m3-3dd8`); real gym containers and
  the LLM L3 judge on `feat/gym-containers-l3-judge`; pilot preparation on
  `feat/pilot-ready` (see docs/STATUS_pilot_ready.md).
- Pilot and headline model: DeepSeek-V4.1-Flash-1017 via OpenRouter, pinned to
  DeepInfra fp8 (`openrouter_ds_1017` in configs/models.yaml, key
  `OPENROUTER_API_KEY`). The same entry is the L3 judge (billed, counted toward
  the cap). Direct `deepseek_flash` is not used for paper numbers.
- A run fails if OpenRouter serves a call from another provider (exit 7).
- Task lists: 7 email-defect tasks swapped by a fixed rule (docs/TASK_RULES.md);
  delegation labels in tasks/delegation_labels.csv.
- prompts/gym_v1 is NOT frozen; the owner freezes it after the pilot
  (`uv run evcarry prompts freeze prompts/gym_v1`). The headline refuses to run
  until it is frozen.

## Next steps (in order)
1. Owner decisions listed in docs/STATUS_pilot_ready.md.
2. Pilot (M4): `uv run evcarry run configs/experiments/pilot_deepseek.yaml`
   (load the key with `set -a; source .env; set +a`; never print .env).
3. Hand-label the judge sample (`scripts/judge_validation.py`), then freeze the
   prompts.
4. Headline (M5): `make headline`.
5. Local model (Qwen3.8-Flash-Next NVFP4 via vLLM) comes LAST (M6); it also
   re-judges the judge-validation sample as a second judge.

## Rules
- Never commit API keys or secrets; read them from environment variables only.
- Local-model runs are last; don't start them before the API milestones.
- Before serving the big local model, the lerobot process (~/isaac-env, PID
  2046889, ~15 GiB VRAM) must be stopped — ask Vinayak first, then confirm
  `nvidia-smi` shows the GPU empty.
