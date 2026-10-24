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
- M0–M3 done, in PR #1, branch `cursor/evidence-harness-m0-m3-3dd8`.
- Toolathlon-Gym task containers are STUBBED (built without Docker).
- L3 support judge is a lexical placeholder; it must be replaced by a real
  judge model before any result counts.

## Next steps (in order)
1. Wire the real Toolathlon-Gym containers with Docker on this machine
   (per-task Postgres + MCP servers + network, run_parallel.sh pattern).
2. Implement the real L3 judge (runs on the local model, never billed to API).
3. DeepSeek smoke run: `uv run evcarry run configs/experiments/deepseek_smoke.yaml`
   (key from env: `DEEPSEEK_API_KEY`).
4. Local model (Qwen3.8-Flash-Next NVFP4 via vLLM) comes LAST (M6).

## Rules
- Never commit API keys or secrets; read them from environment variables only.
- Local-model runs are last; don't start them before the API milestones.
- Before serving the big local model, the lerobot process (~/isaac-env, PID
  2046889, ~15 GiB VRAM) must be stopped — ask Vinayak first, then confirm
  `nvidia-smi` shows the GPU empty.
