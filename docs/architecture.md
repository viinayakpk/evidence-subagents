# Architecture

The plan's data flow, plus OpenRouter as another OpenAI-compatible provider.

```mermaid
flowchart LR
  CFG[Experiment config YAML] --> RM[Run manager: grid, seeds, resume, spend cap]
  TL[Task loader: frozen Toolathlon-Gym subset] --> RM
  RM --> ENV[Toolathlon-Gym env per task: Postgres + MCP servers + container]
  RM --> MA[Main agent]
  MA -->|delegate| SA[Sub-agents]
  SA -->|MCP calls| LED[Tool-call ledger]
  LED --> ENV
  SA -->|claims + pointers| FI[Fault injector]
  FI --> CHK[L0/L1 checks]
  CHK --> SMP[Risk-based audit sampler]
  SMP --> REP[Evidence replayer L2]
  SMP --> JDG[L3 support judge]
  REP --> POL[Policy: accept / reject / re-delegate / fallback]
  JDG --> POL
  POL --> MA
  MA --> PROV[Provider layer: OpenAI-compatible]
  SA --> PROV
  JDG --> PROV
  PROV --> DS[DeepSeek API]
  PROV --> OR[OpenRouter pinned route]
  PROV --> VL[Local vLLM]
  PROV --> MK[Mock scripted model]
  PROV --> FK[Free keys, debug only]
  MA --> TR[Trace store: JSONL per run]
  SA --> TR
  POL --> TR
  ENV --> EV[Evaluator: Toolathlon-Gym eval scripts]
  TR --> LB[Failure labeler]
  EV --> AN[Analysis notebooks and plots]
  LB --> AN
```

M0–M3 runs the same boxes on an in-process fake MCP server and a scripted model. The Docker box is planned, not attached. See the README deviations.
