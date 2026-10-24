#!/usr/bin/env bash
# M6 fallback. vLLM 0.17.0 or later, transformers 5.8.0 or later.
set -euo pipefail
exec vllm serve Qwen/Qwen3.8-27B-FP8 \
  --max-model-len 131072 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --language-model-only \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
