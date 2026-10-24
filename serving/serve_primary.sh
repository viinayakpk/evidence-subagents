#!/usr/bin/env bash
# M6 hook. Do not run this from CI. Stop other GPU jobs before serving.
set -euo pipefail
exec docker run --gpus all --ipc=host --cap-add=SYS_PTRACE \
  -p 8000:8000 \
  -e VLLM_PLE_CPU_OFFLOAD=1 \
  -e VLLM_PLE_OFFLOAD_READY_TIMEOUT=1800 \
  vllm/vllm-openai:qwen38-flash-next \
  --model primitive-ai/Qwen3.8-Flash-Next-NVFP4 \
  --distributed-executor-backend mp \
  --gpu-memory-utilization 0.92 \
  --max-model-len 32768 \
  --max-num-seqs 36 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
