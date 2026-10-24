#!/usr/bin/env bash
# Reports the GPU if nvidia-smi exists. Missing nvidia-smi is not a CI failure.
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is not installed; skipping the GPU check."
  exit 0
fi
nvidia-smi
