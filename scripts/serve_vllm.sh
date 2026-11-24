#!/usr/bin/env bash
# Launch one frozen solver vLLM replica per allocated GPU.
# Usage: MODEL=/path/to/Qwen3-4B-Instruct-2507 GPUS=0,1,2,3 BASE_PORT=8011 bash scripts/serve_vllm.sh
set -euo pipefail

MODEL="${MODEL:?Set MODEL to the local path or HuggingFace ID of Qwen3-4B-Instruct-2507}"
GPUS="${GPUS:-0}"
BASE_PORT="${BASE_PORT:-8011}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.86}"

IFS=',' read -ra GPU_LIST <<< "$GPUS"
for idx in "${!GPU_LIST[@]}"; do
  gpu="${GPU_LIST[$idx]}"
  port=$((BASE_PORT + idx))
  echo "Starting vLLM solver on GPU ${gpu} -> http://127.0.0.1:${port}/v1"
  CUDA_VISIBLE_DEVICES="${gpu}" vllm serve "${MODEL}" \
    --served-model-name solver \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --port "${port}" &
done

wait
