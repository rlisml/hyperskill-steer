#!/bin/bash
# Launch one ALFWorld evaluation run on the EasySteer/vLLM engine (env `alfsteer`).
#   usage: alfworld_launch_vllm.sh <gpu_id> <run_tag> [extra alfworld_eval_vllm.py args...]
set -u
GPU=$1; TAG=$2; shift 2
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=${PY:-python}
export ALFWORLD_DATA=${ALFWORLD_DATA:-${LATENTSKILL_ROOT:-./LatentSkill}/alfworld_data/alfworld}
export OMP_NUM_THREADS=8
export VLLM_USE_FLASHINFER_SAMPLER=0
cd "$PROJ"
mkdir -p logs/alfworld_vllm results/alfworld_vllm

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU \
  setsid nohup "$PY" alfworld_eval_vllm.py \
    --model-name "${MODEL:-Qwen3-8B}" \
    --out-dir results/alfworld_vllm --run-tag "$TAG" "$@" \
    > "logs/alfworld_vllm/$TAG.log" 2>&1 < /dev/null &
echo "$TAG pid=$! gpu=$GPU"
