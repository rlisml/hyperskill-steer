#!/usr/bin/env bash
# Stage 1 launcher: skill-document pretraining (RECON / COMP).
#
# Usage:
#   GPU=2 ./scripts/run_stage1.sh                                  # full preset
#   GPU=2 MODE=smoke ./scripts/run_stage1.sh                       # tiny smoke
#   GPU=2 ./scripts/run_stage1.sh --gen-hidden 1024 --max-steps 200
#
# Env overrides: PY, GPU, MODEL, DATA_ROOT, OUT, GEN_HIDDEN, GEN_LAYERS, GEN_FF,
#                ADAPTER_RANK, BS, GRAD_ACCUM, LR, EPOCHS, MAX_SKILL_LEN, MAX_SEQ_LEN
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY=${PY:-python}
GPU=${GPU:-2}
MODEL=${MODEL:-Qwen3-8B}
DATA_ROOT=${DATA_ROOT:-data/latentskill}
OUT=${OUT:-checkpoints/pretrain}
MODE=${MODE:-full}

# keep compile caches off NFS (Errno 28 incidents)
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU"
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/dev/shm/triton_cache}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/dev/shm/inductor_cache}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/dev/shm/xdg_cache}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/dev/shm/vllm_cache}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

ADAPTER_RANK=${ADAPTER_RANK:-8}
GEN_LAYERS=${GEN_LAYERS:-4}
GEN_FF=${GEN_FF:-8192}
GEN_HIDDEN=${GEN_HIDDEN:-1024}
BS=${BS:-2}
GRAD_ACCUM=${GRAD_ACCUM:-8}
LR=${LR:-5e-5}
EPOCHS=${EPOCHS:-1}
MAX_SKILL_LEN=${MAX_SKILL_LEN:-1024}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-1024}

if [ "$MODE" = "smoke" ]; then
  OUT=${OUT}_smoke
  GEN_LAYERS=1; GEN_FF=2048; GEN_HIDDEN=512
  MAX_SKILL_LEN=256; MAX_SEQ_LEN=256; BS=1; GRAD_ACCUM=1
  mkdir -p logs
  "$PY" train_skill.py --stage pretrain \
    --model-name "$MODEL" --data-root "$DATA_ROOT" --output-dir "$OUT" \
    --adapter-rank "$ADAPTER_RANK" --gen-layers "$GEN_LAYERS" --gen-ff "$GEN_FF" \
    --gen-hidden "$GEN_HIDDEN" --gen-nhead 16 \
    --max-samples 64 --val-samples 16 --val-batches 2 --max-steps 6 \
    --log-every 1 --eval-every 3 --save-every 3 \
    --bs "$BS" --grad-accum "$GRAD_ACCUM" \
    --max-skill-len "$MAX_SKILL_LEN" --max-seq-len "$MAX_SEQ_LEN" \
    --num-workers 0 "$@" 2>&1 | tee logs/stage1_smoke.log
  exit "${PIPESTATUS[0]}"
fi

mkdir -p logs
"$PY" train_skill.py --stage pretrain \
  --model-name "$MODEL" --data-root "$DATA_ROOT" --output-dir "$OUT" \
  --adapter-rank "$ADAPTER_RANK" --gen-layers "$GEN_LAYERS" --gen-ff "$GEN_FF" \
  --gen-hidden "$GEN_HIDDEN" --gen-nhead 32 \
  --bs "$BS" --grad-accum "$GRAD_ACCUM" --lr "$LR" --epochs "$EPOCHS" \
  --max-skill-len "$MAX_SKILL_LEN" --max-seq-len "$MAX_SEQ_LEN" \
  --val-samples 256 --val-batches 16 --eval-every 200 --save-every 200 \
  --num-workers 2 "$@" 2>&1 | tee "logs/$(basename "$OUT").log"
exit "${PIPESTATUS[0]}"
