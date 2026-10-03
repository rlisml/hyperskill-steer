#!/usr/bin/env bash
# Stage 2 launcher: trajectory-supervised fine-tuning (expert-action CE).
#
# Usage:
#   GPU=2 INIT=checkpoints/pretrain/hypernet.pt ./scripts/run_stage2.sh
#   GPU=2 MODE=smoke ./scripts/run_stage2.sh
#   GPU=2 ./scripts/run_stage2.sh --bs 4 --lr 2e-5
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY=${PY:-python}
GPU=${GPU:-2}
MODEL=${MODEL:-Qwen3-8B}
DATA_ROOT=${DATA_ROOT:-data/latentskill}
OUT=${OUT:-checkpoints/sft}
INIT=${INIT:-}
MODE=${MODE:-full}

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
BS=${BS:-4}
GRAD_ACCUM=${GRAD_ACCUM:-4}
LR=${LR:-2e-5}
EPOCHS=${EPOCHS:-3}
MAX_SKILL_LEN=${MAX_SKILL_LEN:-1024}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-1024}

INIT_ARGS=()
[ -n "$INIT" ] && INIT_ARGS=(--init-checkpoint "$INIT")

mkdir -p logs

if [ "$MODE" = "smoke" ]; then
  OUT=${OUT}_smoke
  GEN_LAYERS=1; GEN_FF=2048; GEN_HIDDEN=512
  MAX_SKILL_LEN=256; MAX_SEQ_LEN=256; BS=1; GRAD_ACCUM=1
  "$PY" train_skill.py --stage sft \
    --model-name "$MODEL" --data-root "$DATA_ROOT" --output-dir "$OUT" \
    "${INIT_ARGS[@]}" \
    --adapter-rank "$ADAPTER_RANK" --gen-layers "$GEN_LAYERS" --gen-ff "$GEN_FF" \
    --gen-hidden "$GEN_HIDDEN" --gen-nhead 16 \
    --val-batches 2 --max-steps 6 --log-every 1 --eval-every 3 --save-every 3 \
    --bs "$BS" --grad-accum "$GRAD_ACCUM" \
    --max-skill-len "$MAX_SKILL_LEN" --max-seq-len "$MAX_SEQ_LEN" \
    --num-workers 0 "$@" 2>&1 | tee logs/stage2_smoke.log
  exit "${PIPESTATUS[0]}"
fi

"$PY" train_skill.py --stage sft \
  --model-name "$MODEL" --data-root "$DATA_ROOT" --output-dir "$OUT" \
  "${INIT_ARGS[@]}" \
  --adapter-rank "$ADAPTER_RANK" --gen-layers "$GEN_LAYERS" --gen-ff "$GEN_FF" \
  --gen-hidden "$GEN_HIDDEN" --gen-nhead 32 \
  --bs "$BS" --grad-accum "$GRAD_ACCUM" --lr "$LR" --epochs "$EPOCHS" \
  --max-skill-len "$MAX_SKILL_LEN" --max-seq-len "$MAX_SEQ_LEN" \
  --val-batches 16 --eval-every 50 --save-every 100 \
  --num-workers 2 "$@" 2>&1 | tee "logs/$(basename "$OUT").log"
exit "${PIPESTATUS[0]}"
