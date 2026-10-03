#!/bin/bash
# Stage 1 (skill-document RECON/COMP pretraining) of the LatentSkill-aligned
# steering hypernetwork.  Hyper-parameters = LatentSkill release config
# (lr 1e-5, AdamW wd 0.01, clip 1.0, bs 1 x accum 8, linear schedule);
# warmup is scaled to the (much shorter) run: 12 of 100 optimiser steps.
#
#   usage: GPU=2 STEPS=800 bash scripts/run_ls_stage1.sh
set -u
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=${PY:-python}
GPU=${GPU:-2}
STEPS=${STEPS:-800}
ALPHA=${ALPHA:-1e-2}
INIT=${INIT:-}          # optional: warm-start hypernet
OUT=${OUT:-checkpoints/ls_pretrain}
EXTRA=${EXTRA:-}

cd "$PROJ"
mkdir -p logs "$OUT"

ARGS=(--stage pretrain --mode train --gpu "$GPU"
      --precision fp32 --grad-ckpt 1
      --bs 1 --accum 8 --micro-steps "$STEPS"
      --lr 1e-5 --warmup 12 --weight-decay 0.01 --grad-clip 1.0
      --context-max-length 2048 --conversation-max-length 3072
      --pretrain-train-texts 20000 --pretrain-val-texts 400
      --steer-alpha "$ALPHA" --inject-phase all
      --metalora-ckpt ${LATENTSKILL_ROOT:-./LatentSkill}/checkpoints/latentskill_sft_qwen3_8b/checkpoint-epoch-10
      --eval-batches 8 --eval-every 25 --save-every 50 --log-every 5
      --out-dir "$OUT")
if [ -n "$INIT" ]; then ARGS+=(--init-from "$INIT"); fi
if [ -n "$EXTRA" ]; then ARGS+=($EXTRA); fi

CUDA_DEVICE_ORDER=PCI_BUS_ID setsid nohup "$PY" -u -m ls_align.train "${ARGS[@]}" \
  > "logs/ls_stage1.log" 2>&1 < /dev/null &
echo "ls_stage1 pid=$! gpu=$GPU out=$OUT"
