#!/bin/bash
# Stage 2 (trajectory SFT) of the LatentSkill-aligned steering hypernetwork.
#
# Optimiser settings are LatentSkill's SFT launch script (lr 1e-5, AdamW wd 0.01,
# clip 1.0, bs 1 x accum 8, linear schedule); the warmup is scaled to this
# (much shorter) run: 30 of 300 optimiser steps.
# 1024/1024 caps are pure padding savings: the rendered samples are <=871 /
# <=539 tokens, so nothing is truncated.
#
#   usage: GPU=2 STEPS=2400 INIT=checkpoints/ls_pretrain/hypernet.pt \
#          bash scripts/run_ls_stage2.sh
set -u
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=${PY:-python}
GPU=${GPU:-2}
STEPS=${STEPS:-2400}
WARMUP=${WARMUP:-30}
ALPHA=${ALPHA:-1e-2}
INIT=${INIT:-checkpoints/ls_pretrain/hypernet.pt}
OUT=${OUT:-checkpoints/ls_sft}
EXTRA=${EXTRA:-}

cd "$PROJ"
mkdir -p logs "$OUT"

ARGS=(--stage sft --mode train --gpu "$GPU"
      --precision fp32 --grad-ckpt 1
      --bs 1 --accum 8 --micro-steps "$STEPS"
      --lr 1e-5 --warmup "$WARMUP" --weight-decay 0.01 --grad-clip 1.0
      --context-max-length 1024 --conversation-max-length 1024
      --steer-alpha "$ALPHA" --inject-phase all
      --metalora-ckpt ${LATENTSKILL_ROOT:-./LatentSkill}/checkpoints/latentskill_sft_qwen3_8b/checkpoint-epoch-10
      --eval-batches 16 --eval-every 50 --save-every 200 --log-every 10
      --out-dir "$OUT")
if [ -n "$INIT" ]; then ARGS+=(--init-from "$INIT"); fi
if [ -n "$EXTRA" ]; then ARGS+=($EXTRA); fi

CUDA_DEVICE_ORDER=PCI_BUS_ID setsid nohup "$PY" -u -m ls_align.train "${ARGS[@]}" \
  > "logs/ls_stage2.log" 2>&1 < /dev/null &
echo "ls_stage2 pid=$! gpu=$GPU steps=$STEPS out=$OUT"
