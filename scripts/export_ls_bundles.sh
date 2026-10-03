#!/bin/bash
# Export every LatentSkill IFT skill document from an aligned hypernet checkpoint
# into EasySteer `steerling_lowrank_adapter` bundles.
#
#   usage: HYPERNET=checkpoints/ls_sft/hypernet.pt OUT_ROOT=bundles/ls_skills \
#          GPU=2 bash scripts/export_ls_bundles.sh
set -u
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=${PY:-python}
GPU=${GPU:-2}
HYPERNET=${HYPERNET:-checkpoints/ls_sft/hypernet.pt}
OUT_ROOT=${OUT_ROOT:-bundles/ls_skills}
MAX_SKILL_LEN=${MAX_SKILL_LEN:-1024}

cd "$PROJ"
mkdir -p "$OUT_ROOT"

CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1 "$PY" -u -m ls_align.export \
  --hypernet "$HYPERNET" \
  --model-path "${MODEL:-Qwen3-8B}" \
  --data-root data/latentskill \
  --out-root "$OUT_ROOT" \
  --max-skill-len "$MAX_SKILL_LEN" \
  --gpu "$GPU"
echo "[done] bundles under $OUT_ROOT"
