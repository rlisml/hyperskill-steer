#!/usr/bin/env bash
# Export every LatentSkill IFT skill document into its own EasySteer bundle.
#
# Usage: HYPERNET=checkpoints/sft/hypernet.pt OUT_ROOT=bundles/sft ./scripts/export_all_skills.sh
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY=${PY:-python}
MODEL=${MODEL:-Qwen3-8B}
DATA_ROOT=${DATA_ROOT:-data/latentskill}
HYPERNET=${HYPERNET:-checkpoints/sft/hypernet.pt}
OUT_ROOT=${OUT_ROOT:-bundles/hyper_skills}
N_SKILLS=${N_SKILLS:-9}
DTYPE=${DTYPE:-fp32}

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1

for i in $(seq 0 $((N_SKILLS - 1))); do
  echo "=== exporting skill $i -> $OUT_ROOT/skill$i ==="
  PYTHONPATH="$REPO_ROOT" "$PY" export_skill_bundle.py \
    --hypernet "$HYPERNET" \
    --model-name "$MODEL" \
    --data-root "$DATA_ROOT" \
    --skill-index "$i" \
    --out-dir "$OUT_ROOT/skill$i" \
    --dtype "$DTYPE" || exit 1
done
echo "[done] bundles under $OUT_ROOT"
