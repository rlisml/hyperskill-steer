#!/usr/bin/env bash
# Verify every bundle under a directory against the EasySteer plugin.
# Usage: BUNDLES=bundles/stage1_skills ./scripts/verify_all_bundles.sh
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY=${PY:-python}
BUNDLES=${BUNDLES:-bundles/stage1_skills}

export CUDA_VISIBLE_DEVICES=
export PYTHONUNBUFFERED=1

fail=0
for d in "$BUNDLES"/*/; do
  [ -f "$d/manifest.json" ] || continue
  line=$("$PY" scripts/verify_easysteer_bundle.py --bundle "$d" 2>/dev/null | tail -1)
  echo "$d -> $line"
  case "$line" in
    VERIFY_OK*) ;;
    *) fail=1 ;;
  esac
done
echo "[verify_all_bundles] fail=$fail"
exit $fail
