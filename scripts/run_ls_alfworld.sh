#!/bin/bash
# ALFWorld seen-140 evaluation of the LatentSkill-aligned steering bundles.
# Same protocol as the U10 injection-phase control: LatentSkill chat template,
# thinking on, 2048 new tokens, greedy, and **generation-only** injection.
#
# One vLLM process per task type (the engine steering config is process-level),
# so the 5 arms can be spread over the three A800s.
#
#   usage: bash scripts/run_ls_alfworld.sh <gpu> <tag_suffix> <skill_index> <task_types>
#   e.g.: bash scripts/run_ls_alfworld.sh 2 ls 4 pick_and_place,pick_two_and_place
#
#   usage (single arm, explicit):  bash scripts/run_ls_alfworld.sh 2 lssteer_skill4 \
#                                        ls_skill_4 pick_and_place,pick_two_and_place
set -u
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ"

GPU=${1:?gpu}
TAG=${2:?tag}
BUNDLE=${3:?bundle dir name under BUNDLE_ROOT}
TASKS=${4:?comma separated task types}
BUNDLE_ROOT=${BUNDLE_ROOT:-bundles/ls_skills}
SCALE=${SCALE:-1.0}
SPLIT=${SPLIT:-seen}

bash scripts/alfworld_launch_vllm.sh "$GPU" "$TAG" \
  --split "$SPLIT" --scale "$SCALE" \
  --bundle "$BUNDLE_ROOT/$BUNDLE" \
  --task-types "$TASKS" \
  --concurrency 16 --max-new-tokens 2048 --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --chat-template latentskill --enable-thinking 1 \
  --inject-phases generation
