#!/bin/bash
# Launch all five ALFWorld arms of the LatentSkill-aligned experiment.
#   usage: GPU0=2 GPU1=3 GPU2=7 bash scripts/run_ls_alfworld_all.sh
#
# Bundle -> task-type mapping (verified against
# LatentSkill/evals/alfworld/skills/*.txt):
#   ls_skill_2 cool | ls_skill_3 heat | ls_skill_4 pick(+pick_two)
#   ls_skill_6 clean | ls_skill_8 look_at_obj_in_light
set -u
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ"

G0=${GPU0:-2}
G1=${GPU1:-3}
G2=${GPU2:-7}
TAG=${TAG:-lsb}
BUNDLE_ROOT=${BUNDLE_ROOT:-bundles/ls_skills}
SCALE=${SCALE:-1.0}

bash scripts/run_ls_alfworld.sh "$G0" "${TAG}_skill4" ls_skill_4 \
     pick_and_place,pick_two_and_place
bash scripts/run_ls_alfworld.sh "$G1" "${TAG}_skill6" ls_skill_6 clean
bash scripts/run_ls_alfworld.sh "$G2" "${TAG}_skill2" ls_skill_2 cool
sleep 20
bash scripts/run_ls_alfworld.sh "$G0" "${TAG}_skill3" ls_skill_3 heat
bash scripts/run_ls_alfworld.sh "$G1" "${TAG}_skill8" ls_skill_8 look_at_obj_in_light
echo "launched 5 arms: ${TAG}_skill{2,3,4,6,8}"
