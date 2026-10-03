# HyperSkill: Skill-Text to Post-Block Activation Offsets

Code for reproducing the main experiments: a hypernetwork maps a natural-language
skill description into per-layer **post-block low-rank activation offsets**
(`h' = h + alpha * up(silu(down @ h))`) that are injected into a frozen LLM
residual stream, replacing weight-space LoRA (the LatentSkill parameterization)
with ~1/9 of the per-skill parameters (2.36 M vs 21.8 M).

Two training tracks are included:

1. **Main method** (`skill_hypernet.py`, `skill_data.py`, `train_skill.py`):
   a 94.9 M-parameter hypernetwork (memory encoder + per-layer generator) trained
   in two stages — skill-document RECON/COMP pretraining, then trajectory-action
   SFT — following the LatentSkill recipe.
2. **LatentSkill-aligned round** (`ls_align/`): the LatentSkill data pipeline,
   backbone + frozen MetaLoRA compile end, hypernetwork body, chat template and
   optimizer, with **only** the output parameterization replaced by post-block
   offsets. This is the single-variable comparison against weight-space LoRA.

Bundles are exported in the `steerling_lowrank_adapter` sidecar format
(`manifest.json` + `adapter.safetensors`) and served through the EasySteer/vLLM
steering path via the included `easysteer_parity_plugin` (zero engine changes).

## Layout

```
skill_hypernet.py            hypernetwork + PostBlockSteerer injection module
skill_data.py                datasets / collators / batch samplers (both stages)
train_skill.py               two-stage training entry (stage pretrain | sft)
export_skill_bundle.py       export one skill document to an EasySteer bundle
alfworld_eval.py             ALFWorld evaluation, HF engine
alfworld_eval_vllm.py        ALFWorld evaluation, EasySteer/vLLM engine
searchqa_eval_bundle.py      SearchQA evaluation (7 datasets, EM), vLLM engine
ls_align/                    LatentSkill-aligned round (config/model/data/train/
                             collator_think/export/baseline/ctr)
easysteer_parity_plugin/     vLLM steering plugin (steerling_adapter algorithm)
scripts/                     launchers, export/verify/eval helpers
```

## Requirements

- Python >= 3.10, PyTorch, transformers, safetensors.
- An EasySteer vLLM environment (vLLM 0.26.0 EasySteer fork) for the
  environment-level evaluations and the steering plugin.
- The public **LatentSkill** release, placed at `$LATENTSKILL_ROOT` (default
  `./LatentSkill`). It provides: the datasets, the `latentskill` python package
  used by `ls_align/`, the released MetaLoRA / metanetwork checkpoints, the
  ALFWorld data + task configs, and the E5 retrieval index for SearchQA.
- Base model: Qwen3-8B (36 layers, hidden 4096), locally available; pass its
  path via `--model-name` / `MODEL`. Model weights are **not** included here.

## Environment

```bash
export LATENTSKILL_ROOT=/path/to/LatentSkill
export MODEL=/path/to/Qwen3-8B
```

## Data

```bash
# skill-document pretraining + trajectory SFT data -> data/latentskill/
python scripts/fetch_latentskill_data.py

# ALFWorld: use $LATENTSKILL_ROOT/alfworld_data/alfworld (set ALFWORLD_DATA)
# SearchQA retrieval: assemble the E5 faiss index from the two released parts
#   (part_aa + part_ab) into $LATENTSKILL_ROOT/wiki_index/e5_Flat.index
```

## Reproduction: main method

```bash
# 0) smoke (CPU-light, tiny model config)
GPU=0 MODE=smoke bash scripts/run_stage1.sh
GPU=0 MODE=smoke bash scripts/run_stage2.sh

# 1) stage 1: skill-document RECON/COMP pretraining
GPU=0 bash scripts/run_stage1.sh

# 2) stage 2: trajectory-action SFT (warm start from stage 1)
GPU=0 INIT=checkpoints/pretrain/hypernet.pt bash scripts/run_stage2.sh

# 3) export one bundle per skill document (9 skills)
HYPERNET=checkpoints/sft/hypernet.pt OUT_ROOT=bundles/hyper_skills \
  bash scripts/export_all_skills.sh

# 4) verify bundles against the steering plugin (EasySteer env)
BUNDLES=bundles/hyper_skills bash scripts/verify_all_bundles.sh
```

## Reproduction: ALFWorld (seen 140 games)

Protocol: LatentSkill chat template, thinking enabled, 2048 new tokens, greedy,
generation-only injection; one process per task type (the engine steering config
is process-level). Bundle -> task-type mapping:
`skill_4` pick(+pick_two), `skill_6` clean, `skill_2` cool, `skill_3` heat,
`skill_8` look_at_obj_in_light.

```bash
# single arm (EasySteer vLLM env)
bash scripts/run_ls_alfworld.sh 0 lssteer_skill4 ls_skill_4 \
     pick_and_place,pick_two_and_place

# or all five arms
bash scripts/run_ls_alfworld_all.sh

# aggregate a comparison table
python scripts/alfworld_ls_summary.py
```

For the main-method bundles, use the same commands with `BUNDLE_ROOT` pointing
at the exported bundles. The base arm is the same command without `--bundle`.

## Reproduction: SearchQA (7 datasets, 3125 questions, EM)

```bash
# 1) retrieval server (E5 top-3 over wiki-18)
GPU=0 bash scripts/run_sq_retrieval_server.sh

# 2) evaluation (EasySteer vLLM env); single-hop arm:
python searchqa_eval_bundle.py --bundle bundles/ls_skills/ls_skill_1 \
    --scale 1.0 --datasets nq triviaqa popqa --per-dataset 500 \
    --run-tag sq_sh_skill1
# multi-hop arms use --skills to route one bundle per skill:
#   ls_skill_0 -> hotpotqa/2wiki/musique, ls_skill_7 -> comparison subsets,
#   ls_skill_1 -> bamboogle
```

Protocol: LatentSkill chat template, `enable_thinking=False`, greedy, 2048 new
tokens, max 4 steps, Search-R1 style EM. Injection phase defaults to `both`
(prompt+generation); `--inject-phases generation` reproduces the phase control.

## Reproduction: LatentSkill-aligned round

Requires the `latentskill` package and the released
`checkpoints/latentskill_sft_qwen3_8b/checkpoint-epoch-10` from the LatentSkill
release (found via `LATENTSKILL_ROOT`).

```bash
# 0) self-check: alpha=0 identity, gradient routing, grad-checkpoint consistency
python scripts/smoke_ls_align.py --model-path $MODEL --precision fp32

# 1) stage 1: skill-document RECON/COMP pretraining (800 micro-steps)
GPU=0 STEPS=800 bash scripts/run_ls_stage1.sh

# 2) stage 2: trajectory SFT, warm start (2400 micro-steps)
GPU=0 STEPS=2400 INIT=checkpoints/ls_pretrain/hypernet.pt bash scripts/run_ls_stage2.sh

# 3) token-level evaluation on the 200-item SFT val split (with/base, per-skill)
python scripts/_ls_eval_dump.py

# 4) export 9 bundles + similarity analysis
GPU=0 HYPERNET=checkpoints/ls_sft/hypernet.pt OUT_ROOT=bundles/ls_skills \
  bash scripts/export_ls_bundles.sh
python scripts/_ls_bundle_sim.py bundles/ls_skills

# 5) ALFWorld seen 140 (generation-only injection) - see section above
```

Reference numbers reproduced by this pipeline (200-item SFT val split):
aligned-round activation offsets 0.9843 token accuracy / loss 0.0453 vs
LatentSkill LoRA 0.9921 / 0.0480 (frozen base 0.5559 / 5.1172); ALFWorld seen
overall 41.4% (protocol-matched round) vs base 42.9%; SearchQA all-7 EM 35.7
vs base 31.6.
