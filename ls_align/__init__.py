"""LatentSkill-aligned experiment: identical pipeline, steering-block output head.

Only the output parameterisation differs from LatentSkill:

    LatentSkill : hypernet -> 7 modules x 36 layers x (A, B) LoRA weights (~21.8 M)
    this module : hypernet -> 36 layers x (down (r,H), up (H,r)) post-block offsets
                  (36 x 2 x r x H = 2.36 M for r=8)

Everything else is taken verbatim from the public LatentSkill release:

* backbone          : `latentskill.models.qwen_lora.LatentSkillQwen3ForCausalLM`
                      (MetaLoRA-enabled, memory tokens appended to the *evidence*)
* compile end       : MetaLoRA (r=128) + mem_tokens, loaded from a released
                      LatentSkill checkpoint and frozen by default (plan A)
* hypernet body     : `latentskill.models.hypernetwork.SkillHypernetworkTransformer`
* data pipeline     : `latentskill.data.datasets.{Dataset,Collator}` (RECON/COMP
                      pretraining + trajectory SFT)
* injection         : `skill_hypernet.PostBlockSteerer` (forward hooks
                      on the decoder blocks, h' = h + alpha * up(silu(down h)))
"""

from .config import build_ls_cfg, LS_TRAIN_DEFAULTS  # noqa: F401
from .model import LSSteerHypernet, build_backbone, build_steer_hypernet  # noqa: F401
