"""LatentSkill pipeline with a steering-block output head.

    hypernet body  : latentskill.models.hypernetwork.SkillHypernetworkTransformer
    output head    : reshape of the same flat vector into (down, up) per layer
    injection      : skill_hypernet.PostBlockSteerer (forward hooks)

The number of memory tokens follows LatentSkill's own rule
(`num_mem_token = generated_scalars // (hidden_size * num_layers)`), which for
36 layers x (down rH + up Hr) at r=8 gives 2r = 16 (LatentSkill: 148).
"""

import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

LS_ROOT = os.environ.get("LATENTSKILL_ROOT", "./LatentSkill")
if LS_ROOT not in sys.path:
    sys.path.insert(0, LS_ROOT)

from latentskill.models.hypernetwork import SkillHypernetworkTransformer  # noqa: E402
from latentskill.models.qwen_lora import (  # noqa: E402
    LatentSkillQwen3ForCausalLM,
    Qwen3Config,
)
from latentskill.utils.freeze import freeze_backbone_except_memory  # noqa: E402
from latentskill.training.checkpointing import materialize_trainable_state  # noqa: E402

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from skill_hypernet import HypernetConfig, PostBlockSteerer  # noqa: E402

from .config import REPO_CHAT_TEMPLATE, build_ls_cfg, mem_tokens_for_steer  # noqa: E402

EXTRA_TOKENS = ["<RECON>", "<COMP>", "<NOTHING>"]


def build_tokenizer(model_path: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, padding_side="left", use_fast=True)
    tok.add_tokens(EXTRA_TOKENS)
    tok.chat_template = REPO_CHAT_TEMPLATE
    return tok


def build_backbone(
    model_path: str,
    num_mem_token: int,
    dtype: torch.dtype = torch.float32,
    device: Optional[str] = None,
):
    """LatentSkillQwen3ForCausalLM + memory tokens, exactly as train_compiler.py."""
    config = Qwen3Config.from_pretrained(model_path)
    config.num_mem_token = int(num_mem_token)
    try:
        backbone = LatentSkillQwen3ForCausalLM.from_pretrained(
            model_path, config=config, dtype=dtype)
    except TypeError:
        backbone = LatentSkillQwen3ForCausalLM.from_pretrained(
            model_path, config=config, torch_dtype=dtype)
    backbone.reset_mem_tokens()
    backbone.config.use_cache = False
    if device is not None:
        backbone = backbone.to(device)
    return backbone


def load_metalora(path: str, device: str = "cpu", trainable: bool = False,
                  dtype: Optional[torch.dtype] = None):
    state = torch.load(path, map_location="cpu", weights_only=False)
    # `materialize_trainable_state` detaches -> the released metalora tensors are
    # non-leaf, and `requires_grad_` is only legal on leaves.
    state = materialize_trainable_state(state, "cpu")
    if dtype is not None:
        state = _cast(state, dtype)
    state = _to_device(state, device)
    if not trainable:
        from latentskill.models.lora_ops import freeze_adapter_state
        freeze_adapter_state(state)
    return state


def _cast(obj, dtype):
    if torch.is_tensor(obj) and obj.is_floating_point():
        return obj.detach().to(dtype)
    if isinstance(obj, dict):
        return {k: _cast(v, dtype) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_cast(v, dtype) for v in obj]
    return obj


def _to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_device(v, device) for v in obj]
    return obj


def _set_requires_grad(obj, flag: bool):
    if torch.is_tensor(obj):
        obj.requires_grad_(flag)
        return
    if isinstance(obj, dict):
        for v in obj.values():
            _set_requires_grad(v, flag)
        return
    if isinstance(obj, (list, tuple)):
        for v in obj:
            _set_requires_grad(v, flag)


def load_mem_tokens(path: str):
    return torch.load(path, map_location="cpu", weights_only=False)


class LSSteerHypernet(nn.Module):
    """MetaLoRA-compiled skill encoder -> post-block low-rank activation offsets."""

    def __init__(
        self,
        backbone: nn.Module,
        cfg,
        hcfg: HypernetConfig,
        layer_ids: Optional[Sequence[int]] = None,
        steer_alpha: float = 1.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.cfg = cfg
        self.hcfg = hcfg
        self.rank = int(hcfg.adapter_rank)
        self.hidden = int(hcfg.hidden_size)
        self.num_layers = int(cfg.num_layers)
        self.layer_ids = list(range(self.num_layers)) if layer_ids is None else list(layer_ids)
        self.scale = float(cfg.hypernetwork.transformer_cfg.scale)
        self.steer_alpha = float(steer_alpha)

        # Same partitioning rule as LatentSkill, but with two blocks per layer
        # (down: r*H, up: H*r) instead of seven LoRA modules.
        idx_range = [0, self.rank * self.hidden, 2 * self.rank * self.hidden]
        self.generator = SkillHypernetworkTransformer(cfg, idx_range)
        self.steerer = PostBlockSteerer(hcfg, self.layer_ids)

    # ------------------------------------------------------------------ #
    def register_hooks(self) -> None:
        self.steerer.register_hooks(self.backbone)

    def remove_hooks(self) -> None:
        self.steerer.remove_hooks()

    def hypernet_parameters(self) -> List[nn.Parameter]:
        """Generator + memory tokens (everything that is *not* the frozen base)."""
        params = list(self.generator.parameters())
        mt = getattr(self.backbone.model, "mem_tokens", None)
        if mt is not None:
            params.append(mt)
        return [p for p in params if p.requires_grad]

    def num_hypernet_params(self) -> int:
        return sum(p.numel() for p in self.hypernet_parameters())

    # ------------------------------------------------------------------ #
    def build_delta(
        self,
        evidence_ids: torch.Tensor,
        evidence_attention_mask: torch.Tensor,
        metalora: Any,
        use_gradient_checkpoint: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (down [B,L,r,H], up [B,L,H,r]) for the given skill evidence."""
        # The evidence encoding must run on the *un-steered* backbone, so any
        # delta left over from a previous forward/backward is dropped here.
        self.steerer.clear_delta()
        outputs = self.backbone(
            input_ids=evidence_ids,
            attention_mask=evidence_attention_mask,
            adapter_state=metalora,
            use_gradient_checkpoint=use_gradient_checkpoint,
        )
        memory_states = outputs.memory_states                    # [B, L, N, H]
        flat = self.generator(memory_states)                     # [B, L*N*H]
        B = flat.shape[0]
        L, r, H = self.num_layers, self.rank, self.hidden
        # LatentSkill's `rl` adapter builder multiplies each factor by sqrt(scale).
        flat = flat.view(B, L, 2 * r * H) * math.sqrt(self.scale)
        down = flat[:, :, : r * H].contiguous().view(B, L, r, H)
        up = flat[:, :, r * H: 2 * r * H].contiguous().view(B, L, H, r)
        # S1 (step 13): keep the generated factors (with graph) so the training
        # loop can build the export-space contrastive vector without a re-forward.
        self.last_down, self.last_up = down, up
        return down, up

    def set_steer(self, down: torch.Tensor, up: torch.Tensor) -> None:
        # keep the generated factors in the backbone dtype (bf16 training)
        bdt = next(self.backbone.parameters()).dtype
        down = down.to(bdt)
        up = up.to(bdt)
        alpha = torch.full(
            (len(self.layer_ids),), self.steer_alpha, device=down.device, dtype=bdt
        )
        self.steerer.set_delta(down, up, alpha=alpha)

    def forward(
        self,
        input_ids: torch.Tensor,
        input_attention_mask: torch.Tensor,
        evidence_ids: Optional[torch.Tensor] = None,
        evidence_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        metalora: Any = None,
        use_delta: bool = True,
        use_gradient_checkpoint: bool = False,
    ) -> None:
        # NB: the offset is cleared HERE (not after the forward) because
        # gradient checkpointing re-runs the decoder layers during backward();
        # the hooks must then re-apply the very same offset.
        self.steerer.clear_delta()
        if use_delta:
            assert metalora is not None, "metalora is required when use_delta=True"
            down, up = self.build_delta(
                evidence_ids, evidence_attention_mask, metalora,
                use_gradient_checkpoint=use_gradient_checkpoint,
            )
            self.set_steer(down, up)
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=input_attention_mask,
            labels=labels,
            ignore_mem_token=True,
            use_gradient_checkpoint=use_gradient_checkpoint,
        )
        return outputs

    # ------------------------------------------------------------------ #
    # checkpoint I/O
    # ------------------------------------------------------------------ #
    def save_hypernet(self, path: str, extra=None, metalora=None) -> None:
        from dataclasses import asdict

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        c = self.cfg
        payload = {
            "generator": self.generator.state_dict(),
            "mem_tokens": self.backbone.model.mem_tokens.detach().cpu(),
            "cfg_args": dict(
                num_layers=c.num_layers, hidden_size=c.hidden_size,
                lora_r=c.model.lora_r, metalora_r=c.model.metalora_r,
                num_mem_token=c.num_mem_token,
                gen_num_layers=c.hypernetwork.transformer_cfg.num_layers,
                gen_nhead=c.hypernetwork.transformer_cfg.encoder_cfg["nhead"],
                gen_ff=c.hypernetwork.transformer_cfg.encoder_cfg["dim_feedforward"],
                hypernet_scale=c.hypernetwork.transformer_cfg.scale,
            ),
            "hcfg": asdict(self.hcfg),
            "steer_alpha": self.steer_alpha,
            "layer_ids": list(self.layer_ids),
            "extra": extra or {},
        }
        if metalora is not None:
            payload["metalora"] = metalora
        torch.save(payload, path)

    def load_hypernet(self, path: str):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.generator.load_state_dict(payload["generator"])
        mt = payload.get("mem_tokens")
        if mt is not None:
            with torch.no_grad():
                self.backbone.model.mem_tokens.copy_(
                    mt.to(self.backbone.model.mem_tokens.dtype))
        return payload

    def load_generator_partial(self, path: str):
        """Warm-start the generator from a LatentSkill `metanetwork.pth`.

        Only tensors whose shape matches are copied (the released checkpoint has
        148 memory-token slots, ours 2r=16, so `token_pe` is skipped).
        """
        sd = torch.load(path, map_location="cpu", weights_only=False)
        own = self.generator.state_dict()
        keep, skip = {}, []
        for k, v in sd.items():
            if k in own and own[k].shape == v.shape:
                keep[k] = v
            else:
                skip.append(k)
        self.generator.load_state_dict({**own, **keep})
        return skip


def make_hypernet_config(
    num_layers: int = 36,
    hidden_size: int = 4096,
    rank: int = 8,
    activation: str = "silu",
    inject_phase: str = "all",
) -> HypernetConfig:
    """Minimal `HypernetConfig` for `PostBlockSteerer` (no step-9 extras)."""
    return HypernetConfig(
        num_layers=num_layers,
        hidden_size=hidden_size,
        adapter_rank=rank,
        delta_mode="lowrank",
        inject_target="block",
        activation=activation,
        inject_phase=inject_phase,
        num_classes=0,
        decor_weight=0.0,
    )


def build_steer_hypernet(
    model_path: str,
    rank: int = 8,
    metalora_r: int = 128,
    steer_alpha: float = 1.0,
    gen_num_layers: int = 4,
    gen_nhead: int = 32,
    gen_ff: int = 8192,
    hypernet_scale: float = 0.001,
    dtype: torch.dtype = torch.float32,
    inject_phase: str = "all",
    device: Optional[str] = None,
):
    """Convenience factory used by training / export / smoke tests."""
    num_mem = mem_tokens_for_steer(rank)
    backbone = build_backbone(model_path, num_mem, dtype=dtype, device=device)
    cfg = build_ls_cfg(
        num_layers=int(backbone.config.num_hidden_layers),
        hidden_size=int(backbone.config.hidden_size),
        lora_r=rank,
        metalora_r=metalora_r,
        num_mem_token=num_mem,
        gen_num_layers=gen_num_layers,
        gen_nhead=gen_nhead,
        gen_ff=gen_ff,
        hypernet_scale=hypernet_scale,
    )
    # resize embeddings for the three extra pretraining tokens (as train_compiler.py)
    tok = build_tokenizer(model_path)
    backbone.resize_token_embeddings(len(tok))
    freeze_backbone_except_memory(backbone)
    hcfg = make_hypernet_config(
        num_layers=cfg.num_layers, hidden_size=cfg.hidden_size, rank=rank,
        inject_phase=inject_phase,
    )
    model = LSSteerHypernet(backbone, cfg, hcfg, steer_alpha=steer_alpha)
    model.register_hooks()
    return model, tok, cfg, hcfg
