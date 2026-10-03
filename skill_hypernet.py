"""Skill-text -> post-block activation-offset hypernetwork.

Paradigm:
    skill text --(frozen backbone, layer-wise pooling)--> memory_states
    memory_states --(trainable generator)--> per-layer offsets (down_l, up_l)
    base forward with post-block injection:  h_l' = h_l + s * up_l(sigma(down_l h_l))

The implementation fixes three diagnosed defects of a naive design:

1. *Encoder collapse* (memory_states nearly identical across documents):
   - pooling queries are generated from the document **[CLS]** state
     (`q = codebook + gate * MLP(LN(cls))`) instead of being a global parameter;
   - pooling is **per-paragraph** (markdown headings, `skill_seg`) and the
     paragraph vectors are aggregated by a light learned attention, so a
     document-specific section is not averaged away by the shared preamble;
   - query count 2r -> **4r** (the generator reads those out with a learned
     cross-attention down to the 2r parameter slots).
2. *Train/eval injection-phase mismatch*: `PostBlockSteerer` now
   consumes `phase_mask` from the collator and zeroes the offset on prompt
   positions, matching EasySteer `apply={"prompt": null, "generation": "all"}`.
3. *Missing skill specificity*: InfoNCE over two views of the
   same document (+ MoCo queue of cross-document negatives) and an auxiliary
   skill classifier on top of the pooled memory states.

Optional compile adapter: a light LoRA can be attached to the frozen backbone
and trained together with the hypernetwork (`--lora-rank`; OFF by default
because the exported EasySteer bundle contract assumes a pristine backbone).
"""

import math
from dataclasses import asdict, dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_NEG = -1.0e4


@dataclass
class HypernetConfig:
    """Serializable hypernetwork config (also the checkpoint metadata)."""

    num_layers: int = 36
    hidden_size: int = 4096
    adapter_rank: int = 8
    delta_mode: str = "lowrank"          # "lowrank" | "vector"
    inject_target: str = "block"         # "block" (WUAS post-block) | "attn" | "mlp"
    activation: Optional[str] = "silu"   # None == linear (WUAS `--linear`)
    output_scale: float = 0.1            # delta = alpha_l * up(act(down(h))), alpha_l init
    gen_num_layers: int = 4
    gen_nhead: int = 32
    gen_ff: int = 8192
    gen_hidden: int = 0                  # 0 => d_model = hidden_size (LatentSkill-style)
    gen_dropout: float = 0.0
    down_init_scale: float = -1.0
    up_init_scale: float = -1.0
    model_name: str = "Qwen/Qwen3-8B"
    # --- pooling-specific additions ---------------------------------------- #
    encoder_mode: str = "new"            # "new" (doc-conditioned segment pooling) | "legacy"
    num_query: int = 0                   # 0 => 4r (lowrank) / 2r (vector)
    max_segments: int = 8                # paragraph groups used by segment pooling
    pool_mode: str = "segment"           # "segment" | "flat" (legacy whole-doc pooling)
    inject_phase: str = "all"            # "all" | "gen" (generation-only injection)
    ctr_dim: int = 256                   # contrastive projection dim
    num_classes: int = 0                 # auxiliary classification head (0 => off)
    queue_size: int = 4096               # MoCo queue of cross-document negatives
    ctr_tau: float = 0.07
    delta_norm: str = "row"              # "row" (step 2-8: per-row unit-L2) | "global"
                                         # (one scalar/document; keeps the per-document direction
                                         #  that row-norm discards -> what the bundle cos measures)
    decor_weight: float = 0.0            # direct penalty on pairwise cosine of the
                                         # exported (down, up) vector, i.e. the bundle
                                         # metric itself (0 => off)

    @property
    def num_mem_token(self) -> int:
        return 1 if self.delta_mode == "vector" else 2 * self.adapter_rank

    @property
    def resolved_num_query(self) -> int:
        if self.encoder_mode == "legacy":
            # step 2-8 encoder: N = 2r (lowrank) / 1 (vector), no doc conditioning
            return self.num_mem_token
        if self.num_query and self.num_query > 0:
            return int(self.num_query)
        return 4 * self.adapter_rank if self.delta_mode != "vector" else 2 * self.adapter_rank

    @property
    def per_layer_dim(self) -> int:
        """Number of generated scalars consumed per decoder layer."""
        if self.delta_mode == "vector":
            return self.hidden_size
        return 2 * self.adapter_rank * self.hidden_size


def resolve_decoder_layers(backbone: nn.Module) -> List[nn.Module]:
    """Locate the decoder-layer ModuleList on a HF causal LM."""
    for attr in ("model", "language_model"):
        cand = getattr(backbone, attr, None)
        if cand is not None and hasattr(cand, "layers"):
            return list(cand.layers)
    raise AttributeError("cannot locate decoder layers on the backbone")


def resolve_inject_modules(backbone: nn.Module, target: str) -> List[nn.Module]:
    """Modules whose *output* receives the generated delta."""
    layers = resolve_decoder_layers(backbone)
    t = (target or "block").lower()
    if t in ("block", "post_block", "post-block"):
        return layers
    if t == "attn":
        return [getattr(l, "self_attn") for l in layers]
    if t == "mlp":
        return [getattr(l, "mlp") for l in layers]
    raise ValueError(f"unknown inject_target {target!r}")


# --------------------------------------------------------------------------- #
# optional compile adapter (light LoRA on the frozen backbone)
# --------------------------------------------------------------------------- #
class LoRALinear(nn.Module):
    """Frozen `base` + `x @ A^T @ B^T * scaling`; B zero-init (identity at start)."""

    def __init__(self, base: nn.Module, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad_(False)
        self.rank, self.scaling = int(rank), float(alpha) / max(1, int(rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        in_f = int(getattr(base, "in_features"))
        out_f = int(getattr(base, "out_features"))
        self.lora_a = nn.Parameter(torch.empty(rank, in_f))
        self.lora_b = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    @property
    def weight(self):
        return self.base.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        xa = F.linear(x, self.lora_a.to(x.dtype))
        return out + F.linear(self.dropout(xa), self.lora_b.to(x.dtype)) * self.scaling


_LORA_TARGETS = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
)


def _resolve_parent(layer: nn.Module, dotted: str):
    parts = dotted.split(".")
    parent = layer
    for part in parts[:-1]:
        parent = getattr(parent, part, None)
        if parent is None:
            return None, parts[-1]
    return parent, parts[-1]


def attach_lora(backbone: nn.Module, rank: int, alpha: float,
                targets: Sequence[str] = _LORA_TARGETS, dropout: float = 0.0) -> List[str]:
    """Wrap the given projections of every decoder block in `LoRALinear`."""
    touched = []
    for layer in resolve_decoder_layers(backbone):
        for dotted in targets:
            parent, leaf = _resolve_parent(layer, dotted)
            if parent is None:
                continue
            mod = getattr(parent, leaf, None)
            if mod is None or isinstance(mod, LoRALinear):
                continue
            setattr(parent, leaf, LoRALinear(mod, rank, alpha, dropout))
            touched.append(dotted)
    return sorted(set(touched))


def lora_parameters(backbone: nn.Module):
    return [p for m in backbone.modules() if isinstance(m, LoRALinear)
            for p in m.parameters() if p.requires_grad]


def lora_state_dict(backbone: nn.Module):
    out = {}
    for li, layer in enumerate(resolve_decoder_layers(backbone)):
        for dotted in _LORA_TARGETS:
            parent, leaf = _resolve_parent(layer, dotted)
            if parent is None:
                continue
            mod = getattr(parent, leaf, None)
            if isinstance(mod, LoRALinear):
                out[f"layers.{li}.{dotted}.lora_a"] = mod.lora_a.detach().cpu()
                out[f"layers.{li}.{dotted}.lora_b"] = mod.lora_b.detach().cpu()
    return out


def load_lora_state_dict(backbone: nn.Module, state: dict) -> int:
    loaded = 0
    for li, layer in enumerate(resolve_decoder_layers(backbone)):
        for dotted in _LORA_TARGETS:
            key_a = f"layers.{li}.{dotted}.lora_a"
            if key_a not in state:
                continue
            parent, leaf = _resolve_parent(layer, dotted)
            mod = getattr(parent, leaf, None) if parent is not None else None
            if not isinstance(mod, LoRALinear):
                continue
            with torch.no_grad():
                mod.lora_a.copy_(state[key_a].to(mod.lora_a.device, mod.lora_a.dtype))
                mod.lora_b.copy_(state[f"layers.{li}.{dotted}.lora_b"].to(
                    mod.lora_b.device, mod.lora_b.dtype))
            loaded += 1
    return loaded


# --------------------------------------------------------------------------- #
# skill memory encoder (document-conditioned, paragraph-aware)
# --------------------------------------------------------------------------- #
class SkillMemoryEncoder(nn.Module):
    """Frozen backbone + document-conditioned per-paragraph pooling.

    For every steered layer `l`:
      hidden_l [B,S,H]   backbone under `no_grad`, hidden detached
      queries  [B,N,H]   codebook[l] + gate[l] * MLP(LN(cls_l))
      per-paragraph masked-softmax pooling  -> p[b,n,k,h]  (k = segment)
      learned attention over segments       -> memory[b,n,h]
    Only this module receives gradients; the 8B backbone receives none.
    """

    def __init__(self, cfg: HypernetConfig, layers: Sequence[int]):
        super().__init__()
        self.cfg = cfg
        self.layer_ids = list(layers)
        n = cfg.resolved_num_query
        h = cfg.hidden_size
        self.num_query, self.hidden_size = n, h
        self.max_segments = max(2, int(cfg.max_segments or 8))
        self.pool_mode = (cfg.pool_mode or "segment").lower()
        self.encoder_mode = (getattr(cfg, "encoder_mode", "new") or "new").lower()
        L = len(self.layer_ids)

        self.pool_queries = nn.Parameter(torch.empty(L, n, h))
        nn.init.normal_(self.pool_queries, mean=0.0, std=0.02)
        # document -> query modulation (shared MLP, per-layer/slot gate)
        self.cls_norm = nn.LayerNorm(h)
        self.doc_proj = nn.Sequential(
            nn.Linear(h, cfg.ctr_dim), nn.GELU(), nn.Linear(cfg.ctr_dim, h)
        )
        self.doc_gate = nn.Parameter(torch.full((L, n), 0.1))
        # per-paragraph aggregation queries (also document conditioned)
        self.agg_base = nn.Parameter(torch.empty(L, n, h))
        nn.init.normal_(self.agg_base, mean=0.0, std=0.02)
        self.agg_gate = nn.Parameter(torch.full((L, n), 0.1))
        self.scale = 1.0 / math.sqrt(h)

    def _query_and_agg(self, hidden: torch.Tensor, slot: int):
        """hidden [B,S,H] -> (pooling queries [B,N,H], aggregation queries [B,N,H])."""
        if self.encoder_mode == "legacy":
            # step 2-8 behaviour: a single global learned query per (layer, slot)
            q = self.pool_queries[slot].unsqueeze(0).expand(hidden.shape[0], -1, -1)
            a = self.agg_base[slot].unsqueeze(0).expand(hidden.shape[0], -1, -1)
            return q, a
        cls_h = self.cls_norm(hidden[:, 0, :])                     # [B,H]
        doc = self.doc_proj(cls_h)                                 # [B,H]
        q = self.pool_queries[slot] + self.doc_gate[slot].view(1, -1, 1) * doc[:, None, :]
        a = self.agg_base[slot] + self.agg_gate[slot].view(1, -1, 1) * doc[:, None, :]
        return q, a

    def _pool_flat(self, hidden, mask, slot):
        q, _ = self._query_and_agg(hidden, slot)
        scores = torch.einsum("bsh,bnh->bns", hidden, q) * self.scale
        scores = scores.masked_fill(mask[:, None, :] == 0, _NEG)
        attn = scores.softmax(dim=-1)
        return torch.einsum("bns,bsh->bnh", attn, hidden)          # [B,N,H]

    def _pool_segments(self, hidden, mask, seg, slot):
        n, k = self.num_query, self.max_segments
        q, agg_q = self._query_and_agg(hidden, slot)
        scores = torch.einsum("bsh,bnh->bns", hidden, q) * self.scale        # [B,N,S]
        scores = scores.masked_fill(mask[:, None, :] == 0, _NEG)
        oh = F.one_hot(seg.clamp(0, k - 1), k).to(scores.dtype)              # [B,S,K]
        valid = oh * mask.to(scores.dtype)[:, :, None]                       # pad -> no group
        valid_t = valid.permute(0, 2, 1).unsqueeze(1)                        # [B,1,K,S]
        w = scores.unsqueeze(2).masked_fill(valid_t == 0, _NEG)              # [B,N,K,S]
        w = w.softmax(dim=-1)
        pooled = torch.einsum("bnks,bsh->bnkh", w, hidden)                   # [B,N,K,H]
        cnt = valid.sum(dim=1)                                               # [B,K]
        logit = torch.einsum("bnkh,bnh->bnk", pooled, agg_q) * self.scale
        logit = logit.masked_fill(cnt[:, None, :] == 0, _NEG)
        a = logit.softmax(dim=-1)                                            # [B,N,K]
        return torch.einsum("bnk,bnkh->bnh", a, pooled)                      # [B,N,H]

    def _pool(self, hidden: torch.Tensor, mask: torch.Tensor,
              seg: Optional[torch.Tensor], slot: int) -> torch.Tensor:
        if seg is None or self.pool_mode == "flat" or self.encoder_mode == "legacy":
            return self._pool_flat(hidden, mask, slot)
        return self._pool_segments(hidden, mask, seg, slot)

    def forward(
        self, backbone: nn.Module, input_ids: torch.Tensor,
        attention_mask: torch.Tensor, seg_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        modules = resolve_decoder_layers(backbone)
        buckets: List[Optional[torch.Tensor]] = [None] * len(self.layer_ids)
        handles = []
        for slot, layer_id in enumerate(self.layer_ids):
            def hook(_m, _inp, out, slot=slot):
                h = out[0] if isinstance(out, tuple) else out
                with torch.enable_grad():
                    buckets[slot] = self._pool(h.detach(), attention_mask, seg_ids, slot)
                return None

            handles.append(modules[layer_id].register_forward_hook(hook))
        try:
            with torch.no_grad():
                backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_hidden_states=False,
                )
        finally:
            for h in handles:
                h.remove()
        assert all(b is not None for b in buckets), "pooling hooks did not all fire"
        return torch.stack(buckets, dim=1)                            # [B,L',N,H]


# --------------------------------------------------------------------------- #
# generator: (layer, slot) transformer + learned slot readout
# --------------------------------------------------------------------------- #
class PostBlockGenerator(nn.Module):
    """Transformer encoder over the (layer, slot) grid + learned slot readout.

    Input  : memory_states [B, L', Nq, H] (Nq = `cfg.resolved_num_query` = 4r)
    Output : [B, L', Nmem*H] with Nmem = `cfg.num_mem_token` (2r for lowrank)
    """

    def __init__(self, cfg: HypernetConfig, n_grid: Optional[int] = None):
        super().__init__()
        self.cfg = cfg
        L = int(n_grid or cfg.num_layers)
        Nq, Nm, H = cfg.resolved_num_query, cfg.num_mem_token, cfg.hidden_size
        self.num_layers, self.num_query, self.num_mem_token, self.hidden_size = L, Nq, Nm, H

        self.layer_pe = nn.Parameter(torch.zeros(L, H))
        self.slot_pe = nn.Parameter(torch.zeros(Nq, H))

        d = cfg.gen_hidden or H
        assert d % cfg.gen_nhead == 0, f"gen_hidden {d} not divisible by nhead {cfg.gen_nhead}"
        self.d_model = d
        self.in_proj = nn.Linear(H, d) if d != H else None
        self.out_proj = nn.Linear(d, H) if d != H else None
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d,
                    nhead=cfg.gen_nhead,
                    dim_feedforward=cfg.gen_ff,
                    dropout=cfg.gen_dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=False,
                )
                for _ in range(cfg.gen_num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d)
        # the frozen backbone hidden states can be O(1e3); keep the generator O(1)
        self.mem_norm = nn.LayerNorm(H)
        # Nq -> Nmem readout (cross-attention with per-layer learned queries)
        self.readout = nn.MultiheadAttention(
            embed_dim=d, num_heads=cfg.gen_nhead, batch_first=True)
        self.readout_q = nn.Parameter(torch.empty(L, Nm, d))
        nn.init.normal_(self.readout_q, std=0.02)
        self.readout_norm = nn.LayerNorm(d)
        if (cfg.delta_norm or "row").lower() == "none":
            # raw-output parameterisation: start from an exact zero delta
            with torch.no_grad():
                for head in (self.readout.out_proj, self.out_proj):
                    if head is None:
                        continue
                    head.weight.zero_()
                    if head.bias is not None:
                        head.bias.zero_()

    def zero_output_head(self) -> None:
        """Force the generator to emit exactly zero (delta_norm='none' warm-up)."""
        with torch.no_grad():
            for head in (self.readout.out_proj, self.out_proj):
                if head is None:
                    continue
                head.weight.zero_()
                if head.bias is not None:
                    head.bias.zero_()

    def forward(self, memory_states: torch.Tensor) -> torch.Tensor:
        L, Nq, Nm = self.num_layers, self.num_query, self.num_mem_token
        x = self.mem_norm(memory_states) + self.layer_pe[None, :, None, :] \
            + self.slot_pe[None, None, :, :]
        x = x.flatten(1, 2)                                            # [B, L*Nq, H]
        if self.in_proj is not None:
            x = self.in_proj(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)                                               # [B, L*Nq, d]
        b = x.shape[0]
        kv = x.reshape(b * L, Nq, self.d_model)
        q = self.readout_q.unsqueeze(0).expand(b, L, Nm, self.d_model)
        q = q.reshape(b * L, Nm, self.d_model)
        y, _ = self.readout(q, kv, kv, need_weights=False)             # [B*L, Nm, d]
        y = self.readout_norm(y.reshape(b, L, Nm, self.d_model))
        if self.out_proj is not None:
            y = self.out_proj(y)
        return y.flatten(2, -1)                                        # [B,L,Nm*H]


# --------------------------------------------------------------------------- #
# injector (phase aware)
# --------------------------------------------------------------------------- #
class PostBlockSteerer(nn.Module):
    """Applies generated per-layer offsets at the decoder block output.

    When `cfg.inject_phase == "gen"`, the `phase_mask` built by
    `skill_data.phase_mask_from_labels` restricts the offset to generation
    positions; prompt positions keep the untouched backbone activation, which is
    exactly what EasySteer does at serving time with
    `apply={"prompt": null, "generation": "all"}`.
    """

    def __init__(self, cfg: HypernetConfig, layers: Sequence[int]):
        super().__init__()
        self.cfg = cfg
        self.layer_ids = list(layers)
        self.inject_phase = (cfg.inject_phase or "all").lower()
        self._down: Optional[torch.Tensor] = None   # lowrank [B,L,r,H] | vector [B,L,H]
        self._up: Optional[torch.Tensor] = None     # [B,L,H,r]
        self._alpha: Optional[torch.Tensor] = None  # [L]
        self._phase: Optional[torch.Tensor] = None  # [B,S] generation mask
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        a = self.cfg.activation
        if a is None or a.lower() == "none":
            return x
        name = a.lower()
        if name == "silu":
            return F.silu(x)
        if name == "gelu":
            return F.gelu(x)
        if name == "relu":
            return F.relu(x)
        if name == "tanh":
            return torch.tanh(x)
        raise ValueError(f"unknown activation {a!r}")

    def set_delta(self, down: torch.Tensor, up: Optional[torch.Tensor],
                  alpha: Optional[torch.Tensor] = None,
                  phase_mask: Optional[torch.Tensor] = None) -> None:
        self._down, self._up, self._alpha = down, up, alpha
        if self.inject_phase == "gen" and phase_mask is not None:
            self._phase = phase_mask
        else:
            self._phase = None

    def clear_delta(self) -> None:
        self._down, self._up, self._alpha, self._phase = None, None, None, None

    def _phase_gate(self, hidden: torch.Tensor) -> Optional[torch.Tensor]:
        m = self._phase
        if m is None:
            return None
        s = min(hidden.shape[1], m.shape[1])
        return m[:, :s].to(hidden.dtype)[:, :, None]

    def _apply_delta(self, hidden: torch.Tensor, slot: int) -> torch.Tensor:
        # NOTE: deliberately not named `_apply` (that is nn.Module._apply, used by .to()).
        if self._down is None:
            return hidden
        a = 1.0 if self._alpha is None else self._alpha[slot]
        if self.cfg.delta_mode == "vector":
            v = self._down[:, slot]                                 # [B,H]
            delta = a * v[:, None, :].expand_as(hidden)
        else:
            down = self._down[:, slot]                              # [B,r,H]
            up = self._up[:, slot]                                  # [B,H,r]
            if down.dtype != hidden.dtype:
                down = down.to(hidden.dtype)
            r = torch.bmm(hidden, down.transpose(1, 2))             # [B,S,r]
            r = self._act(r)
            up = up if up.dtype == r.dtype else up.to(r.dtype)
            delta = torch.bmm(r, up.transpose(1, 2)) * a            # [B,S,H]
        gate = self._phase_gate(hidden)
        if gate is not None:
            delta = delta * gate
        return hidden + delta.to(hidden.dtype)

    def register_hooks(self, backbone: nn.Module) -> None:
        self.remove_hooks()
        modules = resolve_inject_modules(backbone, getattr(self.cfg, "inject_target", "block"))
        for slot, layer_id in enumerate(self.layer_ids):
            def hook(_m, _inp, out, slot=slot):
                if self._down is None:
                    return None
                if isinstance(out, torch.Tensor):
                    return self._apply_delta(out, slot)
                if isinstance(out, tuple):
                    return (self._apply_delta(out[0], slot),) + out[1:]
                return None

            self._handles.append(modules[layer_id].register_forward_hook(hook))

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


# --------------------------------------------------------------------------- #
# contrastive / classification auxiliaries
# --------------------------------------------------------------------------- #
class SkillAuxHead(nn.Module):
    """Contrastive projector + skill classifier over the pooled memory states."""

    def __init__(self, cfg: HypernetConfig, num_classes: int = 0):
        super().__init__()
        h, d = cfg.hidden_size, cfg.ctr_dim
        self.pre_norm = nn.LayerNorm(h)
        self.feat = nn.Sequential(nn.Linear(h, d), nn.GELU())
        self.proj_head = nn.Linear(d, d)
        self.classifier = nn.Linear(d, num_classes) if num_classes and num_classes > 0 else None
        self.num_classes = int(num_classes or 0)
        # contrastive head on the *generated delta* itself: the generated down/up
        # must differ per document, not only the encoder state (bundle cosine).
        self.delta_dim = 2 * cfg.adapter_rank * h
        if cfg.delta_mode == "vector":
            self.delta_norm = None
            self.delta_proj = None
        else:
            self.delta_norm = nn.LayerNorm(self.delta_dim)
            self.delta_proj = nn.Sequential(
                nn.Linear(self.delta_dim, d), nn.GELU(), nn.Linear(d, d))

    def _flat(self, memory: torch.Tensor) -> torch.Tensor:
        return self.pre_norm(memory.mean(dim=(1, 2)).float())

    def embed(self, memory: torch.Tensor) -> torch.Tensor:
        """L2-normalised contrastive embedding [B, d]."""
        return F.normalize(self.proj_head(self.feat(self._flat(memory))), dim=-1)

    def classify(self, memory: torch.Tensor) -> Optional[torch.Tensor]:
        if self.classifier is None:
            return None
        return self.classifier(self.feat(self._flat(memory)))

    def embed_delta(self, down: torch.Tensor, up: torch.Tensor) -> Optional[torch.Tensor]:
        """Per-layer (down, up) -> L2-normalised embedding, matching the bundle metric."""
        if self.delta_proj is None:
            return None
        v = torch.cat([down.flatten(2), up.flatten(2)], dim=-1)     # [B,L,2rH]
        z = self.delta_proj(self.delta_norm(v.float())).mean(dim=1)
        return F.normalize(z, dim=-1)


class MoCoQueue(nn.Module):
    """FIFO bank of past contrastive keys (+ labels) used as InfoNCE negatives."""

    def __init__(self, dim: int, size: int):
        super().__init__()
        self.size = int(size)
        self.register_buffer("keys", torch.zeros(size, dim))
        self.register_buffer("labels", torch.full((size,), -1, dtype=torch.long))
        self.register_buffer("ptr", torch.zeros((), dtype=torch.long))
        self.register_buffer("full", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def enqueue(self, new_keys: torch.Tensor, new_labels: torch.Tensor) -> None:
        n = min(int(new_keys.shape[0]), self.size)
        if n == 0:
            return
        keys, labels = new_keys[:n], new_labels[:n]
        p = int(self.ptr)
        idx = (torch.arange(n, device=self.keys.device) + p) % self.size
        self.keys.index_copy_(0, idx, keys.detach().to(self.keys.dtype))
        self.labels.index_copy_(0, idx, labels.detach().to(self.labels.dtype))
        self.ptr.fill_((p + n) % self.size)
        if p + n >= self.size:
            self.full.fill_(True)

    def dump(self):
        if bool(self.full):
            return self.keys, self.labels
        return self.keys[: int(self.ptr)], self.labels[: int(self.ptr)]


def _nce_single(q: torch.Tensor, k_pos: torch.Tensor, neg_keys: torch.Tensor,
                neg_labels: Optional[torch.Tensor], labels: Optional[torch.Tensor],
                tau: float) -> torch.Tensor:
    """InfoNCE: anchor `q`, positive `k_pos` (same doc, other view), queue negatives."""
    pos = (q * k_pos).sum(-1, keepdim=True) / tau                      # [B,1]
    neg = q @ neg_keys.transpose(0, 1) / tau                           # [B,Q]
    if neg_labels is not None and labels is not None and neg.numel() > 0:
        same = (neg_labels[None, :] >= 0) & (neg_labels[None, :] == labels[:, None])
        neg = neg.masked_fill(same, -1.0e9)   # never push views of the same skill apart
    logits = torch.cat([pos, neg], dim=1)
    return (torch.logsumexp(logits, dim=1) - pos.squeeze(1)).mean()


def info_nce(z1: torch.Tensor, z2: torch.Tensor, queue: MoCoQueue, labels: torch.Tensor,
             tau: float, include_batch_neg: bool = True) -> torch.Tensor:
    """Symmetric InfoNCE against in-batch samples + the MoCo queue."""
    q_keys, q_labels = queue.dump()
    parts = []
    for anchor, positive in ((z1, z2), (z2, z1)):
        neg_keys, neg_labels = q_keys, q_labels
        if include_batch_neg:
            neg_keys = torch.cat([positive, neg_keys], dim=0) if neg_keys.numel() else positive
            neg_labels = torch.cat([labels, neg_labels], dim=0) if neg_labels.numel() else labels
        neg_keys = F.normalize(neg_keys.float(), dim=-1)
        if neg_keys.numel() == 0:
            continue
        parts.append(_nce_single(anchor, positive, neg_keys, neg_labels, labels, tau))
    if not parts:
        return torch.zeros((), device=z1.device)
    return sum(parts) / len(parts)


# --------------------------------------------------------------------------- #
# top-level module
# --------------------------------------------------------------------------- #
class StepOutput:
    """Forward result: LM output plus the auxiliary terms."""

    __slots__ = ("loss", "logits", "task_loss", "ctr_loss", "cls_loss",
                 "cls_logits", "labels", "memory", "memory2", "decor_loss")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k, None))


class SkillToPostBlockDelta(nn.Module):
    """Frozen backbone + skill encoder + generator + post-block injector."""

    def __init__(
        self,
        backbone: nn.Module,
        cfg: HypernetConfig,
        layers: Optional[Sequence[int]] = None,
        num_classes: int = 0,
    ):
        super().__init__()
        self.cfg = cfg
        if layers is None:
            layers = list(range(cfg.num_layers))
        self.layer_ids = list(layers)

        for p in backbone.parameters():
            p.requires_grad_(False)
        self.backbone = backbone
        self.delta_enabled = True

        # per-layer steering strength (learnable), init = output_scale
        self.layer_alpha = nn.Parameter(torch.full((cfg.num_layers,), float(cfg.output_scale)))

        self.encoder = SkillMemoryEncoder(cfg, self.layer_ids)
        self.generator = PostBlockGenerator(cfg, n_grid=len(self.layer_ids))
        self.steerer = PostBlockSteerer(cfg, self.layer_ids)
        self.steerer.register_hooks(self.backbone)
        self.num_classes = int(num_classes or cfg.num_classes)
        self.aux = SkillAuxHead(cfg, self.num_classes)
        self.queue = MoCoQueue(cfg.ctr_dim, cfg.queue_size)
        self.delta_queue = MoCoQueue(cfg.ctr_dim, cfg.queue_size)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def encode_skill(self, skill_ids: torch.Tensor, skill_mask: torch.Tensor,
                     skill_seg: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.encoder(self.backbone, skill_ids, skill_mask, skill_seg)

    def generate_delta(
        self, skill_ids: torch.Tensor, skill_mask: torch.Tensor,
        skill_seg: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        return self.delta_from_memory(
            self.encode_skill(skill_ids, skill_mask, skill_seg))

    def delta_from_memory(self, memory: torch.Tensor):
        plain = self.generator(memory)                                 # [B,L',Nmem*H]
        b, n_grid = plain.shape[0], self.generator.num_layers
        if self.cfg.delta_mode == "vector":
            v = plain.view(b, n_grid, self.cfg.hidden_size)
            return v, None, self.layer_alpha
        r, h = self.cfg.adapter_rank, self.cfg.hidden_size
        half = r * h
        down = plain[..., :half].reshape(b, n_grid, r, h)
        up = plain[..., half:].reshape(b, n_grid, h, r)
        mode = getattr(self.cfg, "delta_norm", "row")
        if mode == "none":
            pass                                   # raw output: keeps the doc-specific scale
        elif mode == "global":
            # one scalar per *document*; magnitudes are bounded like the row-norm
            # convention (mean row norm ~ 1) but the per-document direction survives,
            # so the exported bundle no longer collapses to a single shared offset.
            scale = math.sqrt(n_grid * r)
            dn = down.pow(2).flatten(1).sum(1).sqrt().view(b, 1, 1, 1) + 1e-6
            un = up.pow(2).flatten(1).sum(1).sqrt().view(b, 1, 1, 1) + 1e-6
            down = down * (scale / dn)
            up = up * (scale / un)
        else:
            # unit-L2 rows => |delta| ~ alpha * |h| regardless of weight growth
            down = down / (down.pow(2).sum(dim=-1, keepdim=True).sqrt() + 1e-6)
            up = up / (up.pow(2).sum(dim=-1, keepdim=True).sqrt() + 1e-6)
        return down, up, self.layer_alpha

    def auxiliary(self, batch, memory: Optional[torch.Tensor] = None,
                  with_queue: bool = True, memory2: Optional[torch.Tensor] = None,
                  delta=None, delta2=None):
        """Contrastive + classification terms for one micro-batch.

        Two InfoNCE terms: one on the pooled memory state (encoder specificity) and
        one on the generated (down, up) delta (output specificity, which is the
        quantity the exported bundle similarity measures).
        """
        labels = batch.get("skill_label")
        mem1 = memory if memory is not None else self.encode_skill(
            batch["skill_ids"], batch["skill_attention_mask"], batch.get("skill_seg"))
        z1 = self.aux.embed(mem1)
        cls_logits = self.aux.classify(mem1)
        ctr, ctr_delta = None, None
        if batch.get("skill2_ids") is not None:
            if memory2 is None:
                memory2 = self.encode_skill(batch["skill2_ids"],
                                            batch["skill2_attention_mask"],
                                            batch.get("skill2_seg"))
            mem2 = memory2
            z2 = self.aux.embed(mem2)
            if labels is None:
                labels = torch.full((z1.shape[0],), -1, dtype=torch.long, device=z1.device)
            ctr = info_nce(z1, z2, self.queue, labels, tau=self.cfg.ctr_tau)
            if with_queue:
                self.queue.enqueue(z2.detach(), labels.detach())
            if delta is not None and delta2 is not None and delta[0] is not None:
                zd1 = self.aux.embed_delta(delta[0], delta[1])
                zd2 = self.aux.embed_delta(delta2[0], delta2[1])
                if zd1 is not None and zd2 is not None:
                    ctr_delta = info_nce(zd1, zd2, self.delta_queue, labels,
                                         tau=self.cfg.ctr_tau)
                    if with_queue:
                        self.delta_queue.enqueue(zd2.detach(), labels.detach())
        else:
            mem2 = memory2
        if ctr is not None:
            ctr = ctr if ctr_delta is None else ctr + ctr_delta
        cls_loss = None
        if cls_logits is not None and labels is not None:
            keep = labels >= 0
            if bool(keep.any()):
                cls_loss = F.cross_entropy(cls_logits[keep].float(), labels[keep])
        decor = None
        if delta is not None and delta[0] is not None and self.cfg.decor_weight:
            decor = self._delta_cosine_penalty(delta[0], delta[1], labels)
        return {"ctr_loss": ctr, "cls_loss": cls_loss, "cls_logits": cls_logits,
                "memory": mem1, "memory2": mem2, "decor_loss": decor}

    @staticmethod
    def _delta_cosine_penalty(down: torch.Tensor, up: torch.Tensor,
                              labels: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Mean pairwise cosine of the *exported* per-document offset vectors.

        The exported bundle is `concat_l(down_l, up_l)` flattened, and the acceptance
        criterion is its within-group cosine (< 0.95); this term optimises exactly
        that quantity inside a micro-batch. Pairs with the same skill label are
        excluded (same document seen twice should stay close).
        """
        v = torch.cat([down.flatten(2), up.flatten(2)], dim=-1)   # [B, L, 2rH]
        z = F.normalize(v.float().reshape(v.shape[0], -1), dim=-1)
        b = z.shape[0]
        if b < 2:
            return None
        sim = z @ z.T
        off = ~torch.eye(b, dtype=torch.bool, device=sim.device)
        if labels is not None:
            off = off & (labels[:, None] != labels[None, :])
        if int(off.sum()) == 0:
            return None
        return sim[off].mean()

    def forward(
        self,
        skill_ids: torch.Tensor,
        skill_attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        skill_seg: Optional[torch.Tensor] = None,
        phase_mask: Optional[torch.Tensor] = None,
        skill2_ids: Optional[torch.Tensor] = None,
        skill2_attention_mask: Optional[torch.Tensor] = None,
        skill2_seg: Optional[torch.Tensor] = None,
        skill_label: Optional[torch.Tensor] = None,
        return_aux: bool = True,
        **kwargs,
    ):
        need_mem = self.delta_enabled or return_aux
        memory = self.encode_skill(skill_ids, skill_attention_mask, skill_seg) if need_mem else None
        down, up = None, None
        if self.delta_enabled:
            down, up, alpha = self.delta_from_memory(memory)
            self.steerer.set_delta(down, up, alpha, phase_mask)
        else:
            self.steerer.clear_delta()
        try:
            out = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
                **kwargs,
            )
        finally:
            self.steerer.clear_delta()

        if not return_aux:
            return StepOutput(loss=out.loss, logits=out.logits)
        memory2, delta2 = None, None
        if skill2_ids is not None:
            memory2 = self.encode_skill(skill2_ids, skill2_attention_mask, skill2_seg)
            if self.delta_enabled and down is not None:
                d2, u2, _ = self.delta_from_memory(memory2)
                delta2 = (d2, u2)
        aux = self.auxiliary(
            {"skill_ids": skill_ids, "skill_attention_mask": skill_attention_mask,
             "skill_seg": skill_seg, "skill2_ids": skill2_ids,
             "skill2_attention_mask": skill2_attention_mask, "skill2_seg": skill2_seg,
             "skill_label": skill_label},
            memory=memory, memory2=memory2,
            delta=(down, up), delta2=delta2, with_queue=self.training,
        )
        return StepOutput(
            loss=out.loss, logits=out.logits, task_loss=out.loss,
            ctr_loss=aux.get("ctr_loss"), cls_loss=aux.get("cls_loss"),
            cls_logits=aux.get("cls_logits"), labels=labels,
            memory=aux.get("memory"), memory2=aux.get("memory2"),
            decor_loss=aux.get("decor_loss"),
        )

    @torch.no_grad()
    def generate(
        self,
        skill_ids: torch.Tensor,
        skill_attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        skill_seg: Optional[torch.Tensor] = None,
        **gen_kwargs,
    ):
        if self.delta_enabled:
            down, up, alpha = self.generate_delta(skill_ids, skill_attention_mask, skill_seg)
            # no phase mask exists at generation time -> inject everywhere
            self.steerer.set_delta(down, up, alpha, None)
        else:
            self.steerer.clear_delta()
        try:
            return self.backbone.generate(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=True, **gen_kwargs
            )
        finally:
            self.steerer.clear_delta()

    # ---- (de)serialisation ------------------------------------------------ #
    def save_hypernet(self, path: str) -> None:
        payload = {
            "config": asdict(self.cfg),
            "generator": self.generator.state_dict(),
            "encoder": {k: v.detach().cpu() for k, v in self.encoder.state_dict().items()},
            "aux": self.aux.state_dict(),
            "layer_alpha": self.layer_alpha.detach().cpu(),
            "layer_ids": self.layer_ids,
        }
        lora = lora_state_dict(self.backbone)
        if lora:
            payload["lora"] = lora
        torch.save(payload, path)

    @staticmethod
    def load_hypernet_into(module: "SkillToPostBlockDelta", path: str, map_location="cpu",
                           strict: bool = False):
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        try:
            module.generator.load_state_dict(ckpt["generator"])
        except (RuntimeError, KeyError) as exc:  # shape drift (e.g. new query count)
            if strict:
                raise
            missing = module.generator.load_state_dict(ckpt["generator"], strict=False)
            print(f"[load] generator partially restored ({exc.__class__.__name__}): "
                  f"missing={len(getattr(missing, 'missing_keys', []))} "
                  f"unexpected={len(getattr(missing, 'unexpected_keys', []))}")
        if "encoder" in ckpt:
            enc_sd = ckpt["encoder"]
            cur = module.encoder.state_dict()
            compatible = {k: v for k, v in enc_sd.items()
                          if k in cur and tuple(cur[k].shape) == tuple(v.shape)}
            skipped = sorted(k for k in enc_sd if k not in compatible)
            cur.update({k: v.to(cur[k].device, cur[k].dtype) for k, v in compatible.items()})
            module.encoder.load_state_dict(cur)
            if skipped:
                print(f"[load] encoder skipped (shape change): {skipped}")
        elif "pool_queries" in ckpt:  # legacy checkpoint
            pq = ckpt["pool_queries"]
            if tuple(pq.shape) == tuple(module.encoder.pool_queries.shape):
                module.encoder.pool_queries.data.copy_(
                    pq.to(module.encoder.pool_queries.device,
                          module.encoder.pool_queries.dtype))
        if "aux" in ckpt and module.aux is not None:
            try:
                module.aux.load_state_dict(ckpt["aux"])
            except RuntimeError as exc:
                print(f"[load] aux head not restored ({exc})")
        if "layer_alpha" in ckpt:
            la = ckpt["layer_alpha"]
            if tuple(la.shape) == tuple(module.layer_alpha.shape):
                module.layer_alpha.data.copy_(
                    la.to(module.layer_alpha.device, module.layer_alpha.dtype))
        if "lora" in ckpt:
            n = load_lora_state_dict(module.backbone, ckpt["lora"])
            if n:
                print(f"[load] restored {n} LoRA tensors")
        return ckpt["config"]
