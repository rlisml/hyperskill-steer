"""Export-space SupCon on the flattened bundle vector.

The contrastive target lives in the SAME space the acceptance criterion is
measured in: the exported bundle tensor flattened exactly like
`scripts/analyze_bundle_similarity.py::bundle_vecs`
(all layers' `down` raveled, then all layers' `up` raveled, with the scalar
`steer_alpha` folded into `up` exactly like `ls_align.export.delta_for_text`).
The projection head (Linear(flat_dim, dim) + L2 norm) is a training-only
device: it is NOT exported, and the memory-space direct-cosine penalty (whose
gradient vanishes as cos -> 1) is never applied.
"""

import torch


def flatten_export_delta(down: torch.Tensor, up: torch.Tensor,
                         steer_alpha: float) -> torch.Tensor:
    """down [B, L, r, H], up [B, L, H, r] -> [B, L*r*H + L*H*r].

    Layer-major ravel of down-part then up-part, with `steer_alpha` folded into
    the up-part -- the same values, in the same order, that
    `analyze_bundle_similarity.py` concatenates from the exported safetensors.
    """
    return torch.cat([down.reshape(down.shape[0], -1),
                      (up * steer_alpha).reshape(up.shape[0], -1)], dim=1)


def supcon_loss(z: torch.Tensor, labels: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """SupCon (Khosla et al. 2020) with a self-as-positive fallback.

    z: [B, d] L2-normalised; labels: [B] skill-document ids.
    Same-label pairs are positives, different-label pairs are negatives; the
    denominator excludes the anchor itself.  Anchors whose label has no partner
    in the batch (the common case here: the doc sampler guarantees >= 3
    DISTINCT documents per micro-batch) fall back to InfoNCE with the anchor's
    own unit vector as the single positive -- i.e. a pure push-apart term, so
    the loss has a healthy gradient at cos -> 1 (unlike the memory-space direct
    cosine penalty, which is second-order vanishing there).
    """
    n = z.shape[0]
    sims = z @ z.T / tau
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    pos = (labels[:, None] == labels[None, :]) & ~eye
    shift = sims.masked_fill(eye, float("-inf")).max(dim=1, keepdim=True).values.detach()
    shifted = (sims - shift).masked_fill(eye, float("-inf"))
    log_denom = torch.logsumexp(shifted, dim=1)                    # [n]
    has_pos = pos.any(dim=1)
    n_pos = pos.sum(dim=1).clamp_min(1)
    # L_i = log sum_{a != i} exp(s_ia) - mean_p(s_ip);  the shift cancels for the
    # positive mean but must be added back for the no-positive fallback.
    pos_loss = log_denom - (shifted.masked_fill(~pos, 0.0).sum(dim=1) / n_pos)
    self_loss = shift.squeeze(1) + log_denom - (1.0 / tau)          # s_ii/tau = 1/tau
    return torch.where(has_pos, pos_loss, self_loss).mean()
