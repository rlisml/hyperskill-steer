"""The `steerling_adapter` steering algorithm: post-block low-rank adapter.

    delta = alpha * W_up( act(W_down(h)) ),   act = SiLU iff use_silu else identity
    h' = h + scale_factor * delta

Correspondence with the HF intervention (the frozen low-rank adapter
parameterization of WUAS):
  - `down` is nn.Linear(hidden->rank, bias=False)  => down(h)  = h @ down.T
  - `up`   is nn.Linear(rank->hidden, bias=False)  => up(x)    = x @ up.T
  - alpha multiplies the delta AFTER `up`
  - applied to the block output on the residual stream, all decoder layers,
    all token positions of both phases (ApplySpec(prompt="all",
    generation="all") resolves to every row; the fork's delta-form write-back
    adds the delta to the stream exactly like the HF functional add)
  - scale_factor == 0.0 short-circuits to an exact no-op (scale=0 semantics)

Payload: one dict per layer, loaded from a sidecar bundle
(manifest.json + adapter.safetensors, produced by export_skill_bundle.py /
ls_align/export.py) via `load_from_path` — the fork-sanctioned route for
third-party formats. The bundle path travels as VectorSpec.source /
the server-level steering_config.
"""

import json
import os

import torch
import torch.nn.functional as F

from vllm.steer_vectors.algorithms.base import BaseSteerVectorAlgorithm
from vllm.steer_vectors.algorithms.factory import register_algorithm


@register_algorithm("steerling_adapter")
class SteerlingAdapterAlgorithm(BaseSteerVectorAlgorithm):
    """Post-block low-rank adapter (see module docstring)."""

    @classmethod
    def load_from_path(cls, path, device, *, config, target_layers=None, **kwargs):
        """Load the Task-1 sidecar bundle directory.

        Returns {"layer_payloads": {layer_idx: payload}} where each payload is
        {"down", "up", "use_silu", "alpha"} with tensors on `device` in the
        engine's resolved steer-vector dtype (config.adapter_dtype).
        """
        from safetensors.torch import load_file

        manifest_path = os.path.join(path, "manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)
        if manifest.get("kind") != "steerling_lowrank_adapter":
            raise ValueError(
                f"{manifest_path}: expected kind 'steerling_lowrank_adapter', "
                f"got {manifest.get('kind')!r}"
            )
        use_silu = bool(manifest["use_silu"])
        alpha = float(manifest["alpha"])
        layers = list(manifest["layers"])
        if target_layers is not None:
            layers = [l for l in layers if l in target_layers]
        tensors = load_file(os.path.join(path, "adapter.safetensors"))
        dtype = config.adapter_dtype
        layer_payloads = {}
        for layer in layers:
            layer_payloads[int(layer)] = {
                "down": tensors[f"layer{layer}.down"].to(device=device, dtype=dtype),
                "up": tensors[f"layer{layer}.up"].to(device=device, dtype=dtype),
                "use_silu": use_silu,
                "alpha": alpha,
            }
        if not layer_payloads:
            raise ValueError(
                f"bundle {path} targets no requested layer "
                f"(manifest layers {manifest['layers']}, target_layers "
                f"{target_layers})"
            )
        return {"layer_payloads": layer_payloads}

    def _transform(self, hidden_state: torch.Tensor, params: dict) -> torch.Tensor:
        down = params["down"]
        up = params["up"]
        # Tensors are already on the hidden states' device and dtype (loaded
        # in config.adapter_dtype = the model dtype); .to() is a no-op then.
        r = torch.matmul(hidden_state, down.T)      # nn.Linear(hidden->rank)
        if params.get("use_silu", True):
            r = F.silu(r)                            # default activation
        delta = torch.matmul(r, up.T)                # nn.Linear(rank->hidden)
        alpha = params.get("alpha", 1.0)
        if alpha != 1.0:
            delta = delta * alpha
        scale_factor = params.get("scale_factor", 1.0)
        if scale_factor == 0.0:
            return hidden_state                      # exact no-op (scale=0)
        return hidden_state + scale_factor * delta
