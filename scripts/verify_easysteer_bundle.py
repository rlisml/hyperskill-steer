"""Verify a steering bundle against the EasySteer `steerling_adapter` plugin.

Loads the bundle through the plugin's own `load_from_path` (the exact code path
the vLLM fork uses) and checks that `_transform` reproduces
`alpha * up(act(down @ h))` from the safetensors file, for every layer.

Run inside the EasySteer vLLM environment (vLLM fork):
    python scripts/verify_easysteer_bundle.py --bundle bundles/skill0
"""

import argparse
import json
import os
import sys

import torch
from safetensors.torch import load_file


class StubConfig:
    """Minimal stand-in for the fork's steer-vector config."""

    def __init__(self, dtype=torch.float32):
        self.adapter_dtype = dtype


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--layers", type=int, default=3, help="how many layers to check")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import easysteer_parity_plugin  # noqa: F401  (registers the algorithm)
    from easysteer_parity_plugin.steerling_adapter import SteerlingAdapterAlgorithm

    with open(os.path.join(args.bundle, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    print("manifest:", {k: manifest[k] for k in
                        ("kind", "rank", "alpha", "use_silu", "n_layers", "hidden_size")})

    payload = SteerlingAdapterAlgorithm.load_from_path(
        args.bundle, "cpu", config=StubConfig()
    )
    layer_payloads = payload["layer_payloads"]
    print(f"plugin loaded {len(layer_payloads)} layer payloads")

    algo = SteerlingAdapterAlgorithm()
    tensors = load_file(os.path.join(args.bundle, "adapter.safetensors"))
    g = torch.Generator().manual_seed(args.seed)
    h = torch.randn(16, manifest["hidden_size"], generator=g, dtype=torch.float32)

    worst = 0.0
    for lid in sorted(layer_payloads)[: args.layers]:
        p = layer_payloads[lid]
        out = algo._transform(h, dict(p, scale_factor=1.0))
        plugin_delta = out - h
        ref_r = h @ p["down"].T
        if p["use_silu"]:
            ref_r = torch.nn.functional.silu(ref_r)
        ref_delta = (ref_r @ p["up"].T) * p["alpha"]
        d = float((plugin_delta - ref_delta).abs().max())
        worst = max(worst, d)
        print(f"  layer {lid}: max|plugin_delta - reference| = {d:.3e}")
        # also check against the raw safetensors entry
        raw = tensors[f"layer{lid}.up"]
        assert torch.equal(raw, p["up"].to(raw.dtype)), f"layer {lid} up mismatch"

    print(f"VERIFY_OK worst={worst:.3e}")
    assert worst < 1e-5, worst
    return 0


if __name__ == "__main__":
    sys.exit(main())
