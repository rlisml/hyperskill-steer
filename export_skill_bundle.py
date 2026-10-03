"""Materialize a trained skill->post-block-delta hypernetwork into an EasySteer bundle.

For one skill document, run the hypernetwork once (`generate_delta`) and write the
resulting per-layer (down, up) matrices in the sidecar-bundle schema so the
EasySteer/vLLM fast inference path consumes it unchanged:

    <out-dir>/
      manifest.json          schema/provenance/rank/alpha/use_silu/layers
      adapter.safetensors    layer{i}.down (rank, hidden), layer{i}.up (hidden, rank)
      loadback_check.json    tensor + functional equality vs the hypernetwork math

The bundle `alpha` carries `HypernetConfig.output_scale`; the plugin computes
`h' = h + scale * alpha * up(act(down h))`, so serving with `--scale 1.0`
reproduces the trained delta exactly.

Skill selection: `--skill-index N` picks the N-th unique `context` from the
LatentSkill IFT data; `--skill-file path` reads a raw text file instead.

Usage:
    python export_skill_bundle.py --hypernet checkpoints/sft/hypernet.pt \
      --model-name /path/to/Qwen3-8B --data-root data/latentskill \
      --skill-index 0 --out-dir bundles/skill0_multihop
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from skill_data import load_ift_items, make_skill_encoder_inputs  # noqa: E402
from skill_hypernet import HypernetConfig, SkillToPostBlockDelta  # noqa: E402


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def unique_contexts(data_root: str):
    items, _ = load_ift_items(data_root, val_size=0)
    seen, out = set(), []
    for it in items:
        c = it["context"]
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--hypernet", required=True, help="hypernet.pt written by train_skill.py")
    p.add_argument("--model-name", required=True)
    p.add_argument("--data-root", default="data/latentskill")
    p.add_argument("--skill-index", type=int, default=None,
                   help="index into the unique IFT skill documents")
    p.add_argument("--skill-file", default=None, help="raw text file with the skill document")
    p.add_argument("--max-skill-len", type=int, default=2048)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="fp32",
                   help="backbone dtype for the forward (bundle is always fp32)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)

    # ---- skill text --------------------------------------------------------- #
    if args.skill_file:
        with open(args.skill_file, "r", encoding="utf-8") as f:
            skill_text = f.read()
        skill_id = os.path.basename(args.skill_file)
    else:
        if args.skill_index is None:
            raise SystemExit("either --skill-index or --skill-file is required")
        contexts = unique_contexts(args.data_root)
        skill_text = contexts[args.skill_index]
        skill_id = f"ift_skill_{args.skill_index}"
        print(f"[skill] {len(contexts)} unique skills; using index {args.skill_index} "
              f"({len(skill_text.split())} words)")

    # ---- model ------------------------------------------------------------- #
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    ckpt = torch.load(args.hypernet, map_location="cpu", weights_only=False)
    cfg = HypernetConfig(**ckpt["config"])
    layers = ckpt["layer_ids"]

    backbone = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16 if args.dtype == "bf16" else torch.float32
    )
    backbone.eval()
    model = SkillToPostBlockDelta(backbone, cfg, layers)
    SkillToPostBlockDelta.load_hypernet_into(model, args.hypernet)
    model.eval()

    if cfg.delta_mode != "lowrank":
        raise SystemExit(
            "only delta_mode='lowrank' exports to the steerling_lowrank_adapter bundle "
            "schema (delta_mode='vector' bundles are not supported here)"
        )

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    skill_ids, skill_mask, skill_seg = make_skill_encoder_inputs(tok, skill_text, args.max_skill_len)
    skill_ids, skill_mask, skill_seg = skill_ids.to(dev), skill_mask.to(dev), skill_seg.to(dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                         enabled=(args.dtype == "bf16" and dev == "cuda")):
        down, up, alpha = model.generate_delta(skill_ids, skill_mask, skill_seg)
    # single scalar alpha, so the trained per-layer strength must live in up
    up = up * alpha.view(1, -1, 1, 1)
    down = down[0].float().cpu().contiguous()   # [L, r, H]
    up = up[0].float().cpu().contiguous()       # [L, H, r]
    layer_ids = list(layers)
    rank, hidden = down.shape[1], down.shape[2]
    assert down.shape == (len(layer_ids), rank, hidden)
    assert up.shape == (len(layer_ids), hidden, rank)

    # ---- write the bundle --------------------------------------------------- #
    from safetensors.torch import load_file, save_file

    os.makedirs(args.out_dir, exist_ok=True)
    tensor_map = {}
    for i, lid in enumerate(layer_ids):
        tensor_map[f"layer{lid}.down"] = down[i]
        tensor_map[f"layer{lid}.up"] = up[i]
    st_path = os.path.join(args.out_dir, "adapter.safetensors")
    save_file(tensor_map, st_path)

    use_silu = bool(cfg.activation is not None and cfg.activation.lower() == "silu")
    src_sha = sha256_file(args.hypernet)
    manifest = {
        "schema_version": 1,
        "kind": "steerling_lowrank_adapter",
        "rank": int(rank),
        "alpha": 1.0,  # per-layer alpha is folded into `up`
        "use_silu": use_silu,
        "n_layers": len(layer_ids),
        "layers": layer_ids,
        "hidden_size": int(hidden),
        "storage_dtype": "float32",
        "tensor_layout": "down: (rank, hidden) as in nn.Linear(hidden->rank); "
                         "up: (hidden, rank) as in nn.Linear(rank->hidden); "
                         "delta = alpha * up(act(down @ h)), act=SiLU iff use_silu; "
                         "plugin mirrors the HF LowRankAdapter intervention",
        "model_name": args.model_name,
        "dataset_name": "latentskill_skill_ift",
        "harness_id": None,
        "harness_sha256": None,
        "source_checkpoint": os.path.abspath(args.hypernet),
        "source_checkpoint_sha256": src_sha,
        "provenance": {
            "kind": "hypernetwork_generated_postblock_delta",
            "delta_mode": cfg.delta_mode,
            "adapter_rank": cfg.adapter_rank,
            "activation": cfg.activation,
            "output_scale": cfg.output_scale,
            "gen_num_layers": cfg.gen_num_layers,
            "gen_hidden": cfg.gen_hidden,
            "gen_ff": cfg.gen_ff,
            "skill_id": skill_id,
            "skill_words": len(skill_text.split()),
            "skill_sha256": hashlib.sha256(skill_text.encode()).hexdigest(),
        },
        "converted_at": datetime.now(timezone.utc).isoformat(),
        "converted_by": "export_skill_bundle.py",
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # ---- load-back checks --------------------------------------------------- #
    reloaded = load_file(st_path)
    tensor_diff = 0.0
    for i, lid in enumerate(layer_ids):
        for which in ("down", "up"):
            src = tensor_map[f"layer{lid}.{which}"]
            back = reloaded[f"layer{lid}.{which}"]
            assert back.shape == src.shape and back.dtype == src.dtype
            tensor_diff = max(tensor_diff, (back - src).abs().max().item())

    def ref_delta(pair_down, pair_up, h):
        r = h @ pair_down.T
        if use_silu:
            r = torch.nn.functional.silu(r)
        return r @ pair_up.T

    g = torch.Generator().manual_seed(args.seed)
    h = torch.randn(32, hidden, generator=g, dtype=torch.float32)
    func_diff = 0.0
    for lid in layer_ids:
        ref = ref_delta(tensor_map[f"layer{lid}.down"], tensor_map[f"layer{lid}.up"], h)
        ours = ref_delta(reloaded[f"layer{lid}.down"], reloaded[f"layer{lid}.up"], h)
        func_diff = max(func_diff, (ours - ref).abs().max().item())

    report = {
        "bundle_dir": os.path.abspath(args.out_dir),
        "bundle_safetensors": st_path,
        "source_checkpoint": os.path.abspath(args.hypernet),
        "source_checkpoint_sha256": src_sha,
        "rank": int(rank),
        "alpha": 1.0,  # per-layer alpha is folded into `up`
        "use_silu": use_silu,
        "n_layers": len(layer_ids),
        "hidden_size": int(hidden),
        "tensor_max_abs_diff": tensor_diff,
        "tensor_check_pass": tensor_diff == 0.0,
        "functional_max_abs_diff": func_diff,
        "functional_check_pass": func_diff <= 1e-6,
        "skill_id": skill_id,
        "seed": args.seed,
    }
    with open(os.path.join(args.out_dir, "loadback_check.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    assert report["tensor_check_pass"], "tensor load-back check FAILED"
    assert report["functional_check_pass"], "functional load-back check FAILED"
    print(f"[ok] bundle written to {os.path.abspath(args.out_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
