"""Export the LatentSkill-aligned hypernetwork into EasySteer bundles.

Same sidecar-bundle schema as `export_skill_bundle.py`
(`steerling_lowrank_adapter`: `manifest.json` + `adapter.safetensors` with
`layer{i}.down (rank, hidden)` / `layer{i}.up (hidden, rank)`), so the
EasySteer/vLLM `steerling_adapter` plugin consumes it unchanged.

The per-layer scalar alpha is folded into `up` (manifest alpha = 1.0), so
serving with `--scale 1.0` reproduces the trained delta exactly.

Usage:
    python -m ls_align.export --hypernet checkpoints/ls_sft/hypernet.pt \
        --out-root bundles/ls_skills --all
    python -m ls_align.export --hypernet ... --out-dir bundles/x --skill-index 4
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from typing import List, Optional

import torch

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from ls_align.config import build_ls_cfg  # noqa: E402
from ls_align.model import (  # noqa: E402
    LSSteerHypernet,
    build_backbone,
    build_tokenizer,
    load_metalora,
    make_hypernet_config,
)

LS_CKPT_ROOT = os.path.join(os.environ.get("LATENTSKILL_ROOT", "./LatentSkill"),
                            "checkpoints/latentskill_sft_qwen3_8b")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def unique_contexts(data_root: str) -> List[str]:
    path = os.path.join(data_root, "skill_ift", "train.json")
    items = json.load(open(path, encoding="utf-8"))
    seen, out = set(), []
    for it in items:
        c = it["context"]
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def write_lowrank_bundle(out_dir: str, down: torch.Tensor, up: torch.Tensor,
                         layer_ids: List[int], meta: dict) -> dict:
    """Write `steerling_lowrank_adapter` bundle; `down/up` are [L, r, H] / [L, H, r]."""
    from safetensors.torch import load_file, save_file

    os.makedirs(out_dir, exist_ok=True)
    down = down.float().cpu().contiguous()
    up = up.float().cpu().contiguous()
    L, rank, hidden = down.shape
    assert up.shape == (L, hidden, rank)

    tensor_map = {}
    for i, lid in enumerate(layer_ids):
        tensor_map[f"layer{lid}.down"] = down[i]
        tensor_map[f"layer{lid}.up"] = up[i]
    st_path = os.path.join(out_dir, "adapter.safetensors")
    save_file(tensor_map, st_path)

    manifest = {
        "schema_version": 1,
        "kind": "steerling_lowrank_adapter",
        "rank": int(rank),
        "alpha": 1.0,
        "use_silu": bool(meta.get("use_silu", True)),
        "n_layers": len(layer_ids),
        "layers": list(layer_ids),
        "hidden_size": int(hidden),
        "storage_dtype": "float32",
        "tensor_layout": ("down: (rank, hidden) as in nn.Linear(hidden->rank); "
                          "up: (hidden, rank) as in nn.Linear(rank->hidden); "
                          "delta = alpha * up(act(down @ h)), act=SiLU iff use_silu; "
                          "plugin mirrors the HF LowRankAdapter intervention"),
        "model_name": meta.get("model_name", ""),
        "dataset_name": "latentskill_skill_ift",
        "harness_id": None,
        "harness_sha256": None,
        "source_checkpoint": meta.get("source_checkpoint"),
        "source_checkpoint_sha256": meta.get("source_checkpoint_sha256"),
        "provenance": meta.get("provenance", {}),
        "converted_at": datetime.now(timezone.utc).isoformat(),
        "converted_by": "ls_align/export.py",
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # ---- load-back checks ------------------------------------------------ #
    reloaded = load_file(st_path)
    tensor_diff = 0.0
    for lid in layer_ids:
        for which in ("down", "up"):
            src, back = tensor_map[f"layer{lid}.{which}"], reloaded[f"layer{lid}.{which}"]
            assert back.shape == src.shape and back.dtype == src.dtype
            tensor_diff = max(tensor_diff, (back - src).abs().max().item())

    def ref_delta(d, u, h):
        r = h @ d.T
        if manifest["use_silu"]:
            r = torch.nn.functional.silu(r)
        return r @ u.T

    g = torch.Generator().manual_seed(0)
    h = torch.randn(32, hidden, generator=g, dtype=torch.float32)
    func_diff = 0.0
    for lid in layer_ids:
        a = ref_delta(tensor_map[f"layer{lid}.down"], tensor_map[f"layer{lid}.up"], h)
        b = ref_delta(reloaded[f"layer{lid}.down"], reloaded[f"layer{lid}.up"], h)
        func_diff = max(func_diff, (a - b).abs().max().item())

    report = {
        "bundle_dir": os.path.abspath(out_dir),
        "rank": int(rank),
        "alpha": 1.0,
        "use_silu": manifest["use_silu"],
        "n_layers": len(layer_ids),
        "hidden_size": int(hidden),
        "params_per_skill": int(2 * rank * hidden * len(layer_ids)),
        "tensor_max_abs_diff": tensor_diff,
        "tensor_check_pass": tensor_diff == 0.0,
        "functional_max_abs_diff": func_diff,
        "functional_check_pass": func_diff <= 1e-6,
        "skill_id": meta.get("provenance", {}).get("skill_id"),
    }
    with open(os.path.join(out_dir, "loadback_check.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    assert report["tensor_check_pass"], "tensor load-back check FAILED"
    assert report["functional_check_pass"], "functional load-back check FAILED"
    return report


def build_from_checkpoint(ckpt_path: str, model_path: str, device: str,
                          max_skill_len: int, metalora_ckpt: str):
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cargs = payload["cfg_args"]
    hcfg_dict = payload["hcfg"]
    cfg = build_ls_cfg(**cargs)
    hcfg = make_hypernet_config(
        cfg.num_layers, cfg.hidden_size, cargs["lora_r"],
        activation=hcfg_dict.get("activation", "silu"),
        inject_phase=hcfg_dict.get("inject_phase", "all"),
    )
    tok = build_tokenizer(model_path)
    backbone = build_backbone(model_path, cargs["num_mem_token"],
                              dtype=torch.float32, device=device)
    backbone.resize_token_embeddings(len(tok))
    from latentskill.utils.freeze import freeze_backbone_except_memory
    freeze_backbone_except_memory(backbone)
    model = LSSteerHypernet(backbone, cfg, hcfg,
                            steer_alpha=float(payload["steer_alpha"])).to(device)
    model.load_hypernet(ckpt_path)
    model.register_hooks()
    model.eval()
    metalora = load_metalora(os.path.join(metalora_ckpt, "metalora.pth"),
                             device=device, trainable=False, dtype=torch.float32)
    return model, tok, cfg, metalora, payload


@torch.no_grad()
def delta_for_text(model, tok, metalora, text: str, max_skill_len: int, device: str):
    enc = tok(text, max_length=max_skill_len, truncation=True, return_tensors="pt",
              padding="max_length")
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    down, up = model.build_delta(ids, mask, metalora)
    # fold the scalar alpha into `up` (EasySteer only supports a single alpha)
    up = up * model.steer_alpha
    return down[0], up[0]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--hypernet", required=True)
    p.add_argument("--model-path", default="Qwen3-8B")
    p.add_argument("--data-root", default="data/latentskill")
    p.add_argument("--metalora-ckpt", default=os.path.join(LS_CKPT_ROOT, "checkpoint-epoch-10"))
    p.add_argument("--out-root", default=None, help="export every skill under this root")
    p.add_argument("--out-dir", default=None, help="single-bundle output dir")
    p.add_argument("--skill-index", type=int, default=None)
    p.add_argument("--skill-file", default=None)
    p.add_argument("--max-skill-len", type=int, default=1024,
                   help="must match the training-time context_max_length")
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    dev = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model, tok, cfg, metalora, payload = build_from_checkpoint(
        args.hypernet, args.model_path, dev, args.max_skill_len, args.metalora_ckpt)

    if args.skill_file:
        skills = [(os.path.basename(args.skill_file),
                   open(args.skill_file, encoding="utf-8").read())]
    else:
        ctxs = unique_contexts(args.data_root)
        if args.skill_index is not None:
            ctxs = [ctxs[args.skill_index]]
            idxs = [args.skill_index]
        else:
            idxs = list(range(len(ctxs)))
        skills = [(f"ls_skill_{i}", c) for i, c in zip(idxs, ctxs)]

    src_sha = sha256_file(args.hypernet)
    reports = []
    for name, text in skills:
        out_dir = args.out_dir if (args.out_dir and len(skills) == 1) \
            else os.path.join(args.out_root or "bundles/ls_skills", name)
        down, up = delta_for_text(model, tok, metalora, text, args.max_skill_len, dev)
        meta = {
            "model_name": args.model_path,
            "use_silu": True,
            "source_checkpoint": os.path.abspath(args.hypernet),
            "source_checkpoint_sha256": src_sha,
            "provenance": {
                "kind": "latentskill_aligned_hypernetwork_postblock_delta",
                "delta_mode": "lowrank",
                "adapter_rank": int(cfg.model.lora_r),
                "activation": "silu",
                "steer_alpha": float(payload["steer_alpha"]),
                "hypernet_scale": float(cfg.hypernetwork.transformer_cfg.scale),
                "gen_num_layers": cfg.hypernetwork.transformer_cfg.num_layers,
                "num_mem_token": int(cfg.num_mem_token),
                "metalora_ckpt": args.metalora_ckpt,
                "skill_id": name,
                "skill_words": len(text.split()),
                "skill_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "max_skill_len": args.max_skill_len,
            },
        }
        rep = write_lowrank_bundle(out_dir, down, up, list(model.layer_ids), meta)
        rep["delta_down_norm"] = round(float(down.norm().item()), 4)
        rep["delta_up_norm"] = round(float(up.norm().item()), 6)
        reports.append(rep)
        print(json.dumps(rep))
    print(f"[export] {len(reports)} bundles, "
          f"worst functional diff={max(r['functional_max_abs_diff'] for r in reports):.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
