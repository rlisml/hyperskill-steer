"""GPU smoke test for the LatentSkill-aligned steering hypernetwork.

Checks
  1. shapes / parameter counts (per-skill generated = 36 x 2 x r x H = 2.36 M)
  2. alpha=0 (no delta) reproduces the frozen backbone exactly
  3. gradients reach generator + mem_tokens only (base + MetaLoRA stay frozen)
  4. |delta h| / |h| at initialisation, for the chosen --steer-alpha
  5. gradient checkpointing does not change the loss (hook safety)
  6. wall-clock / peak memory of one real micro-batch (bs x 4096 ctx + 4096 conv)
"""

import argparse
import json
import os
import sys
import time

import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ls_align.config import build_ls_cfg, mem_tokens_for_steer  # noqa: E402
from ls_align.model import (  # noqa: E402
    LSSteerHypernet,
    build_backbone,
    build_tokenizer,
    make_hypernet_config,
    load_metalora,
)

LS_CKPT = os.path.join(os.environ.get("LATENTSKILL_ROOT", "./LatentSkill"),
                       "checkpoints/latentskill_sft_qwen3_8b/checkpoint-epoch-10")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="Qwen3-8B")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--steer-alpha", type=float, default=1e-3)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--grad-ckpt", type=int, default=1)
    p.add_argument("--timing-steps", type=int, default=3)
    p.add_argument("--skip-timing", type=int, default=0)
    p.add_argument("--amp", type=int, default=0)
    a = p.parse_args()

    dev = torch.device(f"cuda:{a.gpu}")
    dtype = torch.float32 if a.precision == "fp32" else torch.bfloat16
    torch.manual_seed(0)
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(a.amp))

    tok = build_tokenizer(a.model_path)
    num_mem = mem_tokens_for_steer(a.rank)
    backbone = build_backbone(a.model_path, num_mem, dtype=dtype, device=dev)
    backbone.resize_token_embeddings(len(tok))
    from latentskill.utils.freeze import freeze_backbone_except_memory
    freeze_backbone_except_memory(backbone)

    cfg = build_ls_cfg(
        num_layers=int(backbone.config.num_hidden_layers),
        hidden_size=int(backbone.config.hidden_size),
        lora_r=a.rank, metalora_r=128, num_mem_token=num_mem,
    )
    hcfg = make_hypernet_config(cfg.num_layers, cfg.hidden_size, a.rank)
    model = LSSteerHypernet(backbone, cfg, hcfg, steer_alpha=a.steer_alpha).to(dev)

    n_layers, H = cfg.num_layers, cfg.hidden_size
    print(f"[shapes] mem_tokens={num_mem} generator_out={n_layers*num_mem*H} "
          f"per_skill_params={n_layers*2*a.rank*H} "
          f"({n_layers*2*a.rank*H/1e6:.3f}M)")
    assert n_layers * num_mem * H == n_layers * 2 * a.rank * H
    print(f"[params] generator={sum(p.numel() for p in model.generator.parameters())/1e6:.1f}M "
          f"mem_tokens={backbone.model.mem_tokens.numel()}")

    # ---- probe hooks (registered BEFORE the steerer hooks) ---------------- #
    stats = {}

    def make_probe(lid):
        def hook(_m, _inp, out):
            st = model.steerer
            if st._down is None:
                return
            h = out if isinstance(out, torch.Tensor) else out[0]
            down, up = st._down[:, lid], st._up[:, lid]
            down = down.to(h.dtype); up = up.to(h.dtype)
            r = torch.nn.functional.silu(torch.bmm(h, down.transpose(1, 2)))
            d = torch.bmm(r, up.transpose(1, 2)) * st._alpha[lid]
            stats[lid] = (d.norm(dim=-1).mean() / (h.norm(dim=-1).mean() + 1e-9)).item()
        return hook

    for lid in (0, n_layers // 2, n_layers - 1):
        backbone.model.layers[lid].register_forward_hook(make_probe(lid))
    model.register_hooks()

    metalora = load_metalora(os.path.join(LS_CKPT, "metalora.pth"), device=dev,
                             trainable=False, dtype=dtype)

    # ---- 1. alpha=0 reproduces the backbone ------------------------------- #
    B, S = 1, 256
    ev = torch.randint(100, 5000, (B, S), device=dev)
    evm = torch.ones_like(ev)
    ids = torch.randint(100, 5000, (B, 128), device=dev)
    with torch.no_grad():
        out_base = model(input_ids=ids, input_attention_mask=torch.ones_like(ids),
                         metalora=None, use_delta=False)
        model.steer_alpha = 0.0
        out_zero = model(input_ids=ids, input_attention_mask=torch.ones_like(ids),
                         evidence_ids=ev, evidence_attention_mask=evm,
                         metalora=metalora, use_delta=True)
        model.steer_alpha = a.steer_alpha
        out_on = model(input_ids=ids, input_attention_mask=torch.ones_like(ids),
                       evidence_ids=ev, evidence_attention_mask=evm,
                       metalora=metalora, use_delta=True)
    d_zero = (out_base.logits.float() - out_zero.logits.float()).abs().max().item()
    d_on = (out_base.logits.float() - out_on.logits.float()).abs().max().item()
    print(f"[alpha=0] max|dlogit| vs base = {d_zero:.3e}   (alpha={a.steer_alpha}) "
          f"max|dlogit| = {d_on:.3e}")
    print(f"[|dh|/|h|] " + json.dumps({k: round(v, 5) for k, v in sorted(stats.items())}))

    # ---- 2. gradient routing --------------------------------------------- #
    labels = ids.clone()
    labels[:, :64] = -100
    model.zero_grad(set_to_none=True)
    out = model(input_ids=ids, input_attention_mask=torch.ones_like(ids),
                evidence_ids=ev, evidence_attention_mask=evm, labels=labels,
                metalora=metalora, use_delta=True,
                use_gradient_checkpoint=bool(a.grad_ckpt))
    out.loss.backward()
    g_gen = sum(1 for p in model.generator.parameters()
                if p.grad is not None and p.grad.abs().sum() > 0)
    g_mem = backbone.model.mem_tokens.grad is not None
    g_base = sum(1 for n, p in backbone.named_parameters()
                 if p.grad is not None and "mem_tokens" not in n)
    g_meta = sum(1 for v in metalora[0]["attention"]["q"].values()
                 if torch.is_tensor(v) and v.grad is not None)
    print(f"[grads] generator tensors with grad={g_gen} mem_tokens={g_mem} "
          f"backbone tensors with grad={g_base} (expect 0) metalora with grad={g_meta} "
          f"(expect 0)")
    model.zero_grad(set_to_none=True)

    # ---- 3. gradient checkpointing parity --------------------------------- #
    if not a.skip_timing:
        with torch.no_grad():
            l_ck = model(input_ids=ids, input_attention_mask=torch.ones_like(ids),
                         evidence_ids=ev, evidence_attention_mask=evm, labels=labels,
                         metalora=metalora, use_delta=True,
                         use_gradient_checkpoint=True).loss.item()
            l_no = model(input_ids=ids, input_attention_mask=torch.ones_like(ids),
                         evidence_ids=ev, evidence_attention_mask=evm, labels=labels,
                         metalora=metalora, use_delta=True,
                         use_gradient_checkpoint=False).loss.item()
        print(f"[ckpt parity] loss ckpt={l_ck:.6f} no-ckpt={l_no:.6f} "
              f"diff={abs(l_ck-l_no):.3e}")

    # ---- 4. timing / memory of one real micro-batch ----------------------- #
    if not a.skip_timing:
        S2 = a.seq_len
        big = dict(
            input_ids=torch.randint(100, 5000, (1, S2), device=dev),
            input_attention_mask=torch.ones((1, S2), dtype=torch.long, device=dev),
            evidence_ids=torch.randint(100, 5000, (1, S2), device=dev),
            evidence_attention_mask=torch.ones((1, S2), dtype=torch.long, device=dev),
            labels=torch.randint(100, 5000, (1, S2), device=dev),
        )
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        n = a.timing_steps
        for _ in range(n):
            with amp:
                out = model(**big, metalora=metalora, use_delta=True,
                            use_gradient_checkpoint=bool(a.grad_ckpt))
            (out.loss / 8).backward()
            model.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = (time.time() - t0) / max(n, 1)
        peak = torch.cuda.max_memory_allocated(dev) / 1e9
        print(f"[timing] precision={a.precision} seq={S2} grad_ckpt={a.grad_ckpt} "
              f"{dt:.2f} s/micro-step, peak={peak:.1f} GB")
    print("[done] smoke ok")


if __name__ == "__main__":
    main()
