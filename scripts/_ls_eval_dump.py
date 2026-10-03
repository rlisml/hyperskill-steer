"""Per-sample diagnostic on the 200-item SFT val split.

Explains why the 16-batch (training-time) eval and the 200-sample eval disagree:
groups loss / token accuracy by the skill document (evidence hash) and by
`conversationlen`, for both with-delta and no-delta forwards.
"""

import collections
import hashlib
import json
import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from ls_align.config import build_ls_cfg  # noqa: E402
from ls_align.data import build_sft_data, make_loaders  # noqa: E402
from ls_align.model import (  # noqa: E402
    LSSteerHypernet,
    build_backbone,
    build_tokenizer,
    load_metalora,
    make_hypernet_config,
)
from latentskill.utils.freeze import freeze_backbone_except_memory  # noqa: E402

CKPT = os.path.join(_REPO_ROOT, "checkpoints/ls_sft/hypernet.pt")
LS_CKPT = os.path.join(os.environ.get("LATENTSKILL_ROOT", "./LatentSkill"),
                       "checkpoints/latentskill_sft_qwen3_8b/checkpoint-epoch-10")


@torch.no_grad()
def main():
    dev = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    payload = torch.load(CKPT, map_location="cpu", weights_only=False)
    cargs = payload["cfg_args"]
    cfg = build_ls_cfg(**cargs)
    cfg.data.context_max_length = 1024
    cfg.data.conversation_max_length = 1024
    hcfg = make_hypernet_config(cfg.num_layers, cfg.hidden_size, cargs["lora_r"])

    tok = build_tokenizer("Qwen3-8B")
    backbone = build_backbone("Qwen3-8B", cargs["num_mem_token"],
                              dtype=torch.float32, device=dev)
    backbone.resize_token_embeddings(len(tok))
    freeze_backbone_except_memory(backbone)
    model = LSSteerHypernet(backbone, cfg, hcfg,
                            steer_alpha=float(payload["steer_alpha"])).to(dev)
    model.load_hypernet(CKPT)
    model.register_hooks()
    model.eval()
    metalora = load_metalora(os.path.join(LS_CKPT, "metalora.pth"), device=dev,
                             trainable=False, dtype=torch.float32)

    _, val_ds, collator = build_sft_data(tok, cfg, "data/latentskill", 200)
    _, val_loader = make_loaders(val_ds, val_ds, collator, 1, 1)

    rows = []
    for i, batch in enumerate(val_loader):
        ctx = batch["evidence"][0]
        batch = {k: v.to(dev) for k, v in batch.items() if torch.is_tensor(v)}
        labels = batch["labels"]
        skill = hashlib.sha256(ctx.encode()).hexdigest()[:8]
        ntok = int((labels != -100).sum().item())
        rec = {"i": i, "skill": skill, "ntok": ntok,
               "ctxlen": len(tok(ctx, add_special_tokens=False)["input_ids"])}
        for tag, kwargs in (("with", {"evidence_ids": batch["evidence_ids"],
                                      "evidence_attention_mask": batch["evidence_attention_mask"]}),
                            ("base", {})):
            out = model(input_ids=batch["input_ids"],
                        input_attention_mask=batch["input_attention_mask"],
                        labels=labels, metalora=metalora if tag == "with" else None,
                        use_delta=(tag == "with"), **kwargs)
            logits = out.logits[:, :-1, :].float()
            tgt = labels[:, 1:]
            m = tgt != -100
            rec[f"{tag}_loss"] = out.loss.item()
            rec[f"{tag}_acc"] = float((logits.argmax(-1)[m] == tgt[m]).float().mean())
        rows.append(rec)

    print(f"n={len(rows)}")
    for tag in ("with", "base"):
        print(f"{tag}: loss={sum(r[tag+'_loss'] for r in rows)/len(rows):.4f} "
              f"acc={sum(r[tag+'_acc'] for r in rows)/len(rows):.4f}")
    print("first 16:", {t: round(sum(r[t + '_acc'] for r in rows[:16]) / 16, 4)
                        for t in ("with", "base")})
    print("last 184:", {t: round(sum(r[t + '_acc'] for r in rows[16:]) / 184, 4)
                        for t in ("with", "base")})
    grp = collections.defaultdict(list)
    for r in rows:
        grp[r["skill"]].append(r)
    print("--- by skill document ---")
    for k, v in sorted(grp.items(), key=lambda kv: -len(kv[1])):
        print(f"  skill={k} n={len(v)} with_acc={sum(r['with_acc'] for r in v)/len(v):.4f} "
              f"base_acc={sum(r['base_acc'] for r in v)/len(v):.4f} "
              f"with_loss={sum(r['with_loss'] for r in v)/len(v):.4f} "
              f"ctx_tok={v[0]['ctxlen']}")
    json.dump(rows, open("results/ls_val_dump.json", "w"), indent=1)
    print("wrote results/ls_val_dump.json")


if __name__ == "__main__":
    main()
