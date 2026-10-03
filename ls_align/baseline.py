"""Reference numbers for the *released* LatentSkill (text skill -> LoRA weights).

Runs the genuine `latentskill.models.hypernetwork.SkillHypernetwork` on the same
data, with the same tokenizer / collator / val split and the same metric as our
steering variant, so the two output parameterisations can be compared on one
table ("ALFWorld token-level action accuracy vs LatentSkill training metric").

    python -m ls_align.baseline --gpu 3 --eval-batches 200
"""

import argparse
import json
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from ls_align.config import build_ls_cfg  # noqa: E402
from ls_align.data import build_sft_data, make_loaders  # noqa: E402
from ls_align.model import build_backbone, build_tokenizer, load_metalora  # noqa: E402
from latentskill.models.hypernetwork import SkillHypernetwork  # noqa: E402
from latentskill.utils.freeze import freeze_backbone_except_memory  # noqa: E402

LS_CKPT_ROOT = os.path.join(os.environ.get("LATENTSKILL_ROOT", "./LatentSkill"),
                            "checkpoints/latentskill_sft_qwen3_8b")


@torch.no_grad()
def evaluate(model, loader, device, metalora, max_batches=0):
    model.eval()
    tot_loss, tot_tok, tot_correct, n = 0.0, 0, 0, 0
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        labels = batch["labels"]
        out = model(
            input_ids=batch["input_ids"],
            input_attention_mask=batch["input_attention_mask"],
            evidence_ids=batch["evidence_ids"],
            evidence_attention_mask=batch["evidence_attention_mask"],
            labels=labels,
            metalora=metalora,
            use_generator=True,
        )
        logits = out.logits[:, :-1, :].float()
        tgt = labels[:, 1:]
        mask = tgt != -100
        ntok = int(mask.sum().item())
        tot_loss += out.loss.item() * ntok
        tot_tok += ntok
        tot_correct += int((logits.argmax(-1)[mask] == tgt[mask]).sum().item())
        n += 1
    return {"loss": tot_loss / max(tot_tok, 1), "tok_acc": tot_correct / max(tot_tok, 1),
            "batches": n}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="Qwen3-8B")
    p.add_argument("--data-root", default="data/latentskill")
    p.add_argument("--ckpt", default=os.path.join(LS_CKPT_ROOT, "checkpoint-epoch-10"))
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--context-max-length", type=int, default=1024)
    p.add_argument("--conversation-max-length", type=int, default=1024)
    p.add_argument("--val-size", type=int, default=200)
    p.add_argument("--eval-batches", type=int, default=0)
    p.add_argument("--eval-base", type=int, default=1)
    p.add_argument("--out", default="results/ls_baseline_sft.json")
    args = p.parse_args(argv)

    dev = torch.device(f"cuda:{args.gpu}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    tok = build_tokenizer(args.model_path)
    backbone = build_backbone(args.model_path, 148, dtype=torch.float32, device=dev)
    backbone.resize_token_embeddings(len(tok))
    freeze_backbone_except_memory(backbone)
    backbone.config.use_cache = False

    cfg = build_ls_cfg(
        num_layers=int(backbone.config.num_hidden_layers),
        hidden_size=int(backbone.config.hidden_size),
        lora_r=8, metalora_r=128, num_mem_token=148,
    )
    cfg.data.context_max_length = args.context_max_length
    cfg.data.conversation_max_length = args.conversation_max_length
    cfg.data.skill_ift_val_size = args.val_size

    output_dim = backbone.adapter_params_numel(8)
    model = SkillHypernetwork(backbone, cfg, output_dim).to(dev)
    model.generator.load_state_dict(
        torch.load(os.path.join(args.ckpt, "metanetwork.pth"), map_location="cpu",
                   weights_only=False))
    mt = torch.load(os.path.join(args.ckpt, "mem_tokens.pt"), map_location="cpu",
                    weights_only=False)
    with torch.no_grad():
        backbone.model.mem_tokens.copy_(mt)
    metalora = load_metalora(os.path.join(args.ckpt, "metalora.pth"), device=dev,
                             trainable=False, dtype=torch.float32)
    print(f"[baseline] output_dim={output_dim} ({output_dim/1e6:.2f}M per skill) "
          f"generator={sum(p.numel() for p in model.generator.parameters())/1e6:.1f}M")

    _, val_ds, collator = build_sft_data(tok, cfg, args.data_root, args.val_size)
    _, val_loader = make_loaders(val_ds, val_ds, collator, 1, 1)
    t0 = time.time()
    res = {"latentskill_lora": evaluate(model, val_loader, dev, metalora, args.eval_batches)}
    if args.eval_base:
        res["base"] = evaluate(model, val_loader, dev, metalora, args.eval_batches)
        # base: use_generator=False -> plain backbone, no LoRA, no memory tokens
        res["base"] = _eval_plain(model, val_loader, dev, args.eval_batches)
    print(json.dumps(res, indent=2))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "results": res, "wall": time.time() - t0}, f, indent=2)
    return 0


@torch.no_grad()
def _eval_plain(model, loader, device, max_batches=0):
    """No generator, no MetaLoRA: the frozen backbone alone."""
    tot_loss, tot_tok, tot_correct, n = 0.0, 0, 0, 0
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        labels = batch["labels"]
        out = model.backbone(input_ids=batch["input_ids"],
                             attention_mask=batch["input_attention_mask"],
                             labels=labels, ignore_mem_token=True)
        logits = out.logits[:, :-1, :].float()
        tgt = labels[:, 1:]
        mask = tgt != -100
        ntok = int(mask.sum().item())
        tot_loss += out.loss.item() * ntok
        tot_tok += ntok
        tot_correct += int((logits.argmax(-1)[mask] == tgt[mask]).sum().item())
        n += 1
    return {"loss": tot_loss / max(tot_tok, 1), "tok_acc": tot_correct / max(tot_tok, 1),
            "batches": n}


if __name__ == "__main__":
    raise SystemExit(main())
