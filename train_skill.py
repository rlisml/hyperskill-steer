"""Two-stage training of the skill-text -> post-block activation-offset hypernetwork.

Stage 1 (--stage pretrain): skill-document pretraining (RECON / COMP).
Stage 2 (--stage sft): trajectory-supervised fine-tuning (expert-action CE).

Only the hypernetwork (generator + per-layer pooling queries) is trained; the
Qwen3-8B backbone is frozen throughout.

Examples
--------
# stage 1, smoke
python train_skill.py --stage pretrain \
  --model-name /path/to/Qwen3-8B --data-root data/latentskill \
  --output-dir checkpoints/pretrain_smoke --max-samples 512 --max-steps 20 \
  --max-skill-len 512 --max-seq-len 512 --bs 2 --grad-accum 2 \
  --gen-layers 1 --gen-ff 2048 --gen-hidden 1024

# stage 2, full data
python train_skill.py --stage sft \
  --model-name /path/to/Qwen3-8B --data-root data/latentskill \
  --output-dir checkpoints/sft --init-checkpoint checkpoints/pretrain/hypernet.pt
"""

import argparse
import json
import math
import os
import sys
import time
from typing import List

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from skill_data import (  # noqa: E402
    SkillDocBatchSampler,
    SkillPretrainDataset,
    SkillSFTDataset,
    build_cluster_labels,
    load_ift_holdout,
    load_ift_items,
    load_pretrain_texts,
    make_pretrain_collator,
    make_sft_collator,
)
from skill_hypernet import (  # noqa: E402
    HypernetConfig, SkillToPostBlockDelta, attach_lora, lora_parameters,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--stage", choices=["pretrain", "sft"], required=True)
    p.add_argument("--model-name", required=True, help="HF id or local path of the frozen backbone")
    p.add_argument("--data-root", default="data/latentskill")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--init-checkpoint", default=None, help="hypernet.pt to warm-start from")
    # hypernetwork
    p.add_argument("--adapter-rank", type=int, default=8)
    p.add_argument("--delta-mode", choices=["lowrank", "vector"], default="lowrank")
    p.add_argument("--inject-target", choices=["block", "attn", "mlp"], default="block",
                   help="M3 injection-position ablation; 'block' is the WUAS post-block "
                        "position the bundles are exported for")
    p.add_argument("--linear", action="store_true", help="disable the activation (WUAS --linear)")
    p.add_argument("--output-scale", type=float, default=0.1)
    p.add_argument("--gen-layers", type=int, default=4)
    p.add_argument("--gen-nhead", type=int, default=32)
    p.add_argument("--gen-ff", type=int, default=8192)
    p.add_argument("--gen-hidden", type=int, default=0)
    p.add_argument("--gen-dropout", type=float, default=0.0)
    p.add_argument("--steer-layers", type=str, default="all",
                   help="'all' or comma-separated decoder layer ids to steer")
    # --- step-9: encoder anti-collapse / phase-aware injection / aux losses --- #
    p.add_argument("--encoder", choices=["new", "legacy"], default="new",
                   help="'new' = doc-conditioned segment pooling (step 9); "
                        "'legacy' = step 2-8 global learned queries + flat pooling "
                        "(P-b A/B arm: isolates the encoder change)")
    p.add_argument("--num-query", type=int, default=0,
                   help="encoder query count; 0 => 4r (lowrank) / 2r (vector)")
    p.add_argument("--max-segments", type=int, default=8,
                   help="max paragraph groups for segment pooling")
    p.add_argument("--pool-mode", choices=["segment", "flat"], default="segment")
    p.add_argument("--inject-phase", choices=["all", "gen"], default="gen",
                   help="gen = inject only at generation positions (phase_mask); "
                         "matches EasySteer apply={'prompt': null, 'generation': 'all'}")
    p.add_argument("--lambda-ctr", type=float, default=0.3,
                   help="weight of the InfoNCE contrastive loss (0 disables it)")
    p.add_argument("--lambda-cls", type=float, default=0.1,
                   help="weight of the auxiliary skill-classification loss")
    p.add_argument("--decor-cos-weight", type=float, default=0.0,
                   help="weight of the direct pairwise-cosine penalty on the exported "
                        "per-document offset vector (targets the bundle-cos criterion)")
    p.add_argument("--delta-norm", choices=["row", "global", "none"], default="row",
                   help="'row' = step 2-8 per-row unit-L2 (discards the document-specific "
                        "scale, which collapses the exported bundle); 'global' = one scalar "
                        "per document; 'none' = raw output + zero-init head (keeps the "
                        "document-specific scale in the exported bundle)")
    p.add_argument("--ctr-dim", type=int, default=256)
    p.add_argument("--ctr-tau", type=float, default=0.07)
    p.add_argument("--queue-size", type=int, default=4096)
    p.add_argument("--ctr-aug", choices=["drop", "trunc", "none"], default="drop")
    p.add_argument("--view2-max-len", type=int, default=512)
    p.add_argument("--stage1-label", choices=["cluster", "none"], default="cluster")
    p.add_argument("--stage1-num-classes", type=int, default=64)
    p.add_argument("--min-docs", type=int, default=3,
                   help="document-level sampler: distinct documents guaranteed per batch "
                         "(<=1 disables the sampler)")
    p.add_argument("--lora-rank", type=int, default=0,
                   help="optional compile adapter: light LoRA on the frozen backbone, "
                         "trained jointly with the hypernetwork (0 = off)")
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    # data
    p.add_argument("--max-skill-len", type=int, default=1024)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--max-samples", type=int, default=None, help="cap stage-1 documents")
    p.add_argument("--val-samples", type=int, default=256)
    p.add_argument("--val-batches", type=int, default=16)
    p.add_argument("--ift-val-size", type=int, default=200)
    p.add_argument("--holdout-skill", type=int, default=None,
                   help="leave-one-skill-out (U3): exclude this skill index (first-appearance "
                        "order, see scripts/check_skill_order.py) from stage-2 training; the "
                        "validation set becomes that held-out skill")
    p.add_argument("--sft-thinking", action="store_true",
                   help="stage 2: render+supervise the final assistant turn through the "
                        "chat template with enable_thinking=True (matches the ALFWorld/vLLM "
                        "eval protocol); default = legacy raw-target format")
    p.add_argument("--completion-freq", type=float, default=0.5)
    p.add_argument("--completion-ratio-min", type=float, default=0.3)
    p.add_argument("--completion-ratio-max", type=float, default=0.8)
    # optim
    p.add_argument("--bs", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--grad-clip", type=float, default=1.0)
    # runtime
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--grad-checkpoint", action="store_true")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--tag", default=None)
    return p.parse_args(argv)


def resolve_steer_layers(spec: str, num_layers: int):
    if spec == "all":
        return list(range(num_layers))
    return [int(x) for x in spec.split(",") if x != ""]


def build_tokenizer(model_name, latentskill_chat_template=False):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if latentskill_chat_template:
        # exact parity with the ALFWorld/vLLM eval protocol: the repo template
        # pre-fills '<think>\n' after the assistant header (the stock Qwen3
        # template does not), so the supervised span starts *inside* the block.
        import sys as _sys

        _sys.path.insert(0, os.path.join(
            os.environ.get("LATENTSKILL_ROOT", "./LatentSkill"), "evals_vllm"))
        from qwen_chat_template import REPO_CHAT_TEMPLATE

        tok.chat_template = REPO_CHAT_TEMPLATE
    return tok


def build_backbone(model_name, dtype, grad_checkpoint):
    from transformers import AutoModelForCausalLM

    kwargs = {}
    if dtype == "bf16":
        kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.config.use_cache = False
    if grad_checkpoint:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    return model


def build_hypernet(args, backbone, model_name, num_classes=0):
    encoder_mode = getattr(args, "encoder", "new")
    pool_mode, num_query = args.pool_mode, args.num_query
    if encoder_mode == "legacy":
        pool_mode = "flat"                       # step 2-8: no paragraph grouping
        num_query = 2 * args.adapter_rank if args.delta_mode != "vector" else 1
    cfg = HypernetConfig(
        num_layers=backbone.config.num_hidden_layers,
        hidden_size=backbone.config.hidden_size,
        adapter_rank=args.adapter_rank,
        delta_mode=args.delta_mode,
        inject_target=args.inject_target,
        activation=None if args.linear else "silu",
        output_scale=args.output_scale,
        gen_num_layers=args.gen_layers,
        gen_nhead=args.gen_nhead,
        gen_ff=args.gen_ff,
        gen_hidden=args.gen_hidden,
        gen_dropout=args.gen_dropout,
        model_name=model_name,
        encoder_mode=encoder_mode,
        num_query=num_query,
        max_segments=args.max_segments,
        pool_mode=pool_mode,
        inject_phase=args.inject_phase,
        ctr_dim=args.ctr_dim,
        ctr_tau=args.ctr_tau,
        queue_size=args.queue_size,
        num_classes=num_classes,
        decor_weight=getattr(args, "decor_cos_weight", 0.0),
        delta_norm=getattr(args, "delta_norm", "row"),
    )
    layers = resolve_steer_layers(args.steer_layers, cfg.num_layers)
    model = SkillToPostBlockDelta(backbone, cfg, layers)
    if args.init_checkpoint:
        SkillToPostBlockDelta.load_hypernet_into(model, args.init_checkpoint)
        print(f"[init] warm-started hypernetwork from {args.init_checkpoint}")
    if getattr(args, "delta_norm", "row") == "none":
        model.generator.zero_output_head()
        print("[init] delta_norm=none: output head re-zeroed after warm-start")
    return model, cfg


def lr_at(step, total, base_lr, warmup):
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))


def build_loaders(args, tokenizer):
    """Returns (train_loader, val_loader, num_classes) for the requested stage."""
    num_classes = 0
    if args.stage == "pretrain":
        texts = load_pretrain_texts(args.data_root, "train", args.max_samples)
        val_texts = load_pretrain_texts(args.data_root, "val", args.val_samples)
        collate = make_pretrain_collator(tokenizer, args)
        labels = None
        if args.stage1_label == "cluster" and texts:
            labels = build_cluster_labels(texts, args.stage1_num_classes, seed=args.seed)
            num_classes = max(1, max(labels) + 1)
        train_ds = SkillPretrainDataset(texts, labels)
        val_ds = SkillPretrainDataset(val_texts)
    elif args.holdout_skill is not None:
        train_items, val_items = load_ift_holdout(
            args.data_root, args.holdout_skill, args.val_samples)
        collate = make_sft_collator(tokenizer, args)
        print(f"[holdout] stage-2 skill {args.holdout_skill} excluded from training; "
              f"val = held-out skill only ({len(val_items)} examples)")
        train_ds = SkillSFTDataset(train_items)
        val_ds = SkillSFTDataset(val_items)
        num_classes = max(1, max(train_ds.skill_index.values(), default=-1) + 1)
    else:
        train_items, val_items = load_ift_items(args.data_root, args.ift_val_size)
        collate = make_sft_collator(tokenizer, args)
        train_ds = SkillSFTDataset(train_items)
        val_ds = SkillSFTDataset(val_items)
        num_classes = max(1, max(train_ds.skill_index.values(), default=-1) + 1)
    print(f"[data] stage={args.stage} train={len(train_ds)} val={len(val_ds)} "
          f"num_classes={num_classes} inject_phase={args.inject_phase} "
          f"pool={args.pool_mode} seg={args.max_segments}")

    g = torch.Generator().manual_seed(args.seed)
    sampler = SkillDocBatchSampler(
        [train_ds[i]["label"] for i in range(len(train_ds))],
        batch_size=args.bs, min_docs=args.min_docs, seed=args.seed,
    ) if args.min_docs > 1 and len(train_ds) >= args.bs else None
    if sampler is not None:
        print(f"[sampler] document-level batches (min_docs={args.min_docs}, "
              f"bs={args.bs}, {len(sampler)} batches/epoch)")
        train_loader = DataLoader(
            train_ds,
            batch_sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=collate,
            persistent_workers=args.num_workers > 0,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.bs,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate,
            drop_last=True,
            generator=g,
            persistent_workers=args.num_workers > 0,
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.bs,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )
    return train_loader, val_loader, num_classes


@torch.no_grad()
def evaluate(model, val_loader, tokenizer, args, ablate_skill=False,
             disable_delta=False, shuffle_skill=False, max_batches=16):
    """Teacher-forced CE + token accuracy on supervised (assistant/action) tokens."""
    model.eval()
    model.generator.eval()
    model.delta_enabled = not disable_delta
    tot_loss, tot_tok, tot_correct = 0.0, 0, 0
    n_batches = 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        batch = {k: v.to(args.device) for k, v in batch.items()}
        if ablate_skill:
            batch["skill_ids"] = torch.full_like(batch["skill_ids"], tokenizer.pad_token_id)
            batch["skill_attention_mask"] = torch.zeros_like(batch["skill_attention_mask"])
        if shuffle_skill:
            # strict control: a *different* skill from the same batch
            batch["skill_ids"] = batch["skill_ids"].roll(1, dims=0)
            batch["skill_attention_mask"] = batch["skill_attention_mask"].roll(1, dims=0)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(args.dtype == "bf16")):
            out = model(
                skill_ids=batch["skill_ids"],
                skill_attention_mask=batch["skill_attention_mask"],
                skill_seg=batch.get("skill_seg"),
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                phase_mask=batch.get("phase_mask"),
                return_aux=False,
            )
        labels = batch["labels"]
        # causal shift: logits[:, t] predicts labels[:, t+1]
        shift_logits = out.logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        mask = shift_labels.ne(-100)
        if mask.sum() == 0:
            continue
        logits = shift_logits[mask].float()
        tgt = shift_labels[mask]
        tot_loss += torch.nn.functional.cross_entropy(logits, tgt).item()
        tot_correct += (logits.argmax(-1) == tgt).sum().item()
        tot_tok += int(tgt.numel())
        n_batches += 1
    model.train()
    model.generator.train()
    model.delta_enabled = True
    if tot_tok == 0:
        return {"loss": None, "tok_acc": None, "n_tokens": 0}
    return {
        "loss": tot_loss / max(1, n_batches),
        "tok_acc": tot_correct / tot_tok,
        "n_tokens": tot_tok,
    }


def log_metrics(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")

    tokenizer = build_tokenizer(args.model_name,
                                getattr(args, "sft_thinking", False))
    backbone = build_backbone(args.model_name, args.dtype, args.grad_checkpoint)
    backbone.train()  # no dropout in Qwen3; keeps gradient checkpointing active
    if getattr(args, 'lora_rank', 0) and args.lora_rank > 0:
        touched = attach_lora(backbone, args.lora_rank, args.lora_alpha,
                              dropout=args.lora_dropout)
        print(f'[lora] compile adapter r={args.lora_rank} alpha={args.lora_alpha}')
        print(f'      touched {len(touched)} projections/block')

    train_loader, val_loader, num_classes = build_loaders(args, tokenizer)
    model, cfg = build_hypernet(args, backbone, args.model_name, num_classes=num_classes)
    model.to(args.device)

    n_backbone = sum(p.numel() for p in backbone.parameters())
    n_train = model.num_trainable()
    n_lora = sum(p.numel() for p in lora_parameters(backbone))
    print('=' * 70)
    print(f'stage={args.stage}  model={args.model_name}')
    print(f'delta_mode={cfg.delta_mode} rank={cfg.adapter_rank} layers={len(model.layer_ids)}')
    print(f'encoder={cfg.encoder_mode} queries={cfg.resolved_num_query} '
          f'seg={cfg.max_segments} pool={cfg.pool_mode}')
    print(f'inject_phase={cfg.inject_phase} ctr={args.lambda_ctr} cls={args.lambda_cls} classes={num_classes}')
    print(f'trainable={n_train:,} (lora={n_lora:,})  backbone={n_backbone:,}')
    print(f'ratio={100.0 * n_train / n_backbone:.4f}%')
    print('=' * 70)
    total_steps = args.max_steps or max(
        1, int(len(train_loader) * args.epochs / max(1, args.grad_accum))
    )
    print(f"[train] micro-batches/epoch={len(train_loader)} "
          f"grad_accum={args.grad_accum} total_optim_steps={total_steps}")

    optimizer = torch.optim.AdamW(
        model.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    log_metrics(metrics_path, {"event": "start", "args": vars(args), "config": cfg.__dict__,
                               "trainable_params": n_train, "backbone_params": n_backbone,
                               "total_steps": total_steps})

    step, micro = 0, 0
    t0 = time.time()
    running: List[float] = []
    running_aux: List[float] = []
    stop = False
    for epoch in range(int(math.ceil(args.epochs))):
        for batch in train_loader:
            batch = {k: v.to(args.device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(args.dtype == "bf16")):
                out = model(**batch)
                task = out.task_loss
                reg = None
                if out.ctr_loss is not None and args.lambda_ctr:
                    reg = args.lambda_ctr * out.ctr_loss
                if out.cls_loss is not None and args.lambda_cls:
                    reg = (reg + args.lambda_cls * out.cls_loss) if reg is not None else args.lambda_cls * out.cls_loss
                if out.decor_loss is not None and args.decor_cos_weight:
                    reg = (reg + args.decor_cos_weight * out.decor_loss) if reg is not None \
                        else args.decor_cos_weight * out.decor_loss
                total = task if reg is None else task + reg
                loss = total / args.grad_accum
            loss.backward()
            running.append(float(task.detach()))
            running_aux.append(0.0 if reg is None else float(reg.detach()))
            micro += 1
            if micro % args.grad_accum != 0:
                continue

            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), args.grad_clip)
            lr = lr_at(step, total_steps, args.lr, args.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_every == 0 or step == 1:
                avg = sum(running) / len(running)
                avgaux = sum(running_aux) / max(1, len(running_aux))
                running = []
                running_aux = []
                mem = (torch.cuda.max_memory_allocated() / 2**30) if torch.cuda.is_available() else 0
                print(f'[step {step}/{total_steps}] loss={avg:.4f} aux={avgaux:.4f} '
                      f"lr={lr:.2e} peak_mem={mem:.1f}GiB elapsed={time.time() - t0:.0f}s", flush=True)
                with torch.no_grad():
                    _a = model.layer_alpha.detach().float()
                log_metrics(metrics_path, {"event": "train", "step": step, "loss": avg, "aux": round(avgaux, 4),
                                           "lr": lr, "peak_mem_gib": round(mem, 2),
                                           "alpha_min": round(float(_a.min()), 5),
                                           "alpha_max": round(float(_a.max()), 5)})

            if args.eval_every and step % args.eval_every == 0:
                m = evaluate(model, val_loader, tokenizer, args, max_batches=args.val_batches)
                m0 = evaluate(model, val_loader, tokenizer, args, ablate_skill=True,
                              max_batches=args.val_batches)
                print(f"[eval @{step}] loss={m['loss']:.4f} tok_acc={m['tok_acc']:.4f} | "
                      f"no-skill loss={m0['loss']:.4f} tok_acc={m0['tok_acc']:.4f}", flush=True)
                log_metrics(metrics_path, {"event": "eval", "step": step, "val": m,
                                           "val_no_skill": m0})

            if args.save_every and step % args.save_every == 0:
                model.save_hypernet(os.path.join(args.output_dir, "hypernet.pt"))
                print(f"[save] {os.path.join(args.output_dir, 'hypernet.pt')}", flush=True)

            if step >= total_steps:
                stop = True
                break
        if stop:
            break

    model.save_hypernet(os.path.join(args.output_dir, "hypernet.pt"))
    final = evaluate(model, val_loader, tokenizer, args, max_batches=args.val_batches)
    final0 = evaluate(model, val_loader, tokenizer, args, ablate_skill=True,
                      max_batches=args.val_batches)
    final_base = evaluate(model, val_loader, tokenizer, args, disable_delta=True,
                          max_batches=args.val_batches)
    final_shuf = evaluate(model, val_loader, tokenizer, args, shuffle_skill=True,
                          max_batches=args.val_batches)
    peak = (torch.cuda.max_memory_allocated() / 2**30) if torch.cuda.is_available() else 0
    print(f"[final] loss={final['loss']:.4f} tok_acc={final['tok_acc']:.4f} | "
          f"no-skill loss={final0['loss']:.4f} tok_acc={final0['tok_acc']:.4f} | "
          f"base(no-delta) loss={final_base['loss']:.4f} "
          f"tok_acc={final_base['tok_acc']:.4f} | "
          f"wrong-skill loss={final_shuf['loss']:.4f} "
          f"tok_acc={final_shuf['tok_acc']:.4f} | peak_mem={peak:.1f}GiB")
    log_metrics(metrics_path, {"event": "final", "step": step, "val": final,
                               "val_no_skill": final0, "val_base": final_base,
                               "val_wrong_skill": final_shuf,
                               "peak_mem_gib": round(peak, 2),
                               "wallclock_s": round(time.time() - t0, 1)})
    with open(os.path.join(args.output_dir, "train_args.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "config": cfg.__dict__}, f, indent=2)
    print(f"[done] artifacts in {os.path.abspath(args.output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# __APPEND__
