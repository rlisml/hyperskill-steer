"""Two-stage training for the LatentSkill-aligned steering hypernetwork.

Faithful to `latentskill/training/train_compiler.py` + the released SFT launch
script (lr 1e-5, warmup 400, AdamW wd 0.01, grad-clip 1.0, bs 1 x accum 8,
ctx/conv 4096, gradient checkpointing, linear schedule).  Only the entry point
is re-implemented (argparse instead of hydra, because omegaconf/hydra are not
installed in the `steerling` env and it must not be modified).

Stages
------
pretrain : LatentSkill `SkillPretrainCollator` (RECON / COMP) over skill docs
sft      : LatentSkill `SkillInstructionCollator` over (skill doc, trajectory)
"""

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import torch

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from ls_align.config import LS_TRAIN_DEFAULTS, build_ls_cfg, mem_tokens_for_steer  # noqa: E402
from ls_align.data import (  # noqa: E402
    build_pretrain_data,
    build_sft_data,
    make_loaders,
    FiniteLoader,
)
from ls_align.model import (  # noqa: E402
    LSSteerHypernet,
    build_backbone,
    build_tokenizer,
    load_mem_tokens,
    load_metalora,
    make_hypernet_config,
)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

LS_CKPT_ROOT = os.path.join(os.environ.get("LATENTSKILL_ROOT", "./LatentSkill"),
                            "checkpoints/latentskill_sft_qwen3_8b")


_RANK = 0


def log(msg: str, path: Optional[str] = None):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    if path and _RANK == 0:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def parse_args(argv=None):
    d = LS_TRAIN_DEFAULTS
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--stage", choices=["pretrain", "sft"], default="sft")
    p.add_argument("--mode", choices=["train", "eval"], default="train")
    p.add_argument("--model-path", default="Qwen3-8B")
    p.add_argument("--data-root", default="data/latentskill")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--init-from", default=None, help="hypernet.pt to warm-start from")
    p.add_argument("--metalora-ckpt", default=os.path.join(LS_CKPT_ROOT, "checkpoint-epoch-10"))
    p.add_argument("--mem-init", choices=["zeros", "slice"], default="zeros",
                   help="mem_tokens init: zeros (LatentSkill reset) or first-2r rows "
                        "of the released 148-token mem_tokens.pt")
    p.add_argument("--init-generator", default=None,
                   help="optional metanetwork.pth for partial warm start "
                        "(loads every tensor whose shape matches; token_pe skipped)")
    # --- hypernet ---
    p.add_argument("--rank", type=int, default=d["lora_r"])
    p.add_argument("--metalora-r", type=int, default=d["metalora_r"])
    p.add_argument("--gen-layers", type=int, default=d["gen_num_layers"])
    p.add_argument("--gen-nhead", type=int, default=d["gen_nhead"])
    p.add_argument("--gen-ff", type=int, default=d["gen_ff"])
    p.add_argument("--hypernet-scale", type=float, default=d["hypernet_scale"])
    p.add_argument("--steer-alpha", type=float, default=1e-2,
                   help="scalar alpha of h' = h + alpha * up(silu(down h))")
    p.add_argument("--inject-phase", choices=["all", "gen"], default="all")
    p.add_argument("--sft-thinking", type=int, default=0)
    # --- S1 (step 13): export-space SupCon + doc-level sampling ------------- #
    p.add_argument("--delta-ctr-weight", type=float, default=0.0,
                   help="S1: SupCon weight on the flattened export-space delta "
                        "(0 = off; allocates nothing, control arm stays bitwise "
                        "identical to the pre-S1 path)")
    p.add_argument("--delta-ctr-dim", type=int, default=256,
                   help="S1: projection-head output dim")
    p.add_argument("--delta-ctr-tau", type=float, default=0.1,
                   help="S1: SupCon temperature")
    p.add_argument("--min-docs", type=int, default=0,
                   help="S1: doc-level batch sampler, distinct skill documents "
                        "guaranteed per micro-batch (<=1 disables)")
    # --- optimisation (LatentSkill defaults) ---
    p.add_argument("--lr", type=float, default=d["learning_rate"])
    p.add_argument("--weight-decay", type=float, default=d["weight_decay"])
    p.add_argument("--warmup", type=int, default=d["warmup_steps"])
    p.add_argument("--grad-clip", type=float, default=d["grad_clip_norm"])
    p.add_argument("--epochs", type=int, default=d["num_epochs"])
    p.add_argument("--bs", type=int, default=d["train_batch_size"])
    p.add_argument("--eval-bs", type=int, default=d["eval_batch_size"])
    p.add_argument("--accum", type=int, default=d["gradient_accumulation_steps"])
    p.add_argument("--micro-steps", type=int, default=0,
                   help="cap on the number of micro-batches (0 = full epoch(s))")
    # --- data ---
    p.add_argument("--context-max-length", type=int, default=d["context_max_length"])
    p.add_argument("--conversation-max-length", type=int, default=d["conversation_max_length"])
    p.add_argument("--pretrain-train-texts", type=int, default=20000)
    p.add_argument("--pretrain-val-texts", type=int, default=400)
    p.add_argument("--val-size", type=int, default=d["skill_ift_val_size"])
    p.add_argument("--num-workers", type=int, default=d["num_workers"])
    # --- runtime ---
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--grad-ckpt", type=int, default=1)
    p.add_argument("--train-metalora", type=int, default=0,
                   help="1 = plan B (fine-tune MetaLoRA); 0 = plan A (frozen)")
    p.add_argument("--train-mem-tokens", type=int, default=1)
    p.add_argument("--seed", type=int, default=d["seed"])
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=0)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--eval-batches", type=int, default=0)
    p.add_argument("--eval-base", type=int, default=1,
                   help="also report the no-delta backbone baseline during eval")
    p.add_argument("--eval-wrong-skill", type=int, default=0)
    p.add_argument("--tag", default="")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
def build_model(args, device):
    dtype = torch.float32 if args.precision == "fp32" else torch.bfloat16
    tok = build_tokenizer(args.model_path)
    num_mem = mem_tokens_for_steer(args.rank)
    backbone = build_backbone(args.model_path, num_mem, dtype=dtype)
    backbone.resize_token_embeddings(len(tok))

    cfg = build_ls_cfg(
        num_layers=int(backbone.config.num_hidden_layers),
        hidden_size=int(backbone.config.hidden_size),
        lora_r=args.rank,
        metalora_r=args.metalora_r,
        num_mem_token=num_mem,
        gen_num_layers=args.gen_layers,
        gen_nhead=args.gen_nhead,
        gen_ff=args.gen_ff,
        hypernet_scale=args.hypernet_scale,
    )
    cfg.data.context_max_length = args.context_max_length
    cfg.data.conversation_max_length = args.conversation_max_length
    cfg.data.train_batch_size = args.bs
    cfg.data.eval_batch_size = args.eval_bs
    cfg.data.skill_ift_val_size = args.val_size
    cfg.run.seed = args.seed
    cfg.run.gradient_accumulation_steps = args.accum
    cfg.run.use_gradient_checkpoint = bool(args.grad_ckpt)

    from latentskill.utils.freeze import freeze_backbone_except_memory
    freeze_backbone_except_memory(backbone)

    # --- memory tokens -------------------------------------------------- #
    if args.mem_init == "slice":
        mt = load_mem_tokens(os.path.join(args.metalora_ckpt, "mem_tokens.pt"))
        assert mt.shape[0] >= num_mem, f"mem_tokens {mt.shape} < {num_mem}"
        with torch.no_grad():
            backbone.model.mem_tokens.copy_(mt[:num_mem].to(backbone.model.mem_tokens.dtype))
    else:
        backbone.reset_mem_tokens()

    hcfg = make_hypernet_config(cfg.num_layers, cfg.hidden_size, args.rank,
                                inject_phase=args.inject_phase)
    model = LSSteerHypernet(backbone, cfg, hcfg, steer_alpha=args.steer_alpha)

    if args.init_generator:
        missing = model.load_generator_partial(args.init_generator)
        log(f"[init] generator warm start from {args.init_generator}; skipped={missing}")

    if args.init_from:
        _p = model.load_hypernet(args.init_from)
        # inherit steer_alpha from the checkpoint (an eval-only run must use the
        # alpha the model was trained with, unless explicitly overridden)
        if float(_p.get("steer_alpha", model.steer_alpha)) != model.steer_alpha:
            model.steer_alpha = float(_p["steer_alpha"])
        log(f"[init] hypernet warm start from {args.init_from} "
            f"(steer_alpha={model.steer_alpha})")

    model.to(device)
    model.register_hooks()
    if not args.train_mem_tokens:
        backbone.model.mem_tokens.requires_grad_(False)

    metalora = load_metalora(
        os.path.join(args.metalora_ckpt, "metalora.pth"),
        device=device, trainable=bool(args.train_metalora), dtype=dtype,
    )
    return model, tok, cfg, hcfg, metalora


def _load_generator_partial(self, path: str):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    own = self.generator.state_dict()
    keep, skip = {}, []
    for k, v in sd.items():
        if k in own and own[k].shape == v.shape:
            keep[k] = v
        else:
            skip.append(k)
    self.generator.load_state_dict({**own, **keep})
    return skip




def _save_hypernet(self, path: str, extra: Optional[Dict[str, Any]] = None,
                   metalora=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    c = self.cfg
    payload = {
        "generator": self.generator.state_dict(),
        "mem_tokens": self.backbone.model.mem_tokens.detach().cpu(),
        "cfg_args": dict(
            num_layers=c.num_layers, hidden_size=c.hidden_size,
            lora_r=c.model.lora_r, metalora_r=c.model.metalora_r,
            num_mem_token=c.num_mem_token,
            gen_num_layers=c.hypernetwork.transformer_cfg.num_layers,
            gen_nhead=c.hypernetwork.transformer_cfg.encoder_cfg["nhead"],
            gen_ff=c.hypernetwork.transformer_cfg.encoder_cfg["dim_feedforward"],
            hypernet_scale=c.hypernetwork.transformer_cfg.scale,
        ),
        "hcfg": asdict(self.hcfg),
        "steer_alpha": self.steer_alpha,
        "layer_ids": list(self.layer_ids),
        "extra": extra or {},
    }
    if metalora is not None:
        payload["metalora"] = metalora
    torch.save(payload, path)


def _load_hypernet(self, path: str):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    self.generator.load_state_dict(payload["generator"])
    mt = payload.get("mem_tokens")
    if mt is not None:
        with torch.no_grad():
            self.backbone.model.mem_tokens.copy_(mt.to(self.backbone.model.mem_tokens.dtype))
    return payload




# --------------------------------------------------------------------------- #
def build_doc_bank(tok, data_root: str, device, max_len: int):
    """The 9 unique skill docs encoded once (for wrong/no-skill eval contrasts)."""
    from ls_align.export import unique_contexts
    from ls_align.data import doc_label_index
    ctxs = list(doc_label_index(data_root).keys())
    enc = tok(ctxs, max_length=max_len, truncation=True, return_tensors="pt",
              padding="max_length")
    return {"ids": enc["input_ids"].to(device),
            "mask": enc["attention_mask"].to(device)}


def move_batch(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()
            if torch.is_tensor(v)}


@torch.no_grad()
def evaluate(model, loader, device, metalora, args, max_batches=0, use_delta=True,
             wrong_skill=False, no_skill=False, doc_bank=None):
    model.eval()
    tot_loss, tot_tok, tot_correct, n = 0.0, 0, 0, 0
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=(args.precision == "bf16"))
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        batch = move_batch(batch, device)
        labels = batch["labels"]
        if no_skill:
            batch["evidence_ids"] = torch.zeros_like(batch["evidence_ids"])
            batch["evidence_attention_mask"] = torch.zeros_like(batch["evidence_attention_mask"])
        elif wrong_skill:
            if doc_bank is not None and "doc_label" in batch:
                # always a genuinely different document, independent of the
                # batch's label composition (roll(1) can be a no-op when a
                # batch holds one document only)
                lab = batch["doc_label"].clamp_min(0)
                wrong = (lab + 1) % doc_bank["ids"].shape[0]
                batch["evidence_ids"] = doc_bank["ids"][wrong]
                batch["evidence_attention_mask"] = doc_bank["mask"][wrong]
            else:
                batch["evidence_ids"] = batch["evidence_ids"].roll(1, dims=0)
                batch["evidence_attention_mask"] = batch["evidence_attention_mask"].roll(1, dims=0)
        with amp:
            out = model(
                input_ids=batch["input_ids"],
                input_attention_mask=batch["input_attention_mask"],
                evidence_ids=batch["evidence_ids"],
                evidence_attention_mask=batch["evidence_attention_mask"],
                labels=labels,
                metalora=metalora,
                use_delta=use_delta,
                use_gradient_checkpoint=bool(args.grad_ckpt),
            )
        logits = out.logits[:, :-1, :].float()
        tgt = labels[:, 1:]
        mask = tgt != -100
        ntok = int(mask.sum().item())
        tot_loss += out.loss.item() * ntok
        tot_tok += ntok
        tot_correct += int((logits.argmax(-1)[mask] == tgt[mask]).sum().item())
        n += 1
    model.train()
    if tot_tok == 0:
        return {"loss": float("nan"), "tok_acc": float("nan"), "batches": n}
    return {"loss": tot_loss / tot_tok, "tok_acc": tot_correct / tot_tok, "batches": n}


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, f"{args.stage}_{args.mode}.log")
    jsonl_path = os.path.join(args.out_dir, f"{args.stage}_{args.mode}.jsonl")

    global _RANK
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    _RANK = rank
    is_main = (rank == 0)
    if world > 1:
        import datetime as _dt
        import torch.distributed as dist
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", rank=rank, world_size=world,
                                timeout=_dt.timedelta(minutes=120))
        args.gpu = local_rank

    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    log(f"[args] {json.dumps(vars(args))} world={world} rank={rank}", log_path)
    model, tok, cfg, hcfg, metalora = build_model(args, device)
    log(f"[model] mem_tokens={cfg.num_mem_token} rank={args.rank} "
        f"gen_out_dim={cfg.num_layers * cfg.num_mem_token * cfg.hidden_size} "
        f"hypernet_params={model.num_hypernet_params()/1e6:.1f}M "
        f"per_skill={cfg.num_layers * 2 * args.rank * cfg.hidden_size/1e6:.2f}M", log_path)

    if world > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                        find_unused_parameters=False, broadcast_buffers=False)
        log(f"[ddp] world={world} local_rank={local_rank}", log_path)
    else:
        ddp_model = model

    def _save(path, extra=None, metalora=None):
        if is_main:
            LSSteerHypernet.save_hypernet(model, path, extra=extra, metalora=metalora)

    def _jsonl(rec):
        if is_main:
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

    label_of = None
    if args.stage == "sft" and (args.delta_ctr_weight > 0 or args.min_docs > 1):
        from ls_align.data import doc_label_index
        label_of = doc_label_index(args.data_root)
        log(f"[s1] doc labels: {len(label_of)} unique skill contexts", log_path)
    if args.stage == "pretrain":
        train_ds, val_ds, collator = build_pretrain_data(
            tok, cfg, args.data_root, args.pretrain_train_texts, args.pretrain_val_texts)
    else:
        train_ds, val_ds, collator = build_sft_data(
            tok, cfg, args.data_root, args.val_size,
            thinking=bool(args.sft_thinking), label_of=label_of)
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    train_sampler = None
    if world > 1:
        train_sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank,
                                           shuffle=False, drop_last=False)
    batch_sampler = None
    if args.stage == "sft" and args.min_docs > 1:
        assert world == 1 and train_sampler is None, \
            "doc-level batch sampler is single-GPU only"
        from skill_data import SkillDocBatchSampler  # step-9 implementation, reused
        train_labels = [label_of[train_ds.dataset.item_list[i]["context"]]
                        for i in train_ds.indices]
        batch_sampler = SkillDocBatchSampler(train_labels, batch_size=args.bs,
                                             min_docs=args.min_docs, seed=args.seed)
        log(f"[s1] doc sampler: min_docs={args.min_docs} bs={args.bs} "
            f"batches/epoch={len(batch_sampler)}", log_path)
    if batch_sampler is not None:
        train_loader = DataLoader(train_ds, batch_sampler=batch_sampler,
                                  collate_fn=collator,
                                  num_workers=args.num_workers, pin_memory=False)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.bs,
                                  shuffle=(train_sampler is None),
                                  sampler=train_sampler, collate_fn=collator,
                                  num_workers=args.num_workers, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.eval_bs, shuffle=False,
                            collate_fn=collator, num_workers=0, pin_memory=False)
    log(f"[data] stage={args.stage} train={len(train_ds)} val={len(val_ds)}", log_path)

    if args.mode == "eval":
        bank = build_doc_bank(tok, args.data_root, device, args.context_max_length)
        res = {"with_skill": evaluate(model, val_loader, device, metalora, args,
                                      args.eval_batches)}
        if args.eval_base:
            res["base"] = evaluate(model, val_loader, device, metalora, args,
                                   args.eval_batches, use_delta=False)
        if args.eval_wrong_skill:
            res["wrong_skill"] = evaluate(model, val_loader, device, metalora, args,
                                          args.eval_batches, wrong_skill=True,
                                          doc_bank=bank)
        res["no_skill"] = evaluate(model, val_loader, device, metalora, args,
                                   args.eval_batches, no_skill=True)
        log(f"[eval] {json.dumps(res)}", log_path)
        _jsonl({"event": "eval", **res})
        return 0

    # ---------------- S1 projection head (training-only) ----------------- #
    proj = None
    if args.delta_ctr_weight > 0:
        flat_dim = cfg.num_layers * 2 * args.rank * cfg.hidden_size
        proj = torch.nn.Linear(flat_dim, args.delta_ctr_dim).to(device)
        log(f"[s1] projection head Linear({flat_dim}, {args.delta_ctr_dim}) "
            f"params={sum(p.numel() for p in proj.parameters())/1e6:.1f}M "
            f"tau={args.delta_ctr_tau}", log_path)

    # ---------------- optimizer (LatentSkill grouping) ---------------- #
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight",
                "norm1", "norm2"]
    named = list(model.generator.named_parameters())
    if args.train_mem_tokens:
        named.append(("mem_tokens", model.backbone.model.mem_tokens))
    if proj is not None:
        named.append(("s1_proj.weight", proj.weight))
        named.append(("s1_proj.bias", proj.bias))
    groups = [
        {"params": [p for n, p in named if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in named if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    if args.train_metalora:
        from latentskill.models.lora_ops import iter_trainable_tensors
        groups.append({"params": list(iter_trainable_tensors(metalora)),
                       "weight_decay": args.weight_decay})
    for g in groups:
        for p in g["params"]:
            assert p.requires_grad, "frozen parameter in an optimizer group"

    from transformers import get_linear_schedule_with_warmup
    train_len = len(FiniteLoader(train_loader, args.micro_steps))
    total_steps = args.epochs * max(1, math.ceil(train_len / max(1, args.accum)))
    optimizer = torch.optim.AdamW(groups, lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup, num_training_steps=total_steps)
    log(f"[optim] groups={[len(g['params']) for g in groups]} "
        f"micro_batches={train_len} total_opt_steps={total_steps}", log_path)

    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=(args.precision == "bf16"))
    model.train()
    step, opt_step, t0 = 0, 0, time.time()

    # step-0 sanity check: the delta must start close to (not far above) the base.
    res0 = evaluate(model, val_loader, device, metalora, args, max(1, args.eval_batches or 8))
    base0 = evaluate(model, val_loader, device, metalora, args,
                     max(1, args.eval_batches or 8), use_delta=False)
    log(f"[step0] with_skill={json.dumps(res0)} base={json.dumps(base0)}", log_path)
    _jsonl({"event": "step0", "with_skill": res0, "base": base0})
    running, seen = 0.0, 0
    ctr_running: List[float] = []
    best = float("inf")

    for epoch in range(1, args.epochs + 1):
        if hasattr(train_ds, "set_epoch"):
            train_ds.set_epoch(epoch)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for batch in FiniteLoader(train_loader, args.micro_steps):
            batch = move_batch(batch, device)
            with amp:
                out = ddp_model(
                    input_ids=batch["input_ids"],
                    input_attention_mask=batch["input_attention_mask"],
                    evidence_ids=batch["evidence_ids"],
                    evidence_attention_mask=batch["evidence_attention_mask"],
                    labels=batch["labels"],
                    metalora=metalora,
                    use_delta=True,
                    use_gradient_checkpoint=bool(args.grad_ckpt),
                )
                loss = out.loss
                ctr_val = float("nan")
                if proj is not None:
                    from ls_align.ctr import flatten_export_delta, supcon_loss
                    vec = flatten_export_delta(model.last_down, model.last_up,
                                               model.steer_alpha)
                    z = torch.nn.functional.normalize(proj(vec.float()), dim=-1)
                    ctr = supcon_loss(z, batch["doc_label"], args.delta_ctr_tau)
                    loss = loss + args.delta_ctr_weight * ctr
                    ctr_val = float(ctr.detach())
            if torch.isnan(loss) or torch.isinf(loss):
                log(f"[warn] non-finite loss at micro-step {step}; skipped", log_path)
                optimizer.zero_grad(set_to_none=True)
                step += 1
                continue
            (loss / max(1, args.accum)).backward()
            running += float(out.loss.detach())
            ctr_running.append(ctr_val)
            seen += 1
            step += 1

            if step % 50 == 0:
                with torch.no_grad():
                    from ls_align.ctr import flatten_export_delta
                    v = flatten_export_delta(model.last_down.detach(),
                                             model.last_up.detach(),
                                             model.steer_alpha).float()
                    vn = torch.nn.functional.normalize(v, dim=-1)
                    cs = vn @ vn.T
                    nb = v.shape[0]
                    iu = torch.triu_indices(nb, nb, 1)
                    bcos = float(cs[iu[0], iu[1]].mean()) if nb > 1 else 1.0
                mon = {"event": "ctr_monitor", "micro_step": step,
                       "task_loss": float(out.loss.detach()),
                       "ctr_loss": ctr_val, "delta_batch_cos": round(bcos, 6),
                       "doc_labels": batch["doc_label"].tolist()}
                log(f"[ctr] {json.dumps(mon)}", log_path)
                _jsonl(mon)

            if step % max(1, args.accum) == 0:
                if args.grad_clip and args.grad_clip > 0:
                    for g in optimizer.param_groups:
                        torch.nn.utils.clip_grad_norm_(g["params"], args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                opt_step += 1

                if args.log_every and opt_step % args.log_every == 0:
                    peak = torch.cuda.max_memory_allocated(device) / 1e9
                    ctr_mean = (sum(x for x in ctr_running if x == x)
                                / max(1, sum(1 for x in ctr_running if x == x)))
                    rec = {"event": "train", "opt_step": opt_step, "micro_step": step,
                           "loss": running / max(seen, 1), "ctr_loss": round(ctr_mean, 6),
                           "lr": scheduler.get_last_lr()[0],
                           "peak_gb": round(peak, 2),
                           "s_per_micro": round((time.time() - t0) / step, 3)}
                    log(f"[train] {json.dumps(rec)}", log_path)
                    _jsonl(rec)
                    running, seen = 0.0, 0
                    ctr_running = []

                if args.eval_every and opt_step % args.eval_every == 0:
                    res = evaluate(model, val_loader, device, metalora, args,
                                   args.eval_batches)
                    log(f"[eval@{opt_step}] {json.dumps(res)}", log_path)
                    _jsonl({"event": "eval", "opt_step": opt_step, **res})
                    if res["loss"] < best:
                        best = res["loss"]
                        _save(os.path.join(args.out_dir, "best.pt"),
                                            extra={"opt_step": opt_step, "loss": res["loss"]})

                if args.save_every and opt_step % args.save_every == 0:
                    _save(os.path.join(args.out_dir, f"step{opt_step}.pt"),
                                        extra={"opt_step": opt_step})
        # end of epoch
        _save(os.path.join(args.out_dir, f"epoch{epoch}.pt"),
                            extra={"opt_step": opt_step, "epoch": epoch})

    _save(os.path.join(args.out_dir, "hypernet.pt"),
                        extra={"opt_step": opt_step,
                               "s1": {"delta_ctr_weight": args.delta_ctr_weight,
                                      "delta_ctr_dim": args.delta_ctr_dim,
                                      "delta_ctr_tau": args.delta_ctr_tau,
                                      "min_docs": args.min_docs,
                                      "proj_head": None if proj is None else {
                                          k: v.detach().cpu()
                                          for k, v in proj.state_dict().items()}}},
                        metalora=metalora if args.train_metalora else None)
    res = evaluate(model, val_loader, device, metalora, args, args.eval_batches)
    if args.eval_base:
        base = evaluate(model, val_loader, device, metalora, args, args.eval_batches,
                        use_delta=False)
        log(f"[final] with_skill={json.dumps(res)} base={json.dumps(base)}", log_path)
    else:
        log(f"[final] with_skill={json.dumps(res)}", log_path)
    log(f"[done] out={args.out_dir} wall={time.time()-t0:.0f}s", log_path)
    if world > 1:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
