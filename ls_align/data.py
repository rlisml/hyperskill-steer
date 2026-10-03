"""Data plumbing: LatentSkill's own datasets + collators, wired without hydra."""

import json
import os
import random
from typing import List

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from latentskill.data.datasets import (  # noqa: E402
    DynamicSkillPretrainDataset,
    SkillInstructionCollator,
    SkillInstructionDataset,
    SkillPretrainCollator,
)


def read_jsonl_texts(path: str, limit: int = 0, skip: int = 0) -> List[str]:
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < skip:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                texts.append(json.loads(line)["text"])
            except Exception:
                continue
            if limit and len(texts) >= limit:
                break
    return texts


class _Sub(Dataset):
    """Thin wrapper so a plain list of texts can be sliced like a HF column."""

    def __init__(self, texts: List[str]):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i]


def build_pretrain_data(tokenizer, cfg, data_root: str,
                        train_texts: int = 20000, val_texts: int = 400):
    tr_path = os.path.join(data_root, "skill_pretrain", "train.jsonl")
    va_path = os.path.join(data_root, "skill_pretrain", "val.jsonl")
    train = read_jsonl_texts(tr_path, limit=train_texts)
    val = read_jsonl_texts(va_path, limit=val_texts)
    random.Random(0).shuffle(train)
    train_ds = DynamicSkillPretrainDataset(train, tokenizer, cfg, split="train")
    val_ds = DynamicSkillPretrainDataset(val, tokenizer, cfg, split="val",
                                         force_single_skill=True)
    collator = SkillPretrainCollator(
        tokenizer=tokenizer,
        cfg=cfg,
        conversation_max_length=cfg.data.conversation_max_length,
        context_max_length=cfg.data.context_max_length,
    )
    return train_ds, val_ds, collator


def doc_label_index(data_root: str):
    """context text -> first-appearance index (the rule that fixes the bundle
    index in ls_align.export.unique_contexts / scripts/_sq_recon.py)."""
    from .export import unique_contexts
    return {c: i for i, c in enumerate(unique_contexts(data_root))}


class DocLabelCollator:
    """Wraps a collator and adds `doc_label` [B] (first-appearance skill-doc id)."""

    def __init__(self, base, label_of):
        self.base = base
        self.label_of = label_of

    def __call__(self, batch):
        out = self.base(batch)
        out["doc_label"] = torch.tensor(
            [self.label_of.get(b["evidence"], -1) for b in batch], dtype=torch.long)
        return out


def build_sft_data(tokenizer, cfg, data_root: str, val_size: int = 200,
                   thinking: bool = False, label_of=None):
    path = os.path.join(data_root, "skill_ift", "train.json")
    full = SkillInstructionDataset(
        path, use_exceed=False,
        max_context_len=cfg.data.context_max_length,
        max_conversation_len=cfg.data.conversation_max_length,
    )
    n = len(full)
    val_idx = list(range(n - val_size, n))
    train_idx = list(range(n - val_size))
    if thinking:
        from .collator_think import ThinkingSkillInstructionCollator
        collator = ThinkingSkillInstructionCollator(
            tokenizer=tokenizer,
            context_max_length=cfg.data.context_max_length,
            conversation_max_length=cfg.data.conversation_max_length,
        )
    else:
        collator = SkillInstructionCollator(
            tokenizer=tokenizer,
            context_max_length=cfg.data.context_max_length,
            conversation_max_length=cfg.data.conversation_max_length,
            cfg=cfg,
        )
    if label_of is not None:
        collator = DocLabelCollator(collator, label_of)
    return Subset(full, train_idx), Subset(full, val_idx), collator


def make_loaders(train_ds, val_ds, collator, batch_size: int, eval_batch_size: int,
                 num_workers: int = 0):
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False, collate_fn=collator,
        num_workers=num_workers, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=eval_batch_size, shuffle=False, collate_fn=collator,
        num_workers=0, pin_memory=False,
    )
    return train_loader, val_loader


class FiniteLoader:
    """Yields at most `max_batches` batches from `loader` (then StopIteration)."""

    def __init__(self, loader, max_batches: int = 0):
        self.loader = loader
        self.max_batches = int(max_batches or 0)

    def __iter__(self):
        for i, batch in enumerate(self.loader):
            if self.max_batches and i >= self.max_batches:
                return
            yield batch

    def __len__(self):
        if self.max_batches:
            return min(self.max_batches, len(self.loader))
        return len(self.loader)
