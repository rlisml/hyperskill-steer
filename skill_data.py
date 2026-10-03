"""Datasets, collators and samplers for the two-stage training.

Stage 1 (pretrain): LatentSkill `skill_pretrain/*.jsonl` (`{"text": ...}`),
    RECON / COMP self-supervised tasks. The skill text goes ONLY to the
    hypernetwork (`skill_ids`); the base model sees just a `<RECON>` / `<COMP>`
    marker turn and must reproduce the full skill document.
Stage 2 (sft): LatentSkill `skill_ift/train.json` (`{context, conversations}`):
    `context` (the skill document) -> hypernetwork; `conversations` -> base
    model; supervision is the assistant (expert action) tokens only.

Refactor additions:

* `phase_mask`      -- 1 on the token positions whose *prediction* is a generated
                       (assistant) token, 0 elsewhere. Consumed by
                       `PostBlockSteerer` so training injects the delta only on
                       the generation phase (matches the EasySteer
                       `apply={"prompt": null, "generation": "all"}` serving config).
* CLS + paragraph   -- every skill sequence is `[CLS] + document`, right padded,
                       with a `skill_seg` id per token (markdown `#`/`##`/`###`
                       headings delimit paragraphs). The new `SkillMemoryEncoder`
                       pools per-paragraph and aggregates with attention, and it
                       derives its pooling queries from the [CLS] state, so two
                       different documents can no longer pool to the same vector.
* `skill2_*`        -- second (augmented) view of the same document used as the
                       contrastive positive / MoCo key.
* `skill_label`     -- corpus-cluster id (stage 1) or skill index (stage 2),
                       used by the auxiliary classification loss.
* `SkillDocBatchSampler` -- guarantees >= `min_docs` distinct documents per batch.
"""

import json
import math
import os
import random
import re
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset, Sampler

MARKER_RECON = "<RECON>"
MARKER_COMP = "<COMP>"

# markdown heading used to split a skill document into paragraphs
_HEAD_RE = re.compile(r"(?m)^(?=#{1,6} )")
_WORD_RE = re.compile(r"[a-z0-9_]{2,}")

DEFAULT_MAX_SEGMENTS = 8


# --------------------------------------------------------------------------- #
# raw loaders
# --------------------------------------------------------------------------- #
def load_pretrain_texts(root: str, split: str = "train", max_samples: Optional[int] = None):
    path = os.path.join(root, "skill_pretrain", f"{split}.jsonl")
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            t = obj.get("text")
            if isinstance(t, str) and t.strip():
                texts.append(t)
            if max_samples is not None and len(texts) >= max_samples:
                break
    return texts


def unique_ift_contexts(root: str):
    """Unique skill documents in FIRST-APPEARANCE order (the bundle index order)."""
    path = os.path.join(root, "skill_ift", "train.json")
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    seen, order = set(), []
    for it in items:
        c = it["context"]
        if c not in seen:
            seen.add(c)
            order.append(c)
    return order


def load_ift_holdout(root: str, holdout_skill: int, val_size: int = 200):
    """Leave-one-skill-out split: (train without that skill, val = that skill only)."""
    path = os.path.join(root, "skill_ift", "train.json")
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    contexts = unique_ift_contexts(root)
    held = contexts[holdout_skill]
    train = [it for it in items if it["context"] != held]
    val_all = [it for it in items if it["context"] == held]
    if val_size and val_size > 0 and len(val_all) > val_size:
        step = len(val_all) / val_size
        val = [val_all[min(len(val_all) - 1, round(i * step))] for i in range(val_size)]
    else:
        val = val_all
    return train, val


def load_ift_items(root: str, val_size: int = 200):
    """Split into (train, val); validation indices strided so every skill appears."""
    path = os.path.join(root, "skill_ift", "train.json")
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    if val_size <= 0:
        return items, []
    n = len(items)
    val_size = min(val_size, n)
    val_idx = sorted({round(i * n / val_size) for i in range(val_size)})
    val_set = set(val_idx)
    train = [it for i, it in enumerate(items) if i not in val_set]
    val = [items[i] for i in val_idx]
    return train, val


# --------------------------------------------------------------------------- #
# stage-1 pseudo labels: TF-IDF clusters (no sklearn dependency)
# --------------------------------------------------------------------------- #
def build_cluster_labels(texts: Sequence[str], num_classes: int = 64, seed: int = 0,
                         n_features: int = 4096, iters: int = 15, verbose: bool = True):
    """Cluster skill documents into `num_classes` topical groups.

    Hashing TF-IDF (bit -> bucket) + k-means implemented with torch, so no extra
    dependency is needed. These labels are the target of the stage-1 auxiliary
    classification loss: they force the encoder to keep coarse document identity.
    """
    n = len(texts)
    if n == 0 or num_classes <= 1:
        return [0] * n
    import numpy as np

    k = int(min(num_classes, n))
    rows = []
    for t in texts:
        toks = _WORD_RE.findall(t.lower())
        seen_terms = set(toks)
        rows.append(seen_terms)
    df = defaultdict(int)
    for st in rows:
        for term in st:
            df[term] += 1

    feats = np.zeros((n, n_features), dtype=np.float32)
    for i, st in enumerate(rows):
        for term in st:
            b = (hash(term) & 0x7FFFFFFF) % n_features
            feats[i, b] += math.log(1.0 + n / (1.0 + df[term]))
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    feats /= norms

    x = torch.from_numpy(feats)                                  # [n, F]
    g = torch.Generator().manual_seed(seed)
    # k-means++-ish init: farthest-point seeding on a random subset
    perm = torch.randperm(n, generator=g)[: max(4 * k, min(n, 2000))]
    sub = x[perm]
    cent = [sub[int(torch.randint(len(sub), (1,), generator=g))]]
    for _ in range(k - 1):
        d = torch.stack([(torch.cdist(sub, torch.stack(cent))) .min(dim=1).values for _ in [0]])
        far = int(d.argmax())
        cent.append(sub[far])
    centroids = torch.stack(cent)
    for _ in range(iters):
        dist = torch.cdist(x, centroids)                          # [n, k]
        assign = dist.argmin(dim=1)
        for j in range(k):
            mask = assign == j
            if mask.any():
                centroids[j] = x[mask].mean(dim=0)
            else:  # revive an empty cluster on the worst-fitted point
                centroids[j] = x[int(dist.min(dim=1).values.argmax())]
    assign = torch.cdist(x, centroids).argmin(dim=1)
    labels = [int(v) for v in assign]
    if verbose:
        cnt = defaultdict(int)
        for v in labels:
            cnt[v] += 1
        print(f"[labels] stage-1 TF-IDF clusters: k={k} nonempty={len(cnt)} "
              f"sizes(min/mean/max)={min(cnt.values())}/{n // max(1, len(cnt))}/"
              f"{max(cnt.values())}")
    return labels


# --------------------------------------------------------------------------- #
# datasets
# --------------------------------------------------------------------------- #
class SkillPretrainDataset(Dataset):
    """One skill document per example; the RECON/COMP choice is made in the collator."""

    def __init__(self, texts: List[str], labels: Optional[Sequence[int]] = None):
        self.texts = texts
        self.labels = labels if labels is not None else [-100] * len(texts)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {"text": self.texts[idx], "label": int(self.labels[idx])}


class SkillSFTDataset(Dataset):
    """(skill document, trajectory conversations) pairs + the skill index label."""

    def __init__(self, items: List[Dict[str, Any]],
                 skill_index: Optional[Dict[str, int]] = None):
        self.items = items
        if skill_index is None:
            seen, skill_index = {}, {}
            for it in items:
                c = it["context"]
                if c not in seen:
                    seen[c] = len(seen)
                skill_index[c] = seen[c]
        self.skill_index = skill_index

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        return {
            "context": it["context"],
            "conversations": it["conversations"],
            "label": int(self.skill_index[it["context"]]),
        }


# --------------------------------------------------------------------------- #
# document-level batch sampler
# --------------------------------------------------------------------------- #
class SkillDocBatchSampler(Sampler):
    """Yield batches that always mix >= `min_docs` distinct skill documents.

    Without this, a stage-2 batch of size 4 frequently contains a single skill
    document, and the contrastive/classification objectives have nothing to
    contrast against (all inputs pool to nearly the same vector).
    """

    def __init__(self, labels: Sequence[int], batch_size: int, min_docs: int = 3,
                 seed: int = 0):
        self.batch_size = int(batch_size)
        self.min_docs = int(max(1, min(min_docs, batch_size)))
        self.seed = int(seed)
        self.epoch = 0
        groups: Dict[int, List[int]] = defaultdict(list)
        for i, lab in enumerate(labels):
            groups[int(lab) if int(lab) >= 0 else -i - 1].append(i)
        self.groups = dict(groups)
        self.n = len(labels)

    def __len__(self):
        return max(1, int(math.ceil(self.n / self.batch_size)))

    def __iter__(self):
        rng = random.Random(self.seed + 1000 * self.epoch)
        self.epoch += 1
        pools = {}
        for k, v in self.groups.items():
            lst = list(v)
            rng.shuffle(lst)
            pools[k] = lst
        ptr = {k: 0 for k in pools}
        keys = list(pools.keys())
        rng.shuffle(keys)
        dq = deque(keys)
        batches: List[List[int]] = []
        while len(dq) >= self.min_docs:
            batch, taken = [], []
            for _ in range(min(self.batch_size, len(dq))):
                key = dq.popleft()
                batch.append(pools[key][ptr[key]])
                ptr[key] += 1
                taken.append(key)
            while len(batch) < self.batch_size:  # fill with repeats of used docs
                cands = [k for k in taken if ptr[k] < len(pools[k])]
                if not cands:
                    break
                key = rng.choice(cands)
                batch.append(pools[key][ptr[key]])
                ptr[key] += 1
            for key in taken:
                if ptr[key] < len(pools[key]):
                    dq.append(key)
            if len(batch) < self.batch_size:
                break  # cannot complete a batch -> drop the remainder
            batches.append(batch)
        rng.shuffle(batches)
        return iter(batches)


def build_doc_sampler(dataset: Dataset, batch_size: int, min_docs: int = 3,
                      seed: int = 0, enabled: bool = True):
    labels = []
    for i in range(len(dataset)):
        ex = dataset[i]
        labels.append(int(ex.get("label", -100)) if isinstance(ex, dict) else -100)
    if not enabled or min_docs <= 1:
        return None
    return SkillDocBatchSampler(labels, batch_size, min_docs=min_docs, seed=seed)


# --------------------------------------------------------------------------- #
# tokenization helpers
# --------------------------------------------------------------------------- #
def resolve_cls_token_id(tokenizer) -> int:
    """A surrogate [CLS] id: cls > bos > sep > eos."""
    for name in ("cls_token_id", "bos_token_id", "sep_token_id"):
        v = getattr(tokenizer, name, None)
        if v is not None:
            return int(v)
    return int(tokenizer.eos_token_id or 0)


def split_paragraphs(text: str) -> List[str]:
    parts = _HEAD_RE.split(text)
    return [p for p in parts if p.strip()]


def tokenize_skill_doc(tokenizer, text: str, max_skill_len: int, cls_id: int,
                       max_segments: int = DEFAULT_MAX_SEGMENTS,
                       drop_segments: Sequence[int] = ()) -> Tuple[List[int], List[int]]:
    """`[CLS] + document` -> (ids, seg_ids). Right-padded later by the collator.

    `seg_ids`: 0 = [CLS]/preamble, 1..K-1 = markdown-heading paragraphs
    (everything past `max_segments-1` folds into the last id, which is fine
    because pooling only needs a partition). Long documents keep the *tail*
    tokens (as before).
    """
    max_segments = max(2, int(max_segments))
    ids: List[int] = [int(cls_id)]
    segs: List[int] = [0]
    for pi, part in enumerate(split_paragraphs(text)):
        seg = min(pi + 1, max_segments - 1)
        if seg in drop_segments:
            continue
        toks = tokenizer(part, add_special_tokens=False)["input_ids"]
        if not toks:
            continue
        ids.extend(toks)
        segs.extend([seg] * len(toks))
    if len(ids) == 1:  # empty after filtering
        ids.append(int(tokenizer.eos_token_id or 0))
        segs.append(0)
    budget = max(8, int(max_skill_len) - 1)
    if len(ids) - 1 > budget:  # keep the tail, never drop [CLS]
        cut = len(ids) - budget
        ids = [ids[0]] + ids[cut:]
        segs = [segs[0]] + segs[cut:]
    return ids, segs


def _pad_pair(seqs: List[List[int]], segs: List[List[int]], pad_id: int):
    """Right-pad (so [CLS] is always index 0) -> (ids, mask, seg)."""
    maxlen = max(len(s) for s in seqs)
    maxlen = max(1, maxlen)
    ids, mask, out_seg = [], [], []
    for s, g in zip(seqs, segs):
        pad = maxlen - len(s)
        ids.append(s + [pad_id] * pad)
        mask.append([1] * len(s) + [0] * pad)
        out_seg.append(g + [0] * pad)
    return (
        torch.tensor(ids, dtype=torch.long),
        torch.tensor(mask, dtype=torch.long),
        torch.tensor(out_seg, dtype=torch.long),
    )


def phase_mask_from_labels(labels: Sequence[int]) -> List[int]:
    """1 where the *next* token is generated (assistant) -> where to inject.

    Teacher forcing: `logits[:, t]` predicts `labels[:, t+1]`, so the hidden
    state that must carry the offset is the one at position `t` when token `t+1`
    belongs to the assistant reply. This reproduces EasySteer's
    `apply={"prompt": null, "generation": "all"}` at serving time.
    """
    n = len(labels)
    out = [0] * n
    for t in range(n - 1):
        if labels[t + 1] != -100:
            out[t] = 1
    return out


def _pad_decoder(seqs: List[List[int]], labels: List[List[int]],
                 phases: List[List[int]], pad_id: int):
    maxlen = max(len(s) for s in seqs)
    ids, mask, labs, ph = [], [], [], []
    for s, l, p in zip(seqs, labels, phases):
        pad = maxlen - len(s)
        ids.append(s + [pad_id] * pad)
        mask.append([1] * len(s) + [0] * pad)
        labs.append(l + [-100] * pad)
        ph.append(p + [0] * pad)
    return (
        torch.tensor(ids, dtype=torch.long),
        torch.tensor(mask, dtype=torch.long),
        torch.tensor(labs, dtype=torch.long),
        torch.tensor(ph, dtype=torch.long),
    )


_FULL_TURN_FALLBACKS = [0]


def _prompt_and_target(tokenizer, conversations: List[Dict[str, str]],
                       target_text: Optional[str], max_seq_len: int,
                       enable_thinking: bool = False, full_turn: bool = False):
    """Prompt = templated context (unsupervised); target = expert answer (supervised)."""
    if full_turn:
        full_ids = list(tokenizer.apply_chat_template(
            conversations, add_generation_prompt=False, tokenize=True,
            enable_thinking=enable_thinking))
        prefix_ids = list(tokenizer.apply_chat_template(
            conversations[:-1], add_generation_prompt=True, tokenize=True,
            enable_thinking=enable_thinking))
        if full_ids[:len(prefix_ids)] == prefix_ids:
            n_prefix = len(prefix_ids)
            ids = full_ids
            if len(ids) > max_seq_len:
                drop = min(len(ids) - max_seq_len, max(0, n_prefix - 8))
                ids = ids[drop:]
                n_prefix -= drop
            return ids, [-100] * n_prefix + list(ids[n_prefix:])
        _FULL_TURN_FALLBACKS[0] += 1
        if _FULL_TURN_FALLBACKS[0] % 100 == 1:
            print(f"[warn] chat-template prefix mismatch -> legacy target "
                  f"({_FULL_TURN_FALLBACKS[0]}x)")
        target_text = conversations[-1]["content"]
        conversations = conversations[:-1]
    prompt_ids = tokenizer.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        tokenize=True,
        enable_thinking=enable_thinking,
    )
    target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
    eos = tokenizer.eos_token_id
    if eos is not None:
        target_ids = target_ids + [eos]
    budget = max_seq_len - len(prompt_ids)
    if budget <= 0:  # prompt alone overflows: truncate it from the left
        prompt_ids = prompt_ids[len(prompt_ids) - max_seq_len + 8:]
        budget = max_seq_len - len(prompt_ids)
    target_ids = target_ids[:budget]
    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + list(target_ids)
    return input_ids, labels


# --------------------------------------------------------------------------- #
# collators
# --------------------------------------------------------------------------- #
def _skill_views(tokenizer, text: str, args, cls_id: int, rng: random.Random):
    """Primary view (drives the offset) + augmented view (contrastive positive)."""
    max_seg = int(getattr(args, "max_segments", DEFAULT_MAX_SEGMENTS) or DEFAULT_MAX_SEGMENTS)
    ids, segs = tokenize_skill_doc(tokenizer, text, args.max_skill_len, cls_id, max_seg)
    aug = getattr(args, "ctr_aug", "drop")
    v2_len = int(getattr(args, "view2_max_len", 512) or 512)
    ids2, segs2 = ids, segs
    if aug == "drop":
        drop = rng.randrange(1, max_seg)          # drop one paragraph group
        max_skill2 = max(16, min(v2_len, args.max_skill_len))
        ids2, segs2 = tokenize_skill_doc(tokenizer, text, max_skill2, cls_id, max_seg,
                                         drop_segments=(drop,))
        if (ids2, segs2) == (ids, segs):          # nothing dropped -> truncate
            ids2, segs2 = ids[:max(2, v2_len)], segs[:max(2, v2_len)]
    elif aug == "trunc":
        keep = max(16, min(v2_len, len(ids)))
        ids2, segs2 = ids[-keep:], segs[-keep:]
    return (ids, segs), (ids2, segs2)


def make_pretrain_collator(tokenizer, args):
    """RECON/COMP collator (mirrors LatentSkill's SkillPretrainCollator)."""
    pad_id = tokenizer.pad_token_id
    rng = random.Random(getattr(args, "seed", 0))
    cls_id = resolve_cls_token_id(tokenizer)
    print(f"[collator] CLS token id = {cls_id}  max_segments = "
          f"{int(getattr(args, 'max_segments', DEFAULT_MAX_SEGMENTS) or DEFAULT_MAX_SEGMENTS)}")

    def split_completion_source(text: str):
        toks = text.split()
        if len(toks) < 2:
            return text
        ratio = 1.0 - rng.uniform(args.completion_ratio_min, args.completion_ratio_max)
        cut = round(len(toks) * ratio)
        cut = min(max(cut, 1), len(toks) - 1)
        return " ".join(toks[:cut])

    def collate(batch):
        skill_a, skill_b = [], []
        skill_a_seg, skill_b_seg = [], []
        input_ids, labels_list, phases, out_labels = [], [], [], []
        for ex in batch:
            text = ex["text"]
            out_labels.append(int(ex.get("label", -100)))
            if rng.random() < args.completion_freq:
                evidence, marker = split_completion_source(text), MARKER_COMP
            else:
                evidence, marker = text, MARKER_RECON
            (a_ids, a_seg), (b_ids, b_seg) = _skill_views(
                tokenizer, evidence, args, cls_id, rng)
            skill_a.append(a_ids)
            skill_a_seg.append(a_seg)
            skill_b.append(b_ids)
            skill_b_seg.append(b_seg)
            ids, labs = _prompt_and_target(
                tokenizer, [{"role": "user", "content": marker}], text, args.max_seq_len
            )
            input_ids.append(ids)
            labels_list.append(labs)
            phases.append(phase_mask_from_labels(labs))
        enc_ids, enc_mask, enc_seg = _pad_pair(skill_a, skill_a_seg, pad_id)
        v2_ids, v2_mask, v2_seg = _pad_pair(skill_b, skill_b_seg, pad_id)
        dec_ids, dec_mask, dec_labs, dec_ph = _pad_decoder(
            input_ids, labels_list, phases, pad_id)
        return {
            "skill_ids": enc_ids,
            "skill_attention_mask": enc_mask,
            "skill_seg": enc_seg,
            "skill2_ids": v2_ids,
            "skill2_attention_mask": v2_mask,
            "skill2_seg": v2_seg,
            "skill_label": torch.tensor(out_labels, dtype=torch.long),
            "input_ids": dec_ids,
            "attention_mask": dec_mask,
            "labels": dec_labs,
            "phase_mask": dec_ph,
        }

    return collate


def make_sft_collator(tokenizer, args):
    """Skill document -> hypernetwork; trajectory -> base model (assistant CE)."""
    pad_id = tokenizer.pad_token_id
    rng = random.Random(getattr(args, "seed", 0) + 1)
    cls_id = resolve_cls_token_id(tokenizer)
    thinking = bool(getattr(args, "sft_thinking", False))

    def collate(batch):
        skill_a, skill_b = [], []
        skill_a_seg, skill_b_seg = [], []
        input_ids, labels_list, phases, out_labels = [], [], [], []
        for ex in batch:
            context = ex["context"]
            out_labels.append(int(ex.get("label", -100)))
            (a_ids, a_seg), (b_ids, b_seg) = _skill_views(
                tokenizer, context, args, cls_id, rng)
            skill_a.append(a_ids)
            skill_a_seg.append(a_seg)
            skill_b.append(b_ids)
            skill_b_seg.append(b_seg)
            convs = ex["conversations"]
            last = convs[-1]
            assert last["role"] == "assistant", f"last turn must be assistant: {last['role']}"
            if thinking:
                ids, labs = _prompt_and_target(
                    tokenizer, convs, None, args.max_seq_len,
                    enable_thinking=True, full_turn=True)
            else:
                ids, labs = _prompt_and_target(
                    tokenizer, convs[:-1], last["content"], args.max_seq_len
                )
            input_ids.append(ids)
            labels_list.append(labs)
            phases.append(phase_mask_from_labels(labs))
        enc_ids, enc_mask, enc_seg = _pad_pair(skill_a, skill_a_seg, pad_id)
        v2_ids, v2_mask, v2_seg = _pad_pair(skill_b, skill_b_seg, pad_id)
        dec_ids, dec_mask, dec_labs, dec_ph = _pad_decoder(
            input_ids, labels_list, phases, pad_id)
        return {
            "skill_ids": enc_ids,
            "skill_attention_mask": enc_mask,
            "skill_seg": enc_seg,
            "skill2_ids": v2_ids,
            "skill2_attention_mask": v2_mask,
            "skill2_seg": v2_seg,
            "skill_label": torch.tensor(out_labels, dtype=torch.long),
            "input_ids": dec_ids,
            "attention_mask": dec_mask,
            "labels": dec_labs,
            "phase_mask": dec_ph,
        }

    return collate


def make_skill_encoder_inputs(tokenizer, skill_text: str, max_skill_len: int = 2048,
                              max_segments: int = DEFAULT_MAX_SEGMENTS):
    """Single skill document -> (ids[1,S], mask[1,S], seg[1,S]) for inference / export."""
    cls_id = resolve_cls_token_id(tokenizer)
    ids, segs = tokenize_skill_doc(tokenizer, skill_text, max_skill_len, cls_id, max_segments)
    if len(ids) <= 1:
        ids.append(int(tokenizer.eos_token_id or 0))
        segs.append(0)
    return (
        torch.tensor([ids], dtype=torch.long),
        torch.ones(1, len(ids), dtype=torch.long),
        torch.tensor([segs], dtype=torch.long),
    )
