"""Download the public LatentSkill datasets used by this codebase.

Fetches skill-document pretraining data (train/val) and trajectory SFT data
(skill_ift/train.json) from the HF dataset repo AofaYu71/LatentSkill into
<repo>/data/latentskill/ (same relative layout LatentSkill expects).

Usage:
    python scripts/fetch_latentskill_data.py [--root data/latentskill]
"""

import argparse
import os
import sys

from huggingface_hub import hf_hub_download

REPO_ID = "AofaYu71/LatentSkill"
FILES = [
    "skill_pretrain/train.jsonl",
    "skill_pretrain/val.jsonl",
    "skill_ift/train.json",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/latentskill")
    args = p.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    root = args.root if os.path.isabs(args.root) else os.path.join(repo_root, args.root)
    os.makedirs(root, exist_ok=True)

    for fn in FILES:
        dst = os.path.join(root, fn)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            print(f"[skip] {fn} exists ({os.path.getsize(dst)} bytes)")
            continue
        print(f"[download] {fn} ...", flush=True)
        path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=fn,
            local_dir=root,
        )
        print(f"[ok] {path} ({os.path.getsize(path)} bytes)", flush=True)

    for fn in FILES:
        dst = os.path.join(root, fn)
        assert os.path.exists(dst) and os.path.getsize(dst) > 0, dst
    print("[done] all files present under", root)


if __name__ == "__main__":
    sys.exit(main())
