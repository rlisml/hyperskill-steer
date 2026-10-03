"""Pairwise cosine of the LatentSkill-aligned bundles (CPU, numpy/safetensors)."""

import glob
import itertools
import os
import sys

import numpy as np
from safetensors.torch import load_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "bundles/ls_skills")

vecs = {}
for d in sorted(glob.glob(os.path.join(root, "ls_skill_*"))):
    f = os.path.join(d, "adapter.safetensors")
    if not os.path.isfile(f):
        continue
    sd = load_file(f)
    parts = [sd[k].float().numpy().ravel()
             for k in sorted(sd.keys(), key=lambda s: (int(s.split(".")[0][5:]),
                                                       s.split(".")[1]))]
    vecs[os.path.basename(d)] = np.concatenate(parts)

names = sorted(vecs)
print("bundles:", len(names))


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


alf = [n for n in names if n.split("_")[-1] in ("2", "3", "4", "6", "8")]
search = [n for n in names if n.split("_")[-1] in ("0", "1", "5", "7")]
for label, group in (("ALFWorld group", alf), ("SearchQA group", search),
                     ("all 9", names)):
    cs = [cos(vecs[a], vecs[b]) for a, b in itertools.combinations(group, 2)]
    ds = [float(np.linalg.norm(vecs[a] - vecs[b]) /
                (0.5 * (np.linalg.norm(vecs[a]) + np.linalg.norm(vecs[b]))))
          for a, b in itertools.combinations(group, 2)]
    if cs:
        print(f"{label}: n_pairs={len(cs)} cosine mean={np.mean(cs):.4f} "
              f"min={np.min(cs):.4f} max={np.max(cs):.4f} | "
              f"relL2 mean={np.mean(ds):.4f} max={np.max(ds):.4f}")

print("--- ALFWorld pairwise ---")
for a, b in itertools.combinations(alf, 2):
    print(f"  {a} vs {b}: cos={cos(vecs[a], vecs[b]):.4f}")
