"""Aggregate the LatentSkill-protocol ALFWorld arms into one comparison table.

Arms (all `--chat-template latentskill --enable-thinking 1 --max-new-tokens 2048`,
seen split, 140 games, greedy, concurrency 16):

  base        = tag `ls_base`                       (no injection, no skill text)
  M2 (ours)   = tags `ls_m2_skill{2,3,4,6,8}`       (our hypernet bundle per task type)
  in-context  = tag `ls_incontext`                  (skill document inside the prompt)

Reads the detail jsonl files (written incrementally, so it works mid-run) and prints
per-task-type + overall success, mean steps and parse-valid rate, next to the
LatentSkill paper numbers (Table 1 / Table 11) from their REPRODUCTION_LOG.md.

Usage:
  python scripts/alfworld_ls_summary.py
"""
import argparse
import json
import os
from collections import defaultdict

TASK_ORDER = ["pick_and_place", "pick_two_and_place", "clean", "heat", "cool",
              "look_at_obj_in_light"]
TASK_TYPE_NAMES = {
    "pick_and_place": "Pick&Place", "pick_two_and_place": "PickTwo", "clean": "Clean",
    "heat": "Heat", "cool": "Cool", "look_at_obj_in_light": "Look",
}

REF_PAPER = {"pick_and_place": 97.1, "pick_two_and_place": 75.0, "clean": 63.0,
             "heat": 43.8, "cool": 64.0, "look_at_obj_in_light": 92.3}      # Table 1, a=0.6
REF_PAPER_BASE = {"pick_and_place": 82.9, "pick_two_and_place": 29.2, "clean": 18.5,
                  "heat": 37.5, "cool": 32.0, "look_at_obj_in_light": 46.2}  # Table 11, a=0
REF_VLLM_REPRO = {"pick_and_place": 100.0, "pick_two_and_place": 54.2, "clean": 51.9,
                  "heat": 18.8, "cool": 48.0, "look_at_obj_in_light": 69.2}  # local repro a=0.6
REF_VLLM_BASE = {"pick_and_place": 82.9, "pick_two_and_place": 16.7, "clean": 33.3,
                 "heat": 31.3, "cool": 24.0, "look_at_obj_in_light": 23.1}   # local repro a=0


def load_detail(path):
    recs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                recs[r["gamefile"]] = r
    return list(recs.values())


def stats(paths):
    per_task = defaultdict(lambda: [0, 0])
    steps, parse_ok, parse_n, n = [], 0, 0, 0
    for p in paths:
        for r in load_detail(p):
            per_task[r["task_type"]][0] += int(bool(r["won"]))
            per_task[r["task_type"]][1] += 1
            steps.append(r["steps"])
            for s in r["steps_log"]:
                parse_n += 1
                parse_ok += int(bool(s.get("is_parse_valid")))
            n += 1
    return {
        "per_task": {k: tuple(v) for k, v in per_task.items()},
        "overall": (sum(v[0] for v in per_task.values()), n),
        "steps": (sum(steps) / len(steps)) if steps else None,
        "parse": (parse_ok, parse_n) if parse_n else None,
    }


def pct(n, d):
    return 100.0 * n / d if d else float("nan")


def cell(n, d):
    return f"{n}/{d} ({pct(n, d):.1f})" if d else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/alfworld_vllm")
    ap.add_argument("--split", default="seen")
    ap.add_argument("--base-tag", default="ls_base")
    ap.add_argument("--incontext-tag", default="ls_incontext")
    ap.add_argument("--m2-prefix", default="ls_m2t_skill",
                    help="M2 arm detail-file prefix (thinking-aligned bundles)")
    args = ap.parse_args()
    rd = args.results_dir

    arms = [
        ("base", [f"{rd}/detail_{args.split}_{args.base_tag}.jsonl"]),
        ("M2 (ours)", [f"{rd}/detail_{args.split}_{args.m2_prefix}{k}.jsonl"
                       for k in (2, 3, 4, 6, 8)]),
        ("in-context skill", [f"{rd}/detail_{args.split}_{args.incontext_tag}.jsonl"]),
    ]

    data = {}
    for name, paths in arms:
        paths = [p for p in paths if os.path.exists(p)]
        if not paths:
            print(f"[warn] no detail files for arm {name}")
            continue
        data[name] = stats(paths)

    hdr = "| arm | " + " | ".join(TASK_TYPE_NAMES[t] for t in TASK_ORDER) + \
          " | overall | mean steps | parse-valid |"
    print(hdr)
    print("|" + "---|" * (len(TASK_ORDER) + 4))
    for name, d in data.items():
        cells = [cell(*d["per_task"].get(t, (0, 0))) for t in TASK_ORDER]
        n, tot = d["overall"]
        st = f"{d['steps']:.1f}" if d["steps"] else "-"
        pv = f"{pct(*d['parse']):.1f}%" if d["parse"] else "-"
        print(f"| {name} | " + " | ".join(cells) + f" | **{cell(n, tot)}** | {st} | {pv} |")

    # reference rows (percentages only)
    for label, ref in [("LatentSkill paper a=0.6 (Table 1)", REF_PAPER),
                       ("LatentSkill paper a=0 (Table 11)", REF_PAPER_BASE),
                       ("LatentSkill vLLM repro a=0.6", REF_VLLM_REPRO),
                       ("LatentSkill vLLM repro a=0", REF_VLLM_BASE)]:
        avg = sum(ref.values()) / len(ref)
        cells = [f"{ref[t]:.1f}" for t in TASK_ORDER]
        print(f"| {label} | " + " | ".join(cells) + f" | **{avg:.1f}** | - | - |")

    print("\noverall ranking")
    for name, d in data.items():
        n, tot = d["overall"]
        print(f"  {name:<18} {n}/{tot}  {pct(n, tot):.1f}%")


if __name__ == "__main__":
    main()
