"""SearchQA evaluation of a shipped EasySteer bundle — **EasySteer/vLLM engine**.

Protocol is byte-for-byte the one of LatentSkill's own evaluator
(`evals/searchqa/evaluate.py`, ported to vLLM in
`LatentSkill/evals_vllm/searchqa_eval_vllm_offline.py`):

  * prompt: `evals/searchqa/prompts.py::SEARCH_TEMPLATE[_NO_HIS]` (imported, not copied)
  * chat template: `evals_vllm/qwen_chat_template.py::REPO_CHAT_TEMPLATE`
  * `enable_thinking=False`, greedy, `max_new_tokens=2048`
  * E5 retriever top-3 over wiki-18, `max_steps=4`
  * EM: `evals/searchqa/skillrl_utils.py::em_check` (Search-R1 normalisation)

The *only* difference from the reference: the skill is injected as a **post-block
activation offset** through the engine's own steering path (the deployment
configuration this project promises) instead of a PEFT LoRA via `--enable-lora`:

    LLM(enable_steer_vector=True, steer_algorithms=["steerling_adapter"],
        steering_config=json.dumps({"vectors": [VectorSpec(source=<bundle>, ...)]}))

Skill routing is by the record's `skill` field. Single-hop QA (nq / triviaqa /
popqa) is entirely `direct_retrieval` -> `bundles/ls_skills/ls_skill_1`, so one
bundle serves the whole arm (the engine steering config is process-level).

Env: an EasySteer vLLM environment (vLLM 0.26.0 fork + `steerling_adapter` plugin).

Usage:
  CUDA_VISIBLE_DEVICES=2 python \
    searchqa_eval_bundle.py --bundle bundles/ls_skills/ls_skill_1 --scale 1.0 \
    --datasets nq triviaqa popqa --per-dataset 500 --run-tag sq_sh_skill1
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

LATENTSKILL_ROOT = os.environ.get("LATENTSKILL_ROOT", "./LatentSkill")

# ---- protocol components imported from the reference implementation (no copies) ----
sys.path.insert(0, LATENTSKILL_ROOT)
sys.path.insert(0, os.path.join(LATENTSKILL_ROOT, "evals_vllm"))
from evals.searchqa.prompts import (  # noqa: E402
    SEARCH_TEMPLATE,
    SEARCH_TEMPLATE_NO_HIS,
)
from evals.searchqa.skillrl_utils import em_check, extract_solution  # noqa: E402

# skill name -> bundle subdirectory (first-appearance order of skill_ift/train.json,
# verified by scripts/_sq_recon.py: skill0=multi_hop_reasoning, skill1=direct_retrieval,
# skill7=comparison)
DEFAULT_SKILL_BUNDLE = {
    "direct_retrieval": "ls_skill_1",
    "multi_hop_reasoning": "ls_skill_0",
    "comparison": "ls_skill_7",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model-name", default="Qwen3-8B")
    p.add_argument("--bundle", default=None,
                   help="EasySteer bundle dir; omit for the un-steered base arm")
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--inject-phases", choices=["both", "generation"], default="both",
                   help="'both' = prompt+generation; 'generation' = generation-only (U10)")
    p.add_argument("--test-data",
                   default=os.path.join(LATENTSKILL_ROOT, "data/search_test/search_test_all.jsonl"))
    p.add_argument("--datasets", type=str, nargs="+", default=["nq", "triviaqa", "popqa"])
    p.add_argument("--skills", type=str, nargs="+", default=None,
                   help="keep only records whose `skill` field is in this set; used "
                        "for multi-hop, where one process-level bundle can only serve "
                        "one skill")
    p.add_argument("--per-dataset", type=int, default=500,
                   help="cap per dataset (paper: 500 for the 7 sets, Bamboogle full 125)")
    p.add_argument("--retrieval-url", type=str,
                   default=os.environ.get("RETRIEVAL_URL", "http://localhost:8200/retrieve"))
    p.add_argument("--retrieval-topk", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--conversation-max-length", type=int, default=4096)
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--steer-graph-mode", default="auto")
    p.add_argument("--out-dir", default="results/searchqa_vllm")
    p.add_argument("--run-tag", required=True)
    return p.parse_args(argv)


def sha256_file(path: str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# retrieval helpers (same request/response contract as the reference evaluator)
# --------------------------------------------------------------------------- #
def format_docs(docs):
    if not docs:
        return ""
    parts = []
    for i, doc in enumerate(docs, 1):
        title = doc.get("title", "").strip()
        content = doc.get("contents", doc.get("text", "")).strip()
        parts.append(f"Doc {i}: {title}\n{content}")
    return "\n\n".join(parts)


def retrieve_batch(queries, retrieval_url, topk, timeout=3600):
    """One index scan for all pending queries of a wave."""
    import requests

    if not queries:
        return []
    try:
        resp = requests.post(
            f"{retrieval_url.rsplit('/', 1)[0]}/retrieve_batch",
            json={"queries": queries, "topk": topk},
            timeout=timeout,
        )
        resp.raise_for_status()
        return [format_docs(docs) for docs in resp.json()["result"]]
    except Exception as e:  # noqa: BLE001
        print(f"[retrieval warning] batch of {len(queries)}: {e}", flush=True)
        return [""] * len(queries)


class Question:
    """Mirrors LatentSkill `evals_vllm/searchqa_eval_vllm_offline.py::Question`."""

    def __init__(self, record, args):
        self.record = record
        self.args = args
        self.qid = record["id"]
        self.dataset = record["dataset"]
        self.skill = record.get("skill", "direct_retrieval")
        self.question = record["question"]
        self.golden = record["golden_answers"]
        self.history_str = ""
        self.step = 0
        self.final_answer = ""
        self.steps_log = []
        self.finished = False
        self.pending_query = None

    def build_prompt_ids(self, tokenizer):
        if self.step == 0:
            prompt_text = SEARCH_TEMPLATE_NO_HIS.format(task_description=self.question)
        else:
            prompt_text = SEARCH_TEMPLATE.format(
                task_description=self.question,
                step_count=self.step,
                memory_context=self.history_str,
            )
        prompt_text = prompt_text.lstrip("\n")
        enc = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            max_length=self.args.conversation_max_length,
            truncation=True,
            padding=False,
            enable_thinking=False,
        )
        return enc["input_ids"]

    def apply_output(self, output_text):
        output_text = output_text.strip()
        step_record = {"step": self.step + 1, "output": output_text}

        if "<answer>" in output_text and "</answer>" in output_text:
            ans = extract_solution(output_text)
            self.final_answer = ans if ans is not None else ""
            step_record["action"] = "answer"
            step_record["answer"] = self.final_answer
            self.steps_log.append(step_record)
            self.finished = True
            return

        m = re.search(r"<search>(.*?)</search>", output_text, re.DOTALL)
        if m:
            query = m.group(1).strip()
            step_record["action"] = "search"
            step_record["query"] = query
            self.steps_log.append(step_record)
            self.pending_query = query
        else:
            step_record["action"] = "invalid"
            self.steps_log.append(step_record)
            self.finished = True
            return

        self.step += 1
        if self.step >= self.args.max_steps:
            self.finished = True

    def apply_retrieval(self, docs):
        self.pending_query = None
        if self.finished:
            return
        docs_str = docs if docs else ""
        self.history_str += f"<search>{self.steps_log[-1]['query']}</search>\n\n"
        if docs_str:
            self.history_str += f"<information>\n{docs_str}\n</information>\n"
        else:
            self.history_str += "<information>\nNo results found.\n</information>\n"

    def finalize(self):
        if not self.final_answer and self.steps_log:
            ans = extract_solution(self.steps_log[-1].get("output", ""))
            self.final_answer = ans if ans is not None else ""
        self.em = em_check(self.final_answer, self.golden)
        return {
            "id": self.qid,
            "dataset": self.dataset,
            "skill": self.skill,
            "question": self.question,
            "golden_answers": self.golden,
            "pred_answer": self.final_answer,
            "em": self.em,
            "n_steps": len(self.steps_log),
            "steps": self.steps_log,
        }


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- data -----
    per_ds = defaultdict(list)
    with open(args.test_data, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["dataset"] not in set(args.datasets):
                continue
            if args.skills and r.get("skill", "direct_retrieval") not in set(args.skills):
                continue
            per_ds[r["dataset"]].append(r)
    records = []
    for ds in args.datasets:
        recs = per_ds.get(ds, [])
        records.extend(recs[: args.per_dataset])
    print(f"[eval] {len(records)} questions: "
          f"{ {ds: min(len(per_ds.get(ds, [])), args.per_dataset) for ds in args.datasets} }",
          flush=True)

    # ----- tokenizer (reference chat template, thinking off) -----
    from transformers import AutoTokenizer
    from qwen_chat_template import REPO_CHAT_TEMPLATE  # noqa: E402

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left",
                                              use_fast=True)
    tokenizer.chat_template = REPO_CHAT_TEMPLATE

    # ----- engine (EasySteer steering path) -----
    import easysteer_parity_plugin  # noqa: E402

    easysteer_parity_plugin.register()
    from vllm import LLM, SamplingParams  # noqa: E402
    from vllm.inputs import TokensPrompt  # noqa: E402

    llm_kwargs = dict(
        model=args.model_name, dtype=args.dtype, max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager, seed=0,
    )
    fork_config = {"engine": "vllm_easysteer", "algorithm": None, "scale": args.scale,
                   "bundle": None, "inject_phases": args.inject_phases}
    if args.bundle:
        with open(os.path.join(args.bundle, "manifest.json"), encoding="utf-8") as f:
            manifest = json.load(f)
        layers = manifest.get("layers") or list(range(int(manifest["n_layers"])))
        vector_spec = {
            "source": os.path.abspath(args.bundle),
            "algorithm": "steerling_adapter",
            "scale": args.scale,
            "layers": list(layers),
            "apply": ({"generation": "all"} if args.inject_phases == "generation"
                      else {"prompt": "all", "generation": "all"}),
            "name": args.run_tag,
        }
        llm_kwargs.update(
            enable_steer_vector=True,
            steer_algorithms=["steerling_adapter"],
            steering_config=json.dumps({"vectors": [vector_spec], "conflict": "priority"}),
            steer_graph_mode=args.steer_graph_mode,
        )
        fork_config.update({
            "algorithm": "steerling_adapter",
            "bundle": os.path.abspath(args.bundle),
            "bundle_sha256": sha256_file(os.path.join(args.bundle, "adapter.safetensors")),
            "use_silu": bool(manifest["use_silu"]), "rank": manifest["rank"],
            "alpha": manifest["alpha"], "n_layers": len(layers),
        })
    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
    print(f"[eval] engine ready | inject={json.dumps(fork_config)}", flush=True)

    # ----- wave-batched agent loop -----
    active = [Question(r, args) for r in records]
    queue = list(active)
    done = []
    detail_path = out_dir / f"detail_{args.run_tag}.jsonl"
    detail_f = open(detail_path, "w", encoding="utf-8")

    def finalize(q):
        rec = q.finalize()
        detail_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        detail_f.flush()
        done.append(q)

    t0 = time.time()
    wave = 0
    while queue:
        wave += 1
        batch = queue[: args.concurrency]
        del queue[: len(batch)]

        prompts = [TokensPrompt(prompt_token_ids=q.build_prompt_ids(tokenizer)) for q in batch]
        outs = llm.generate(prompts, sampling, use_tqdm=False)

        to_retrieve, next_round = [], []
        for q, out in zip(batch, outs):
            q.apply_output(out.outputs[0].text)
            if q.pending_query:
                to_retrieve.append(q)
            elif q.finished:
                finalize(q)

        if to_retrieve:
            docs_list = retrieve_batch([q.pending_query for q in to_retrieve],
                                       args.retrieval_url, args.retrieval_topk)
            for q, docs in zip(to_retrieve, docs_list):
                q.apply_retrieval(docs)
                if q.finished:
                    finalize(q)
                else:
                    next_round.append(q)

        queue = next_round + queue
        print(f"[wave {wave}] done={len(done)}/{len(active)} live={len(queue)} "
              f"searched={len(to_retrieve)} elapsed={time.time()-t0:.0f}s", flush=True)

    detail_f.close()

    # ----- summary + behaviour stats -----
    per = defaultdict(list)
    for q in done:
        per[q.dataset].append(q.em)
    all_em = [e for v in per.values() for e in v]

    n_steps = [q.steps_log.__len__() for q in done]
    n_invalid = sum(1 for q in done for s in q.steps_log if s.get("action") == "invalid")
    n_all_steps = sum(len(q.steps_log) for q in done)
    chars = [len(s.get("output", "")) for q in done for s in q.steps_log]

    summary = {"run_tag": args.run_tag, "bundle": args.bundle, "scale": args.scale,
               "inject_phases": args.inject_phases, "datasets": args.datasets,
               "per_dataset": {}, "config": {"max_steps": args.max_steps,
               "max_new_tokens": args.max_new_tokens, "topk": args.retrieval_topk,
               "concurrency": args.concurrency}}
    print(f"\n{'='*52}")
    print(f"{'Dataset':<20} {'EM':>8} {'Count':>8}")
    print(f"{'='*52}")
    for ds in args.datasets:
        ems = per.get(ds, [])
        if not ems:
            continue
        summary["per_dataset"][ds] = {"em": round(sum(ems) / len(ems), 4), "count": len(ems)}
        print(f"{ds:<20} {sum(ems)/len(ems):>8.4f} {len(ems):>8}")
    summary["average"] = {"em": round(sum(all_em) / len(all_em), 4), "count": len(all_em)}
    print(f"{'-'*52}")
    print(f"{'average':<20} {sum(all_em)/len(all_em):>8.4f} {len(all_em):>8}")
    summary["wall_time_sec"] = round(time.time() - t0, 1)
    summary["behaviour"] = {
        "mean_steps": round(sum(n_steps) / max(1, len(n_steps)), 2),
        "invalid_action_rate": round(n_invalid / max(1, n_all_steps), 4),
        "mean_chars_per_step": round(sum(chars) / max(1, len(chars)), 1),
    }
    print(f"[behaviour] {json.dumps(summary['behaviour'])}")

    summary_path = out_dir / f"summary_{args.run_tag}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved:\n  detail:  {detail_path}\n  summary: {summary_path}")


if __name__ == "__main__":
    main()
