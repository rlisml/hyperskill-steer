"""ALFWorld environment-level evaluation of a shipped bundle — **EasySteer/vLLM engine**.

Same protocol, prompts, parsing and episode bookkeeping as `alfworld_eval.py`
(imported from it, so the two engines cannot drift), but generation runs on the
EasySteer fork of vLLM 0.26.0 with the bundle injected through the engine's own
steering path — i.e. the deployment configuration:

    LLM(enable_steer_vector=True, steer_algorithms=["steerling_adapter"],
        steering_config=json.dumps({"vectors": [VectorSpec(source=<bundle>, ...)]}))

Generation is issued in lockstep "waves": at every step all live episodes'
prompts go into ONE `llm.generate()` call (the pattern of LatentSkill's
`evals_vllm/alfworld_eval_vllm_offline.py`), so vLLM prefills/decodes for the
whole live batch continuously. That is what makes this engine ~an order of
magnitude faster than the HF path (`alfworld_eval.py`).

Env: an EasySteer vLLM environment (vLLM 0.26.0 EasySteer fork + the
`steerling_adapter` plugin + ALFWorld 0.4.2).

Usage:
  CUDA_VISIBLE_DEVICES=2 ALFWORLD_DATA=<...>/alfworld_data/alfworld \
  python alfworld_eval_vllm.py \
    --model-name Qwen3-8B --split seen --scale 1.0 \
    --bundle bundles/stage2_skills/skill4 \
    --task-types pick_and_place,pick_two_and_place \
    --concurrency 16 --max-new-tokens 512 --run-tag vllm_skill4
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

# identical protocol / episode logic as the HF evaluator
from alfworld_eval import (  # noqa: E402
    TASK_ORDER,
    TASK_TYPE_NAMES,
    Episode,
    detect_task_type,
    load_skill_docs,
    load_skill_docs,
    parse_args as _hf_parse_args,  # noqa: F401  (kept for reference only)
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model-name", default="Qwen3-8B")
    p.add_argument("--split", choices=["seen", "unseen"], default="seen")
    p.add_argument("--bundle", default=None)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--task-types", default="all")
    p.add_argument("--max-games", type=int, default=None)
    p.add_argument("--game-offset", type=int, default=0,
                   help="skip the first N games of each task type (splits one arm across GPUs)")
    p.add_argument("--alfworld-data",
                   default=os.path.join(os.environ.get("LATENTSKILL_ROOT", "./LatentSkill"),
                                  "alfworld_data/alfworld"))
    p.add_argument("--alfworld-config",
                   default=os.path.join(os.environ.get("LATENTSKILL_ROOT", "./LatentSkill"),
                                  "evals/alfworld/config_tw.yaml"))
    p.add_argument("--latentskill-root",
                   default=os.environ.get("LATENTSKILL_ROOT", "./LatentSkill"))
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--history-length", type=int, default=5)
    p.add_argument("--conversation-max-length", type=int, default=4096)
    p.add_argument("--concurrency", type=int, default=16)
    # --- protocol knobs (same semantics as alfworld_eval.py) ---
    p.add_argument("--chat-template", choices=["default", "latentskill"], default="default")
    p.add_argument("--enable-thinking", type=int, choices=[0, 1], default=0)
    p.add_argument("--skill-in-context", action="store_true")
    p.add_argument("--skill-doc-dir",
                   default=os.path.join(os.environ.get("LATENTSKILL_ROOT", "./LatentSkill"),
                                  "evals/alfworld/skills"))
    # U10: which phases the steering vector applies to. The engine's SelectSpec accepts
    # only "all" | None per phase (vllm.steer_vectors.api), and an omitted phase is left
    # untouched, so generation-only == {"generation": "all"}.
    p.add_argument("--inject-phases", choices=["both", "generation"], default="both",
                   help="'both' (default, prompt+generation) or 'generation' (U10: leave "
                        "the prompt pass un-steered so the thinking prefix is not perturbed)")
    # engine (same pre-registered defaults)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--steer-graph-mode", default="auto")
    # bookkeeping
    p.add_argument("--out-dir", default="results/alfworld_vllm")
    p.add_argument("--run-tag", required=True)
    p.add_argument("--force", action="store_true")
    return p.parse_args(argv)


def sha256_file(path: str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    args = parse_args(argv)
    os.environ["ALFWORLD_DATA"] = args.alfworld_data

    sys.path.insert(0, args.latentskill_root)
    from evals.alfworld.prompts import (  # noqa: E402
        ALFWORLD_TEMPLATE,
        ALFWORLD_TEMPLATE_NO_HIS,
        ALFWORLD_TEMPLATE_WITH_MEMORY,
        ALFWORLD_TEMPLATE_NO_HIS_WITH_MEMORY,
    )

    import easysteer_parity_plugin  # noqa: E402

    easysteer_parity_plugin.register()
    import yaml  # noqa: E402
    from alfworld.agents.environment import get_environment  # noqa: E402
    from transformers import AutoTokenizer  # noqa: E402
    from vllm import LLM, SamplingParams  # noqa: E402
    from vllm.inputs import TokensPrompt  # noqa: E402

    if args.task_types == "all":
        wanted = list(TASK_ORDER)
    else:
        wanted = [t.strip() for t in args.task_types.split(",") if t.strip()]
        for t in wanted:
            assert t in TASK_ORDER, f"unknown task type {t!r}"

    # ----- engine (+ server-level steering config, the EasySteer way) -----
    fork_config = {"engine": "vllm_easysteer", "algorithm": None, "scale": args.scale}
    llm_kwargs = dict(
        model=args.model_name, dtype=args.dtype, max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager, seed=0,
    )
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
        server_spec = {"vectors": [vector_spec], "conflict": "priority"}
        llm_kwargs.update(
            enable_steer_vector=True,
            steer_algorithms=["steerling_adapter"],
            steering_config=json.dumps(server_spec),
            steer_graph_mode=args.steer_graph_mode,
        )
        fork_config.update({
            "algorithm": "steerling_adapter",
            "inject_phases": args.inject_phases,
            "adapter_bundle": os.path.abspath(args.bundle),
            "adapter_bundle_sha256": sha256_file(
                os.path.join(args.bundle, "adapter.safetensors")),
            "use_silu": bool(manifest["use_silu"]),
            "rank": manifest["rank"], "alpha": manifest["alpha"],
            "n_layers": len(layers),
            "source_checkpoint": manifest.get("source_checkpoint"),
            "phases": ["prompt", "generation"],
        })
    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left",
                                              use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.chat_template == "latentskill":
        sys.path.insert(0, os.path.join(args.latentskill_root, "evals_vllm"))
        from qwen_chat_template import REPO_CHAT_TEMPLATE  # noqa: E402
        tokenizer.chat_template = REPO_CHAT_TEMPLATE
    if args.skill_in_context:
        args.skill_docs = load_skill_docs(args.skill_doc_dir)
        print(f"[in-context] skill docs loaded: {sorted(args.skill_docs)}", flush=True)

    # ----- game list -----
    alf_config = yaml.safe_load(open(args.alfworld_config))
    train_eval = ("eval_out_of_distribution" if args.split == "unseen"
                  else "eval_in_distribution")
    probe_env = get_environment(alf_config["env"]["type"])(alf_config, train_eval=train_eval)
    all_games = list(probe_env.game_files)
    del probe_env
    by_type = defaultdict(list)
    for g in all_games:
        by_type[detect_task_type(g)].append(g)
    game_files = []
    for t in TASK_ORDER:
        if t not in wanted:
            continue
        sel = by_type[t][args.game_offset:]
        game_files.extend(sel[: args.max_games] if args.max_games else sel)
    print(f"[eval] engine=vllm_easysteer split={args.split} task_types={wanted} "
          f"n_games={len(game_files)}", flush=True)
    print(f"[inject] {json.dumps(fork_config)}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / f"detail_{args.split}_{args.run_tag}.jsonl"
    if args.force and detail_path.exists():
        detail_path.unlink()
    done_games = set()
    if detail_path.exists():
        with open(detail_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done_games.add(json.loads(line)["gamefile"])
        print(f"[resume] {len(done_games)} games already recorded", flush=True)
    todo_games = [g for g in game_files if g not in done_games]

    templates = {
        "no_his": ALFWORLD_TEMPLATE_NO_HIS,
        "his": ALFWORLD_TEMPLATE,
        "no_his_mem": ALFWORLD_TEMPLATE_NO_HIS_WITH_MEMORY,
        "his_mem": ALFWORLD_TEMPLATE_WITH_MEMORY,
    }
    C = min(args.concurrency, max(1, len(todo_games)))
    slots = [{"games": todo_games[w::C], "idx": 0, "env": None, "episode": None}
             for w in range(C)] if todo_games else []

    def make_env(games):
        env_obj = get_environment(alf_config["env"]["type"])(alf_config, train_eval=train_eval)
        env_obj.game_files = list(games)
        env_obj.num_games = len(games)
        return env_obj.init_env(batch_size=1)

    for s in slots:
        s["env"] = make_env(s["games"][s["idx"]:])

    task_results = defaultdict(list)
    detail_f = open(detail_path, "a", encoding="utf-8")
    n_done, n_total = len(done_games), len(game_files)
    gen_calls = total_new_tokens = 0
    t0 = time.time()

    while n_done < n_total:
        for s in slots:
            if s["episode"] is None and s["idx"] < len(s["games"]):
                ep = Episode(s, s["games"][s["idx"]], tokenizer, args, templates)
                ep.reset(s["env"])
                s["episode"] = ep
                s["idx"] += 1
        live = [s for s in slots if s["episode"] is not None]
        if not live:
            break

        # text prompts (what the engine receives): on
        # transformers 5.x `apply_chat_template(tokenize=True)` does not return
        # list[int] for Qwen2/3, and `TokensPrompt` then trips vLLM's input
        # validation with `TypeError: '>' not supported between 'str' and 'int'`.
        prompts = [TokensPrompt(prompt_token_ids=s["episode"].build_prompt_ids())
                   for s in live]
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        gen_calls += 1
        total_new_tokens += sum(len(o.outputs[0].token_ids) for o in outputs)

        for s, out in zip(live, outputs):
            ep = s["episode"]
            try:
                ep.apply_output(s["env"], out.outputs[0].text)
            except Exception as exc:
                print(f"[WARN] env error ({ep.gamefile}): {exc!r}", flush=True)
                ep.won = False
                ep.finished = True
                ep.env_error = repr(exc)[:300]
                try:
                    s["env"].close()
                except Exception:
                    pass
                s["env"] = make_env(s["games"][s["idx"]:])
            if ep.finished:
                rec = ep.record()
                detail_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                detail_f.flush()
                task_results[ep.task_type].append(ep.won)
                s["episode"] = None
                n_done += 1
                rate = sum(sum(v) for v in task_results.values()) / max(n_done, 1)
                print(f"[{n_done}/{n_total}] {ep.task_type} won={ep.won} "
                      f"steps={len(ep.steps_log)} rate={rate:.3f} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
    detail_f.close()

    summary = {"run_tag": args.run_tag, "split": args.split, "scale": args.scale,
               "bundle": args.bundle, "engine": "vllm_easysteer",
               "chat_template": args.chat_template, "enable_thinking": args.enable_thinking,
               "skill_in_context": bool(getattr(args, "skill_in_context", False)),
               "inject_phases": args.inject_phases,
               "fork_config": fork_config, "task_types": wanted,
               "max_games_per_type": args.max_games, "concurrency": args.concurrency,
               "max_new_tokens": args.max_new_tokens, "max_steps": args.max_steps,
               "history_length": args.history_length, "model": args.model_name,
               "max_model_len": args.max_model_len,
               "gpu_memory_utilization": args.gpu_memory_utilization,
               "n_generate_calls": gen_calls, "total_new_tokens": total_new_tokens,
               "wall_time_sec": round(time.time() - t0, 1), "per_task": {}}
    all_res = []
    for t in TASK_ORDER:
        rs = task_results.get(t, [])
        if rs:
            summary["per_task"][t] = {"success": int(sum(rs)), "total": len(rs),
                                      "rate": round(sum(rs) / len(rs), 4)}
            all_res.extend(rs)
    if all_res:
        summary["overall"] = {"success": int(sum(all_res)), "total": len(all_res),
                              "rate": round(sum(all_res) / len(all_res), 4)}
    with open(out_dir / f"summary_{args.split}_{args.run_tag}.json", "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 64)
    for t, d in summary["per_task"].items():
        print(f"{TASK_TYPE_NAMES[t]:<24} {d['success']:>4}/{d['total']:<4} {d['rate']:.4f}")
    if all_res:
        o = summary["overall"]
        print(f"{'Overall':<24} {o['success']:>4}/{o['total']:<4} {o['rate']:.4f}")
    print(f"wall={summary['wall_time_sec']}s gen_calls={gen_calls} "
          f"new_tokens={total_new_tokens}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
