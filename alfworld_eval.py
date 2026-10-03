"""ALFWorld environment-level evaluation of a *shipped bundle* (environment-level, HF engine).

Instead of
teacher-forced token accuracy on the stage-2 val split, the model is put inside
the real ALFWorld/TextWorld environment and scored by **task success rate**.

Protocol (frozen, mirrors LatentSkill `evals_vllm/alfworld_eval_vllm_offline.py`
so the numbers are comparable with that reference implementation):

  * prompts        : `evals/alfworld/prompts.py::ALFWORLD_TEMPLATE[_NO_HIS]`
                     (imported from the LatentSkill checkout, `--latentskill-root`)
  * decoding       : greedy, no sampling, `--max-new-tokens`, early stop on `</action>`
  * action parsing : last `<action>...</action>`; matched against the admissible
                     list after whitespace/lowercase normalisation; otherwise the
                     raw parsed action is passed through, else the literal "look"
  * episode        : `--max-steps` 50, rolling `--history-length` 5 observations
  * env            : ALFWorld TextWorld, `--split seen|unseen`, one env per
                     concurrency slot, games strided across slots; at every step the
                     live slots are generated in ONE batch (lockstep "waves")

What is NEW here (vs the LatentSkill script): the engine is plain HF transformers
(no vLLM, no LoRA) and the intervention is our **post-block low-rank activation
offset** injected by `skill_hypernet.PostBlockSteerer` from an exported bundle
(`bundles/stage2_skills/skill<K>`), i.e. exactly the artifact the EasySteer/vLLM
path consumes. `--scale 0` gives the
frozen-backbone control on the same games.

Because the steering tensors are set once per process, one run covers ONE bundle:
use `--task-types` to select which ALFWorld task types to run (default: the types
that the bundle's skill document matches). Aggregation across task types is done
by `scripts/alfworld_ls_summary.py`.

Usage:
  CUDA_VISIBLE_DEVICES=2 ALFWORLD_DATA=<...>/alfworld_data/alfworld \
  python alfworld_eval.py \
    --model-name Qwen3-8B --split seen --scale 1.0 \
    --bundle bundles/stage2_skills/skill4 --task-types pick_and_place,pick_two_and_place \
    --concurrency 16 --max-new-tokens 512 \
    --out-dir results/alfworld --run-tag m2_skill4
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

TASK_ORDER = [
    "pick_and_place", "pick_two_and_place", "clean",
    "heat", "cool", "look_at_obj_in_light",
]
TASK_TYPE_NAMES = {
    "pick_and_place": "Pick & Place",
    "pick_two_and_place": "Pick Two & Place",
    "clean": "Clean & Place",
    "heat": "Heat & Place",
    "cool": "Cool & Place",
    "look_at_obj_in_light": "Look At Obj In Light",
}

# stage-2 skill index -> ALFWorld task types it is the "correct" skill document for.
# The index is the order of FIRST APPEARANCE in `skill_ift/train.json` (the
# convention `export_skill_bundle.py --skill-index` uses); verified content-identical
# (exact string equality) to LatentSkill `evals/alfworld/skills/*.txt` by
# `scripts/check_skill_order.py`:
#   skill0 searchqa   skill1 searchqa   skill2 cool      skill3 heat
#   skill4 pick_and_place             skill5 examine   skill6 clean
#   skill7 searchqa   skill8 look_at_obj_in_light
SKILL_TASK_TYPES = {
    2: ["cool"],
    3: ["heat"],
    4: ["pick_and_place", "pick_two_and_place"],
    6: ["clean"],
    8: ["look_at_obj_in_light"],
}

# task type -> skill document basename (LatentSkill's rule; pick_two reuses pick_and_place).
# Used by the E3 "skill in context" baseline (skill text placed in the prompt).
TASK_SKILL_DOC = {
    "pick_and_place": "pick_and_place",
    "pick_two_and_place": "pick_and_place",
    "clean": "clean",
    "heat": "heat",
    "cool": "cool",
    "look_at_obj_in_light": "look_at_obj_in_light",
}


def load_skill_docs(skill_dir: str) -> dict:
    """Load the 5 ALFWorld skill documents (the files the LatentSkill eval uses)."""
    docs = {}
    for name in sorted(set(TASK_SKILL_DOC.values())):
        with open(os.path.join(skill_dir, f"{name}.txt"), encoding="utf-8") as f:
            docs[name] = f.read().strip()
    return docs


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model-name", default="Qwen3-8B")
    p.add_argument("--split", choices=["seen", "unseen"], default="seen")
    p.add_argument("--bundle", default=None,
                   help="bundle dir (manifest.json + adapter.safetensors); omit/`--scale 0` "
                        "for the frozen-backbone control")
    p.add_argument("--scale", type=float, default=1.0,
                   help="0.0 = no injection (base); 1.0 = full generated offset")
    p.add_argument("--task-types", default="all",
                   help="'all' or comma-separated subset of " + ",".join(TASK_ORDER))
    p.add_argument("--max-games", type=int, default=None,
                   help="cap games PER TASK TYPE (deterministic: first N in env order)")
    # env / protocol (frozen: same defaults as the LatentSkill reference script)
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
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    # --- protocol knobs (defaults reproduce the pre-registered run) ---
    p.add_argument("--chat-template", choices=["default", "latentskill"], default="default",
                   help="'latentskill' = the reference REPO_CHAT_TEMPLATE (appends '<think>' to "
                        "the generation prompt); 'default' = the tokenizer's own template")
    p.add_argument("--enable-thinking", type=int, choices=[0, 1], default=0,
                   help="Qwen3 thinking mode flag handed to apply_chat_template")
    p.add_argument("--skill-in-context", action="store_true",
                   help="E3 baseline: put the skill document in the prompt "
                        "(LatentSkill's ALFWORLD_TEMPLATE[_NO_HIS]_WITH_MEMORY); implies no bundle")
    p.add_argument("--skill-doc-dir",
                   default=os.path.join(os.environ.get("LATENTSKILL_ROOT", "./LatentSkill"),
                                  "evals/alfworld/skills"))
    # bookkeeping
    p.add_argument("--out-dir", default="results/alfworld")
    p.add_argument("--run-tag", required=True)
    p.add_argument("--force", action="store_true")
    p.add_argument("--verify-injection", action="store_true",
                   help="forward one fixed prompt twice (delta on/off) and print the "
                        "max |logit| difference — proves the bundle reaches the forward pass")
    return p.parse_args(argv)


VERIFY_PROMPT = (
    "You are an expert agent operating in the ALFRED Embodied Environment. "
    "Your task is to: cool an apple and put it in the fridge.\n\n"
    "## Current Progress\nYour current observation is: You are in the middle of a room. "
    "Looking quickly around you, you see a cabinet 1, a countertop 1, a fridge 1, a "
    "sinkbasin 1, and a table 1.\nYour admissible actions of the current situation are: "
    "[go to fridge 1, look, go to countertop 1, take apple 1, cool apple 1 with fridge 1].\n\n"
    "Now it's your turn to take an action."
)


def verify_injection(backbone, tokenizer, steerer, device):
    """Max |Δlogits| between delta-on and delta-off on one fixed prompt."""
    import torch

    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": VERIFY_PROMPT}], add_generation_prompt=True,
        tokenize=True, enable_thinking=False)
    ids = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        on = backbone(input_ids=ids, use_cache=False).logits[:, -1, :].float()
        saved_d, saved_u = steerer._down, steerer._up
        steerer._down = steerer._up = None
        off = backbone(input_ids=ids, use_cache=False).logits[:, -1, :].float()
        steerer._down, steerer._up = saved_d, saved_u
    return float((on - off).abs().max().item())


# --------------------------------------------------------------------------- #
# prompt / parsing helpers (identical to the LatentSkill reference script)
# --------------------------------------------------------------------------- #
def detect_task_type(gamefile: str) -> str:
    if "pick_clean_then_place" in gamefile:
        return "clean"
    if "pick_heat_then_place" in gamefile:
        return "heat"
    if "pick_cool_then_place" in gamefile:
        return "cool"
    if "look_at_obj_in_light" in gamefile:
        return "look_at_obj_in_light"
    if "pick_two_obj_and_place" in gamefile:
        return "pick_two_and_place"
    return "pick_and_place"


def extract_task_description(obs: str) -> str:
    match = re.search(r"Your task is to:\s*(.+?)(?:\n|$)", obs)
    return match.group(1).strip() if match else ""


def _norm_action(x: str) -> str:
    return re.sub(r"\s+", " ", x or "").strip().lower()


def _first_str(x):
    if isinstance(x, (list, tuple)):
        return _first_str(x[0]) if x else ""
    return str(x)


class Episode:
    """One in-flight episode driven by an env pool slot."""

    def __init__(self, slot, gamefile, tokenizer, args, templates):
        self.slot = slot
        self.gamefile = gamefile
        self.tokenizer = tokenizer
        self.args = args
        # `templates`: dict(no_his, his, no_his_mem, his_mem). The old 2-tuple is still
        # accepted so older callers keep working.
        if isinstance(templates, dict):
            self.templates = templates
        else:
            self.templates = {"no_his": templates[0], "his": templates[1],
                              "no_his_mem": None, "his_mem": None}
        self.task_type = detect_task_type(gamefile)
        self.enable_thinking = bool(getattr(args, "enable_thinking", False))
        docs = getattr(args, "skill_docs", None)
        self.skill_text = docs[TASK_SKILL_DOC[self.task_type]] if docs else None

    def reset(self, env):
        obs_list, info_list = env.reset()
        self.obs = _first_str(obs_list)
        self.info = info_list
        self.task_description = extract_task_description(self.obs)
        self.admissible = list(info_list["admissible_commands"][0])
        self.history = []
        self.steps_log = []
        self.step_count = 0
        self.won = False
        self.finished = False
        self.env_error = None

    def build_prompt_text(self):
        """The templated user turn (before tokenisation)."""
        admissible_str = ", ".join(self.admissible)
        mem = self.skill_text  # not None only for the E3 "skill in context" baseline
        if not self.history:
            tpl = self.templates["no_his_mem"] if mem is not None else self.templates["no_his"]
            kwargs = dict(task_description=self.task_description,
                          current_observation=self.obs,
                          admissible_actions=admissible_str)
            if mem is not None:
                kwargs["retrieved_memories"] = mem
            prompt_text = tpl.format(**kwargs)
        else:
            recent = self.history[-self.args.history_length:]
            history_str = "\n".join(
                f"[Observation {i+1}: '{h_obs[:300]}', Action {i+1}: '{h_action}']"
                for i, (h_obs, h_action) in enumerate(recent)
            )
            tpl = self.templates["his_mem"] if mem is not None else self.templates["his"]
            kwargs = dict(task_description=self.task_description,
                          step_count=self.step_count,
                          history_length=len(recent),
                          action_history=history_str,
                          current_step=self.step_count + 1,
                          current_observation=self.obs,
                          admissible_actions=admissible_str)
            if mem is not None:
                kwargs["retrieved_memories"] = mem
            prompt_text = tpl.format(**kwargs)
        return prompt_text.lstrip("\n")

    def build_prompt_for_engine(self):
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": self.build_prompt_text()}],
            add_generation_prompt=True, tokenize=False,
            enable_thinking=self.enable_thinking)

    def build_prompt_ids(self):
        # same chat template + enable_thinking flag the SFT stage trained with.
        # NOTE: on transformers 5.x `tokenize=True` does NOT return list[int] for the
        # Qwen2/3 tokenizers (it returns tokenizer `Encoding` objects), so the id path
        # must go through `encode(text)`. `alfworld_eval_vllm.py` simply passes the
        # text of `build_prompt_text()` to vLLM instead.
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": self.build_prompt_text()}],
            add_generation_prompt=True, tokenize=False,
            enable_thinking=self.enable_thinking)
        return self.tokenizer.encode(text, add_special_tokens=False)

    def apply_output(self, env, output_text):
        output_text = output_text.strip()
        step = self.step_count + 1
        matches = re.findall(r"<action>(.*?)</action>", output_text,
                             flags=re.IGNORECASE | re.DOTALL)
        if matches:
            parsed_action, is_parse_valid = matches[-1].strip(), True
        else:
            parsed_action, is_parse_valid = output_text[-30:].strip(), False

        admissible_map = {_norm_action(a): a for a in self.admissible}
        in_admissible = is_parse_valid and _norm_action(parsed_action) in admissible_map
        step_log = {"step": step, "obs": self.obs[:500], "output": output_text,
                    "parsed_action": parsed_action, "is_parse_valid": is_parse_valid,
                    "in_admissible": in_admissible}
        if in_admissible:
            action = admissible_map[_norm_action(parsed_action)]
        else:
            action = parsed_action if is_parse_valid and parsed_action else "look"
            step_log["fallback"] = True
        step_log["final_action"] = action

        text_obs_list, _scores, dones_list, info_list = env.step([action])
        next_obs = _first_str(text_obs_list)
        done = dones_list[0] if isinstance(dones_list, (list, tuple)) else dones_list
        won = bool(info_list["won"][0])
        step_log["won"] = won
        step_log["next_obs"] = next_obs[:500]
        self.steps_log.append(step_log)
        self.history.append((self.obs, action))
        self.step_count += 1
        self.obs = next_obs
        self.admissible = list(info_list["admissible_commands"][0])
        self.won = won
        if done or step >= self.args.max_steps:
            self.finished = True

    def record(self):
        return {
            "gamefile": self.gamefile,
            "task_type": self.task_type,
            "task_description": self.task_description,
            "won": self.won,
            "steps": len(self.steps_log),
            "steps_log": self.steps_log,
            "env_error": self.env_error,
            "run_tag": self.args.run_tag,
            "scale": self.args.scale,
            "bundle": self.args.bundle,
        }


# --------------------------------------------------------------------------- #
# bundle loading / injection
# --------------------------------------------------------------------------- #
def build_backbone(model_name, dtype):
    import torch
    from transformers import AutoModelForCausalLM

    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                   "fp32": torch.float32}[dtype]
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
    model.config.use_cache = True
    model.eval()
    return model


def inject_bundle(backbone, bundle, scale, device):
    """Register the exported (down, up) as a post-block offset on every layer."""
    import torch
    from safetensors.torch import load_file

    from skill_hypernet import HypernetConfig, PostBlockSteerer

    class FrozenBundleSteerer(PostBlockSteerer):
        """Bundle tensors (batch 1, fp32) broadcast to whatever batch `generate` uses.

        `model.generate` changes the effective batch size across waves (episodes
        finish, so a later wave may have B=1 while the previous had B=16), so the
        per-batch view is rebuilt from the stored batch-1 tensors on every call.
        """

        _down0 = None
        _up0 = None

        def load_bundle(self, down, up):
            self._down0, self._up0 = down, up
            self.set_delta(down, up, self._alpha)

        def _apply_delta(self, hidden, slot):
            if self._down is None:
                return hidden
            d0, u0 = self._down0, self._up0
            if d0.dtype != hidden.dtype:
                d0, u0 = d0.to(hidden.dtype), u0.to(hidden.dtype)
                self._down0, self._up0 = d0, u0
            b = hidden.shape[0]
            self._down = d0 if d0.shape[0] == b else d0.expand(b, -1, -1, -1)
            self._up = u0 if u0.shape[0] == b else u0.expand(b, -1, -1, -1)
            return super()._apply_delta(hidden, slot)


    with open(os.path.join(bundle, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    tensors = load_file(os.path.join(bundle, "adapter.safetensors"))
    layers = list(manifest["layers"])
    rank, alpha = int(manifest["rank"]), float(manifest["alpha"])

    cfg = HypernetConfig(
        num_layers=backbone.config.num_hidden_layers,
        hidden_size=backbone.config.hidden_size,
        adapter_rank=rank,
        delta_mode="lowrank",
        activation="silu" if manifest["use_silu"] else None,
        output_scale=1.0,
    )
    steerer = FrozenBundleSteerer(cfg, layers)
    down = torch.stack([tensors[f"layer{i}.down"] for i in layers]).unsqueeze(0).to(device)
    up = torch.stack([tensors[f"layer{i}.up"] for i in layers]).unsqueeze(0).to(device)
    alpha_t = torch.full((len(layers),), alpha * scale, device=device)
    steerer._alpha = alpha_t
    steerer.load_bundle(down, up)
    steerer.register_hooks(backbone)
    info = {"bundle": os.path.abspath(bundle), "layers": len(layers), "rank": rank,
            "alpha": alpha, "scale": scale, "use_silu": bool(manifest["use_silu"]),
            "source_checkpoint": manifest.get("source_checkpoint")}
    return steerer, info


class StopOnAction:
    """Stop the wave as soon as every live sequence has emitted `</action>`."""

    def __init__(self, tokenizer, window=8):
        self.tokenizer = tokenizer
        self.window = window
        self.NEED = "</action>"

    def __call__(self, input_ids, scores, **kwargs):
        for row in input_ids:
            tail = self.tokenizer.decode(row[-self.window:], skip_special_tokens=True)
            if self.NEED not in tail:
                return False
        return True


# --------------------------------------------------------------------------- #
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

    import torch
    import yaml

    from alfworld.agents.environment import get_environment
    from transformers import AutoTokenizer

    torch.manual_seed(0)
    device = "cuda"
    if args.task_types == "all":
        wanted = list(TASK_ORDER)
    else:
        wanted = [t.strip() for t in args.task_types.split(",") if t.strip()]
        for t in wanted:
            assert t in TASK_ORDER, f"unknown task type {t!r}"

    # ----- tokenizer + backbone + injection -----
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left",
                                              use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.chat_template == "latentskill":
        sys.path.insert(0, os.path.join(args.latentskill_root, "evals_vllm"))
        from qwen_chat_template import REPO_CHAT_TEMPLATE  # noqa: E402
        tokenizer.chat_template = REPO_CHAT_TEMPLATE
    backbone = build_backbone(args.model_name, args.dtype).to(device)
    inject_info = None
    if args.bundle and args.scale != 0.0:
        _steerer, inject_info = inject_bundle(backbone, args.bundle, args.scale, device)
    print(f"[engine] HF transformers, dtype={args.dtype}, bundle={args.bundle}, "
          f"scale={args.scale}", flush=True)
    print(f"[inject] {json.dumps(inject_info)}", flush=True)
    if args.verify_injection and inject_info is not None:
        d = verify_injection(backbone, tokenizer, _steerer, device)
        inject_info["verify_max_abs_logit_diff"] = d
        print(f"[verify] max|Δlogit| (delta on vs off, fixed prompt) = {d:.4f}", flush=True)

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
        if t in wanted:
            game_files.extend(by_type[t][: args.max_games] if args.max_games else by_type[t])
    print(f"[eval] split={args.split} task_types={wanted} n_games={len(game_files)} "
          f"({ {t: len(by_type[t][:args.max_games] if args.max_games else by_type[t]) for t in wanted} })",
          flush=True)

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

    if args.skill_in_context:
        args.skill_docs = load_skill_docs(args.skill_doc_dir)
        print(f"[in-context] skill docs loaded: {sorted(args.skill_docs)}", flush=True)
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
    n_done = len(done_games)
    n_total = len(game_files)
    gen_calls = total_new_tokens = 0
    t0 = time.time()
    stop_criteria = StopOnAction(tokenizer)

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

        prompts = [s["episode"].build_prompt_ids() for s in live]
        maxlen = max(len(p) for p in prompts)
        input_ids = torch.full((len(prompts), maxlen), tokenizer.pad_token_id,
                               dtype=torch.long)
        attn = torch.zeros((len(prompts), maxlen), dtype=torch.long)
        for i, p in enumerate(prompts):
            input_ids[i, maxlen - len(p):] = torch.tensor(p, dtype=torch.long)
            attn[i, maxlen - len(p):] = 1
        input_ids, attn = input_ids.to(device), attn.to(device)
        with torch.no_grad():
            out = backbone.generate(
                input_ids=input_ids, attention_mask=attn, do_sample=False,
                max_new_tokens=args.max_new_tokens,
                stopping_criteria=[stop_criteria], use_cache=True,
            )
        new = out[:, maxlen:]
        gen_calls += 1
        total_new_tokens += int(new.numel())
        texts = tokenizer.batch_decode(new, skip_special_tokens=True)

        for s, text in zip(live, texts):
            ep = s["episode"]
            try:
                ep.apply_output(s["env"], text)
            except Exception as exc:  # env died: record failure, rebuild the slot env
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

    # ----- summary over this run's task types -----
    summary = {"run_tag": args.run_tag, "split": args.split, "scale": args.scale,
               "bundle": args.bundle, "inject": inject_info,
               "chat_template": args.chat_template, "enable_thinking": args.enable_thinking,
               "skill_in_context": bool(getattr(args, "skill_in_context", False)),
               "task_types": wanted, "max_games_per_type": args.max_games,
               "concurrency": args.concurrency, "max_new_tokens": args.max_new_tokens,
               "max_steps": args.max_steps, "history_length": args.history_length,
               "model": args.model_name, "engine": "hf_transformers",
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
    summary_path = out_dir / f"summary_{args.split}_{args.run_tag}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 64)
    for t, d in summary["per_task"].items():
        print(f"{TASK_TYPE_NAMES[t]:<24} {d['success']:>4}/{d['total']:<4} {d['rate']:.4f}")
    if all_res:
        o = summary["overall"]
        print(f"{'Overall':<24} {o['success']:>4}/{o['total']:<4} {o['rate']:.4f}")
    print(f"wall={summary['wall_time_sec']}s gen_calls={gen_calls}")
    print(f"[saved] {detail_path}\n[saved] {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
