"""Config objects that mirror `configs/models/qwen3_8b.yaml` + the SFT launch script.

`latentskill.models.hypernetwork.SkillHypernetworkTransformer` only reads a handful
of attributes, so a plain `SimpleNamespace` tree is enough -- no omegaconf/hydra
needed (neither is installed in the `steerling` env and it must not be modified).
"""

from types import SimpleNamespace
from typing import Any, Dict


class NS(SimpleNamespace):
    """SimpleNamespace that also supports `key in cfg`
    (LatentSkill's collators test `\"pretrain\" in self.cfg`)."""

    def __contains__(self, key):
        return key in self.__dict__

# Values taken verbatim from
#   LatentSkill/configs/models/qwen3_8b.yaml                  (architecture + optim)
#   .../checkpoint-epoch-10/launch_script.sh                  (SFT launch overrides)
#   .../training_log.txt  (pretrain: completion_freq/ratios)
LS_TRAIN_DEFAULTS: Dict[str, Any] = {
    # ---- optim (launch script) ----
    "learning_rate": 1e-5,
    "weight_decay": 0.01,
    "warmup_steps": 400,
    "grad_clip_norm": 1.0,
    "num_epochs": 1,
    # ---- run ----
    "seed": 42,
    "gradient_accumulation_steps": 8,
    "use_gradient_checkpoint": True,
    # ---- data (launch script) ----
    "train_batch_size": 1,
    "eval_batch_size": 1,
    "context_max_length": 4096,
    "conversation_max_length": 4096,
    "num_workers": 0,
    "skill_ift_val_size": 200,
    # ---- model ----
    "lora_r": 8,
    "metalora_r": 128,
    "hypernet_type": "transformer",
    "hypernet_method": "rl",
    "gen_num_layers": 4,
    "gen_nhead": 32,
    "gen_ff": 8192,
    "hypernet_scale": 0.001,
    # ---- pretrain collator (training_log.txt) ----
    "completion_freq": 0.5,
    "max_completion_ratio": 0.3,
    "min_completion_ratio": 0.1,
}

# Chat template used by LatentSkill's `train_compiler.py` (copied verbatim so the
# rendered training sequences are byte-identical to the reference run).
REPO_CHAT_TEMPLATE = (
    "{%- if tools %}\n"
    "    {{- '<|im_start|>system\\n' }}\n"
    "    {%- if messages[0].role == 'system' %}\n"
    "        {{- messages[0].content + '\\n\\n' }}\n"
    "    {%- endif %}\n"
    '    {{- "# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\n'
    'You are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}\n'
    "    {%- for tool in tools %}\n"
    '        {{- "\\n" }}\n'
    "        {{- tool | tojson }}\n"
    "    {%- endfor %}\n"
    '    {{- "\\n</tools>\\n\\nFor each function call, return a json object with function name '
    'and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\"name\\": '
    '<function-name>, \\"arguments\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}\n'
    "{%- else %}\n"
    "    {%- if messages[0].role == 'system' %}\n"
    "        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n"
    "    {%- endif %}\n"
    "{%- endif %}\n"
    "{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n"
    "{%- for message in messages[::-1] %}\n"
    "    {%- set index = (messages|length - 1) - loop.index0 %}\n"
    '    {%- if ns.multi_step_tool and message.role == "user" and message.content is string '
    "and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n"
    "        {%- set ns.multi_step_tool = false %}\n"
    "        {%- set ns.last_query_index = index %}\n"
    "    {%- endif %}\n"
    "{%- endfor %}\n"
    "{%- for message in messages %}\n"
    "    {%- if message.content is string %}\n"
    "        {%- set content = message.content %}\n"
    "    {%- else %}\n"
    "        {%- set content = '' %}\n"
    "    {%- endif %}\n"
    '    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}\n'
    "        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>\\n' }}\n"
    '    {%- elif message.role == "assistant" %}\n'
    "        {%- set reasoning_content = '' %}\n"
    "        {%- if message.reasoning_content is string %}\n"
    "            {%- set reasoning_content = message.reasoning_content %}\n"
    "        {%- else %}\n"
    "            {%- if '</think>' in content %}\n"
    "                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n')"
    ".split('<think>')[-1].lstrip('\\n') %}\n"
    "                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}\n"
    "            {%- endif %}\n"
    "        {%- endif %}\n"
    "        {%- if loop.index0 > ns.last_query_index %}\n"
    "            {%- if (loop.last or (not loop.last and reasoning_content)) and "
    "(enable_thinking is not defined or enable_thinking != false) %}\n"
    "                {{- '<|im_start|>' + message.role + '\\n<think>\\n' + "
    "reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}\n"
    "            {%- else %}\n"
    "                {{- '<|im_start|>' + message.role + '\\n' + content }}\n"
    "            {%- endif %}\n"
    "        {%- else %}\n"
    "            {{- '<|im_start|>' + message.role + '\\n' + content }}\n"
    "        {%- endif %}\n"
    "        {%- if message.tool_calls %}\n"
    "            {%- for tool_call in message.tool_calls %}\n"
    "                {%- if (loop.first and content) or (not loop.first) %}\n"
    "                    {{- '\\n' }}\n"
    "                {%- endif %}\n"
    "                {%- if tool_call.function %}\n"
    "                    {%- set tool_call = tool_call.function %}\n"
    "                {%- endif %}\n"
    '                {{- \'<tool_call>\\n{"name": "\' }}\n'
    "                {{- tool_call.name }}\n"
    '                {{- \'", "arguments": \' }}\n'
    "                {%- if tool_call.arguments is string %}\n"
    "                    {{- tool_call.arguments }}\n"
    "                {%- else %}\n"
    "                    {{- tool_call.arguments | tojson }}\n"
    "                {%- endif %}\n"
    "                {{- '}\\n</tool_call>' }}\n"
    "            {%- endfor %}\n"
    "        {%- endif %}\n"
    "        {{- '<|im_end|>\\n' }}\n"
    '    {%- elif message.role == "tool" %}\n'
    "        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n"
    "            {{- '<|im_start|>user' }}\n"
    "        {%- endif %}\n"
    '        {{- \'\\n<tool_response>\\n\' }}\n'
    "        {{- content }}\n"
    '        {{- \'\\n</tool_response>\' }}\n'
    "        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n"
    "            {{- '<|im_end|>\\n' }}\n"
    "        {%- endif %}\n"
    "    {%- endif %}\n"
    "{%- endfor %}\n"
    "{%- if add_generation_prompt %}\n"
    "    {{- '<|im_start|>assistant\\n' }}\n"
    "    {%- if enable_thinking is not defined or enable_thinking != false %}\n"
    "        {{- '<think>\\n\\n</think>\\n\\n' }}\n"
    "    {%- endif %}\n"
    "{%- endif %}"
)


def _encoder_cfg(d_model: int, nhead: int, ff: int) -> Dict[str, Any]:
    return dict(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=ff,
        dropout=0.0,
        activation="gelu",
        layer_norm_eps=1e-5,
        batch_first=True,
        norm_first=False,
        bias=True,
    )


def build_ls_cfg(
    num_layers: int = 36,
    hidden_size: int = 4096,
    lora_r: int = 8,
    metalora_r: int = 128,
    num_mem_token: int = 16,
    gen_num_layers: int = 4,
    gen_nhead: int = 32,
    gen_ff: int = 8192,
    hypernet_scale: float = 0.001,
    method: str = "rl",
    **train_overrides,
) -> SimpleNamespace:
    """Namespace with the same attribute paths as LatentSkill's hydra config."""
    enc = _encoder_cfg(hidden_size, gen_nhead, gen_ff)
    tcfg = NS(
        encoder_cfg=dict(enc),
        couple_encoder_cfg=NS(**enc),
        layer_transformer_first=True,
        mean_pool_size=1,
        num_layers=gen_num_layers,
        couple_num_layers=0,
        scale=hypernet_scale,
    )
    cfg = NS(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_mem_token=num_mem_token,
        model=NS(
            lora_r=lora_r,
            metalora_r=metalora_r,
            ift_additional_metalora_r=-1,
            num_mem_token=num_mem_token,
        ),
        hypernetwork=NS(
            type="transformer",
            method=method,
            transformer_cfg=tcfg,
            linear_cfg=NS(num_layers=4, linear_hidden_dim=8192,
                                       scale=hypernet_scale, bias=False),
            linear_gate_cfg=NS(num_layers=4, linear_hidden_dim=8192,
                                            scale=hypernet_scale, bias=False),
        ),
        optim=NS(adapter_reg=0.0),
        run=NS(seed=42, use_amp=False, gradient_accumulation_steps=8,
                            device="cuda", use_gradient_checkpoint=True),
        pretrain=NS(completion_freq=0.5, max_completion_ratio=0.3,
                                 min_completion_ratio=0.1),
        data=NS(
            context_max_length=4096,
            conversation_max_length=4096,
            train_batch_size=1,
            eval_batch_size=1,
            num_workers=0,
            skill_ift_val_size=200,
        ),
        curriculum=NS(
            stages=[NS(start_epoch=1, end_epoch=2, ratios={1: 1.0})]
        ),
    )
    for k, v in train_overrides.items():
        setattr(cfg, k, v)
    return cfg


def steer_output_numel(num_layers: int, rank: int, hidden_size: int) -> int:
    """Generated scalars per sample: 36 layers x (down rH + up Hr)."""
    return num_layers * 2 * rank * hidden_size


def mem_tokens_for_steer(rank: int) -> int:
    """LatentSkill's own rule: num_mem_token = output_numel // (hidden * layers)."""
    return 2 * rank
