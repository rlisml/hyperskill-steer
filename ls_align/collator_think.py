"""Thinking-protocol stage-2 collator.

The native LatentSkill SkillInstructionCollator renders the trajectory with
add_generation_prompt=False and enable_thinking=False: the assistant turn is
emitted as an assistant header followed directly by the action, and the
template drops the expert think block.  Serving (alfworld_eval_vllm.py with
--chat-template latentskill --enable-thinking 1) instead uses
add_generation_prompt=True and enable_thinking=True, so the generation prompt
ends with an already-closed empty think block.  Training and inference thus
disagreed on the token distribution, which is
exactly the mismatch fixed here.

Protocol of this collator (same semantics as skill_data.py
_prompt_and_target(full_turn=True), the --sft-thinking path):

    prompt = tokenizer.apply_chat_template(
                 conversations[:-1], add_generation_prompt=True,
                 tokenize=True, enable_thinking=True)
    labels = -100 over the prompt, then the expert reply tokens plus EOS.

The prompt is byte-identical to the ALFWorld serving prompt, and the
supervised tokens are exactly the ones the model must generate at inference.
"""

import torch


class ThinkingSkillInstructionCollator:
    """Skill document -> hypernet; trajectory -> base model (thinking protocol)."""

    def __init__(self, tokenizer, context_max_length=1024,
                 conversation_max_length=1024):
        self.tokenizer = tokenizer
        self.context_max_length = int(context_max_length)
        self.conversation_max_length = int(conversation_max_length)

    def __call__(self, batch):
        tok = self.tokenizer
        evidence_texts = [item["evidence"] for item in batch]
        convs_list = [item["conversations"] for item in batch]
        if not isinstance(convs_list[0], list):
            convs_list = [list(c) for c in convs_list]
        evidence_enc = tok(
            evidence_texts,
            max_length=self.context_max_length,
            truncation=True,
            return_tensors="pt",
            padding="max_length",
        )
        eos = tok.eos_token_id
        seqs, labs = [], []
        for convs in convs_list:
            assert convs[-1]["role"] == "assistant", "last turn must be assistant"
            prompt_ids = list(tok.apply_chat_template(
                convs[:-1], add_generation_prompt=True, tokenize=True,
                enable_thinking=True))
            target_ids = list(tok(convs[-1]["content"],
                                  add_special_tokens=False)["input_ids"])
            if eos is not None:
                target_ids = target_ids + [eos]
            budget = self.conversation_max_length - len(prompt_ids)
            if budget <= 0:
                prompt_ids = prompt_ids[len(prompt_ids) - self.conversation_max_length + 8:]
                budget = self.conversation_max_length - len(prompt_ids)
            target_ids = target_ids[:budget]
            seqs.append(list(prompt_ids) + list(target_ids))
            labs.append([-100] * len(prompt_ids) + list(target_ids))
        maxlen = max(1, max(len(s) for s in seqs))
        pad_id = tok.pad_token_id
        input_ids, labels, attn = [], [], []
        for seq, lab in zip(seqs, labs):
            pad = maxlen - len(seq)
            input_ids.append(seq + [pad_id] * pad)
            labels.append(lab + [-100] * pad)
            attn.append([1] * len(seq) + [0] * pad)
        return {
            "evidence": evidence_texts,
            "evidence_ids": evidence_enc["input_ids"],
            "evidence_attention_mask": evidence_enc["attention_mask"],
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "input_attention_mask": torch.tensor(attn, dtype=torch.long),
        }
