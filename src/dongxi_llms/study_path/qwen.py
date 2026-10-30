"""Qwen3 plumbing shared by the study-path notebooks (the parts that are not exercises).

Data loaders for the two cached public preference datasets, chat rendering, sampling and
per-token log-probabilities. The notebooks print these functions with ``inspect.getsource``
when they are first used, so nothing important stays hidden. Behaviour matches Day 15 · 04
(``notebooks/day-15/04_preference_alignment_end_to_end.ipynb``), including its fixes:
sampling at temperature 1 from the full distribution, explicit position ids under left
padding, and reward-model text that ends on ``<|im_end|>``.

Precision: the loaders default to bf16, which is right for frozen models (reference, grader).
A model that AdamW trains needs fp32 master weights: load it with ``dtype=torch.float32`` and
run forward passes under ``torch.autocast("cuda", dtype=torch.bfloat16)``. In bf16 a weight near
0.02 has a spacing of about 1.2e-4, so steps of 1e-6 to 1e-5 round back to the same value and
most updates are lost (measured in S1, S3 and S7: 2–10% of weights change instead of most).
``rm_scores`` runs under ``no_grad``; training code needs its own scoring call.
"""
from __future__ import annotations

import gc
import gzip
import json
import re

import torch

MODEL = "Qwen/Qwen3-0.6B"


# --------------------------------------------------------------------------- models
def load_tokenizer(model: str = MODEL):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model, local_files_only=True)


def load_chat_model(model: str = MODEL, device: str = "cuda", dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(model, local_files_only=True, dtype=dtype).to(device)


def load_reward_model(tok, model: str = MODEL, device: str = "cuda", dtype=torch.bfloat16):
    """Qwen3 body + a new Linear(hidden → 1) head; ``lm_head`` is discarded (D30)."""
    from transformers import AutoModelForSequenceClassification
    rm = AutoModelForSequenceClassification.from_pretrained(model, num_labels=1, local_files_only=True,
                                                            dtype=dtype).to(device)
    rm.config.pad_token_id = tok.pad_token_id
    return rm


def free():
    """Release GPU memory after ``del``-ing models at the top level."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def gib(n_bytes: int) -> float:
    return n_bytes / 2**30


# --------------------------------------------------------------------------- data
def parse_transcript(text: str) -> list[dict]:
    """HH-RLHF '\\n\\nHuman: …\\n\\nAssistant: …' → chat messages."""
    parts = re.split(r"\n\n(Human|Assistant): ", text)[1:]
    return [{"role": "user" if who == "Human" else "assistant", "content": said.strip()}
            for who, said in zip(parts[0::2], parts[1::2])]


def to_pair(row: dict):
    """{prompt, chosen, rejected}, or None unless the transcripts differ only in the last reply."""
    c, r = parse_transcript(row["chosen"]), parse_transcript(row["rejected"])
    if len(c) < 2 or c[:-1] != r[:-1] or c[-1]["role"] != "assistant" or r[-1]["role"] != "assistant":
        return None
    if c[-1]["content"] == r[-1]["content"]:
        return None
    return {"prompt": c[:-1], "chosen": c[-1]["content"], "rejected": r[-1]["content"]}


def hh_rows(split: str) -> list[dict]:
    """Anthropic HH-RLHF helpful-base (MIT), from the local Hugging Face cache."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("Anthropic/hh-rlhf", f"helpful-base/{split}.jsonl.gz", repo_type="dataset")
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f]


def hh_pairs(split: str) -> list[dict]:
    return [p for p in map(to_pair, hh_rows(split)) if p]


def first_question(row: dict):
    turns = parse_transcript(row["chosen"])
    return turns[0]["content"] if turns and turns[0]["role"] == "user" else None


def hh_questions(split: str, start: int = 0, min_len: int = 15, max_len: int = 200) -> list[str]:
    """Single-turn user questions (phase-2 RL prompts); train[20000:] is never used by reward models."""
    return [q for q in map(first_question, hh_rows(split)[start:]) if q and min_len < len(q) < max_len]


def ultrafeedback_test_prefs():
    """UltraFeedback binarized test_prefs (MIT): GPT-4-judged pairs with scores, as a DataFrame."""
    import pandas as pd
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("HuggingFaceH4/ultrafeedback_binarized", "data/test_prefs-00000-of-00001.parquet",
                           repo_type="dataset")
    return pd.read_parquet(path)


# --------------------------------------------------------------------------- text
def render(tok, prompt: list[dict], answer: str) -> str:
    """Full conversation with Qwen3's template, thinking off, ending on <|im_end|> (D28)."""
    text = tok.apply_chat_template(prompt + [{"role": "assistant", "content": answer}], tokenize=False,
                                   enable_thinking=False)
    return text.removesuffix("\n")


def chat_prompt(tok, question) -> str:
    """Generation prompt for one question (str) or a message list, thinking off."""
    messages = [{"role": "user", "content": question}] if isinstance(question, str) else question
    return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)


@torch.no_grad()
def rm_scores(rm, tok, prompts, answers, bs: int = 16, max_length: int = 320) -> torch.Tensor:
    """One reward-model score per (prompt, answer); read at the last non-pad token (D28)."""
    out = []
    for i in range(0, len(answers), bs):
        texts = [render(tok, p, a) for p, a in zip(prompts[i:i + bs], answers[i:i + bs])]
        enc = tok(texts, return_tensors="pt", padding=True, padding_side="right", truncation=True,
                  max_length=max_length).to(rm.device)
        out.append(rm(**enc).logits.squeeze(-1).float())
    return torch.cat(out)


# --------------------------------------------------------------------------- sampling
@torch.no_grad()
def rollout(model, tok, questions, max_new: int = 96, temperature: float = 1.0) -> dict:
    """Sample one answer per question from π itself (temperature 1, no top-k/top-p).

    Returns ids [B, P+T], attention mask, position ids, prompt length P, response mask [B, T]
    (1 through the first <|im_end|>), index of the last response token, and truncation flags.
    """
    im_end, pad = tok.convert_tokens_to_ids("<|im_end|>"), tok.pad_token_id
    enc = tok([chat_prompt(tok, q) for q in questions], return_tensors="pt", padding=True,
              padding_side="left").to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=temperature, top_k=0,
                         top_p=1.0, pad_token_id=pad)
    P, resp = enc.input_ids.shape[1], out[:, enc.input_ids.shape[1]:]
    ended = (resp == im_end) | (resp == pad)
    last = torch.where(ended.any(1), ended.float().argmax(1),
                       torch.full_like(ended[:, 0], resp.shape[1] - 1, dtype=torch.long))
    mask = (torch.arange(resp.shape[1], device=out.device)[None] <= last[:, None]).float()
    attn = torch.cat([enc.attention_mask, mask.long()], 1)
    pos = (attn.cumsum(1) - 1).clamp(min=0)        # left padding: positions start at the first real token
    return {"ids": out, "attn": attn, "pos": pos, "P": P, "mask": mask, "last": last,
            "questions": list(questions), "truncated": ~ended.any(1)}


def response_texts(tok, ro: dict) -> list[str]:
    P = ro["P"]
    return [tok.decode(ro["ids"][i, P:P + int(ro["last"][i]) + 1], skip_special_tokens=True)
            for i in range(ro["ids"].shape[0])]


def token_logps(model, ro: dict) -> torch.Tensor:
    """log π(a_t | s_t) for every response token, [B, T]: logits at P−1+t predict token P+t."""
    logits = model(input_ids=ro["ids"], attention_mask=ro["attn"],
                   position_ids=ro["pos"]).logits[:, ro["P"] - 1:-1].float()
    return torch.log_softmax(logits, -1).gather(-1, ro["ids"][:, ro["P"]:, None]).squeeze(-1)


@torch.no_grad()
def score_rollout(rm, ro: dict) -> torch.Tensor:
    """Reward-model score of each sampled conversation, [B] (policy and RM share the tokenizer)."""
    return rm(input_ids=ro["ids"], attention_mask=ro["attn"], position_ids=ro["pos"]).logits.squeeze(-1).float()


def answer_logps(model, tok, pairs, max_len: int = 320) -> torch.Tensor:
    """Σ log π(answer tokens + <|im_end|>) for chosen and rejected answers, [B, 2] (Day 15 · 04 Part G)."""
    pad = tok.pad_token_id
    seqs, starts = [], []
    for p in pairs:
        prompt_ids = tok(chat_prompt(tok, p["prompt"]), add_special_tokens=False).input_ids
        for ans in (p["chosen"], p["rejected"]):
            ans_ids = tok(ans + "<|im_end|>", add_special_tokens=False).input_ids
            seq = (prompt_ids + ans_ids)[-max_len:]
            seqs.append(seq)
            starts.append(max(1, len(seq) - len(ans_ids)))
    L = max(map(len, seqs))
    ids = torch.full((len(seqs), L), pad)
    attn = torch.zeros_like(ids)
    amask = torch.zeros(len(seqs), L)
    for i, (s, st) in enumerate(zip(seqs, starts)):
        ids[i, :len(s)] = torch.tensor(s)
        attn[i, :len(s)] = 1
        amask[i, st:len(s)] = 1
    dev = next(model.parameters()).device
    ids, attn, amask = ids.to(dev), attn.to(dev), amask.to(dev)
    logits = model(input_ids=ids, attention_mask=attn).logits[:, :-1].float()
    lp = torch.log_softmax(logits, -1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    return (lp * amask[:, 1:]).sum(1).view(-1, 2)
