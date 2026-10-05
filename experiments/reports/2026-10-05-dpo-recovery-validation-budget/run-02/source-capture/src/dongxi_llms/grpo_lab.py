"""Readable CPU GRPO microscope, including a real autoregressive decoder update.

The objective is explicitly response-mean token clipping, population group std,
and exact forward categorical KL on sampled prefix states. It is a teaching
variant, not a compatibility promise for a framework's current GRPO defaults.
"""
from copy import deepcopy
import hashlib
import re

import torch
from torch.nn import functional as F

from dongxi_llms.decoder_lab import DecoderConfig, TinyDecoder

PAD, BOS, EOS, EQ, NUMBER_START, VOCAB = 0, 1, 2, 3, 4, 12
VERIFIER_VERSION = "one-integer-then-eos-v1"
DEV_PAIRS = ((0, 2), (1, 3), (2, 0), (3, 1))
TRAIN_PAIRS = tuple((a, b) for a in range(4) for b in range(4)
                    if (a, b) not in DEV_PAIRS)


def group_advantages(rewards, normalize=True, eps=1e-8):
    """rewards [prompts,G]; detach the complete reward/baseline calculation."""
    if rewards.ndim != 2 or rewards.shape[1] < 2 or eps <= 0:
        raise ValueError("Need [prompts,G>=2] and positive epsilon")
    if not torch.isfinite(rewards).all():
        raise ValueError("Rewards must be finite")
    centered = rewards.detach() - rewards.detach().mean(-1, keepdim=True)
    std = rewards.detach().std(-1, correction=0, keepdim=True)
    advantages = centered / (std + eps) if normalize else centered
    # An exactly constant group has no relative policy signal.
    return torch.where(std == 0, torch.zeros_like(advantages), advantages)


def exact_kl(logits, reference_logits):
    """D_KL(policy || reference), over all vocabulary outcomes at each state."""
    logp = logits.log_softmax(-1)
    logq = reference_logits.detach().log_softmax(-1)
    return (logp.exp() * (logp - logq)).sum(-1)


def clipped_objective(logp, old_logp, advantages, mask, epsilon=.2,
                      kl=None, beta=0.):
    """All token tensors [N,T]; advantages [N]; every response has >=1 token.

    Returns a loss for minimization. Prompt/padding positions are excluded.
    old_logp and advantages are ALWAYS detached; kl retains its policy gradient.
    """
    if logp.shape != old_logp.shape or logp.shape != mask.shape:
        raise ValueError("Token shapes must agree")
    if not (0 < epsilon < 1 and beta >= 0) or advantages.shape != logp.shape[:1]:
        raise ValueError("Invalid objective geometry or coefficients")
    counts = mask.sum(-1)
    if (counts <= 0).any():
        raise ValueError("Empty responses cannot be silently normalized")
    ratio = (logp - old_logp.detach()).exp()
    advantage = advantages.detach()[:, None]
    surrogate = torch.minimum(ratio * advantage,
                              ratio.clamp(1-epsilon, 1+epsilon) * advantage)
    terms = surrogate if kl is None else surrogate - beta * kl
    return -((terms * mask).sum(-1) / counts).mean()


def verify_integer(text, answer):
    """Strict demonstration parser: one signed base-10 integer, surrounding space.

    No eval, substring credit, extra answer, decimal, Unicode digit, or rationale.
    """
    if not isinstance(text, str) or len(text) > 64:
        return False
    if re.fullmatch(r"\s*[+-]?[0-9]+\s*", text, flags=re.ASCII) is None:
        return False
    return int(text.strip()) == answer


def verify_tokens(tokens, answer):
    active = list(tokens)
    if EOS not in active:
        return 0.
    end = active.index(EOS)
    return float(end == 1 and active[0] == NUMBER_START + answer)


def prompt_tensor(pairs):
    return torch.tensor([[BOS, NUMBER_START+a, NUMBER_START+b, EQ]
                         for a, b in pairs], dtype=torch.long)


def make_policy(seed=2223):
    config = DecoderConfig(vocab=VOCAB, width=16, heads=4, kv_heads=2,
                           head_dim=4, layers=1, hidden=32, max_length=12,
                           modern=True, tied=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return TinyDecoder(config)


def frozen_copy(model):
    snapshot = deepcopy(model).eval()
    for parameter in snapshot.parameters():
        parameter.requires_grad_(False)
    return snapshot


def response_logits(model, prompts, responses):
    """Logits predicting each response token, correctly shifted by one position."""
    complete = torch.cat((prompts, responses), -1)
    start = prompts.shape[1] - 1
    return model(complete[:, :-1])[:, start:start+responses.shape[1]]


@torch.no_grad()
def rollout(model, prompts, generator, max_new_tokens=3):
    if max_new_tokens < 1 or prompts.shape[1]+max_new_tokens > model.cfg.max_length:
        raise ValueError("Generation must fit model context")
    current = prompts.clone()
    finished = torch.zeros(len(prompts), dtype=torch.bool)
    tokens, valid = [], []
    for _ in range(max_new_tokens):
        probabilities = model(current)[:, -1].softmax(-1)
        token = torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)
        token = torch.where(finished, torch.full_like(token, PAD), token)
        valid.append(~finished)
        tokens.append(token)
        finished |= token == EOS
        current = torch.cat((current, token[:, None]), -1)
    responses = torch.stack(tokens, -1)
    mask = torch.stack(valid, -1)
    logits = response_logits(model, prompts, responses)
    old_logp = logits.log_softmax(-1).gather(-1, responses[..., None]).squeeze(-1)
    return responses, mask, old_logp.detach()


def warm_start(model, updates=60):
    """Original 12 procedural demonstrations; held-out prompt pairs excluded."""
    prompts = prompt_tensor(TRAIN_PAIRS)
    responses = torch.tensor([[NUMBER_START+a+b, EOS] for a, b in TRAIN_PAIRS])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.015, weight_decay=0.)
    history = []
    for _ in range(updates):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(response_logits(model, prompts, responses).flatten(0, 1),
                               responses.flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        history.append(float(loss.detach()))
    return history


@torch.no_grad()
def evaluate(model, pairs):
    prompts = prompt_tensor(pairs)
    responses = []
    current = prompts
    for _ in range(3):
        token = model(current)[:, -1].argmax(-1)
        responses.append(token)
        current = torch.cat((current, token[:, None]), -1)
    responses = torch.stack(responses, -1)
    rewards = [verify_tokens(row.tolist(), a+b)
               for row, (a, b) in zip(responses, pairs)]
    return {"accuracy": sum(rewards)/len(rewards), "n": len(rewards),
            "rows": [{"prompt": f"{a}+{b}", "expected": a+b,
                      "tokens": row.tolist(), "reward": reward}
                     for row, (a, b), reward in zip(responses, pairs, rewards)]}


def decoder_rlvr(group_size=4, updates=12, seed=2223, beta=.02,
                 normalize=True, max_new_tokens=3):
    """Bounded real decoder GRPO, fresh snapshot and ONE update per rollout group.

    Exact KL is measured on states visited by the detached behavior rollout.
    The returned evaluation is descriptive on four held-out arithmetic prompts.
    No parameter/recipe selection is based on those held-out measurements.
    """
    if group_size < 2 or updates < 1:
        raise ValueError("Need positive updates and group_size >=2")
    model = make_policy(seed)
    sft_history = warm_start(model)
    reference = frozen_copy(model)
    initial = {"train": evaluate(model, TRAIN_PAIRS), "heldout": evaluate(model, DEV_PAIRS)}
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=0.)
    generator = torch.Generator().manual_seed(seed+1)
    history = []
    for update in range(updates):
        # Fixed rotating prompt schedule, identical initial weights across ablations.
        pairs = [TRAIN_PAIRS[(2*update+j) % len(TRAIN_PAIRS)] for j in range(2)]
        repeated = [pair for pair in pairs for _ in range(group_size)]
        prompts = prompt_tensor(repeated)
        old = frozen_copy(model)
        responses, mask, old_logp = rollout(old, prompts, generator, max_new_tokens)
        rewards = torch.tensor([verify_tokens(row.tolist(), a+b)
                                for row, (a, b) in zip(responses, repeated)])
        advantages = group_advantages(rewards.reshape(2, group_size), normalize).flatten()
        logits = response_logits(model, prompts, responses)
        with torch.no_grad():
            reference_logits = response_logits(reference, prompts, responses)
        logp = logits.log_softmax(-1).gather(-1, responses[..., None]).squeeze(-1)
        kl = exact_kl(logits, reference_logits)
        loss = clipped_objective(logp, old_logp, advantages, mask, kl=kl, beta=beta)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
        optimizer.step()
        with torch.no_grad():
            after = response_logits(model, prompts, responses)
            p = after.softmax(-1)
            entropy = -(p * after.log_softmax(-1)).sum(-1)
            after_logp = after.log_softmax(-1).gather(-1, responses[..., None]).squeeze(-1)
            ratios = (after_logp-old_logp).exp()
            n = mask.sum()
            history.append({"update": update+1, "behavior_version": update,
                            "policy_version": update+1, "loss": float(loss.detach()),
                            "mean_reward": float(rewards.mean()),
                            "zero_variance_fraction": float((rewards.reshape(2, group_size).std(-1, correction=0)==0).float().mean()),
                            "mean_length": float(mask.sum(-1).float().mean()),
                            "valid_response_tokens": int(n), "gradient_norm": gradient,
                            "entropy_nats": float((entropy*mask).sum()/n),
                            "exact_kl_nats": float((exact_kl(after, reference_logits)*mask).sum()/n),
                            "ratio_min": float(ratios[mask].min()),
                            "ratio_max": float(ratios[mask].max()),
                            "clip_fraction": float(((ratios-1).abs()>.2)[mask].float().mean())})
    return {"mode": "cpu-mechanism", "architecture": "one-layer TinyDecoder,16-wide",
            "seed": seed, "group_size": group_size, "updates": updates,
            "warm_start_updates": len(sft_history), "sft_first_loss": sft_history[0],
            "sft_final_loss": sft_history[-1], "beta": beta,
            "advantage_std": "population" if normalize else "none",
            "length_reduction": "per-response mean then batch mean",
            "max_new_tokens": max_new_tokens,
            "verifier": VERIFIER_VERSION,
            "data_sha256": hashlib.sha256(repr((TRAIN_PAIRS, DEV_PAIRS)).encode()).hexdigest(),
            "initial": initial, "final": {"train": evaluate(model, TRAIN_PAIRS),
                                           "heldout": evaluate(model, DEV_PAIRS)},
            "history": history,
            "limits": "Original tiny arithmetic tokens; no natural-language reasoning or Qwen result."}
