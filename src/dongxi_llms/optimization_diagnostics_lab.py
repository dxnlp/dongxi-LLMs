"""Chapter 14 exact toy optimization and rollout-contract diagnostics."""
from dataclasses import dataclass
import math

import torch

from dongxi_llms.grpo_lab import verify_integer

HACK_RESPONSES = ("35", "35 0", "0")


def hacking_experiment(updates=60, lr=.4, repair=False):
    """Optimize a real three-action categorical policy under a flawed or strict reward.

    Both comparisons start from identical zero logits. Exact expectation replaces
    Monte Carlo noise so this isolates proxy misspecification rather than GRPO.
    """
    logits = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))
    optimizer = torch.optim.SGD([logits], lr=lr)
    truth = torch.tensor([float(verify_integer(text, 35)) for text in HACK_RESPONSES])
    # Broken rule: contains answer, with a bonus for apparent "work" (extra fields).
    proxy = truth if repair else torch.tensor([1., 2., 0.])
    history = []
    for step in range(updates+1):
        p = logits.softmax(-1)
        history.append({"update": step, "proxy_reward": float((p*proxy).sum().detach()),
                        "strict_accuracy": float((p*truth).sum().detach()),
                        "entropy": float(-(p*p.log()).sum().detach()),
                        "probabilities": p.detach().tolist()})
        if step < updates:
            optimizer.zero_grad(set_to_none=True)
            (-(p*proxy).sum()).backward()
            optimizer.step()
    return {"repair": repair, "responses": list(HACK_RESPONSES), "history": history}


def length_weights(lengths, reduction="response"):
    """Total token weight per response under the selected loss reduction."""
    lengths = torch.as_tensor(lengths, dtype=torch.float64)
    if (lengths <= 0).any():
        raise ValueError("All response lengths must be positive")
    if reduction == "response":
        return torch.ones_like(lengths)/len(lengths)
    if reduction == "token":
        return lengths/lengths.sum()
    raise ValueError("Unknown reduction")


@dataclass(frozen=True)
class RolloutIdentity:
    prompt_id: str
    behavior_version: int
    tokenizer_hash: str
    verifier_hash: str
    group_id: str
    response_tokens: int


def accept_rollout(row, policy_version, tokenizer_hash, verifier_hash, max_lag=0):
    """Contract check, not a statistical proof that accepted off-policy data are safe."""
    if policy_version < row.behavior_version or policy_version-row.behavior_version > max_lag:
        raise ValueError("Behavior policy version is outside the allowed lag")
    if row.tokenizer_hash != tokenizer_hash or row.verifier_hash != verifier_hash:
        raise ValueError("Token/reward identity changed")
    if row.response_tokens < 1:
        raise ValueError("Empty rollout")
    return True


def pipeline_budget(prompts=4, group=8, mean_length=128, rollout_tps=128.,
                    learner_seconds=6., verifier_seconds=.1, sync_seconds=.5):
    """Analytical budget, explicitly projected (not measured hardware throughput)."""
    if min(prompts, group, mean_length, rollout_tps) <= 0:
        raise ValueError("Positive rollout geometry and throughput required")
    tokens = prompts*group*mean_length
    generation = tokens/rollout_tps
    total = generation+learner_seconds+verifier_seconds+sync_seconds
    return {"projected_tokens": tokens, "projected_generation_seconds": generation,
            "projected_update_seconds": total,
            "projected_useful_tps": tokens/total,
            "projected_generation_fraction": generation/total}


def diagnostic_flags(row):
    """Teaching thresholds; reasons, not automatic causal diagnoses."""
    reasons = []
    required = ("loss", "reward", "kl", "entropy", "host_available_gib")
    if any(not math.isfinite(float(row[key])) for key in required):
        reasons.append("nonfinite measurement")
    if row["host_available_gib"] < 25:
        reasons.append("host reserve below configured 25 GiB")
    if row.get("policy_lag", 0) > 0:
        reasons.append("rollout stale under synchronous contract")
    if row["entropy"] < .1:
        reasons.append("low entropy: inspect exploration and validity")
    if row["kl"] > .2:
        reasons.append("KL above predeclared toy review threshold")
    if row.get("reward_delta", 0) > 0 and row.get("evaluation_delta", 0) < 0:
        reasons.append("proxy/evaluation disagreement")
    return reasons
