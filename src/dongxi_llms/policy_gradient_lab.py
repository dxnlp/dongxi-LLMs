"""Chapter 12 finite policy microscopes, with exactly inspectable expectations."""
import itertools
import torch


def exact_gradient(logits, reward):
    p = logits.softmax(-1)
    return p * (reward - (p * reward).sum(-1, keepdim=True))


def score_vectors(probability):
    return torch.eye(len(probability), dtype=probability.dtype) - probability[None, :]


def estimator_moments(logits, reward, baseline=0.):
    """Exact one-sample mean/covariance trace; no Monte Carlo error."""
    p = logits.softmax(-1)
    samples = (reward - torch.as_tensor(baseline))[:, None] * score_vectors(p)
    mean = (p[:, None] * samples).sum(0)
    variance = float((p[:, None] * (samples - mean).square()).sum())
    return {"mean": mean, "variance_trace": variance}


def rloo_advantages(rewards):
    """Final axis groups independent samples from ONE prompt; G>=2."""
    if rewards.ndim < 1 or rewards.shape[-1] < 2:
        raise ValueError("RLOO needs at least two completions per prompt")
    return rewards - (rewards.sum(-1, keepdim=True) - rewards) / (rewards.shape[-1] - 1)


def group_estimator_moments(logits, reward, group=3, leave_one_out=True):
    """Enumerate every iid group to expose unbiasedness and self-baseline scaling."""
    if not 2 <= group <= 5:
        raise ValueError("Bounded enumeration: group 2..5")
    p, scores = logits.softmax(-1), score_vectors(logits.softmax(-1))
    estimates, weights = [], []
    for indices in itertools.product(range(len(p)), repeat=group):
        ids = torch.tensor(indices)
        r = reward[ids]
        advantage = rloo_advantages(r) if leave_one_out else r - r.mean()
        estimates.append((advantage[:, None] * scores[ids]).mean(0))
        weights.append(p[ids].prod())
    samples, mass = torch.stack(estimates), torch.stack(weights)
    mean = (mass[:, None] * samples).sum(0)
    return {"mean": mean, "variance_trace": float((mass[:, None] * (samples - mean).square()).sum())}


def ppo_surrogate(ratio, advantage, epsilon=.2):
    if not 0 < epsilon < 1:
        raise ValueError("Clip epsilon must be in (0,1)")
    return torch.minimum(ratio * advantage, ratio.clamp(1 - epsilon, 1 + epsilon) * advantage)


def kl_estimators(p, q):
    """Actions sampled from p; all entries p,q positive, normalized.

    k1 and k3 expectations equal KL(p||q) on common full support. k2 has a
    local quadratic interpretation and is generally biased for KL.
    """
    if p.shape != q.shape or (p <= 0).any() or (q <= 0).any():
        raise ValueError("Require matching strictly positive distributions")
    if not torch.allclose(p.sum(), p.new_tensor(1.)) or not torch.allclose(q.sum(), q.new_tensor(1.)):
        raise ValueError("Distributions must sum to one")
    k1 = p.log() - q.log()
    k2 = .5 * k1.square()
    k3 = torch.expm1(-k1) + k1
    out = {}
    for name, samples in (("k1", k1), ("k2", k2), ("k3", k3)):
        mean = (p * samples).sum()
        out[name] = {"samples": samples, "mean": float(mean),
                     "variance": float((p * (samples - mean).square()).sum())}
    return out


def procedural_policy(mode="rloo", seed=1921, updates=160, group=4, epochs=3):
    """Six contexts are (a,b), a in 0..2,b in 0..1; action=(a+b)%3.

    Independent logits per context deliberately remove representation learning.
    PPO uses a detached exact pre-rollout value, making critic error zero. The
    result is a mechanism control, never an LLM benchmark or generalization test.
    """
    if mode not in ("reinforce", "baseline", "rloo", "ppo"):
        raise ValueError(mode)
    generator = torch.Generator().manual_seed(seed)
    logits = torch.zeros(6, 3, dtype=torch.float64, requires_grad=True)
    correct = torch.tensor([(a + b) % 3 for a in range(3) for b in range(2)])
    history = []
    gradient_steps = 0
    for _ in range(updates):
        old_logp = logits.detach().log_softmax(-1)
        old_p = old_logp.exp()
        actions = torch.multinomial(old_p, group, replacement=True, generator=generator)
        rewards = (actions == correct[:, None]).to(logits.dtype)
        if mode == "rloo":
            advantage = rloo_advantages(rewards)
        elif mode in ("baseline", "ppo"):
            advantage = rewards - old_p.gather(1, correct[:, None])
        else:
            advantage = rewards
        for _epoch in range(epochs if mode == "ppo" else 1):
            selected = logits.log_softmax(-1).gather(1, actions)
            if mode == "ppo":
                ratio = (selected - old_logp.gather(1, actions)).exp()
                loss = -ppo_surrogate(ratio, advantage).mean()
            else:
                loss = -(selected * advantage).mean()
            (gradient,) = torch.autograd.grad(loss, logits)
            with torch.no_grad():
                logits -= 1.5 * gradient
            gradient_steps += 1
        history.append(float(logits.detach().softmax(-1).gather(1, correct[:, None]).mean()))
    return {"mode": mode, "seed": seed, "history": history,
            "success_probability": history[-1], "rollout_updates": updates,
            "sampled_completions": updates * 6 * group, "gradient_steps": gradient_steps,
            "probability": logits.detach().softmax(-1).tolist()}
