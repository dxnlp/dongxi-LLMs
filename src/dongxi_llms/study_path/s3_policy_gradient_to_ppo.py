"""Checks and reference implementations for study-path stage S3 (REINFORCE → RLOO → PPO).

Notebook: ``notebooks/study-path/s3_policy_gradient_to_ppo.ipynb`` (Day 21 · Chapter 12).

Each ``check_*`` function receives the learner's function and prints one line per test: ⏳ when
the exercise is not attempted yet, ✅ when it passes and ❌ with a hint when it fails. Learner
checks never raise, so an unfinished notebook still runs top to bottom. ``strict=True`` turns
failures into ``AssertionError``; the reference cells use it.

The ``ref_*`` functions are the reference solutions. Conventions shared by all of them:

- one-decision toy: ``logits`` is ``[K]`` (one prompt, K complete answers), ``actions`` ``[N]``
- token tensors are ``[B, T]`` over **response positions**; ``mask`` is 1 for tokens the policy
  wrote (through its first ``<|im_end|>``) and 0 for padding; ``last`` ``[B]`` is the index of the
  last real token of each row
- all advantages, returns and old log-probabilities are data (no gradient)
"""
from __future__ import annotations

import itertools

import torch

from .common import Report, call, close, not_attempted

GARLIC_TOKENS = ["No", ", raw garlic is", "toxic", ".", "<end>"]
GARLIC_V = [0.5, 0.1, 0.0, -1.8, -1.9]
GARLIC_R = -2.0
GARLIC_A_LAM0 = [-0.4, -0.1, -1.8, -0.1, -0.1]
GARLIC_A_LAM1 = [-2.5, -2.1, -2.0, -0.2, -0.1]


def _f64(x):
    return torch.as_tensor(x, dtype=torch.float64)


# =========================================================================== Part A · one decision
def ref_exact_grad(logits: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
    """∂J/∂z for J = Σ_a softmax(z)_a R_a: the vector p ⊙ (R − J)."""
    p = logits.softmax(-1)
    return p * (rewards - (p * rewards).sum(-1, keepdim=True))


def ref_reinforce_loss(logits: torch.Tensor, actions: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Surrogate loss whose gradient is minus the REINFORCE estimate: −mean_i w_i log π(a_i)."""
    logp = logits.log_softmax(-1)[actions]
    return -(weights.detach() * logp).mean()


def score_vectors(p: torch.Tensor) -> torch.Tensor:
    """Row a is ∇_z log softmax(z)_a = e_a − p, shape [K, K]."""
    return torch.eye(len(p), dtype=p.dtype) - p[None, :]


def check_exact_grad(fn, strict: bool = False) -> bool:
    if not_attempted(fn):
        return False
    r = Report("exact_grad", strict)
    logits = torch.log(_f64([0.4, 0.3, 0.2, 0.1]))
    rewards = _f64([1.0, 2.0, 3.0, 6.0])
    ok, g = call(r, fn, logits, rewards)
    if not ok:
        return r.done()
    z = logits.clone().requires_grad_(True)
    (auto,) = torch.autograd.grad((z.softmax(-1) * rewards).sum(), z)
    r.expect(close(g, auto, atol=1e-8), f"matches autograd of J on p=[.4,.3,.2,.1], R=[1,2,3,6]: {auto.tolist()}",
             "∂J/∂z_i = p_i (R_i − J), with J = Σ_a p_a R_a")
    ok2, g2 = call(r, fn, logits, rewards + 10)
    if ok2:
        r.expect(close(g2, auto, atol=1e-8), "adding 10 to every reward leaves the gradient unchanged",
                 "you used p·R: subtract the expected reward J, so only R − J matters")
    r.expect(close(torch.as_tensor(g).sum(), 0.0, atol=1e-8), "coordinates sum to zero (softmax ignores a common shift)",
             "multiply (R − J) by p, coordinate by coordinate")
    return r.done()


def check_reinforce_loss(fn, strict: bool = False) -> bool:
    if not_attempted(fn):
        return False
    r = Report("reinforce_loss", strict)
    base = torch.log(_f64([0.4, 0.3, 0.2, 0.1]))
    actions = torch.tensor([3, 0, 1, 3, 2])
    w = _f64([6.0, 1.0, 2.0, 6.0, 3.0])
    z = base.clone().requires_grad_(True)
    ok, loss = call(r, fn, z, actions, w)
    if not ok:
        return r.done()
    if not r.expect(isinstance(loss, torch.Tensor) and loss.dim() == 0, "returns a scalar loss",
                    "average over the N samples with .mean()"):
        return r.done()
    p = base.softmax(-1)
    expected_grad = -(w[:, None] * score_vectors(p)[actions]).mean(0)
    (g,) = torch.autograd.grad(loss, z, allow_unused=True)
    if g is None:
        r.expect(False, "the loss depends on the logits", "use logits.log_softmax(-1)[actions]")
        return r.done()
    sign_ok = close(g, expected_grad, atol=1e-8)
    if not sign_ok and close(g, -expected_grad, atol=1e-8):
        r.expect(False, "gradient of the loss is −mean w(e_a − p)",
                 "minimize the NEGATIVE: loss = −mean(w · log π(a)); gradient descent then ascends J")
    elif not sign_ok and close(g, expected_grad * len(actions), atol=1e-8):
        r.expect(False, "gradient of the loss is −mean w(e_a − p)", "use .mean() over samples, not .sum()")
    else:
        r.expect(sign_ok, "gradient of the loss is −mean_i w_i (e_{a_i} − p) (the REINFORCE estimate, negated)",
                 "weight the LOG-probability of each sampled action, log_softmax(logits)[actions], not the probability")
    z2 = base.clone().requires_grad_(True)
    w_live = (z2.softmax(-1)[actions] * 10).clone()         # a weight that depends on θ (must be treated as data)
    ok, loss2 = call(r, fn, z2, actions, w_live)
    if ok:
        (g2,) = torch.autograd.grad(loss2, z2)
        want = -(w_live.detach()[:, None] * score_vectors(p)[actions]).mean(0)
        r.expect(close(g2, want, atol=1e-8), "weights are treated as data (no gradient through them)",
                 "use weights.detach(): a reward or advantage is a number, not a function of θ")
    return r.done()


def exact_moments(logits: torch.Tensor, rewards: torch.Tensor, group: int, advantage_fn) -> dict:
    """Exact mean and covariance trace of g = (1/G) Σ_i A_i (e_{a_i} − p) over all K^G ordered groups."""
    p = logits.softmax(-1)
    scores = score_vectors(p)
    est, mass = [], []
    for idx in itertools.product(range(len(p)), repeat=group):
        ids = torch.tensor(idx)
        adv = advantage_fn(rewards[ids])
        est.append((adv[:, None] * scores[ids]).mean(0))
        mass.append(p[ids].prod())
    est, mass = torch.stack(est), torch.stack(mass)
    mean = (mass[:, None] * est).sum(0)
    return {"mean": mean, "var": float((mass[:, None] * (est - mean) ** 2).sum())}


# =========================================================================== Part B · baselines
def ref_rloo_advantages(R: torch.Tensor) -> torch.Tensor:
    """A_i = R_i − mean_{j≠i} R_j over the last axis (G ≥ 2 answers to one prompt)."""
    G = R.shape[-1]
    return R - (R.sum(-1, keepdim=True) - R) / (G - 1)


def check_rloo(fn, strict: bool = False) -> bool:
    if not_attempted(fn):
        return False
    r = Report("rloo", strict)
    ok, a = call(r, fn, _f64([[0.0, 1.0, 3.0]]))
    if not ok:
        return r.done()
    want = _f64([[-2.0, -0.5, 2.5]])
    inclusive = _f64([[-4 / 3, -1 / 3, 5 / 3]])
    if close(a, inclusive):
        r.expect(False, "R = [0, 1, 3] → A = [−2, −0.5, 2.5]",
                 "you subtracted the mean INCLUDING answer i; leave it out: (sum − R_i) / (G − 1)")
    else:
        r.expect(close(a, want), "R = [0, 1, 3] → A = [−2, −0.5, 2.5]",
                 "baseline for answer i is the mean of the OTHER G − 1 rewards")
    ok, a = call(r, fn, _f64([[1.0, 1.0, 1.0, 1.0]]))
    if ok:
        r.expect(close(a, torch.zeros(1, 4)), "equal rewards in a group give zero advantages")
    batch = _f64([[0.0, 1.0, 3.0, 0.0], [2.0, 2.0, 0.0, 4.0]])
    ok, a = call(r, fn, batch)
    if ok:
        r.expect(close(a, ref_rloo_advantages(batch)), "works row by row on a [B, G] batch",
                 "reduce over the last axis only: R.sum(-1, keepdim=True)")
    return r.done()


# =========================================================================== Part C · values, GAE
def ref_exact_values(probs: list, terminal_reward: torch.Tensor):
    """Exact Q and V of a tabular token policy with terminal reward only (γ = 1).

    ``probs[d]`` is ``[K**d, K]``: π(a | s) for every prefix s of length d (row index in base K).
    ``terminal_reward`` is ``[K**T]``: R of every complete sequence. Returns lists Q[d] ``[K**d, K]``
    and V[d] ``[K**d]``. Child of prefix i by action a has index i*K + a, so V_{d+1}.view(K**d, K)
    lists Q(s, ·) for every prefix s of length d.
    """
    T, K = len(probs), probs[0].shape[-1]
    Q, V = [None] * T, [None] * T
    V_next = terminal_reward
    for d in reversed(range(T)):
        Q[d] = V_next.view(K ** d, K)
        V[d] = (probs[d] * Q[d]).sum(-1)
        V_next = V[d]
    return Q, V


def _brute_values(probs, terminal_reward):
    T, K = len(probs), probs[0].shape[-1]
    V0 = 0.0
    for seq in itertools.product(range(K), repeat=T):
        pr, idx = 1.0, 0
        for d, a in enumerate(seq):
            pr *= float(probs[d][idx, a])
            idx = idx * K + a
        V0 += pr * float(terminal_reward[idx])
    return V0


def check_exact_values(fn, strict: bool = False) -> bool:
    if not_attempted(fn):
        return False
    r = Report("exact_values", strict)
    g = torch.Generator().manual_seed(3)
    K, T = 3, 3
    probs = [torch.randn(K ** d, K, generator=g, dtype=torch.float64).mul(1.5).softmax(-1) for d in range(T)]
    R = torch.randn(K ** T, generator=g, dtype=torch.float64)
    ok, out = call(r, fn, probs, R)
    if not ok:
        return r.done()
    try:
        Q, V = out
        shapes = [tuple(q.shape) for q in Q] == [(K ** d, K) for d in range(T)] and \
                 [tuple(v.shape) for v in V] == [(K ** d,) for d in range(T)]
    except Exception:
        shapes = False
    if not r.expect(shapes, "Q[d] has shape [K**d, K] and V[d] has shape [K**d]",
                    "Q[d] = V_next.view(K**d, K); V[d] = (probs[d] * Q[d]).sum(-1)"):
        return r.done()
    Qr, Vr = ref_exact_values(probs, R)
    r.expect(close(V[0], _f64([_brute_values(probs, R)]), atol=1e-8),
             "V(prompt) equals Σ_y π(y) R(y) by brute-force enumeration of all 27 sequences",
             "weight each child by π(a|s): V = (probs[d] * Q[d]).sum(-1), not Q.mean(-1)")
    r.expect(all(close(a, b, atol=1e-8) for a, b in zip(Q, Qr)) and all(close(a, b, atol=1e-8) for a, b in zip(V, Vr)),
             "Q and V match at every depth", "with γ = 1 and no intermediate reward, Q(s, a) = V(s·a)")
    A0 = Q[0] - V[0][:, None]
    r.expect(close((probs[0] * A0).sum(-1), torch.zeros(1), atol=1e-8),
             "the policy-weighted mean advantage Σ_a π(a|s) A(s, a) is 0 at the prompt")
    return r.done()


def ref_gae(rewards: torch.Tensor, values: torch.Tensor, mask: torch.Tensor, gamma: float = 1.0,
            lam: float = 0.95) -> torch.Tensor:
    """A_t = Σ_k (γλ)^k δ_{t+k}, δ_t = r_t + γ V(s_{t+1}) − V(s_t); V after the last real token is 0."""
    T = rewards.shape[1]
    adv = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[:, 0])
    for t in reversed(range(T)):
        next_v = values[:, t + 1] * mask[:, t + 1] if t + 1 < T else torch.zeros_like(running)
        delta = rewards[:, t] + gamma * next_v - values[:, t]
        running = (delta + gamma * lam * running) * mask[:, t]
        adv[:, t] = running
    return adv


def check_gae(fn, strict: bool = False) -> bool:
    if not_attempted(fn):
        return False
    r = Report("gae", strict)
    V = _f64([GARLIC_V])
    rew = _f64([[0, 0, 0, 0, GARLIC_R]])
    m = torch.ones(1, 5, dtype=torch.float64)
    ok, a0 = call(r, fn, rew, V, m, 1.0, 0.0)
    if ok:
        r.expect(close(a0, _f64([GARLIC_A_LAM0])), f"garlic, λ = 0: A = δ = {GARLIC_A_LAM0}",
                 "δ_t = r_t + γ·V(s_{t+1}) − V(s_t), with V = 0 after the last token")
    ok, a1 = call(r, fn, rew, V, m, 1.0, 1.0)
    if ok:
        good = close(a1, _f64([GARLIC_A_LAM1]))
        r.expect(good, f"garlic, λ = 1: A = R − V(s_t) = {GARLIC_A_LAM1}",
                 "run the recursion BACKWARDS (t = T−1 … 0): running = δ_t + γλ·running")
    ok, a95 = call(r, fn, rew, V, m, 1.0, 0.95)
    if ok:
        r.expect(close(a95, ref_gae(rew, V, m, 1.0, 0.95)), "garlic, λ = 0.95 matches the reference")
    rew2 = _f64([[0.1, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 2.0]])
    V2 = _f64([[0.5, 0.4, 0.3, 9.0, 9.0], [0.2, 0.3, 0.4, 0.5, 0.6]])
    m2 = _f64([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    ok, a2 = call(r, fn, rew2, V2, m2, 0.9, 0.8)
    if ok:
        want = ref_gae(rew2, V2, m2, 0.9, 0.8)
        pad_ok = close(a2[0, 3:], torch.zeros(2), atol=1e-8)
        r.expect(pad_ok, "padding positions get advantage 0", "multiply the running sum by mask[:, t]")
        r.expect(close(a2, want), "padding values (9.0) never leak into the last real token; γ = 0.9 handled",
                 "use V(s_{t+1}) · mask[:, t+1], so the value after the last real token counts as 0; multiply both V and the running term by γ")
    return r.done()


# =========================================================================== Part D · PPO pieces
def ref_kl_shaped_rewards(R: torch.Tensor, logp: torch.Tensor, ref_logp: torch.Tensor, mask: torch.Tensor,
                          last: torch.Tensor, beta: float):
    """Per-token rewards r_t = −β (log π_old − log π_ref) on every response token, + R on the last one."""
    kl = (logp - ref_logp) * mask
    rewards = -beta * kl
    rewards[torch.arange(len(R)), last] += R
    return rewards, kl


def check_kl_shaped_rewards(fn, strict: bool = False) -> bool:
    if not_attempted(fn):
        return False
    r = Report("kl_shaped_rewards", strict)
    logp = _f64([[-1.0, -0.5, -2.0, -0.1], [-0.3, -0.2, -0.7, -0.4]])
    ref = _f64([[-1.2, -0.5, -1.0, -0.4], [-0.3, -0.6, -0.9, -0.4]])
    mask = _f64([[1, 1, 1, 1], [1, 1, 0, 0]])
    last = torch.tensor([3, 1])
    R = _f64([2.0, -1.0])
    ok, out = call(r, fn, R, logp.clone(), ref.clone(), mask, last, 0.1)
    if not ok:
        return r.done()
    want_r, want_kl = ref_kl_shaped_rewards(R, logp.clone(), ref.clone(), mask, last, 0.1)
    try:
        rew, kl = out
    except Exception:
        r.expect(False, "returns (rewards, kl)", "return both tensors")
        return r.done()
    r.expect(close(kl, want_kl), "kl = (log π − log π_ref) · mask, per token",
             "log-ratio policy minus reference, zero on padding")
    if close(rew, -want_r + 2 * (want_r + 0.1 * want_kl)):   # sign of the KL charge flipped
        r.expect(False, "per-token rewards match", "the KL term is a COST: rewards = −β · kl")
    else:
        r.expect(close(rew, want_r), "rewards = −β·kl everywhere, plus R at index `last` of each row",
                 "add R only at rewards[arange(B), last]; row 1 ends at index 1, not at mask.sum() = 2")
    r.expect(close(rew[1, 2:], torch.zeros(2), atol=1e-12), "padding positions get reward 0")
    return r.done()


def ref_clipped_policy_loss(logp, old_logp, adv, mask, eps: float = 0.2):
    """PPO policy loss −mean_t min(ρ_t A_t, clip(ρ_t, 1−ε, 1+ε) A_t) over real tokens, and the clip fraction."""
    ratio = torch.exp(logp - old_logp)
    surrogate = torch.min(ratio * adv, ratio.clamp(1 - eps, 1 + eps) * adv)
    loss = -(surrogate * mask).sum() / mask.sum()
    clip_frac = (((ratio - 1).abs() > eps).float() * mask).sum() / mask.sum()
    return loss, clip_frac


def check_clipped_policy_loss(fn, strict: bool = False) -> bool:
    if not_attempted(fn):
        return False
    r = Report("clipped_policy_loss", strict)
    rho = _f64([[1.5, 1.5, 0.5, 0.5, 0.9, 1.1]])
    adv = _f64([[1.0, -1.0, -1.0, 1.0, 1.0, -1.0]])
    old = torch.zeros(1, 6, dtype=torch.float64)
    mask = torch.ones(1, 6, dtype=torch.float64)
    logp = rho.log().clone().requires_grad_(True)
    ok, out = call(r, fn, logp, old, adv, mask, 0.2)
    if not ok:
        return r.done()
    try:
        loss, clip_frac = out
    except Exception:
        r.expect(False, "returns (loss, clip_frac)", "keep the two return values of the scaffold")
        return r.done()
    want, want_frac = ref_clipped_policy_loss(rho.log(), old, adv, mask, 0.2)
    if not r.expect(close(loss, want), f"loss on 6 test tokens = {float(want):+.4f}",
                    "ratio = exp(logp − old_logp); surrogate = min(ratio·A, ratio.clamp(1−ε, 1+ε)·A); loss = −mean"):
        if close(loss, -want):
            print("   (your loss has the opposite sign: PPO MAXIMIZES the surrogate, so the loss is its negative)")
        return r.done()
    (g,) = torch.autograd.grad(loss, logp)
    g = g[0].tolist()
    r.expect(abs(g[0]) < 1e-12, "ρ = 1.5, A = +1: gradient 0 (helpful move already beyond 1 + ε)",
             "clip the ratio inside the min, so the objective is flat above 1 + ε when A > 0")
    r.expect(abs(g[1]) > 1e-6, "ρ = 1.5, A = −1: gradient ≠ 0 (harmful move is still pulled back)",
             "take torch.min of the clipped AND unclipped terms; clamping alone kills this gradient")
    r.expect(abs(g[2]) < 1e-12 and abs(g[3]) > 1e-6, "ρ = 0.5: zero for A = −1, nonzero for A = +1",
             "use min (pessimistic), not max")
    r.expect(close(clip_frac, want_frac), f"clip fraction = {float(want_frac):.3f} (4 of 6 tokens outside [0.8, 1.2])")
    logp2 = rho.log().clone()
    mask2 = _f64([[1, 1, 1, 0, 0, 0]])
    ok, out2 = call(r, fn, logp2, old, adv, mask2, 0.2)
    if ok:
        r.expect(close(out2[0], ref_clipped_policy_loss(logp2, old, adv, mask2, 0.2)[0]),
                 "padding tokens are excluded from the mean")
    return r.done()


def ref_value_loss(new_values, old_values, adv, mask):
    """Value targets are the λ-returns A + V_old (data); loss is the masked mean squared error."""
    returns = (adv + old_values).detach()
    return ((new_values - returns) ** 2 * mask).sum() / mask.sum()


def check_value_loss(fn, strict: bool = False) -> bool:
    if not_attempted(fn):
        return False
    r = Report("value_loss", strict)
    V = _f64([GARLIC_V])
    rew = _f64([[0, 0, 0, 0, GARLIC_R]])
    m = torch.ones(1, 5, dtype=torch.float64)
    zero = torch.zeros(1, 5, dtype=torch.float64)
    a1 = ref_gae(rew, V, m, 1.0, 1.0)
    ok, l1 = call(r, fn, zero, V, a1, m)          # new V = 0: loss = mean(target²)
    if not ok:
        return r.done()
    want1 = float((_f64([GARLIC_R] * 5) ** 2).mean())
    if close(l1, (a1 ** 2).mean()):
        r.expect(False, "λ = 1 targets are the Monte-Carlo returns G_t = R = −2.0",
                 "the target is A + V_old (the λ-return), not the advantage alone")
    else:
        r.expect(close(l1, want1), "λ = 1 targets are the Monte-Carlo returns G_t = R = −2.0 (loss vs V=0 is 4.0)",
                 "returns = adv + old_values")
    a0 = ref_gae(rew, V, m, 1.0, 0.0)
    ok, l0 = call(r, fn, zero, V, a0, m)
    if ok:
        td = _f64([0.1, 0.0, -1.8, -1.9, -2.0])        # r_t + V(s_{t+1})
        r.expect(close(l0, (td ** 2).mean()), "λ = 0 targets are the TD targets r_t + V(s_{t+1}) = [0.1, 0, −1.8, −1.9, −2]")
    nv = _f64([[0.0, 1.0, 2.0, 50.0]]).requires_grad_(True)
    ov, ad, mk = _f64([[0.5, 0.5, 0.5, 0.5]]), _f64([[0.5, 0.5, 0.5, 0.5]]), _f64([[1, 1, 1, 0]])
    ok, l2 = call(r, fn, nv, ov, ad, mk)
    if ok:
        r.expect(close(l2, _f64((1 + 0 + 1) / 3)), "masked mean over real tokens (padding value 50 ignored)",
                 "multiply by mask and divide by mask.sum()")
        (g,) = torch.autograd.grad(l2, nv)
        r.expect(close(g, _f64([[-2 / 3, 0.0, 2 / 3, 0.0]])), "gradient 2(V − target)/n reaches only the new values")
    return r.done()


# =========================================================================== Part E · real PPO
def ref_rule_reward(lengths: torch.Tensor, finished: torch.Tensor, target: int = 48) -> torch.Tensor:
    """1.0 for a finished answer within `target` tokens, 0.5 finished but longer, 0.0 truncated."""
    return torch.where(finished, torch.where(lengths <= target, 1.0, 0.5), 0.0).float()


def check_rule_reward(fn, strict: bool = False) -> bool:
    if not_attempted(fn):
        return False
    r = Report("rule_reward", strict)
    lengths = torch.tensor([10, 48, 49, 96, 30, 96])
    finished = torch.tensor([True, True, True, True, False, False])
    ok, R = call(r, fn, lengths, finished, 48)
    if not ok:
        return r.done()
    want = torch.tensor([1.0, 1.0, 0.5, 0.5, 0.0, 0.0])
    r.expect(isinstance(R, torch.Tensor) and R.shape == (6,), "one reward per answer, shape [B]")
    if not close(R, want):
        if close(R[:4], want[:4]):
            r.expect(False, "R = [1, 1, 0.5, 0.5, 0, 0]", "a truncated answer earns 0 even when it is short")
        elif float(R[1]) == 0.5:
            r.expect(False, "R = [1, 1, 0.5, 0.5, 0, 0]", "the target is inclusive: lengths <= target")
        else:
            r.expect(False, "R = [1, 1, 0.5, 0.5, 0, 0]",
                     "torch.where(finished, torch.where(lengths <= target, 1.0, 0.5), 0.0)")
    else:
        r.expect(True, "R = [1, 1, 0.5, 0.5, 0, 0] for lengths [10, 48, 49, 96, 30, 96]")
    return r.done()


def ref_ppo_signals(R, old_logp, ref_logp, V, mask, last, beta, lam, kl_fn=ref_kl_shaped_rewards,
                    gae_fn=ref_gae, gamma: float = 1.0):
    """rewards → GAE advantages → λ-return targets → advantages whitened over real tokens."""
    m = mask.bool()
    rewards, kl = kl_fn(R, old_logp, ref_logp, mask, last, beta)
    adv = gae_fn(rewards, V, mask, gamma, lam)
    returns = adv + V
    adv = (adv - adv[m].mean()) / (adv[m].std() + 1e-8) * mask
    return rewards, adv, returns, kl


def _toy_signals_case():
    g = torch.Generator().manual_seed(7)
    B, T = 3, 6
    mask = _f64([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0]])
    last = torch.tensor([5, 2, 3])
    old = -torch.rand(B, T, generator=g, dtype=torch.float64) * mask
    ref = -torch.rand(B, T, generator=g, dtype=torch.float64) * mask
    V = torch.randn(B, T, generator=g, dtype=torch.float64) * mask
    R = _f64([1.0, 0.0, 0.5])
    return R, old, ref, V, {"mask": mask, "last": last}


def check_ppo_signals(fn, beta: float = 0.05, lam: float = 0.95, strict: bool = False) -> bool:
    if not_attempted(fn):
        return False
    r = Report("ppo_signals", strict)
    R, old, ref, V, ro = _toy_signals_case()
    ok, out = call(r, fn, R, old.clone(), ref.clone(), V.clone(), ro)
    if not ok:
        return r.done()
    try:
        rewards, adv, returns, kl = out
    except Exception:
        r.expect(False, "returns (rewards, adv, returns, kl)", "keep the scaffold's return line")
        return r.done()
    want = ref_ppo_signals(R, old.clone(), ref.clone(), V.clone(), ro["mask"], ro["last"], beta, lam)
    r.expect(close(rewards, want[0]), "rewards: KL charge on every real token, R on the last",
             "rewards, kl = kl_shaped_rewards(R, old_logp, ref_logp, mask, ro['last'], beta)")
    r.expect(close(returns, want[2]), "value targets = raw GAE advantage + V (computed BEFORE whitening)",
             "returns = adv + V, then whiten adv on the next line")
    m = ro["mask"].bool()
    if not close(adv, want[1]):
        a_all = (want[2] - V)
        if close(adv, (a_all - a_all.mean()) / (a_all.std() + 1e-8)):
            r.expect(False, "advantages whitened over real tokens only", "use adv[m].mean() and adv[m].std(), then * mask")
        else:
            r.expect(False, "advantages whitened over real tokens only",
                     "adv = (adv − adv[m].mean()) / (adv[m].std() + 1e-8) * mask")
    else:
        r.expect(True, "advantages whitened over real tokens only: mean 0, std 1, padding 0")
    r.expect(close(kl, want[3]), "kl returned unchanged for logging")
    return r.done()
