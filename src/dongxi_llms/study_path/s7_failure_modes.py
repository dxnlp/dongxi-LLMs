"""Checks, references and toy worlds for study-path stage S7 (when optimization goes wrong).

The notebook ``notebooks/study-path/s7_failure_modes.ipynb`` asks the learner to write the key
lines of reward-overoptimization analysis: the KL-regularized optimum, exact KL, the best-of-n
distribution and its KL, a length-controlled score, token entropy, a policy-gradient loss with KL
and entropy terms, a conservative reward-model ensemble, a training-log diagnostic and a masked
entropy for a real Qwen3 policy. Each ``check_*`` function prints ⏳ when the exercise is not
attempted, ✅ per passing property, and ❌ with a hint naming the likely mistake. Checks never
raise unless ``strict=True`` (used by the reference cells).

The toy worlds are finite answer sets with a known gold reward, so every toy number is exact for
that set. They support the mechanisms in Chapter 14; they are not measurements of a language model.
"""
from __future__ import annotations

import math
from typing import Callable

import torch

from .common import Report, call, close, not_attempted

DT = torch.float64


# =========================================================================== Part A · toy world
def answer_pool(K: int = 2**20, seed: int = 0, base_len: float = 60.0, len_spread: float = 0.5) -> dict:
    """K candidate answers to one prompt, each with helpfulness ``h`` and log-length score ``z``.

    ``h`` and ``z`` are independent N(0, 1) draws; the answer length in tokens is
    ``base_len · exp(len_spread · z)``. The reference policy is uniform over the K answers.
    """
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(K, generator=g, dtype=DT)
    z = torch.randn(K, generator=g, dtype=DT)
    return {"h": h, "z": z, "length": base_len * torch.exp(len_spread * z),
            "logp_ref": torch.full((K,), -math.log(K), dtype=DT)}


def gold_reward(pool: dict, a: float = 0.5) -> torch.Tensor:
    """Gold quality: helpfulness, minus a penalty for lengths far from typical (zero mean under π_ref)."""
    return pool["h"] - a * (pool["z"] ** 2 - 1)


def proxy_reward(pool: dict, b: float = 1.0) -> torch.Tensor:
    """A reward model that sees helpfulness and also credits length linearly with slope b."""
    return pool["h"] + b * pool["z"]


def expect(logp: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """E_π[x] for a policy given as log-probabilities over the last dimension."""
    return (logp.exp() * x).sum(-1)


def ref_tilt(logp_ref: torch.Tensor, R: torch.Tensor, beta: float) -> torch.Tensor:
    """log π_β with π_β(y) ∝ π_ref(y)·exp(R(y)/β), the maximizer of E_π[R] − β·KL(π‖π_ref)."""
    return torch.log_softmax(logp_ref + R / beta, -1)


def ref_kl(logp: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    """Exact KL(π‖π_ref) = Σ_y π(y)·(log π(y) − log π_ref(y)) over the last dimension."""
    return (logp.exp() * (logp - logp_ref)).sum(-1)


def sweep(tilt: Callable, kl: Callable, logp_ref: torch.Tensor, R: torch.Tensor, betas, measures: dict) -> dict:
    """For each β: KL(π_β‖π_ref) and E_{π_β}[m] for every named measure m (exact over the pool)."""
    out = {"beta": [], "kl": [], **{name: [] for name in measures}}
    for beta in betas:
        logp = tilt(logp_ref, R, float(beta))
        out["beta"].append(float(beta))
        out["kl"].append(float(kl(logp, logp_ref)))
        for name, m in measures.items():
            out[name].append(float(expect(logp, m)))
    return {k: torch.tensor(v, dtype=DT) for k, v in out.items()}


def gaussian_theory(b: float, a: float = 0.5, t=None) -> dict:
    """K → ∞ limit: tilting N(0, I) by exp((h + b z)/β) shifts its mean to t·(1, b), t = 1/β."""
    t = torch.linspace(0, 4, 200, dtype=DT) if t is None else torch.as_tensor(t, dtype=DT)
    return {"t": t, "kl": t ** 2 * (1 + b * b) / 2, "gold": t - a * b * b * t ** 2, "proxy": t * (1 + b * b)}


# =========================================================================== Part B · best-of-n
def ref_bon_probs(p_sorted: torch.Tensor, n: int) -> torch.Tensor:
    """P(best-of-n picks answer i), answers sorted by ascending proxy: C_i^n − C_{i−1}^n."""
    C = p_sorted.cumsum(0)
    return C ** n - (C - p_sorted).clamp(min=0) ** n


def ref_bon_kl(n: int) -> float:
    """KL(π_bon‖π_ref) when every answer has vanishing reference probability: log n − (n−1)/n."""
    return math.log(n) - (n - 1) / n


def bon_policy(bon_probs: Callable, logp_ref: torch.Tensor, R: torch.Tensor, n: int) -> torch.Tensor:
    """log π_bon in the original answer order (ties have probability zero for continuous R)."""
    order = R.argsort()
    p = bon_probs(logp_ref.exp()[order], n)
    out = torch.empty_like(p)
    out[order] = p
    return torch.log(out.clamp_min(1e-300))


# =========================================================================== Part C · length
def ref_length_controlled(r: torch.Tensor, z: torch.Tensor, p: torch.Tensor):
    """Remove the linear length channel: r − b̂·(z − z̄), b̂ = Cov_p(r, z)/Var_p(z). Returns (r_lc, b̂)."""
    zc = z - (p * z).sum()
    b_hat = (p * zc * r).sum() / (p * zc ** 2).sum()
    return r - b_hat * zc, b_hat


# =========================================================================== Part D · entropy
def ref_entropy(logits: torch.Tensor) -> torch.Tensor:
    """H(softmax(logits)) over the last dimension, stable for very negative logits."""
    logp = torch.log_softmax(logits, -1)
    return -(logp.exp() * logp).sum(-1)


# Tabular sequence policy: position t = 0..T−1, vocabulary V, token 0 = EOS (ends the answer).
TOY_T, TOY_V = 16, 6
TOY_PROXY = torch.tensor([0.0, 0.5, 0.35, 0.2, 0.05, -0.1], dtype=DT)    # per content token; EOS has none
TOY_GOLD = torch.tensor([0.0, 0.3, 0.45, 0.2, 0.05, -0.1], dtype=DT)


def toy_reference_logits(seed: int = 0) -> torch.Tensor:
    """π_ref logits [T, V]: P(EOS) ≈ 0.2 per position (mean length ≈ 4), content tokens near uniform."""
    g = torch.Generator().manual_seed(seed)
    logits = 0.3 * torch.randn(TOY_T, TOY_V, generator=g, dtype=DT)
    logits[:, 0] = math.log(0.2 / 0.8 * (TOY_V - 1))
    return logits


def toy_stats(logits: torch.Tensor, ref_logits: torch.Tensor, entropy: Callable = ref_entropy) -> dict:
    """Exact sequence quantities of a position-indexed policy (differentiable in ``logits``).

    reach_t = P(the answer is still running at position t) = Π_{s<t} (1 − π_s(EOS)).
    Sequence KL and entropy are Σ_t reach_t·(per-position value) by the chain rule;
    ``token_entropy`` is the reach-weighted mean over generated positions, as logged in practice.
    """
    logp, logp_ref = torch.log_softmax(logits, -1), torch.log_softmax(ref_logits, -1)
    stay = 1 - logp[:, 0].exp()
    reach = torch.cat([torch.ones(1, dtype=logits.dtype), torch.cumprod(stay, 0)[:-1]])
    H = entropy(logits)
    kl_t = (logp.exp() * (logp - logp_ref)).sum(-1)
    p = logp.exp()
    content = (p[:, 1:] * TOY_PROXY[1:]).sum(-1)
    return {"reach": reach, "kl": (reach * kl_t).sum(), "seq_entropy": (reach * H).sum(),
            "token_entropy": (reach * H).sum() / reach.sum(),
            "length": (reach * stay).sum(), "proxy": (reach * content).sum()}


def toy_sample(logits: torch.Tensor, n: int, gen: torch.Generator):
    """n answers: tokens [n, T], content length L (tokens before EOS), Σ log π over generated tokens."""
    logp = torch.log_softmax(logits, -1)
    with torch.no_grad():
        tokens = torch.multinomial(logp.exp().repeat(n, 1), 1, generator=gen).view(n, TOY_T)
    is_eos = tokens == 0
    L = torch.where(is_eos.any(1), is_eos.float().argmax(1), torch.full((n,), TOY_T))
    pos = torch.arange(TOY_T)
    generated = pos[None] <= L[:, None]                      # content tokens and the EOS itself
    content = pos[None] < L[:, None]
    lp_tok = logp.gather(1, tokens.T).T                      # [n, T]: log π_t(y_t)
    return tokens, L, (lp_tok * generated).sum(1), content


def toy_rewards(tokens: torch.Tensor, content: torch.Tensor, L: torch.Tensor):
    """Proxy: Σ token scores (longer earns more). Gold: tokens graded differently, minus 0.05·(L − 6)² beyond 6 tokens."""
    proxy = (TOY_PROXY[tokens] * content).sum(1)
    gold = (TOY_GOLD[tokens] * content).sum(1) - 0.05 * (L.to(DT) - 6).clamp(min=0) ** 2
    return proxy, gold


def ref_pg_loss(logp_seq, adv, kl, ent, beta, lam):
    """REINFORCE with a baseline, plus β·KL(π‖π_ref), minus an entropy bonus λ·H."""
    return -(adv * logp_seq).mean() + beta * kl - lam * ent


def train_toy(pg_loss: Callable, entropy: Callable = ref_entropy, beta: float = 0.0, lam: float = 0.0,
              steps: int = 300, n: int = 64, lr: float = 0.05, seed: int = 0) -> dict:
    """Policy-gradient training of the tabular policy from π_ref; returns a per-step log."""
    ref_logits = toy_reference_logits()
    logits = ref_logits.clone().requires_grad_(True)
    opt = torch.optim.Adam([logits], lr=lr)
    gen = torch.Generator().manual_seed(seed)
    log = {k: [] for k in ("reward", "kl", "entropy", "length", "gold")}
    for _ in range(steps):
        tokens, L, logp_seq, content = toy_sample(logits, n, gen)
        R, gold = toy_rewards(tokens, content, L)
        adv = R - R.mean()                                    # group baseline
        st = toy_stats(logits, ref_logits, entropy)
        loss = pg_loss(logp_seq, adv, st["kl"], st["token_entropy"], beta, lam)
        opt.zero_grad()
        loss.backward()
        opt.step()
        log["reward"].append(float(R.mean()))
        log["kl"].append(st["kl"].item())
        log["entropy"].append(st["token_entropy"].item())
        log["length"].append(float(L.to(DT).mean()))
        log["gold"].append(float(gold.mean()))
    return log


# =========================================================================== Part E · ensembles
def ensemble_world(pool: dict, M: int = 2, b: float = 1.0, shared: float = 1.0, a: float = 0.5, seed: int = 1):
    """M reward models with independent errors, optionally plus a shared length credit.

    Each answer gets M extra spurious features w_k ~ N(0, 1). Reward model k credits its own
    feature (an error the other models do not share) and, with weight ``shared``, the common length
    feature z. Gold penalizes every feature far from typical. Returns (scores [M, K], gold [K]).
    """
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(M, pool["h"].shape[0], generator=g, dtype=DT)
    gold = pool["h"] - a * (pool["z"] ** 2 - 1) - a * (w ** 2 - 1).sum(0)
    scores = pool["h"] + shared * b * pool["z"] + b * w
    return scores, gold


def ref_aggregate(scores: torch.Tensor, mode: str) -> torch.Tensor:
    """Combine M reward-model scores [M, K] into one reward [K]."""
    if mode == "mean":
        return scores.mean(0)
    if mode == "min":
        return scores.min(0).values
    if mode == "lcb":
        return scores.mean(0) - scores.std(0, unbiased=False)
    raise ValueError(mode)


# =========================================================================== Part F · diagnostics
FLAGS = ("reward up while KL past limit", "length rising", "entropy collapsing")


def early(x, frac: float = 0.2) -> float:
    x = torch.as_tensor(x, dtype=DT)
    return float(x[:max(1, int(len(x) * frac))].mean())


def late(x, frac: float = 0.2) -> float:
    x = torch.as_tensor(x, dtype=DT)
    return float(x[-max(1, int(len(x) * frac)):].mean())


def trend_t(x) -> float:
    """t-statistic of the least-squares slope of x against step (white-noise residuals assumed)."""
    y = torch.as_tensor(x, dtype=DT)
    n = len(y)
    s = torch.arange(n, dtype=DT)
    sc = s - s.mean()
    slope = (sc * (y - y.mean())).sum() / (sc ** 2).sum()
    resid = y - y.mean() - slope * sc
    se = torch.sqrt((resid ** 2).sum() / max(n - 2, 1) / (sc ** 2).sum())
    return float(slope / se) if se > 0 else (math.inf if slope > 0 else (-math.inf if slope < 0 else 0.0))


def ref_diagnose(log: dict, kl_limit: float, ent_ratio: float = 0.5, t_crit: float = 3.0) -> dict:
    """Warning signs in a training log (lists per step). Flags call for reading samples, not verdicts."""
    return {
        FLAGS[0]: trend_t(log["reward"]) > t_crit and late(log["kl"]) > kl_limit,
        FLAGS[1]: trend_t(log["length"]) > t_crit,
        FLAGS[2]: late(log["entropy"]) < ent_ratio * early(log["entropy"]),
    }


# =========================================================================== Part G · real model
def ref_masked_entropy(logits: torch.Tensor, mask: torch.Tensor, entropy: Callable = ref_entropy) -> torch.Tensor:
    """Mean next-token entropy over response tokens: logits [B, T, V], mask [B, T] → scalar."""
    return (entropy(logits) * mask).sum() / mask.sum()


# =========================================================================== checks
def _logp(p):
    return torch.log(torch.as_tensor(p, dtype=DT))


def check_tilt(fn, strict: bool = False) -> bool:
    """fn(logp_ref, R, beta) -> log π_β."""
    if not_attempted(fn):
        return False
    r = Report("tilt", strict)
    lr, R = _logp([0.5, 0.3, 0.2]), torch.tensor([1.0, 0.0, -1.0], dtype=DT)
    ok, got = call(r, fn, lr, R, 0.5)
    if not ok:
        return r.done()
    want = ref_tilt(lr, R, 0.5)
    if isinstance(got, torch.Tensor) and bool((got > 0).all()) and close(got.sum(), 1.0):
        r.expect(False, "returns log-probabilities", "you returned probabilities: return log_softmax(...), not softmax(...)")
        return r.done()
    r.expect(close(torch.logsumexp(got, -1), 0.0), "the result is normalized: logsumexp(log π_β) = 0",
             "add log π_ref and R/β, then normalize with log_softmax")
    r.expect(close(got, want), f"π_β = {[round(v, 4) for v in want.exp().tolist()]} for π_ref = [0.5, 0.3, 0.2], R = [1, 0, −1], β = 0.5",
             "the exponent is R/β (divide by β); multiplying by β flattens instead of sharpens")
    ok, big = call(r, fn, lr, R, 1e6)
    if ok:
        r.expect(close(big, lr), "β → ∞ returns π_ref", "with huge β the reward term vanishes")
    ok, flat = call(r, fn, lr, torch.full((3,), 5.0, dtype=DT), 0.1)
    if ok:
        r.expect(close(flat, lr), "a constant reward leaves π_ref unchanged", "normalization must remove any constant shift")
    return r.done()


def check_kl(fn, strict: bool = False) -> bool:
    """fn(logp, logp_ref) -> KL(π‖π_ref) over the last dimension."""
    if not_attempted(fn):
        return False
    r = Report("kl", strict)
    lp, lr = _logp([0.7, 0.2, 0.1]), _logp([0.5, 0.3, 0.2])
    ok, got = call(r, fn, lp, lr)
    if not ok:
        return r.done()
    reverse = float(ref_kl(lr, lp))
    if close(got, reverse):
        r.expect(False, "KL(π‖π_ref), not KL(π_ref‖π)",
                 f"you got {reverse:.4f} = KL(π_ref‖π): weight the log-ratio by π = exp(logp), the first argument")
        return r.done()
    r.expect(close(got, 0.085122, atol=1e-5), "D35 worked example: KL = 0.0851 nats",
             "Σ_y π(y)·(log π(y) − log π_ref(y)); did you weight by π and sum?")
    ok, same = call(r, fn, lr, lr)
    if ok:
        r.expect(close(same, 0.0), "KL of a distribution with itself is 0")
    batch_p = torch.stack([lp, lr])
    ok, b = call(r, fn, batch_p, torch.stack([lr, lr]))
    if ok:
        r.expect(close(b, torch.tensor([0.085122, 0.0], dtype=DT), atol=1e-5), "works on a batch [2, V] → [2]",
                 "sum over the last dimension only: .sum(-1)")
    return r.done()


def check_bon_probs(fn, strict: bool = False) -> bool:
    """fn(p_sorted, n) -> P(best-of-n picks answer i) for answers sorted by ascending proxy."""
    if not_attempted(fn):
        return False
    r = Report("bon_probs", strict)
    u = torch.full((3,), 1 / 3, dtype=DT)
    ok, got = call(r, fn, u, 1)
    if not ok:
        return r.done()
    r.expect(close(got, u), "n = 1 returns π_ref (one sample, nothing to choose)")
    ok, got = call(r, fn, u, 2)
    if ok:
        r.expect(close(got, torch.tensor([1, 3, 5], dtype=DT) / 9), "uniform over 3 answers, n = 2: [1/9, 3/9, 5/9]",
                 "P(max = i) = P(all n ≤ i) − P(all n < i) = C_i^n − C_{i−1}^n with C the cumulative sum")
    p = torch.tensor([0.5, 0.3, 0.2], dtype=DT)
    ok, got = call(r, fn, p, 3)
    if ok:
        r.expect(close(got, torch.tensor([0.125, 0.387, 0.488], dtype=DT)), "π_ref = [0.5, 0.3, 0.2], n = 3: [0.125, 0.387, 0.488]",
                 "C_{i−1} = C_i − p_i; raise both cumulative sums to the power n")
    g = torch.Generator().manual_seed(0)
    q = torch.rand(50, generator=g, dtype=DT)
    q = q / q.sum()
    ok, got = call(r, fn, q, 7)
    if ok:
        r.expect(close(got.sum(), 1.0) and bool((got >= -1e-12).all()), "a valid distribution for random π_ref, n = 7")
    return r.done()


def check_bon_kl(fn, strict: bool = False) -> bool:
    """fn(n) -> log n − (n−1)/n."""
    if not_attempted(fn):
        return False
    r = Report("bon_kl", strict)
    ok, got = call(r, fn, 1)
    if not ok:
        return r.done()
    r.expect(close(got, 0.0), "n = 1: KL = 0")
    for n in (2, 16, 1024):
        ok, got = call(r, fn, n)
        if not ok:
            break
        if close(got, math.log(n)):
            r.expect(False, f"n = {n}", "log n alone is only an upper bound: subtract (n − 1)/n")
            break
        r.expect(close(got, ref_bon_kl(n)), f"n = {n}: KL = {ref_bon_kl(n):.4f} nats",
                 "KL = log n + (n−1)·E[log C] with E[log C] = −1/n under best-of-n")
    return r.done()


def check_length_controlled(fn, strict: bool = False) -> bool:
    """fn(r, z, p) -> (r_lc, b_hat)."""
    if not_attempted(fn):
        return False
    rep = Report("length_controlled", strict)
    z = torch.tensor([-1.0, 0.0, 1.0], dtype=DT)
    r = torch.tensor([0.0, 1.0, 4.0], dtype=DT)
    u = torch.full((3,), 1 / 3, dtype=DT)
    ok, got = call(rep, fn, r, z, u)
    if not ok:
        return rep.done()
    try:
        r_lc, b_hat = got
    except Exception:
        rep.expect(False, "returns a pair (r_lc, b_hat)", "end with `return r_lc, b_hat`")
        return rep.done()
    rep.expect(close(b_hat, 2.0), "uniform weights: b̂ = Cov(r, z)/Var(z) = 2",
               "slope = Σ p (z − z̄)(r − r̄) / Σ p (z − z̄)²; Var(r) in the denominator is wrong")
    rep.expect(close(r_lc, torch.tensor([2.0, 1.0, 2.0], dtype=DT)), "r_lc = r − b̂·(z − z̄) = [2, 1, 2]",
               "subtract b̂ times the centered length, not b̂·z with an uncentered z")
    p = torch.tensor([0.2, 0.2, 0.6], dtype=DT)
    want, wb = ref_length_controlled(r, z, p)
    ok, got = call(rep, fn, r, z, p)
    if ok:
        r_lc, b_hat = got
        rep.expect(close(b_hat, wb) and close(r_lc, want), f"weighted by π_ref = [0.2, 0.2, 0.6]: b̂ = {float(wb):.4f}",
                   "use the weights p for the means and sums (z̄ = Σ p z), not plain .mean()")
        zc = z - (p * z).sum()
        rep.expect(close((p * zc * r_lc).sum(), 0.0), "r_lc has zero covariance with length under p")
    return rep.done()


def check_entropy(fn, strict: bool = False) -> bool:
    """fn(logits) -> entropy over the last dimension."""
    if not_attempted(fn):
        return False
    r = Report("entropy", strict)
    ok, got = call(r, fn, torch.zeros(4))
    if not ok:
        return r.done()
    r.expect(close(got, math.log(4)), "uniform over 4 tokens: H = log 4 = 1.3863 nats",
             "H = −Σ p log p with p = softmax(logits)")
    x = torch.tensor([2.0, 1.0, 0.0])
    ok, got = call(r, fn, x)
    if ok:
        r.expect(close(got, float(ref_entropy(x.double()))), f"logits [2, 1, 0]: H = {float(ref_entropy(x.double())):.4f}",
                 "did you forget the minus sign or the softmax?")
    ok, got = call(r, fn, x + 100)
    if ok:
        r.expect(close(got, float(ref_entropy(x.double()))), "adding a constant to every logit leaves H unchanged")
    ok, got = call(r, fn, torch.tensor([0.0, -200.0, -200.0]))
    if ok:
        r.expect(bool(torch.isfinite(torch.as_tensor(got)).all()) and close(got, 0.0, atol=1e-6),
                 "a near-certain float32 distribution gives H ≈ 0, not nan",
                 "log(softmax(x)) is −inf where softmax underflows to 0, and 0·(−inf) = nan: use log_softmax")
    ok, got = call(r, fn, torch.randn(2, 3, 5))
    if ok:
        r.expect(isinstance(got, torch.Tensor) and tuple(got.shape) == (2, 3), "logits [2, 3, 5] → entropies [2, 3]",
                 "sum over the last dimension only")
    return r.done()


def check_pg_loss(fn, strict: bool = False) -> bool:
    """fn(logp_seq, adv, kl, ent, beta, lam) -> scalar loss to minimize."""
    if not_attempted(fn):
        return False
    r = Report("pg_loss", strict)
    lp = torch.tensor([-1.0, -2.0, -0.5, -3.0], dtype=DT, requires_grad=True)
    adv = torch.tensor([1.0, -0.5, 0.25, -0.75], dtype=DT)
    kl = torch.tensor(0.3, dtype=DT, requires_grad=True)
    ent = torch.tensor(1.2, dtype=DT, requires_grad=True)
    ok, loss = call(r, fn, lp, adv, kl, ent, 0.5, 0.1)
    if not ok:
        return r.done()
    want = float(ref_pg_loss(lp.detach(), adv, kl.detach(), ent.detach(), 0.5, 0.1))
    r.expect(isinstance(loss, torch.Tensor) and close(loss.detach(), want), f"loss = {want:.4f} on the fixed example",
             "−mean(A·log π) + β·KL − λ·H")
    try:
        loss.backward()
        r.expect(close(lp.grad, -adv / 4), "∂loss/∂log π(y_i) = −A_i/N",
                 "average over the N answers (.mean()), and the policy term has a minus sign")
        r.expect(close(kl.grad, 0.5), "∂loss/∂KL = β: the KL term is a penalty (added)")
        r.expect(close(ent.grad, -0.1), "∂loss/∂H = −λ: the entropy bonus is subtracted",
                 "you minimize the loss, so a bonus for entropy enters with a minus sign")
    except Exception as error:
        r.expect(False, f"backward failed: {error}", "return a scalar tensor built from the inputs")
    return r.done()


def check_aggregate(fn, strict: bool = False) -> bool:
    """fn(scores [M, K], mode) -> [K] for mode in mean / min / lcb."""
    if not_attempted(fn):
        return False
    r = Report("aggregate", strict)
    s = torch.tensor([[1.0, 5.0, 0.0], [2.0, 0.0, 0.0], [3.0, 1.0, 0.0]], dtype=DT)
    for mode in ("mean", "min", "lcb"):
        ok, got = call(r, fn, s, mode)
        if not ok:
            return r.done()
        want = ref_aggregate(s, mode)
        hint = {"mean": "average over models: dim 0",
                "min": "torch.min over dim 0 returns (values, indices): take .values",
                "lcb": "mean − std over models, population std (unbiased=False)"}[mode]
        r.expect(isinstance(got, torch.Tensor) and close(got, want), f"{mode}: {[round(v, 3) for v in want.tolist()]}", hint)
    two = s[:2]
    ok, got = call(r, fn, two, "lcb")
    if ok:
        r.expect(close(got, two.min(0).values), "with two models, mean − std equals the minimum")
    return r.done()


def _synthetic_logs():
    g = torch.Generator().manual_seed(3)
    n = 50
    s = torch.arange(n, dtype=DT)
    noise = lambda scale: (scale * torch.randn(n, generator=g, dtype=DT)).tolist()
    healthy = {"reward": noise(0.3), "kl": [0.2 + v for v in noise(0.02)],
               "entropy": [1.0 + v for v in noise(0.02)], "length": [40 + v for v in noise(2.0)]}
    hacked = {"reward": (0.05 * s + torch.tensor(noise(0.3))).tolist(), "kl": (0.2 * s).tolist(),
              "entropy": (1.0 * torch.exp(-s / 10)).tolist(), "length": (40 + 2 * s + torch.tensor(noise(2.0))).tolist()}
    learning = {"reward": (0.05 * s + torch.tensor(noise(0.3))).tolist(), "kl": (0.01 * s).tolist(),
                "entropy": [0.8 + v for v in noise(0.02)], "length": [40 + v for v in noise(2.0)]}
    return healthy, hacked, learning


def check_diagnose(fn, strict: bool = False) -> bool:
    """fn(log, kl_limit, ent_ratio=0.5, t_crit=3.0) -> {flag name: bool}."""
    if not_attempted(fn):
        return False
    r = Report("diagnose", strict)
    healthy, hacked, learning = _synthetic_logs()
    cases = [("flat, healthy log", healthy, {f: False for f in FLAGS}),
             ("reward ↑, KL → 9.8 nats, length ↑, entropy ×0.01", hacked, {f: True for f in FLAGS}),
             ("reward ↑ with KL only 0.45 nats (limit 2)", learning, {f: False for f in FLAGS})]
    for label, log, want in cases:
        ok, got = call(r, fn, log, 2.0)
        if not ok:
            return r.done()
        if not isinstance(got, dict) or set(got) != set(FLAGS):
            r.expect(False, f"{label}: returns a dict with the three flag names",
                     f"keep the keys exactly as in the scaffold: {list(FLAGS)}")
            return r.done()
        for f in FLAGS:
            hint = {FLAGS[0]: "both conditions: trend_t(reward) > t_crit AND late(kl) > kl_limit",
                    FLAGS[1]: "trend_t(length) > t_crit",
                    FLAGS[2]: "late(entropy) < ent_ratio · early(entropy)"}[f]
            r.expect(bool(got[f]) == want[f], f"{label}: '{f}' is {want[f]}", hint)
    return r.done()


def check_masked_entropy(fn, entropy: Callable = ref_entropy, strict: bool = False) -> bool:
    """fn(logits [B, T, V], mask [B, T]) -> mean entropy over response tokens."""
    if not_attempted(fn):
        return False
    r = Report("masked_entropy", strict)
    logits = torch.zeros(2, 3, 4)
    logits[0, :, 0] = 50.0                                     # row 0: 3 near-certain tokens, H ≈ 0
    logits[1, 1:, 0] = 50.0                                    # row 1: 1 uniform token (H = log 4), then 2 pads
    mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]])
    ok, got = call(r, fn, logits, mask)
    if not ok:
        return r.done()
    want = math.log(4) / 4
    if close(got, ref_entropy(logits).mean(), atol=1e-4):
        r.expect(False, "pads excluded", "multiply by the mask and divide by mask.sum(), not .mean() over all positions")
        return r.done()
    if close(got, math.log(4) / 2, atol=1e-4):
        r.expect(False, "token-weighted mean", "you averaged per answer first: divide the masked sum by the total number of response tokens")
        return r.done()
    r.expect(close(got, want, atol=1e-4), f"3 near-certain tokens + 1 uniform token: H̄ = log 4 / 4 = {want:.4f}",
             "(entropy(logits) * mask).sum() / mask.sum()")
    return r.done()
