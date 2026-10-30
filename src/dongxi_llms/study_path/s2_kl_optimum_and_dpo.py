"""Checks and reference implementations for study-path stage S2: the KL-regularized optimum → DPO.

The notebook ``notebooks/study-path/s2_kl_optimum_and_dpo.ipynb`` asks the learner to write the
key lines of each mechanism: the KL divergence, the closed-form optimum π* ∝ π_ref·exp(R/β), the
implicit reward and its partition-function cancellation, the DPO loss and its gradient weight,
tabular DPO, the masked sequence log-probability, one DPO step on Qwen3, and the estimates
(pair accuracy, sampled KL) with their standard errors.

Each ``check_*`` function receives the learner's object and prints one line per test: ⏳ when the
exercise is not attempted yet, ✅ when it passes, and ❌ with a hint when it fails. Checks never
raise, so an unfinished notebook still runs; ``strict=True`` turns failures into
``AssertionError`` (the reference cells use it). The ``ref_*`` functions are the references; the
non-exercise helpers (pair encoding, the vectorised tabular fit) are printed by the notebook.
"""
from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import Report, call, close, not_attempted

D64 = torch.float64

# --------------------------------------------------------------------------- toy fixtures
# One prompt, five complete answers. The reference favours answer 0; the reward favours answer 3.
TOY_REF = torch.tensor([0.40, 0.25, 0.15, 0.12, 0.08], dtype=D64)
TOY_REWARD = torch.tensor([0.0, 1.0, 0.5, 2.0, -1.0], dtype=D64)
TOY_BETA = 1.0
# D35's worked example (three tokens) and the chapter-11 fixture (three answers).
D35_REF = torch.tensor([0.5, 0.3, 0.2], dtype=D64)
D35_PI = torch.tensor([0.7, 0.2, 0.1], dtype=D64)
CH11_REF = torch.tensor([0.6, 0.3, 0.1], dtype=D64)
CH11_REWARD = torch.tensor([0.0, 0.5, 1.0], dtype=D64)


# =========================================================================== Part A · KL
def ref_kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """KL(p‖q) = Σ_y p(y) log(p(y)/q(y)) over the last dimension (0·log 0 = 0)."""
    return torch.special.xlogy(p, p / q).sum(-1)


def check_kl(fn, strict: bool = False) -> bool:
    if not_attempted(fn):
        return False
    r = Report("kl", strict)
    ok, got = call(r, fn, D35_PI, D35_REF)
    if not ok:
        return r.done()
    want = float(ref_kl(D35_PI, D35_REF))
    r.expect(close(got, want), f"D35 example: KL(π‖π_ref) = {want:.4f}",
             _kl_hint(fn, got, want))
    ok, same = call(r, fn, D35_REF, D35_REF)
    if ok:
        r.expect(close(same, 0.0, atol=1e-10), "KL(p‖p) = 0", "log(p/p) = 0 for every term")
    ok, back = call(r, fn, D35_REF, D35_PI)
    if ok:
        rev = float(ref_kl(D35_REF, D35_PI))
        r.expect(close(back, rev), f"reverse direction KL(π_ref‖π) = {rev:.4f} (≠ {want:.4f}: not symmetric)",
                 "the first argument is the weighting distribution: Σ p·log(p/q)")
    batch_p = torch.stack([D35_PI, D35_REF])
    batch_q = torch.stack([D35_REF, D35_PI])
    ok, both = call(r, fn, batch_p, batch_q)
    if ok:
        r.expect(close(both, ref_kl(batch_p, batch_q)), "a batch [2, 3] gives one KL per row, shape [2]",
                 "sum over the last dimension only: .sum(-1)")
    return r.done()


def _kl_hint(fn, got, want) -> str:
    try:
        if close(got, -want):
            return "sign flipped: the log-ratio is log(p/q), not log(q/p)"
        if close(got, float(ref_kl(D35_REF, D35_PI))):
            return "arguments swapped: weight by p (the first argument), log(p/q)"
        if close(got, float((D35_PI.log() - D35_REF.log()).sum())):
            return "weight each log-ratio by p(y) before summing"
    except Exception:
        pass
    return "KL(p‖q) = (p * (p / q).log()).sum(-1)"


# =========================================================================== Part B · optimum
def ref_optimal_policy(ref: torch.Tensor, reward: torch.Tensor, beta: float) -> torch.Tensor:
    """π*(y) = π_ref(y)·exp(R(y)/β) / Z, computed as softmax(log π_ref + R/β)."""
    return torch.softmax(ref.log() + reward / beta, -1)


def check_optimal_policy(fn, strict: bool = False) -> bool:
    if not_attempted(fn):
        return False
    r = Report("optimal_policy", strict)
    ok, got = call(r, fn, CH11_REF, CH11_REWARD, 0.5)
    if not ok:
        return r.done()
    want = torch.tensor([0.278501, 0.378522, 0.342977], dtype=D64)
    hint = "π* = softmax(log π_ref + R/β)"
    if close(got, torch.softmax(CH11_REWARD / 0.5, -1), atol=1e-5):
        hint = "you dropped the reference: add log π_ref inside the softmax"
    elif close(got, CH11_REF * torch.exp(CH11_REWARD / 0.5), atol=1e-5):
        hint = "normalize: divide π_ref·exp(R/β) by its sum Z (or use softmax)"
    elif close(got, torch.softmax(CH11_REF.log() + 0.5 * CH11_REWARD, -1), atol=1e-5):
        hint = "divide the reward by β, don't multiply"
    r.expect(close(got, want, atol=1e-5), "chapter 11 fixture, β = 0.5: π* = (0.2785, 0.3785, 0.3430)", hint)
    ok, got1 = call(r, fn, CH11_REF, CH11_REWARD, 1.0)
    if ok:
        r.expect(close(got1, torch.tensor([0.439096, 0.361973, 0.198931], dtype=D64), atol=1e-5),
                 "β = 1: π* = (0.4391, 0.3620, 0.1989), closer to π_ref", hint)
    ok, shifted = call(r, fn, TOY_REF, TOY_REWARD + 50.0, TOY_BETA)
    if ok:
        r.expect(close(shifted, ref_optimal_policy(TOY_REF, TOY_REWARD, TOY_BETA), atol=1e-6),
                 "adding 50 to every reward leaves π* unchanged (and stays finite)",
                 "exp(50) overflows less than you think, but normalization must cancel it: use softmax")
    return r.done()


def ref_objective(pi, ref, reward, beta, kl=ref_kl):
    """J(π) = E_π[R] − β·KL(π‖π_ref)."""
    return (pi * reward).sum(-1) - beta * kl(pi, ref)


def check_objective(fn, strict: bool = False) -> bool:
    """fn(pi, ref, reward, beta) -> J."""
    if not_attempted(fn):
        return False
    r = Report("objective", strict)
    beta = 0.5
    ok, got = call(r, fn, D35_PI, D35_REF, CH11_REWARD, beta)
    if not ok:
        return r.done()
    want = float(ref_objective(D35_PI, D35_REF, CH11_REWARD, beta))
    hint = "J = (pi * reward).sum(-1) - beta * kl(pi, ref)"
    if close(got, float((D35_PI * CH11_REWARD).sum() + beta * ref_kl(D35_PI, D35_REF))):
        hint = "the KL term is a cost: subtract β·KL"
    elif close(got, float((D35_PI * CH11_REWARD).sum() - beta * ref_kl(D35_REF, D35_PI))):
        hint = "KL direction: the policy being optimized is the first argument, kl(pi, ref)"
    r.expect(close(got, want), f"J(π) for the D35 policy with rewards (0, 0.5, 1), β = 0.5: {want:.4f}", hint)
    ok, at_ref = call(r, fn, D35_REF, D35_REF, CH11_REWARD, beta)
    if ok:
        r.expect(close(at_ref, float((D35_REF * CH11_REWARD).sum())), "J(π_ref) = E_ref[R] (no KL cost)", hint)
    star = ref_optimal_policy(TOY_REF, TOY_REWARD, TOY_BETA)
    Z = float((TOY_REF * torch.exp(TOY_REWARD / TOY_BETA)).sum())
    ok, at_star = call(r, fn, star, TOY_REF, TOY_REWARD, TOY_BETA)
    if ok:
        r.expect(close(at_star, TOY_BETA * math.log(Z)), f"J(π*) = β·log Z = {TOY_BETA * math.log(Z):.4f} on the toy", hint)
    return r.done()


# =========================================================================== Part C · implicit reward
def ref_implicit_reward(pi, ref, beta):
    """r̂(y) = β·log(π(y)/π_ref(y)); for π = π* this equals R(y) − β·log Z."""
    return beta * (pi.log() - ref.log())


def ref_pair_prob(scores, i, j):
    """Bradley–Terry P(answer i ≻ answer j) = σ(s_i − s_j)."""
    return torch.sigmoid(scores[..., i] - scores[..., j])


def check_implicit_reward(reward_fn, prob_fn, strict: bool = False) -> bool:
    if not_attempted(reward_fn) or not_attempted(prob_fn):
        return False
    r = Report("implicit reward", strict)
    beta = TOY_BETA
    star = ref_optimal_policy(TOY_REF, TOY_REWARD, beta)
    ok, imp = call(r, reward_fn, star, TOY_REF, beta)
    if not ok:
        return r.done()
    want = ref_implicit_reward(star, TOY_REF, beta)
    hint = "r̂ = β·(log π − log π_ref)"
    if close(imp, star.log() - TOY_REF.log()) and beta != 1.0:
        hint = "multiply the log-ratio by β"
    elif close(imp, -want):
        hint = "sign: log π − log π_ref, the policy on top"
    r.expect(close(imp, want), "implicit reward of π* matches β·log(π*/π_ref)", hint)
    ok, imp2 = call(r, reward_fn, ref_optimal_policy(TOY_REF, TOY_REWARD, 0.25), TOY_REF, 0.25)
    if ok:
        r.expect(close(imp2, ref_implicit_reward(ref_optimal_policy(TOY_REF, TOY_REWARD, 0.25), TOY_REF, 0.25)),
                 "also at β = 0.25", "multiply the log-ratio by β")
    if ok and close(imp, want):
        gap = TOY_REWARD - imp
        r.expect(close(gap, gap.mean().expand_as(gap)), f"R − r̂ is the same constant β·log Z = {float(gap[0]):.4f} for every answer")
    s = torch.tensor([0.3, 1.5, -0.2], dtype=D64)
    ok, p = call(r, prob_fn, s, 1, 0)
    if ok:
        hint = "P(i ≻ j) = σ(s_i − s_j)"
        if close(p, torch.sigmoid(s[0] - s[1])):
            hint = "order: σ(s_i − s_j), the first index is the winner"
        elif close(p, s[1] - s[0]):
            hint = "pass the difference through torch.sigmoid"
        r.expect(close(p, torch.sigmoid(torch.tensor(1.2, dtype=D64))), "σ(1.5 − 0.3) = 0.7685", hint)
    return r.done()


# =========================================================================== Part D · DPO loss
def ref_dpo_loss(policy_logps: torch.Tensor, ref_logps: torch.Tensor, beta: float):
    """Mean DPO loss over pairs and the implicit rewards [B, 2] (column 0 chosen, 1 rejected)."""
    implicit = beta * (policy_logps - ref_logps.detach())
    margin = implicit[:, 0] - implicit[:, 1]
    return -F.logsigmoid(margin).mean(), implicit


def _dpo_cases():
    pol = torch.tensor([[-10.0, -12.0], [-30.0, -25.0], [-7.0, -7.5]], dtype=D64)
    ref = torch.tensor([[-11.0, -11.0], [-28.0, -26.0], [-7.0, -7.0]], dtype=D64)
    return pol, ref


def check_dpo_loss(fn, strict: bool = False) -> bool:
    """fn(policy_logps [B,2], ref_logps [B,2], beta) -> (loss, implicit [B,2])."""
    if not_attempted(fn):
        return False
    r = Report("dpo_loss", strict)
    pol, ref = _dpo_cases()
    ok, out = call(r, fn, ref.clone(), ref, 0.1)
    if not ok:
        return r.done()
    if not r.expect(isinstance(out, (tuple, list)) and len(out) == 2, "returns (loss, implicit)",
                    "return loss, implicit"):
        return r.done()
    r.expect(close(out[0], math.log(2)), "policy = reference → loss = log 2 = 0.6931",
             "loss = −log σ(margin), margin = implicit[:, 0] − implicit[:, 1]")
    ok, out = call(r, fn, pol, ref, 0.5)
    if not ok:
        return r.done()
    want_loss, want_imp = ref_dpo_loss(pol, ref, 0.5)
    loss, imp = out
    r.expect(close(imp, want_imp), "implicit rewards β·(log π − log π_ref), shape [B, 2]",
             "implicit = beta * (policy_logps - ref_logps)")
    hint = "loss = -F.logsigmoid(implicit[:, 0] - implicit[:, 1]).mean()"
    m = want_imp[:, 0] - want_imp[:, 1]
    if close(loss, -F.logsigmoid(-m).mean()):
        hint = "sign of the margin: chosen minus rejected"
    elif close(loss, -F.logsigmoid(m).sum()):
        hint = "average over pairs: .mean(), not .sum()"
    elif close(loss, -F.logsigmoid(0.5 * ((pol[:, 0] - pol[:, 1]))).mean()):
        hint = "subtract the reference log-probs: the margin is relative to π_ref"
    r.expect(close(loss, want_loss), f"three pairs, β = 0.5: loss = {float(want_loss):.4f}", hint)
    far = torch.tensor([[-500.0, 0.0]], dtype=torch.float32)
    ok, out = call(r, fn, far, torch.zeros_like(far), 1.0)
    if ok:
        r.expect(bool(torch.isfinite(torch.as_tensor(out[0]))) and close(out[0], 500.0, atol=1e-2),
                 "margin −500 in float32: loss = 500, finite",
                 "log(sigmoid(x)) underflows to −inf; use F.logsigmoid")
    return r.done()


def ref_dpo_grads(policy_logps, ref_logps, beta):
    """∂(mean loss)/∂ log π for chosen and rejected: ∓ β·σ(−m)/B, shape [B, 2]."""
    m = beta * ((policy_logps[:, 0] - ref_logps[:, 0]) - (policy_logps[:, 1] - ref_logps[:, 1]))
    w = torch.sigmoid(-m)
    B = policy_logps.shape[0]
    return torch.stack([-beta * w / B, beta * w / B], 1)


def check_dpo_grads(fn, strict: bool = False) -> bool:
    if not_attempted(fn):
        return False
    r = Report("dpo gradient", strict)
    pol, ref = _dpo_cases()
    beta = 0.5
    ok, got = call(r, fn, pol, ref, beta)
    if not ok:
        return r.done()
    leaf = pol.clone().requires_grad_()
    ref_dpo_loss(leaf, ref, beta)[0].backward()
    want = leaf.grad
    hint = "w = σ(−m); grads = (−β·w/B, +β·w/B)"
    try:
        m = beta * ((pol[:, 0] - ref[:, 0]) - (pol[:, 1] - ref[:, 1]))
        if close(got, torch.stack([-beta * torch.sigmoid(m) / 3, beta * torch.sigmoid(m) / 3], 1)):
            hint = "the weight is σ(−m), not σ(m): pairs already ordered correctly (m > 0) get a small push"
        elif close(got, want * 3):
            hint = "the loss is a mean over B pairs: divide by B"
        elif close(got, want / beta):
            hint = "the chain rule brings a factor β (∂m/∂ log π = ±β)"
        elif close(got, -want):
            hint = "sign: gradient descent raises log π(chosen), so ∂L/∂ log π(chosen) < 0"
    except Exception:
        pass
    r.expect(close(got, want), "matches autograd on three pairs", hint)
    return r.done()


# =========================================================================== Part E · tabular DPO
PAIR_I, PAIR_J = torch.triu_indices(5, 5, 1)


def sample_pairs(n: int, reward: torch.Tensor, seed: int):
    """n preference pairs over the K answers: a uniformly random unordered pair, winner by BT on `reward`."""
    K = reward.shape[0]
    I, J = torch.triu_indices(K, K, 1)
    g = torch.Generator().manual_seed(seed)
    k = torch.randint(len(I), (n,), generator=g)
    i, j = I[k], J[k]
    win = torch.rand(n, generator=g, dtype=D64) < torch.sigmoid(reward[i] - reward[j])
    return torch.where(win, i, j), torch.where(win, j, i)


def ref_tabular_dpo_loss(theta, W, L, ref_logp, beta, dpo_loss=ref_dpo_loss):
    """DPO loss of a tabular policy π_θ = softmax(θ) over K answers on pairs (W[n] ≻ L[n])."""
    logp = theta.log_softmax(-1)
    policy = torch.stack([logp[W], logp[L]], 1)
    ref = torch.stack([ref_logp[W], ref_logp[L]], 1)
    return dpo_loss(policy, ref, beta)[0]


def check_tabular_dpo_loss(fn, strict: bool = False) -> bool:
    """fn(theta [K], W [N], L [N], ref_logp [K], beta) -> scalar loss with a gradient for theta."""
    if not_attempted(fn):
        return False
    r = Report("tabular dpo loss", strict)
    ref_logp = TOY_REF.log()
    W, L = torch.tensor([3, 1, 3, 0]), torch.tensor([0, 2, 4, 1])
    theta = torch.tensor([0.2, -0.1, 0.0, 1.0, -0.5], dtype=D64, requires_grad=True)
    ok, loss = call(r, fn, ref_logp.clone(), W, L, ref_logp, 0.5)
    if not ok:
        return r.done()
    r.expect(close(loss, math.log(2)), "θ = log π_ref → loss = log 2", "start from the reference: every margin is 0")
    ok, loss = call(r, fn, theta, W, L, ref_logp, 0.5)
    if not ok:
        return r.done()
    t2 = theta.detach().clone().requires_grad_()
    want = ref_tabular_dpo_loss(t2, W, L, ref_logp, 0.5)
    want.backward()
    hint = "logp = theta.log_softmax(-1); stack logp[W], logp[L] as the [N, 2] policy log-probs"
    if close(loss, ref_tabular_dpo_loss(theta.detach(), L, W, ref_logp, 0.5)):
        hint = "column 0 is the winner W, column 1 the loser L"
    elif close(loss, ref_dpo_loss(torch.stack([theta.detach().softmax(-1)[W], theta.detach().softmax(-1)[L]], 1),
                                  torch.stack([TOY_REF[W], TOY_REF[L]], 1), 0.5)[0]):
        hint = "use log-probabilities (log_softmax), not probabilities"
    r.expect(close(loss.detach() if isinstance(loss, torch.Tensor) else loss, want.detach()),
             f"four pairs, β = 0.5: loss = {float(want.detach()):.4f}", hint)
    try:
        g, = torch.autograd.grad(loss, theta)
        r.expect(close(g, t2.grad), "gradient flows to θ and matches", "build logp from theta inside the function")
    except Exception as error:
        r.expect(False, f"no gradient to θ ({type(error).__name__})", "don't detach theta; compute logp from it")
    return r.done()


def pair_counts(W, L, K: int = 5) -> torch.Tensor:
    """C[w, l] = number of pairs where answer w beat answer l."""
    C = torch.zeros(K, K, dtype=D64)
    C.index_put_((W, L), torch.ones(len(W), dtype=D64), accumulate=True)
    return C


def fit_tabular_dpo(counts: torch.Tensor, ref: torch.Tensor, beta: float, steps: int = 1500, lr: float = 0.1):
    """Minimise the DPO loss of tabular policies on pair counts, many datasets at once.

    counts: [M, K, K] (C[m, w, l] = wins of w over l in dataset m) or [K, K]. Same loss as
    ``ref_tabular_dpo_loss`` on the individual pairs, averaged per dataset; Adam on θ from log π_ref.
    Returns π [M, K] (or [K]) and the final loss per dataset.
    """
    single = counts.ndim == 2
    C = counts[None] if single else counts
    M, K, _ = C.shape
    ref_logp = ref.log()
    theta = ref_logp.expand(M, K).clone().requires_grad_()
    opt = torch.optim.Adam([theta], lr=lr)
    N = C.sum((1, 2)).clamp(min=1e-12)
    for _ in range(steps):
        imp = beta * (theta.log_softmax(-1) - ref_logp)                  # [M, K]
        margin = imp[:, :, None] - imp[:, None, :]                       # [M, w, l]
        loss = ((C * F.softplus(-margin)).sum((1, 2)) / N)
        opt.zero_grad()
        loss.sum().backward()
        opt.step()
    pi = theta.detach().softmax(-1)
    return (pi[0], loss.detach()[0]) if single else (pi, loss.detach())


def tv(p, q):
    """Total-variation distance ½·Σ|p − q| over the last dimension."""
    return 0.5 * (p - q).abs().sum(-1)


# =========================================================================== Part F · sequence log-probs
def ref_sequence_logps(logits: torch.Tensor, ids: torch.Tensor, answer_mask: torch.Tensor) -> torch.Tensor:
    """Σ_t log π(ids[t] | ids[<t]) over answer positions t (answer_mask[t] = 1), shape [N].

    logits[:, t] predicts ids[:, t + 1], so drop the last logit and the first id/mask entry.
    """
    lp = logits[:, :-1].log_softmax(-1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    return (lp * answer_mask[:, 1:]).sum(1)


def _seq_case():
    g = torch.Generator().manual_seed(3)
    logits = torch.randn(2, 6, 7, generator=g, dtype=D64)
    ids = torch.tensor([[1, 4, 2, 5, 6, 0], [3, 3, 1, 2, 6, 6]])
    mask = torch.tensor([[0, 0, 1, 1, 1, 0], [0, 0, 0, 1, 1, 1]], dtype=D64)   # answer + end token
    return logits, ids, mask


def check_sequence_logps(fn, strict: bool = False) -> bool:
    if not_attempted(fn):
        return False
    r = Report("sequence_logps", strict)
    logits, ids, mask = _seq_case()
    ok, got = call(r, fn, logits, ids, mask)
    if not ok:
        return r.done()
    want = ref_sequence_logps(logits, ids, mask)
    lsm = logits.log_softmax(-1)
    hint = "lp = logits[:, :-1].log_softmax(-1).gather(-1, ids[:, 1:, None]).squeeze(-1); (lp * mask[:, 1:]).sum(1)"
    try:
        noshift = (lsm.gather(-1, ids[..., None]).squeeze(-1) * mask).sum(1)
        lp = lsm[:, :-1].gather(-1, ids[:, 1:, None]).squeeze(-1)
        if close(got, noshift):
            hint = "shift by one: logits at position t predict ids[t + 1]"
        elif close(got, (lp * mask[:, :-1]).sum(1)):
            hint = "shift the mask with the labels: use answer_mask[:, 1:]"
        elif close(got, lp.sum(1)):
            hint = "you summed over prompt tokens too: multiply by the answer mask"
        elif close(got, (lp * mask[:, 1:]).sum(1) / mask[:, 1:].sum(1)):
            hint = "sum, don't average: DPO's sequence probability is a product of token probabilities"
        elif close(got, (logits[:, :-1].gather(-1, ids[:, 1:, None]).squeeze(-1) * mask[:, 1:]).sum(1)):
            hint = "take log_softmax over the vocabulary before gathering"
    except Exception:
        pass
    r.expect(hasattr(got, "shape") and tuple(got.shape) == (2,), "one value per sequence, shape [N]",
             "sum over positions only: .sum(1)")
    r.expect(close(got, want), "two sequences: masked, shifted sums match", hint)
    return r.done()


def encode_pairs(tok, pairs, max_len: int = 256) -> dict:
    """Token ids, attention mask and answer mask for chosen/rejected of each pair, rows 2b and 2b+1.

    Same layout as ``qwen.answer_logps``: prompt = chat_prompt (thinking off), answer = text +
    <|im_end|>, right padding, left-truncated to ``max_len``. answer_mask marks the answer tokens
    themselves (positions in ``ids``), so the logit that predicts each lies one position earlier.
    """
    from .qwen import chat_prompt
    pad = tok.pad_token_id
    seqs, starts = [], []
    for p in pairs:
        prompt_ids = tok(chat_prompt(tok, p["prompt"]), add_special_tokens=False).input_ids
        for ans in (p["chosen"], p["rejected"]):
            ans_ids = tok(ans + "<|im_end|>", add_special_tokens=False).input_ids
            seq = (prompt_ids + ans_ids)[-max_len:]
            seqs.append(seq)
            starts.append(max(1, len(seq) - len(ans_ids)))
    T = max(map(len, seqs))
    ids = torch.full((len(seqs), T), pad)
    attn = torch.zeros_like(ids)
    amask = torch.zeros(len(seqs), T)
    for n, (s, st) in enumerate(zip(seqs, starts)):
        ids[n, :len(s)] = torch.tensor(s)
        attn[n, :len(s)] = 1
        amask[n, st:len(s)] = 1
    return {"ids": ids, "attn": attn, "amask": amask}


def pair_logps(model, batch: dict, seq_logps=ref_sequence_logps) -> torch.Tensor:
    """Σ log π(answer + <|im_end|>) for chosen and rejected, [B, 2]; one forward pass for both."""
    dev = next(model.parameters()).device
    ids, attn, amask = (batch[k].to(dev) for k in ("ids", "attn", "amask"))
    logits = model(input_ids=ids, attention_mask=attn).logits.float()
    return seq_logps(logits, ids, amask).view(-1, 2)


# =========================================================================== Part F · DPO step
def ref_dpo_step(policy, ref, batch, opt, beta, logps=pair_logps, dpo_loss=ref_dpo_loss):
    """One DPO update: frozen reference without a graph, policy with one, loss, step."""
    with torch.no_grad():
        ref_lp = logps(ref, batch)
    policy_lp = logps(policy, batch)
    loss, implicit = dpo_loss(policy_lp, ref_lp, beta)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
    return loss.item(), implicit.detach()


class TinyLM(nn.Module):
    """A two-layer causal 'language model' over a 12-token vocabulary, for CPU checks."""

    def __init__(self, vocab: int = 12, dim: int = 8, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.emb, self.out = nn.Embedding(vocab, dim), nn.Linear(dim, vocab)
        self.grad_modes = []                       # torch.is_grad_enabled() at each forward

    def forward(self, input_ids, attention_mask=None):
        self.grad_modes.append(torch.is_grad_enabled())
        h = self.emb(input_ids).cumsum(1)          # position t sees tokens ≤ t
        return SimpleNamespace(logits=self.out(torch.tanh(h)))


def tiny_batch() -> dict:
    ids = torch.tensor([[1, 2, 5, 6, 11, 0], [1, 2, 7, 11, 0, 0], [3, 4, 8, 9, 11, 0], [3, 4, 10, 11, 0, 0]])
    attn = (torch.arange(6)[None] < torch.tensor([5, 4, 5, 4])[:, None]).long()
    amask = torch.zeros(4, 6)
    amask[0, 2:5] = amask[1, 2:4] = amask[2, 2:5] = amask[3, 2:4] = 1
    return {"ids": ids, "attn": attn, "amask": amask}


def check_dpo_step(fn, strict: bool = False) -> bool:
    """fn(policy, ref, batch, opt, beta, logps) -> (loss float, implicit [B, 2])."""
    if not_attempted(fn):
        return False
    r = Report("dpo step", strict)
    policy = TinyLM()
    ref = copy.deepcopy(policy)                   # requires_grad left on, to catch a graph through ref
    before = [p.detach().clone() for p in policy.parameters()]
    ref_before = [p.detach().clone() for p in ref.parameters()]
    opt = torch.optim.SGD(policy.parameters(), lr=0.5)
    batch = tiny_batch()
    ok, out = call(r, fn, policy, ref, batch, opt, 0.5, pair_logps)
    if not ok:
        return r.done()
    try:
        loss, implicit = out
    except Exception:
        r.expect(False, "returns (loss, implicit)", "return loss.item(), implicit.detach()")
        return r.done()
    r.expect(close(float(loss), math.log(2), atol=1e-5), "first step, policy = copy of ref → loss = log 2",
             "loss, implicit = dpo_loss(policy_lp, ref_lp, beta)")
    r.expect(close(implicit, torch.zeros(2, 2), atol=1e-6), "implicit rewards all 0 at the first step, shape [2, 2]",
             "pass (policy_lp, ref_lp) in this order")
    r.expect(len(ref.grad_modes) > 0 and not any(ref.grad_modes),
             "the reference forward runs without a graph", "compute ref_lp = logps(ref, batch) inside torch.no_grad()")
    r.expect(all(p.grad is None or float(p.grad.abs().sum()) == 0 for p in ref.parameters()),
             "no gradient reaches the reference", "compute ref_lp inside torch.no_grad()")
    r.expect(all(torch.equal(a, b) for a, b in zip(ref_before, ref.parameters())), "reference weights unchanged")
    moved = any(not torch.equal(a, b) for a, b in zip(before, policy.parameters()))
    r.expect(moved, "the policy took a step", "policy_lp must keep its graph: no torch.no_grad() around it")
    if moved:
        with torch.no_grad():
            lp, rl = pair_logps(policy, batch), pair_logps(ref, batch)
        r.expect(bool(((lp - rl)[:, 0] - (lp - rl)[:, 1] > 0).all()),
                 "after the step both pairs have a positive margin", "the margin is chosen minus rejected")
    return r.done()


# =========================================================================== Part F · estimates
def ref_pair_accuracy(implicit: torch.Tensor):
    """Fraction of pairs whose chosen implicit reward is higher (ties count ½), and its binomial SE."""
    m = implicit[:, 0] - implicit[:, 1]
    acc = ((m > 0).float() + 0.5 * (m == 0).float()).mean()
    se = (acc * (1 - acc) / m.shape[0]).sqrt()
    return float(acc), float(se)


def check_pair_accuracy(fn, strict: bool = False) -> bool:
    if not_attempted(fn):
        return False
    r = Report("pair accuracy", strict)
    imp = torch.tensor([[0.3, 0.1], [-0.2, 0.4], [0.0, 0.0], [1.0, -1.0], [0.5, 0.2], [0.1, 0.3], [0.2, 0.2], [0.9, 0.8]])
    ok, out = call(r, fn, imp)
    if not ok:
        return r.done()
    try:
        acc, se = (float(x) for x in out)
    except Exception:
        r.expect(False, "returns (accuracy, standard error)", "return acc, se")
        return r.done()
    want_acc, want_se = ref_pair_accuracy(imp)
    hint = "ties (equal implicit rewards) count ½"
    if close(acc, 4 / 8):
        hint = "two pairs tie at 0 margin: count each tie as ½, so an untrained policy scores exactly 50%"
    r.expect(close(acc, want_acc), f"8 pairs (4 right, 2 ties): accuracy = {want_acc:.4f}", hint)
    hint = "binomial SE = sqrt(acc·(1 − acc)/n)"
    if close(se, math.sqrt(want_acc * (1 - want_acc) / 7)):
        hint = "divide by n, the number of pairs (not n − 1)"
    elif close(se, want_acc * (1 - want_acc) / 8):
        hint = "take the square root"
    r.expect(close(se, want_se), f"standard error = {want_se:.4f}", hint)
    return r.done()


def ref_kl_estimate(logp_pi: torch.Tensor, logp_ref: torch.Tensor, mask: torch.Tensor):
    """Monte Carlo KL(π‖π_ref) per sampled answer: mean and SE of Σ_t (log π − log π_ref) over response tokens."""
    k1 = ((logp_pi - logp_ref) * mask).sum(1)
    return float(k1.mean()), float(k1.std() / math.sqrt(k1.shape[0])), k1


def check_kl_estimate(fn, strict: bool = False) -> bool:
    if not_attempted(fn):
        return False
    r = Report("kl estimate", strict)
    lp = torch.tensor([[-0.5, -1.0, -2.0, -9.0], [-0.1, -0.2, -0.3, -0.4], [-1.0, -0.5, -9.0, -9.0]])
    lr = torch.tensor([[-0.7, -1.1, -1.5, -1.0], [-0.3, -0.2, -0.6, -0.1], [-1.2, -0.9, -1.0, -1.0]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.float32)
    ok, out = call(r, fn, lp, lr, mask)
    if not ok:
        return r.done()
    try:
        mean, se = float(out[0]), float(out[1])
    except Exception:
        r.expect(False, "returns (mean, se, per-answer values)", "return k1.mean(), k1.std() / sqrt(n), k1")
        return r.done()
    want_mean, want_se, k1 = ref_kl_estimate(lp, lr, mask)
    hint = "k1 per answer = ((logp_pi − logp_ref) * mask).sum(1); then its mean"
    if close(mean, float((lp - lr).sum(1).mean())):
        hint = "multiply by the response mask: padding after <|im_end|> is not part of the answer"
    elif close(mean, float(((lp - lr) * mask).sum() / mask.sum())):
        hint = "sequence KL sums over an answer's tokens; then average over answers"
    elif close(mean, -want_mean):
        hint = "sign: log π − log π_ref (samples come from π)"
    r.expect(close(mean, want_mean), f"three answers: KL estimate = {want_mean:.4f}", hint)
    r.expect(close(se, want_se), f"standard error = std(k1)/√n = {want_se:.4f}",
             "SE of a mean: k1.std() / sqrt(number of answers)")
    return r.done()
