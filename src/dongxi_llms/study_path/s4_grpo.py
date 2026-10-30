"""Checks and reference implementations for study-path stage S4 (GRPO).

Notebook: ``notebooks/study-path/s4_grpo.ipynb`` (Day 22 · Chapter 13).

Each ``check_*`` function receives the learner's function and prints one line per test: ⏳ when
the exercise is not attempted yet, ✅ when it passes and ❌ with a hint when it fails. Learner
checks never raise, so an unfinished notebook still runs top to bottom. ``strict=True`` turns
failures into ``AssertionError``; the reference cells use it.

The ``ref_*`` functions are the reference solutions. Conventions shared by all of them:

- rewards are ``R`` ``[P, G]``: P prompts, G sampled answers per prompt (one group per row)
- token tensors are ``[N, T]`` over **response positions**; ``mask`` is 1 for tokens the policy
  wrote (through its first ``<|im_end|>``) and 0 for padding; N = P·G, rows grouped by prompt
- ``logp``, ``old_logp``, ``ref_logp`` are log-probabilities of the sampled tokens under the
  current policy, the behaviour policy that sampled them, and the frozen reference
- advantages ``A`` are one number per answer, ``[N]``, and carry no gradient
"""
from __future__ import annotations

import math

import torch

from .common import Report, call, close, not_attempted

# D35 worked example: three-token vocabulary
D35_PI = [0.7, 0.2, 0.1]
D35_REF = [0.5, 0.3, 0.2]
D35_KL = 0.7 * math.log(0.7 / 0.5) + 0.2 * math.log(0.2 / 0.3) + 0.1 * math.log(0.1 / 0.2)


def _f64(x):
    return torch.as_tensor(x, dtype=torch.float64)


# =========================================================================== Part A · group advantages
def ref_group_advantages(R: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-group z-score with the population std; exactly 0 for a constant group. [P, G] → [P, G]."""
    R = R.detach()
    centered = R - R.mean(-1, keepdim=True)
    std = R.std(-1, correction=0, keepdim=True)
    return torch.where(std > 0, centered / (std + eps), torch.zeros_like(centered))


def check_group_advantages(fn, strict: bool = False) -> bool:
    if not_attempted(fn):
        return False
    r = Report("group_advantages", strict)
    ok, a = call(r, fn, _f64([[0.0, 0.0, 1.0, 1.0]]))
    if not ok:
        return r.done()
    if close(a, [[-0.8660254, -0.8660254, 0.8660254, 0.8660254]], atol=1e-5):
        r.expect(False, "rewards [0,0,1,1] → advantages [−1,−1,+1,+1]",
                 "you used the sample std (divides by G−1): use R.std(-1, correction=0, keepdim=True)")
    else:
        r.expect(close(a, [[-1.0, -1.0, 1.0, 1.0]], atol=1e-6), "rewards [0,0,1,1] → advantages [−1,−1,+1,+1]",
                 f"got {a}: subtract the group mean, divide by the group's population std")
    ok, a = call(r, fn, _f64([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]))
    if ok:
        finite = isinstance(a, torch.Tensor) and bool(torch.isfinite(a).all())
        r.expect(finite and close(a, torch.zeros(2, 4)), "all-success and all-failure groups give exactly 0 (no NaN)",
                 "a constant group has std 0: keep the `torch.where(std > 0, …, 0)` line")
    ok, a = call(r, fn, _f64([[0.0, 1.0], [10.0, 12.0]]))
    if ok:
        r.expect(close(a, [[-1.0, 1.0], [-1.0, 1.0]], atol=1e-6), "statistics are per group (rows [0,1] and [10,12])",
                 "reduce over the last dimension only, with keepdim=True, so each row uses its own mean and std")
    g = torch.Generator().manual_seed(0)
    R = torch.randint(0, 2, (64, 8), generator=g).double()
    ok, a = call(r, fn, R)
    if ok and isinstance(a, torch.Tensor):
        r.expect(close(a.sum(-1), torch.zeros(64), atol=1e-6), "advantages sum to zero in every group (64 random 0/1 groups)",
                 "subtract the mean of the group, not of the whole batch")
        ok2, a2 = call(r, fn, 3.0 * R + 5.0)
        if ok2:
            r.expect(close(a2, a, atol=1e-5), "unchanged when every reward becomes 3R + 5",
                     "centre AND divide by the std: then shifting and scaling R cancel")
    return r.done()


def ref_zero_signal_prob(p, G):
    """Probability that G independent 0/1 rewards with success rate p are all equal."""
    return p ** G + (1 - p) ** G


def check_zero_signal_prob(fn, strict: bool = False) -> bool:
    if not_attempted(fn):
        return False
    r = Report("zero_signal_prob", strict)
    cases = [(0.1, 4, 0.6562), (0.9, 8, 0.43046729 + 1e-8), (0.5, 2, 0.5), (0.0, 8, 1.0)]
    for p, G, want in cases:
        ok, got = call(r, fn, p, G)
        if not ok:
            return r.done()
        hint = "a group is also constant when all G answers succeed: add p**G" \
            if close(got, (1 - p) ** G, atol=1e-6) and p > 0 else "P(all fail) + P(all succeed)"
        r.expect(close(got, want, atol=1e-6), f"p = {p}, G = {G} → {want:.6g}", hint)
    return r.done()


# =========================================================================== Part B · loss normalisation
def ref_seq_mean(u: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Original GRPO: average each answer's token terms over its own length, then average answers."""
    return ((u * m).sum(1) / m.sum(1)).mean()


def ref_token_mean(u: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """DAPO: one average over every real token in the batch."""
    return (u * m).sum() / m.sum()


def ref_constant_norm(u: torch.Tensor, m: torch.Tensor, L_max: int) -> torch.Tensor:
    """Dr. GRPO: sum of real token terms divided by a constant, N answers × the generation budget."""
    return (u * m).sum() / (u.shape[0] * L_max)


def _agg_fixture():
    u = _f64([[2.0, 9.0, 9.0, 9.0], [1.0, 2.0, 3.0, 9.0]])      # 9s sit on padding
    m = _f64([[1, 0, 0, 0], [1, 1, 1, 0]])
    return u, m


def check_aggregations(seq_fn, tok_fn, const_fn, strict: bool = False) -> bool:
    if not_attempted(seq_fn) or not_attempted(tok_fn) or not_attempted(const_fn):
        return False
    r = Report("aggregations", strict)
    u, m = _agg_fixture()
    # seq: (2 + 2)/2 = 2 ; token: (2+1+2+3)/4 = 2 ; constant (L_max 4): 8/8 = 1
    ok, v = call(r, seq_fn, u, m)
    if ok:
        r.expect(close(v, 2.0), "per-sequence mean: ((2)/1 + (1+2+3)/3) / 2 = 2.0",
                 f"got {v}: divide each row by its own count m.sum(1) (not the padded width), then average rows")
    ok, v = call(r, tok_fn, u, m)
    if ok:
        r.expect(close(v, 2.0), "token mean: (2 + 1 + 2 + 3) / 4 real tokens = 2.0",
                 f"got {v}: one sum over all real tokens, divided by m.sum()")
    ok, v = call(r, const_fn, u, m, 4)
    if ok:
        r.expect(close(v, 1.0), "constant normaliser: 8 / (2 answers × L_max 4) = 1.0",
                 f"got {v}: divide the masked sum by u.shape[0] * L_max, a constant that doesn't depend on lengths")
    # gradient weights on a second fixture separate the three
    w = {}
    for name, fn, extra in (("seq", seq_fn, ()), ("tok", tok_fn, ()), ("const", const_fn, (4,))):
        uu = torch.ones(2, 4, dtype=torch.float64, requires_grad=True)
        ok, v = call(r, fn, uu, m, *extra)
        if not ok or not isinstance(v, torch.Tensor) or not v.requires_grad:
            r.expect(False, f"{name}: result is a differentiable scalar", "return a torch scalar built from u")
            return r.done()
        v.backward()
        w[name] = uu.grad
    r.expect(close(w["seq"], [[1 / 2, 0, 0, 0], [1 / 6, 1 / 6, 1 / 6, 0]]),
             "per-sequence mean: per-token weight 1/(N·n_i) = 1/2 (short) vs 1/6 (long), 0 on padding",
             "multiply by the mask before summing")
    r.expect(close(w["tok"], [[1 / 4, 0, 0, 0], [1 / 4, 1 / 4, 1 / 4, 0]]),
             "token mean: every real token weighs 1/4", "multiply by the mask before summing")
    r.expect(close(w["const"], [[1 / 8, 0, 0, 0], [1 / 8, 1 / 8, 1 / 8, 0]]),
             "constant: every real token weighs 1/(N·L_max) = 1/8", "multiply by the mask before summing")
    return r.done()


def ref_centered_advantages(R: torch.Tensor) -> torch.Tensor:
    """Dr. GRPO advantage: reward minus the group mean, no division by the std."""
    R = R.detach()
    return R - R.mean(-1, keepdim=True)


def check_centered_advantages(fn, strict: bool = False) -> bool:
    if not_attempted(fn):
        return False
    r = Report("centered_advantages", strict)
    ok, a = call(r, fn, _f64([[0.0, 0.0, 0.0, 1.0], [2.0, 2.0, 2.0, 2.0]]))
    if not ok:
        return r.done()
    zs = ref_group_advantages(_f64([[0.0, 0.0, 0.0, 1.0], [2.0, 2.0, 2.0, 2.0]]))
    hint = "Dr. GRPO keeps the mean subtraction and drops the division by the std" if close(a, zs, atol=1e-5) \
        else "subtract each row's mean: R - R.mean(-1, keepdim=True)"
    r.expect(close(a, [[-0.25, -0.25, -0.25, 0.75], [0, 0, 0, 0]]),
             "[0,0,0,1] → [−0.25,−0.25,−0.25,+0.75]; a constant group → 0", hint)
    ok, a = call(r, fn, _f64([[0.0, 1.0], [10.0, 12.0]]))
    if ok:
        r.expect(close(a, [[-0.5, 0.5], [-1.0, 1.0]]), "per group, and the reward scale is kept ([10,12] → [−1,+1])",
                 "reduce over the last dimension with keepdim=True; don't rescale")
    return r.done()


# =========================================================================== Part C · KL estimators
def ref_k1(logp: torch.Tensor, ref_logp: torch.Tensor) -> torch.Tensor:
    """k1 = log π − log π_ref per sampled token (unbiased for KL(π‖π_ref), can be negative)."""
    return logp - ref_logp


def ref_k3(logp: torch.Tensor, ref_logp: torch.Tensor) -> torch.Tensor:
    """k3 = e^d − d − 1 with d = log π_ref − log π (unbiased for KL(π‖π_ref), always ≥ 0)."""
    d = ref_logp - logp
    return torch.exp(d) - d - 1


def check_k1_k3(k1_fn, k3_fn, strict: bool = False) -> bool:
    if not_attempted(k1_fn) or not_attempted(k3_fn):
        return False
    r = Report("k1_k3", strict)
    lp, lq = _f64(D35_PI).log(), _f64(D35_REF).log()
    ok, v1 = call(r, k1_fn, lp, lq)
    if ok:
        hint = "k1 = log π − log π_ref (policy first)" if close(v1, -(lp - lq)) else "k1 = logp − ref_logp"
        r.expect(close(v1, lp - lq), "k1 on the D35 tokens = [+0.336, −0.405, −0.693]", hint)
    ok, v3 = call(r, k3_fn, lp, lq)
    if ok:
        want = ref_k3(lp, lq)
        if close(v3, ref_k3(lq, lp)):
            hint = ("you used d = log π − log π_ref: k3 needs d = log π_ref − log π; the other sign does not "
                    "estimate KL(π‖π_ref) from tokens sampled from π")
        else:
            hint = "d = ref_logp − logp, then k3 = exp(d) − d − 1"
        r.expect(close(v3, want), "k3 on the D35 tokens = [0.0508, 0.0945, 0.3069]", hint)
        g = torch.Generator().manual_seed(1)
        a, b = torch.randn(1000, generator=g, dtype=torch.float64) * 3, torch.randn(1000, generator=g, dtype=torch.float64) * 3
        ok2, v = call(r, k3_fn, a, b)
        if ok2:
            r.expect(isinstance(v, torch.Tensor) and bool((v >= -1e-12).all()), "k3 ≥ 0 on 1,000 random (logp, ref_logp) pairs",
                     "e^d − d − 1 ≥ 0 for every real d: check the formula")
            ok3, v0 = call(r, k3_fn, a, a)
            if ok3:
                r.expect(close(v0, torch.zeros(1000)), "k3 = 0 when π and π_ref agree on the token", "e^0 − 0 − 1 = 0")
    return r.done()


def ref_kl_exact(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """KL(p‖q) = Σ_v p_v log(p_v / q_v) over the last dimension."""
    return (p * (p.log() - q.log())).sum(-1)


def check_kl_exact(fn, strict: bool = False) -> bool:
    if not_attempted(fn):
        return False
    r = Report("kl_exact", strict)
    p, q = _f64(D35_PI), _f64(D35_REF)
    ok, v = call(r, fn, p, q)
    if not ok:
        return r.done()
    hint = "you computed KL(π_ref‖π): weight the log-ratio by p, the first argument" \
        if close(v, ref_kl_exact(q, p)) else "Σ_v p_v (log p_v − log q_v), summed over the last dimension"
    r.expect(close(v, D35_KL), f"KL(π‖π_ref) on the D35 example = {D35_KL:.4f}", hint)
    ok, v = call(r, fn, p, p)
    if ok:
        r.expect(close(v, 0.0), "KL(p‖p) = 0", "log p − log p = 0 at every token")
    P = torch.tensor([[0.98, 0.01, 0.01], [0.25, 0.25, 0.5]], dtype=torch.float64)
    Qm = torch.tensor([[0.2, 0.4, 0.4], [0.5, 0.25, 0.25]], dtype=torch.float64)
    ok, v = call(r, fn, P, Qm)
    if ok:
        r.expect(close(v, ref_kl_exact(P, Qm)), "works row by row on a [2, V] batch",
                 "sum over the last dimension only: .sum(-1)")
    return r.done()


def ref_k3_sampled_surrogate(z: torch.Tensor, ref_logp: torch.Tensor, k3=ref_k3) -> torch.Tensor:
    """Σ_v p_v k3(v) with p detached: the expected k3 loss term when the sampled token is held fixed."""
    logp = z.log_softmax(-1)
    return (logp.exp().detach() * k3(logp, ref_logp)).sum()


def check_k3_sampled_surrogate(fn, strict: bool = False) -> bool:
    if not_attempted(fn):
        return False
    r = Report("k3_sampled_surrogate", strict)
    z = _f64(D35_PI).log().requires_grad_(True)
    q = _f64(D35_REF)
    ok, s = call(r, fn, z, q.log())
    if not ok:
        return r.done()
    if not isinstance(s, torch.Tensor) or not s.requires_grad or s.numel() != 1:
        r.expect(False, "returns a differentiable scalar", "sum p.detach() * k3 over the vocabulary")
        return r.done()
    r.expect(close(s, D35_KL), f"its value is the expected k3 = KL(π‖π_ref) = {D35_KL:.4f}",
             "weight each token's k3 by its probability p_v and sum")
    (g,) = torch.autograd.grad(s, z)
    p = z.detach().softmax(-1)
    fwd = p * (p.log() - q.log() - D35_KL)
    if close(g, fwd, atol=1e-6):
        r.expect(False, "its gradient w.r.t. the logits is p − q (the gradient of KL(π_ref‖π))",
                 "you let the gradient flow through the sampling weights p: in training the sampled token is "
                 "fixed, so use p.detach()")
    else:
        r.expect(close(g, p - q, atol=1e-6), "its gradient w.r.t. the logits is p − q (the gradient of KL(π_ref‖π))",
                 f"got {g}: weights p.detach(), and k3 evaluated with logp = z.log_softmax(-1)")
    return r.done()


# =========================================================================== Part D · the GRPO loss
def ref_grpo_loss(logp, old_logp, ref_logp, A, mask, eps: float = 0.2, beta: float = 0.04,
                  k3=ref_k3, agg=ref_seq_mean) -> torch.Tensor:
    """−clipped surrogate + β·k3 per token, sequence advantage broadcast, then aggregated (seq-mean)."""
    ratio = torch.exp(logp - old_logp.detach())
    adv = A.detach()[:, None]
    surr = torch.minimum(ratio * adv, ratio.clamp(1 - eps, 1 + eps) * adv)
    per_token = -surr + beta * k3(logp, ref_logp.detach())
    return agg(per_token, mask)


def _loss_fixture():
    mask = _f64([[1, 1, 0], [1, 1, 1]])
    ref = _f64([[-1.0, -2.0, -5.0], [-0.5, -1.5, -1.0]])
    logp = _f64([[-1.2, -1.8, 7.0], [-0.7, -1.4, -1.3]])          # 7.0 on padding
    A = _f64([1.0, -1.0])
    return logp, ref, A, mask


def check_grpo_loss(fn, strict: bool = False) -> bool:
    if not_attempted(fn):
        return False
    r = Report("grpo_loss", strict)
    logp0, ref, A, mask = _loss_fixture()
    # (1) single fresh update: old = current, ρ = 1
    logp = logp0.clone().requires_grad_(True)
    ok, loss = call(r, fn, logp, logp0.clone(), ref, A, mask, 0.2, 0.04)
    if not ok:
        return r.done()
    if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
        r.expect(False, "returns a differentiable scalar", "build the loss from logp (it carries the gradient)")
        return r.done()
    want = ref_grpo_loss(logp0, logp0, ref, A, mask, 0.2, 0.04)
    r.expect(close(loss, want), f"ρ = 1: loss = −mean_i A_i + β·mean k3 = {float(want):.5f}",
             "per token: −min(ρA, clip(ρ, 1−ε, 1+ε)A) + β·k3, then the given seq_mean; the KL enters with a + sign")
    (g,) = torch.autograd.grad(loss, logp)
    d = (ref - logp0)
    n = mask.sum(1, keepdim=True)
    g_want = mask * (-A[:, None] + 0.04 * (1 - d.exp())) / (2 * n)
    r.expect(close(g, g_want, atol=1e-6), "ρ = 1: ∂loss/∂logp = (−A_i + β(1 − π_ref/π)) / (N·n_i); 0 on padding",
             "the ratio must be exp(logp − old_logp) so it has a gradient even when it equals 1; mask padding")
    # (2) clipping: ratios 1.5 (A>0), 0.5 (A<0) → flat; ratios 0.5 (A>0), 1.5 (A<0) → not flat
    old = _f64([[0.0, 0.0], [0.0, 0.0]])
    lp = _f64([[math.log(1.5), math.log(0.5)], [math.log(0.5), math.log(1.5)]]).requires_grad_(True)
    m2 = torch.ones(2, 2, dtype=torch.float64)
    ok, l2 = call(r, fn, lp, old, lp.detach().clone(), _f64([1.0, -1.0]), m2, 0.2, 0.0)
    if ok and isinstance(l2, torch.Tensor) and l2.requires_grad:
        (g2,) = torch.autograd.grad(l2, lp)
        flat = bool(abs(g2[0, 0]) < 1e-12 and abs(g2[1, 0]) < 1e-12)
        alive = bool(abs(g2[0, 1]) > 1e-6 and abs(g2[1, 1]) > 1e-6)
        r.expect(flat, "no gradient once ρ has left [0.8, 1.2] in the helpful direction (A>0, ρ=1.5; A<0, ρ=0.5)",
                 "take torch.minimum of the unclipped term ρ·A and the clipped term clip(ρ)·A")
        r.expect(alive, "gradient kept on the harmful side (A>0, ρ=0.5; A<0, ρ=1.5)",
                 "the min keeps the unclipped term when it is the smaller one; don't clip both terms")
    return r.done()


def ref_reinforce_loss(logp, R, mask, agg=ref_seq_mean) -> torch.Tensor:
    """REINFORCE with no baseline: per-token term −R_i·log π, raw reward as the weight."""
    return agg(-R.detach()[:, None] * logp, mask)


def check_reinforce_loss(fn, strict: bool = False) -> bool:
    if not_attempted(fn):
        return False
    r = Report("reinforce_loss", strict)
    logp0, _, _, mask = _loss_fixture()
    R = _f64([1.0, 0.0])
    logp = logp0.clone().requires_grad_(True)
    ok, loss = call(r, fn, logp, R, mask)
    if not ok:
        return r.done()
    if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
        r.expect(False, "returns a differentiable scalar", "build it from logp")
        return r.done()
    r.expect(close(loss, ref_reinforce_loss(logp0, R, mask)), "value = seq_mean(−R·logp) = 1.5",
             "per-token term −R[:, None] * logp, aggregated with the given seq_mean (padding masked)")
    (g,) = torch.autograd.grad(loss, logp)
    r.expect(close(g, _f64([[-0.25, -0.25, 0], [0, 0, 0]])),
             "∂loss/∂logp = −R_i/(N·n_i): the R = 0 answer gets no gradient at all",
             "the weight is the raw reward R, with no baseline subtracted")
    return r.done()


# =========================================================================== Part F · real Qwen3 run
def ref_length_reward(text: str, truncated: bool) -> float:
    """1.0 if the answer stopped by itself (emitted <|im_end|>) and has 5–20 words, else 0.0."""
    n = len(text.split())
    return float((not truncated) and 5 <= n <= 20)


def check_length_reward(fn, strict: bool = False) -> bool:
    if not_attempted(fn):
        return False
    r = Report("length_reward", strict)
    cases = [("Garlic is toxic to dogs, so don't feed it to them.", False, 1.0, "11 words, stopped → 1"),
             ("Garlic is toxic to dogs, so don't feed it to them.", True, 0.0,
              "same text but truncated at the token budget → 0"),
             ("No.", False, 0.0, "1 word → 0 (below 5)"),
             ("one two three four five", False, 1.0, "exactly 5 words → 1 (bounds included)"),
             (" ".join(["w"] * 20), False, 1.0, "exactly 20 words → 1"),
             (" ".join(["w"] * 21), False, 0.0, "21 words → 0"),
             ("Line one\nline two\n\n- item three four", False, 1.0, "newlines and bullets: words = text.split()")]
    for text, trunc, want, what in cases:
        ok, got = call(r, fn, text, trunc)
        if not ok:
            return r.done()
        r.expect(isinstance(got, (int, float)) and float(got) == want, what,
                 f"got {got!r}: n = len(text.split()); return float(not truncated and 5 <= n <= 20)")
    return r.done()


def ref_grpo_terms(R, logp, ref_logp, mask, G, group_advantages=ref_group_advantages, grpo_loss=ref_grpo_loss,
                   eps: float = 0.2, beta: float = 0.04):
    """One fresh update: A from the group z-score, loss with old_logp = logp.detach() (so ρ = 1)."""
    A = group_advantages(R.view(-1, G)).flatten()
    loss = grpo_loss(logp, logp.detach(), ref_logp, A, mask, eps, beta)
    return A, loss


def check_grpo_terms(fn, strict: bool = False) -> bool:
    if not_attempted(fn):
        return False
    r = Report("grpo_terms", strict)
    G = 2
    R = _f64([0.0, 1.0, 1.0, 1.0, 1.0, 0.0])      # rows grouped by prompt: [q0, q0, q1, q1, q2, q2]
    mask = _f64([[1, 1], [1, 0], [1, 1], [1, 1], [1, 1], [1, 0]])
    lp0 = _f64([[-1.0, -2.0], [-0.5, 3.0], [-1.0, -1.0], [-2.0, -0.3], [-0.2, -0.9], [-1.5, 2.0]])
    ref = _f64([[-1.1, -2.0], [-0.4, -9.0], [-1.0, -1.2], [-2.0, -0.3], [-0.3, -0.8], [-1.4, -7.0]])
    logp = lp0.clone().requires_grad_(True)
    ok, out = call(r, fn, R, logp, ref, mask, G)
    if not ok:
        return r.done()
    if not (isinstance(out, tuple) and len(out) == 2):
        r.expect(False, "returns (A, loss)", "return A, loss")
        return r.done()
    A, loss = out
    hint = "rows are grouped by prompt, G consecutive answers each: use R.view(-1, G), then .flatten()" \
        if close(A, ref_group_advantages(R.view(G, -1)).flatten()) else "A = group_advantages(R.view(-1, G)).flatten()"
    r.expect(close(A, [-1.0, 1.0, 0.0, 0.0, 1.0, -1.0], atol=1e-6),
             "A = [−1, +1, 0, 0, +1, −1] (prompts 0 and 2 mixed, prompt 1 all-success)", hint)
    if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
        r.expect(False, "loss is differentiable", "pass logp itself as the current log-probs")
        return r.done()
    want = ref_grpo_terms(R, lp0.clone().requires_grad_(True), ref, mask, G)[1]
    r.expect(close(loss, want), f"loss value at ρ = 1 = {float(want.detach()):.5f}",
             "call grpo_loss(logp, old_logp, ref_logp, A, mask, EPS, BETA)")
    (g,) = torch.autograd.grad(loss, logp)
    lpw = lp0.clone().requires_grad_(True)
    (gw,) = torch.autograd.grad(ref_grpo_terms(R, lpw, ref, mask, G)[1], lpw)
    if close(g, torch.zeros_like(g)) or (not close(g, gw, atol=1e-6) and close(g[2:4], gw[2:4], atol=1e-6)):
        hint = "old_logp must be logp.detach(): if it carries a gradient, ρ = exp(logp − logp) is a constant and " \
               "the policy term has no gradient"
    else:
        hint = "old_logp = logp.detach() (fresh batch, one update)"
    r.expect(close(g, gw, atol=1e-6), "gradient: −A_i/(N·n_i) from the surrogate plus the k3 term", hint)
    return r.done()
