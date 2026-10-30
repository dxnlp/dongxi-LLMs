"""Checks and reference implementations for study-path stage S6 (feedback sources × optimizers).

The notebook ``notebooks/study-path/s6_combinations.ipynb`` combines the pieces of S1–S5. The
learner writes the key lines of: best-of-n selection with a reward model, the exact best-of-n
distribution, the supervised loss of rejection-sampling fine-tuning, reward-model pair labelling
(deterministic with a margin, and Bradley–Terry sampled), the KL-regularized optimum, group
z-normalisation of reward-model scores, two ways to mix a reward model with a verifier (sum and
gate), a KL-anchored RLVR loss, best-vs-worst pair construction for online DPO, and a paired
standard error.

Each ``check_*`` function receives the learner's object and prints one line per test: ⏳ when the
exercise is not attempted yet, ✅ when it passes and ❌ with a hint when it fails. Checks never raise;
``strict=True`` turns failures into ``AssertionError`` (the reference cells use it). Everything here
runs on CPU in a fraction of a second, without model weights.
"""
from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn.functional as F

from dongxi_llms.study_path.common import Report, call, close, not_attempted

D = torch.float64


# --------------------------------------------------------------------------- Part A · best-of-n
def ref_best_of_n(samples: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    """samples [B, n] answer indices, scores [K] → [B] the answer with the highest score in each row."""
    return samples.gather(1, scores[samples].argmax(1, keepdim=True)).squeeze(1)


def check_best_of_n(fn: Callable, strict: bool = False) -> bool:
    if not_attempted(fn):
        return False
    r = Report("best_of_n", strict)
    scores = torch.tensor([0.5, -1.0, 2.0, 0.0])
    samples = torch.tensor([[0, 1, 3], [3, 2, 1], [1, 1, 1], [2, 0, 2]])
    ok, out = call(r, fn, samples, scores)
    if ok:
        expected = torch.tensor([0, 2, 1, 2])
        good_shape = r.expect(isinstance(out, torch.Tensor) and tuple(out.shape) == (4,),
                              "returns one answer index per row, shape [B]",
                              "reduce over dim 1 and squeeze, e.g. .squeeze(1)")
        if good_shape:
            positions = torch.tensor([0, 1, 0, 0])
            r.expect(torch.equal(out.long(), expected), "picks the highest-scored answer of each row",
                     "you returned the position inside the row; map it back to the answer index with "
                     "samples.gather(1, pos)" if torch.equal(out.long(), positions) else
                     "score each sample with scores[samples], take argmax over dim 1, then gather the answer")
    return r.done()


def ref_bon_probs(p: torch.Tensor, scores: torch.Tensor, n: int) -> torch.Tensor:
    """Exact distribution of the best of n i.i.d. draws from p, ranked by scores (distinct scores)."""
    order = scores.argsort()
    p_sorted = p[order]
    cdf = p_sorted.cumsum(0)
    p_best_sorted = cdf ** n - (cdf - p_sorted) ** n
    out = torch.empty_like(p)
    out[order] = p_best_sorted
    return out


def check_bon_probs(fn: Callable, strict: bool = False) -> bool:
    if not_attempted(fn):
        return False
    r = Report("bon_probs", strict)
    p = torch.tensor([0.5, 0.3, 0.2], dtype=D)
    scores = torch.tensor([0.0, 2.0, 1.0], dtype=D)
    ok1, one = call(r, fn, p, scores, 1)
    ok2, two = call(r, fn, p, scores, 2)
    if ok1 and ok2:
        r.expect(close(one, p), "n = 1 gives back p (one draw, nothing to select)",
                 "at n = 1 your formula must reduce to p_sorted itself")
        # sorted by score: answer 0 (0.5), answer 2 (0.2), answer 1 (0.3); cdf 0.5, 0.7, 1.0
        expected = torch.tensor([0.25, 1 - 0.49, 0.49 - 0.25], dtype=D)
        if close(two, (torch.tensor([0.5, 0.7, 1.0], dtype=D) ** 2)[[0, 2, 1]]):
            hint = "cdf**n is P(best ≤ k-th answer); subtract P(best ≤ (k−1)-th) = (cdf − p_sorted)**n"
        else:
            hint = "P(best = k-th) = P(all n draws ≤ k-th) − P(all n draws < k-th) = cdf**n − (cdf − p_sorted)**n"
        r.expect(close(two, expected), "n = 2 matches the exact values [0.25, 0.51, 0.24]", hint)
        ok8, eight = call(r, fn, p, scores, 8)
        if ok8:
            r.expect(close(float(eight.sum()), 1.0), "probabilities sum to 1 at n = 8")
            r.expect(float(eight[1]) > 0.94, "at n = 8 the top-scored answer gets 1 − 0.7⁸ ≈ 0.94",
                     "the best answer's probability is 1 − P(no draw is the best answer)")
    return r.done()


def ref_sft_loss(logits: torch.Tensor, picked: torch.Tensor) -> torch.Tensor:
    """Mean negative log-likelihood of the selected answers under a tabular policy (logits [K])."""
    return -torch.log_softmax(logits, -1)[picked].mean()


def check_sft_loss(fn: Callable, strict: bool = False) -> bool:
    if not_attempted(fn):
        return False
    r = Report("sft_loss", strict)
    logits = torch.tensor([0.0, 1.0, -1.0, 0.5], dtype=D, requires_grad=True)
    picked = torch.tensor([1, 1, 3, 0, 1])
    ok, loss = call(r, fn, logits, picked)
    if ok:
        expected = ref_sft_loss(logits.detach(), picked)
        good = r.expect(isinstance(loss, torch.Tensor) and loss.dim() == 0, "returns a scalar tensor",
                        "average over the selected samples with .mean()")
        if good:
            sum_version = expected * len(picked)
            r.expect(close(loss.detach(), expected), f"value matches −mean log π(picked) = {expected:.4f}",
                     "you summed instead of averaging" if close(loss.detach(), sum_version)
                     else "use log_softmax(logits), index it with picked, negate and average")
            if loss.requires_grad:
                (g,) = torch.autograd.grad(loss, logits)
                freq = torch.bincount(picked, minlength=4).to(D) / len(picked)
                r.expect(close(g, torch.softmax(logits.detach(), -1) - freq),
                         "gradient = π − empirical frequency of the picked answers (so the optimum is that frequency)")
            else:
                r.expect(False, "the loss is differentiable", "don't detach the logits")
    return r.done()


# --------------------------------------------------------------------------- Part B · online vs offline DPO
def ref_label_hard(s_a: torch.Tensor, s_b: torch.Tensor, margin: float):
    """Deterministic RM labelling: a wins iff s_a > s_b; keep only pairs whose gap is at least margin."""
    a_wins = s_a > s_b
    keep = (s_a - s_b).abs() >= margin
    return a_wins, keep


def ref_label_bt(s_a: torch.Tensor, s_b: torch.Tensor, generator=None):
    """Bradley–Terry sampled labelling: a wins with probability σ(s_a − s_b); every pair is kept."""
    a_wins = torch.rand(s_a.shape, generator=generator, dtype=s_a.dtype) < torch.sigmoid(s_a - s_b)
    keep = torch.ones_like(a_wins)
    return a_wins, keep


def check_label_hard(fn: Callable, strict: bool = False) -> bool:
    if not_attempted(fn):
        return False
    r = Report("label_hard", strict)
    s_a = torch.tensor([1.0, 0.0, 2.0, -1.0, 0.3])
    s_b = torch.tensor([0.0, 1.0, 1.8, -1.0, 0.0])
    ok, out = call(r, fn, s_a, s_b, 0.25)
    if ok:
        try:
            a_wins, keep = out
            a_wins, keep = torch.as_tensor(a_wins).bool(), torch.as_tensor(keep).bool()
        except Exception:
            r.expect(False, "returns (a_wins, keep)", "return two boolean tensors")
            return r.done()
        r.expect(torch.equal(a_wins, torch.tensor([True, False, True, False, True])),
                 "a_wins is True exactly where s_a > s_b", "compare the two scores with >")
        expected_keep = torch.tensor([True, True, False, False, True])
        one_sided = torch.tensor([True, False, False, False, True])
        r.expect(torch.equal(keep, expected_keep), "keep drops pairs whose |gap| < margin, in both directions",
                 "use the absolute gap: (s_a - s_b).abs() >= margin" if torch.equal(keep, one_sided)
                 else "keep = |s_a − s_b| ≥ margin")
    return r.done()


def check_label_bt(fn: Callable, strict: bool = False) -> bool:
    if not_attempted(fn):
        return False
    r = Report("label_bt", strict)
    n = 40000
    s_a = torch.cat([torch.full((n,), 1.0), torch.full((n,), -0.5)]).to(D)
    s_b = torch.zeros(2 * n, dtype=D)
    ok, out = call(r, fn, s_a, s_b, torch.Generator().manual_seed(0))
    if ok:
        try:
            a_wins, keep = out
            a_wins, keep = torch.as_tensor(a_wins).bool(), torch.as_tensor(keep).bool()
        except Exception:
            r.expect(False, "returns (a_wins, keep)", "return two boolean tensors")
            return r.done()
        f1, f2 = a_wins[:n].double().mean().item(), a_wins[n:].double().mean().item()
        hint = ("these are deterministic labels; draw them: torch.rand(...) < torch.sigmoid(s_a - s_b)"
                if f1 in (0.0, 1.0) else "a wins with probability σ(s_a − s_b): compare a uniform draw with it")
        r.expect(abs(f1 - 1 / (1 + math.exp(-1.0))) < 0.01, f"gap +1.0: a wins {f1:.3f} of the time (σ(1) = 0.731)", hint)
        r.expect(abs(f2 - 1 / (1 + math.exp(0.5))) < 0.01, f"gap −0.5: a wins {f2:.3f} of the time (σ(−0.5) = 0.378)", hint)
        r.expect(bool(keep.all()), "every pair is kept (no margin filter for sampled labels)",
                 "keep = torch.ones_like(a_wins)")
        ok2, out2 = call(r, fn, s_a[n - 200:n + 200], s_b[:400], torch.Generator().manual_seed(1))
        ok3, out3 = call(r, fn, s_a[n - 200:n + 200], s_b[:400], torch.Generator().manual_seed(1))
        if ok2 and ok3:
            r.expect(torch.equal(torch.as_tensor(out2[0]), torch.as_tensor(out3[0])),
                     "uses the generator it is given (same seed → same labels)",
                     "pass generator=generator to torch.rand")
    return r.done()


def ref_kl_optimum(p_ref: torch.Tensor, r: torch.Tensor, beta: float) -> torch.Tensor:
    """π*(y) ∝ π_ref(y) · exp(r(y) / β), over the last dimension."""
    return torch.softmax(p_ref.log() + r / beta, -1)


def check_kl_optimum(fn: Callable, strict: bool = False) -> bool:
    if not_attempted(fn):
        return False
    rep = Report("kl_optimum", strict)
    p_ref = torch.tensor([0.5, 0.25, 0.25], dtype=D)
    r = torch.tensor([0.0, 1.0, -1.0], dtype=D)
    ok, out = call(rep, fn, p_ref, r, 0.5)
    if ok:
        expected = ref_kl_optimum(p_ref, r, 0.5)
        if close(out, ref_kl_optimum(p_ref, r, 2.0)):
            hint = "divide the reward by β (you multiplied): small β must mean a large move"
        elif close(out, torch.softmax(r / 0.5, -1)):
            hint = "you dropped the reference: start from log π_ref, then add r / β"
        else:
            hint = "softmax(log π_ref + r / β)"
        rep.expect(close(out, expected), "matches softmax(log π_ref + r/β) on a 3-answer case", hint)
        ok2, big = call(rep, fn, p_ref, r, 1e6)
        if ok2:
            rep.expect(close(big, p_ref, atol=1e-5), "β → ∞ gives back π_ref (the penalty dominates)")
        ok3, batched = call(rep, fn, torch.stack([p_ref, p_ref.flip(0)]), torch.stack([r, r]), 1.0)
        if ok3:
            rep.expect(close(batched.sum(-1), [1.0, 1.0]), "works row-wise on [P, K] (one distribution per prompt)",
                       "normalise over the last dimension: softmax(..., -1)")
    return rep.done()


# --------------------------------------------------------------------------- Part C · RM + verifier
def ref_group_z(s: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """z-score of each row [B, G]: (s − row mean) / (row std + eps)."""
    return (s - s.mean(-1, keepdim=True)) / (s.std(-1, keepdim=True) + eps)


def check_group_z(fn: Callable, strict: bool = False) -> bool:
    if not_attempted(fn):
        return False
    r = Report("group_z", strict)
    s = torch.tensor([[1.0, 2.0, 3.0, 6.0], [10.0, 10.0, 30.0, 30.0], [5.0, 5.0, 5.0, 5.0]], dtype=D)
    ok, z = call(r, fn, s)
    if ok:
        good = r.expect(isinstance(z, torch.Tensor) and z.shape == s.shape, "same shape as the scores [B, G]")
        if good:
            r.expect(bool(torch.isfinite(z).all()) and close(z[2], torch.zeros(4)),
                     "a constant group gives zeros, not NaN", "add a small eps to the std")
            r.expect(close(z[:2].mean(-1), [0.0, 0.0]), "each group has mean 0",
                     "normalise per row: mean(-1, keepdim=True), not over the whole tensor")
            std_ok = any(close(z[:2].std(-1, correction=c), [1.0, 1.0], atol=1e-4) for c in (0, 1))
            r.expect(std_ok, "each group has standard deviation 1",
                     "divide by the row std: s.std(-1, keepdim=True) + eps")
            ok2, z2 = call(r, fn, 3.0 * s + 7.0)
            if ok2:
                r.expect(close(z2[:2], z[:2], atol=1e-4), "invariant to the RM's offset and scale (3·s + 7 gives the same z)")
    return r.done()


def ref_combine_sum(v: torch.Tensor, z: torch.Tensor, w: float) -> torch.Tensor:
    """Weighted sum: verifier reward plus w times the normalised RM score."""
    return v + w * z


def ref_combine_gate(v: torch.Tensor, z: torch.Tensor, w: float) -> torch.Tensor:
    """Verifier-gated RM: 0 when the verifier fails, 1 + w·σ(z) ∈ (1, 1 + w) when it passes."""
    return v * (1 + w * torch.sigmoid(z))


def check_combine(fn_sum: Callable, fn_gate: Callable, strict: bool = False) -> bool:
    if not_attempted(fn_sum) or not_attempted(fn_gate):
        return False
    r = Report("combine", strict)
    v = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=D)
    z = torch.tensor([-1.0, 0.0, 2.0, -0.5], dtype=D)
    ok1, rs = call(r, fn_sum, v, z, 1.0)
    ok2, rg = call(r, fn_gate, v, z, 1.0)
    if ok1:
        ok4, rs_half = call(r, fn_sum, v, z, 0.5)
        r.expect(close(rs, [0.0, 1.0, 2.0, -0.5]) and ok4 and close(rs_half, [0.5, 1.0, 1.0, -0.25]),
                 "sum: v + w·z on a fixed case (w = 1 and w = 0.5)", "R_sum = v + w * z")
        r.expect(float(rs[2]) > float(rs[0]),
                 "sum: the wrong answer the RM likes (z = +2) outscores a correct one (z = −1)")
    if ok2:
        expected = ref_combine_gate(v, z, 1.0)
        if close(rg, v + torch.sigmoid(z)):
            hint = "that adds the RM term to wrong answers too; multiply by v so it only counts when v = 1"
        elif close(rg, v * torch.sigmoid(z)):
            hint = "keep the verifier's 1: v * (1 + w * sigmoid(z))"
        else:
            hint = "R_gate = v * (1 + w * torch.sigmoid(z))"
        r.expect(close(rg, expected), "gate: v·(1 + w·σ(z)) on a fixed case", hint)
        g = torch.Generator().manual_seed(0)
        zz = 4 * torch.randn(2000, dtype=D, generator=g)
        vv = (torch.rand(2000, generator=g) < 0.5).to(D)
        ok3, big = call(r, fn_gate, vv, zz, 3.0)
        if ok3:
            worst_correct = big[vv == 1].min().item()
            best_wrong = big[vv == 0].max().item()
            r.expect(worst_correct > best_wrong,
                     f"gate: every correct answer outscores every wrong one (min correct {worst_correct:.2f} > "
                     f"max wrong {best_wrong:.2f}) for any RM score")
    return r.done()


# --------------------------------------------------------------------------- Part D · pipeline
def ref_rlvr_loss(logits: torch.Tensor, ref_logp: torch.Tensor, y: torch.Tensor, R: torch.Tensor,
                  beta: float) -> torch.Tensor:
    """Tabular RLVR loss for one prompt: group-mean advantage policy gradient + β·KL(π ‖ π_ref).

    logits [K], ref_logp [K] (log-probs of the anchor), y [B, G] sampled answers, R [B, G] rewards.
    """
    logp = torch.log_softmax(logits, -1)
    A = R - R.mean(1, keepdim=True)
    kl = (logp.exp() * (logp - ref_logp)).sum()
    return -(A * logp[y]).mean() + beta * kl


def check_rlvr_loss(fn: Callable, strict: bool = False) -> bool:
    if not_attempted(fn):
        return False
    r = Report("rlvr_loss", strict)
    logits = torch.tensor([0.2, -0.3, 0.5, 0.0], dtype=D)
    ref_logp = torch.log_softmax(torch.tensor([0.0, 0.0, 1.0, -1.0], dtype=D), -1)
    y = torch.tensor([[0, 2, 2, 3], [1, 1, 0, 2]])
    R = torch.tensor([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 1.0]], dtype=D)
    lg = logits.clone().requires_grad_(True)
    ok, loss = call(r, fn, lg, ref_logp, y, R, 0.3)
    if ok:
        expected = ref_rlvr_loss(logits, ref_logp, y, R, 0.3)
        logp = torch.log_softmax(logits, -1)
        no_base = -(R * logp[y]).mean() + 0.3 * (logp.exp() * (logp - ref_logp)).sum()
        rev_kl = -((R - R.mean(1, keepdim=True)) * logp[y]).mean() + 0.3 * (ref_logp.exp() * (ref_logp - logp)).sum()
        if not isinstance(loss, torch.Tensor) or loss.dim() != 0:
            r.expect(False, "returns a scalar tensor", "average the per-sample terms with .mean()")
            return r.done()
        if close(loss.detach(), no_base):
            hint = "subtract the group mean: A = R − R.mean(1, keepdim=True)"
        elif close(loss.detach(), rev_kl):
            hint = "KL(π ‖ π_ref) = Σ π · (log π − log π_ref); you wrote KL(π_ref ‖ π)"
        else:
            hint = "−mean(A · log π(y)) + β · Σ_y π(y)(log π(y) − log π_ref(y))"
        r.expect(close(loss.detach(), expected), f"value matches the reference ({expected:.4f})", hint)
        if loss.requires_grad:
            (g,) = torch.autograd.grad(loss, lg)
            lg2 = logits.clone().requires_grad_(True)
            (g_ref,) = torch.autograd.grad(ref_rlvr_loss(lg2, ref_logp, y, R, 0.3), lg2)
            r.expect(close(g, g_ref), "gradient with respect to the logits matches the reference")
            ok0, l0 = call(r, fn, ref_logp.clone().requires_grad_(True), ref_logp, y, torch.ones_like(R), 0.3)
            if ok0:
                r.expect(abs(float(l0.detach())) < 1e-9, "π = π_ref and equal rewards → loss 0 (no advantage, no KL)")
        else:
            r.expect(False, "the loss is differentiable", "don't detach the logits")
    return r.done()


# --------------------------------------------------------------------------- Part E · Qwen3 online DPO
def ref_best_vs_worst(scores: torch.Tensor, margin: float):
    """scores [P, n] → index of the best and worst sample per prompt, and keep = (max − min ≥ margin)."""
    best = scores.argmax(1)
    worst = scores.argmin(1)
    keep = (scores.max(1).values - scores.min(1).values) >= margin
    return best, worst, keep


def check_best_vs_worst(fn: Callable, strict: bool = False) -> bool:
    if not_attempted(fn):
        return False
    r = Report("best_vs_worst", strict)
    scores = torch.tensor([[0.1, 0.9, -0.4, 0.2], [0.0, 0.1, 0.05, 0.12], [2.0, -1.0, 1.9, 0.0]])
    ok, out = call(r, fn, scores, 0.5)
    if ok:
        try:
            best, worst, keep = (torch.as_tensor(t) for t in out)
        except Exception:
            r.expect(False, "returns (best, worst, keep)", "return three tensors of shape [P]")
            return r.done()
        r.expect(torch.equal(best.long(), torch.tensor([1, 3, 0])), "best = argmax over the n samples of each prompt",
                 "argmax over dim 1")
        r.expect(torch.equal(worst.long(), torch.tensor([2, 0, 1])), "worst = argmin over the n samples",
                 "argmin over dim 1")
        top2 = torch.tensor([True, False, False])
        r.expect(torch.equal(keep.bool(), torch.tensor([True, False, True])),
                 "keep = (best score − worst score) ≥ margin",
                 "the gap is max − min, not best − second best" if torch.equal(keep.bool(), top2)
                 else "keep = scores.max(1).values − scores.min(1).values ≥ margin")
    return r.done()


def ref_paired_diff(before: torch.Tensor, after: torch.Tensor):
    """Mean per-prompt change and its standard error (sample std with n − 1, over √n)."""
    d = after - before
    return d.mean(), d.std() / math.sqrt(len(d))


def check_paired_diff(fn: Callable, strict: bool = False) -> bool:
    if not_attempted(fn):
        return False
    r = Report("paired_diff", strict)
    before = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=D)
    after = torch.tensor([1.5, 2.2, 3.9, 4.1, 5.8], dtype=D)
    ok, out = call(r, fn, before, after)
    if ok:
        try:
            mean, se = (float(x) for x in out)
        except Exception:
            r.expect(False, "returns (mean, se)", "return two numbers")
            return r.done()
        d = after - before
        r.expect(abs(mean - float(d.mean())) < 1e-9, f"mean change = {float(d.mean()):.3f}", "mean of after − before")
        se_ref = float(d.std() / math.sqrt(5))
        unpaired = float(math.sqrt(before.var() / 5 + after.var() / 5))
        if abs(se - unpaired) < 1e-6:
            hint = ("that is the unpaired SE; the same prompts appear in both, so take the SE of the "
                    "per-prompt differences d = after − before")
        elif abs(se - float(d.std(correction=0) / math.sqrt(5))) < 1e-9:
            hint = "use the sample standard deviation (n − 1), torch's default d.std()"
        else:
            hint = "se = d.std() / sqrt(len(d))"
        r.expect(abs(se - se_ref) < 1e-9, f"paired standard error = {se_ref:.4f}", hint)
    return r.done()
