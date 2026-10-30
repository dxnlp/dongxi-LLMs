"""Checks and reference implementations for study-path stage S1 · Reward models.

Notebook: ``notebooks/study-path/s1_reward_models.ipynb`` (Chapter 10 · Day 16, Spark GPU first).

Each ``check_*`` function receives the learner's object (usually a function) and prints one line
per test: ⏳ when the exercise is not attempted yet, ✅ when it passes and ❌ with a hint when it
fails. Checks never raise, so an unfinished notebook still runs top to bottom; ``strict=True``
turns failures into ``AssertionError`` (the reference cells use it). The ``ref_*`` functions are
the reference solutions the notebook prints in its "Reference solution" cells.

Parts: A Bradley–Terry loss and gradient · B a linear reward model under label noise ·
C pooling one score from a sequence · D pair accuracy and the pairwise training loss ·
E length audit · F outcome and process labels on toy arithmetic solutions.
"""
from __future__ import annotations

import math
import operator

import torch
import torch.nn.functional as F

from .common import Report, call, close, not_attempted

F64 = torch.float64


# =========================================================================== Part A · Bradley–Terry
def ref_bt_prob(gap: torch.Tensor) -> torch.Tensor:
    """P(chosen ≻ rejected) = σ(r_chosen − r_rejected)."""
    return torch.sigmoid(gap)


def ref_bt_loss(r_chosen: torch.Tensor, r_rejected: torch.Tensor) -> torch.Tensor:
    """Mean pairwise loss −log σ(r_chosen − r_rejected), computed stably with logsigmoid."""
    return -F.logsigmoid(r_chosen - r_rejected).mean()


def check_bt(prob_fn, loss_fn, strict: bool = False) -> bool:
    """prob_fn(gap) -> σ(gap); loss_fn(r_chosen [N], r_rejected [N]) -> scalar mean loss."""
    if not_attempted(prob_fn) or not_attempted(loss_fn):
        return False
    r = Report("Bradley–Terry", strict)
    gap = torch.tensor([-2.0, 0.0, 1.0, 4.0], dtype=F64)
    ok, p = call(r, prob_fn, gap)
    if not ok:
        return r.done()
    r.expect(close(p, torch.sigmoid(gap)), "bt_prob: gaps −2, 0, 1, 4 → 0.119, 0.500, 0.731, 0.982",
             "P(chosen wins) is torch.sigmoid(gap)")
    rc = torch.tensor([0.0, 1.0, 2.0, 5.0], dtype=F64)
    rr = torch.tensor([2.0, 1.0, 1.0, 1.0], dtype=F64)
    want = ref_bt_loss(rc, rr)
    ok, got = call(r, loss_fn, rc, rr)
    if not ok:
        return r.done()
    total = -F.logsigmoid(rc - rr).sum()
    swapped = -F.logsigmoid(rr - rc).mean()
    hint = ("you returned the sum: take the mean over pairs" if close(got, total)
            else "chosen and rejected are swapped: the gap is r_chosen − r_rejected" if close(got, swapped)
            else "loss = −log σ(r_chosen − r_rejected), averaged over pairs")
    r.expect(close(got, want), f"bt_loss on four pairs = {float(want):.4f} (mean over pairs)", hint)
    r.expect(close(loss_fn(rc + 10, rr + 10), want), "adding 10 to both scores leaves the loss unchanged",
             "the loss must depend on the scores only through their difference")
    big = torch.tensor([-200.0], dtype=torch.float32)
    ok, far = call(r, loss_fn, big, torch.zeros(1))
    if ok:
        far = torch.as_tensor(far, dtype=F64)
        r.expect(bool(torch.isfinite(far)) and close(far, torch.tensor(200.0, dtype=F64), atol=1e-3),
                 "stable for a confidently wrong pair: gap −200 (float32) gives loss 200, not inf",
                 "torch.log(torch.sigmoid(gap)) underflows to log 0; use F.logsigmoid(gap)")
    return r.done()


def ref_bt_grads(r_chosen: torch.Tensor, r_rejected: torch.Tensor):
    """Per-pair gradient of the summed loss Σ_i −log σ(gap_i) w.r.t. r_chosen and r_rejected."""
    g_chosen = -(1 - torch.sigmoid(r_chosen - r_rejected))
    g_rejected = 1 - torch.sigmoid(r_chosen - r_rejected)
    return g_chosen, g_rejected


def check_bt_grads(fn, strict: bool = False) -> bool:
    """fn(r_chosen, r_rejected) -> (d L/d r_chosen, d L/d r_rejected) with L the summed loss."""
    if not_attempted(fn):
        return False
    r = Report("Bradley–Terry gradient", strict)
    rc = torch.tensor([0.0, 1.0, 2.0, 5.0, -3.0], dtype=F64, requires_grad=True)
    rr = torch.tensor([2.0, 1.0, 1.0, 1.0, 3.0], dtype=F64, requires_grad=True)
    (-F.logsigmoid(rc - rr)).sum().backward()
    ok, got = call(r, fn, rc.detach(), rr.detach())
    if not ok:
        return r.done()
    if not r.expect(isinstance(got, (tuple, list)) and len(got) == 2, "returns (g_chosen, g_rejected)",
                    "return two tensors, one per score"):
        return r.done()
    gc, gr = got
    p = torch.sigmoid(rc.detach() - rr.detach())
    hint_c = ("sign flipped: lowering the loss raises r_chosen, so d loss/d r_chosen is negative"
              if close(gc, 1 - p) else "you used σ(gap) instead of 1 − σ(gap)" if close(gc, -p)
              else "d/d r_chosen of −log σ(gap) is −(1 − σ(gap))")
    r.expect(close(gc, rc.grad), "g_chosen matches autograd: −(1 − σ(gap)) per pair", hint_c)
    r.expect(close(gr, rr.grad), "g_rejected matches autograd: +(1 − σ(gap)) per pair",
             "the gap has −r_rejected in it, so its gradient is minus g_chosen")
    return r.done()


# =========================================================================== Part B · linear RM
def make_linear_pairs(n: int, w_true=(2.0, 0.5), seed: int = 0, eps: float = 0.0, judge: str = "random"):
    """Synthetic comparisons between answers A and B with features [helpfulness, length].

    The true reward is r*(f) = w_true · f. The clean label says "A wins" with probability
    σ(r*(A) − r*(B)) (Bradley–Terry is the true model). Then the judge corrupts a fraction ε:
    ``judge="random"`` flips the label with probability ε; ``judge="length"`` replaces the label
    with "the longer answer wins" with probability ε. Returns chosen/rejected features after the
    corrupted label, the clean label, and the true gap.
    """
    g = torch.Generator().manual_seed(seed)
    w = torch.tensor(w_true, dtype=F64)
    fa = torch.randn(n, len(w), generator=g, dtype=F64)
    fb = torch.randn(n, len(w), generator=g, dtype=F64)
    gap_true = (fa - fb) @ w
    a_wins_clean = torch.rand(n, generator=g, dtype=F64) < torch.sigmoid(gap_true)
    corrupt = torch.rand(n, generator=g, dtype=F64) < eps
    if judge == "random":
        a_wins = a_wins_clean ^ corrupt
    elif judge == "length":
        a_wins = torch.where(corrupt, fa[:, 1] > fb[:, 1], a_wins_clean)
    else:
        raise ValueError("judge must be 'random' or 'length'")
    pick = a_wins[:, None]
    return {"fc": torch.where(pick, fa, fb), "fr": torch.where(pick, fb, fa), "fa": fa, "fb": fb,
            "a_wins": a_wins, "a_wins_clean": a_wins_clean, "gap_true": gap_true, "w_true": w}


def ref_fit_linear_rm(fc, fr, steps: int = 300, lr: float = 1.0, loss_fn=ref_bt_loss):
    """Full-batch gradient descent on the Bradley–Terry loss of a linear score r(f) = w · f."""
    w = torch.zeros(fc.shape[1], dtype=fc.dtype, requires_grad=True)
    history = []
    for _ in range(steps):
        loss = loss_fn(fc @ w, fr @ w)
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            w.grad.zero_()
        history.append(loss.item())
    return w.detach(), history


def check_fit_linear_rm(fn, strict: bool = False) -> bool:
    """fn(fc, fr, steps, lr) -> (w [d], history) on clean synthetic pairs."""
    if not_attempted(fn):
        return False
    r = Report("linear reward model", strict)
    data = make_linear_pairs(20_000, seed=11)
    ok, got = call(r, fn, data["fc"], data["fr"], 300, 1.0)
    if not ok:
        return r.done()
    if not r.expect(isinstance(got, (tuple, list)) and len(got) == 2, "returns (w, history)"):
        return r.done()
    w, history = got
    w = torch.as_tensor(w, dtype=F64).detach()
    want, _ = ref_fit_linear_rm(data["fc"], data["fr"], 300, 1.0)
    neg = bool((w[0] < 0).item()) if w.numel() else False
    r.expect(w.shape == want.shape and close(w, want, atol=1e-3),
             f"fitted w = {[round(v, 3) for v in want.tolist()]} on 20,000 clean pairs",
             "w came out negative: the loss must be bt_loss(fc @ w, fr @ w), chosen first" if neg
             else "loss = bt_loss(fc @ w, fr @ w): linear scores of the chosen and rejected features")
    r.expect(close(w, data["w_true"], atol=0.12), "recovers the true weights (2.0, 0.5) within 0.12",
             "with clean labels and 20,000 pairs the fit should land near w_true")
    r.expect(len(history) == 300 and history[-1] < history[0], "loss history has 300 decreasing entries")
    return r.done()


def ref_noisy_label_prob(gap_true: torch.Tensor, eps: float) -> torch.Tensor:
    """P(label says A wins) when a clean BT label is flipped with probability ε."""
    return eps + (1 - 2 * eps) * torch.sigmoid(gap_true)


def check_noisy_label_prob(fn, strict: bool = False) -> bool:
    if not_attempted(fn):
        return False
    r = Report("noisy label probability", strict)
    gap = torch.tensor([-50.0, -1.0, 0.0, 1.0, 50.0], dtype=F64)
    for eps in (0.0, 0.2, 0.5):
        ok, got = call(r, fn, gap, eps)
        if not ok:
            return r.done()
        want = ref_noisy_label_prob(gap, eps)
        missing = (1 - eps) * torch.sigmoid(gap)
        r.expect(close(got, want), f"ε = {eps}: P(A) at gaps −50, −1, 0, 1, 50 = "
                 + ", ".join(f"{v:.3f}" for v in want.tolist()),
                 "you dropped the flipped-in mass: a clean 'B wins' label flipped to 'A' adds ε·(1 − σ)"
                 if close(got, missing) else "P(A) = (1 − ε)·σ(gap) + ε·(1 − σ(gap)) = ε + (1 − 2ε)·σ(gap)")
    return r.done()


def ref_reliability(p: torch.Tensor, y: torch.Tensor, n_bins: int = 10) -> dict:
    """Bin predicted probabilities; per non-empty bin: mean prediction, observed frequency, count; ECE."""
    p, y = p.detach().double().flatten(), y.detach().double().flatten()
    bins = torch.clamp((p * n_bins).long(), max=n_bins - 1)
    conf, freq, count = [], [], []
    for b in range(n_bins):
        m = bins == b
        if m.any():
            conf.append(p[m].mean().item())
            freq.append(y[m].mean().item())
            count.append(int(m.sum()))
    ece = sum(c / len(p) * abs(a - f) for a, f, c in zip(conf, freq, count))
    return {"conf": conf, "freq": freq, "count": count, "ece": ece}


def check_reliability(fn, strict: bool = False) -> bool:
    """fn(p, y, n_bins) -> {"conf", "freq", "count", "ece"}."""
    if not_attempted(fn):
        return False
    r = Report("reliability", strict)
    p = torch.tensor([0.05, 0.15, 0.12, 0.55, 0.58, 0.95, 0.91, 0.99], dtype=F64)
    y = torch.tensor([0, 0, 1, 1, 0, 1, 0, 1], dtype=F64)
    want = ref_reliability(p, y, 10)
    ok, got = call(r, fn, p, y, 10)
    if not ok:
        return r.done()
    if not r.expect(isinstance(got, dict) and {"conf", "freq", "count", "ece"} <= set(got),
                    "returns a dict with conf, freq, count, ece"):
        return r.done()
    r.expect(list(got["count"]) == want["count"], f"counts per non-empty bin {want['count']}",
             "keep only bins that contain predictions; count = m.sum()")
    r.expect(close(got["conf"], want["conf"]), "conf = mean predicted probability in each bin",
             "conf is p[m].mean(), not the bin centre")
    r.expect(close(got["freq"], want["freq"]), "freq = fraction of label 1 in each bin", "freq is y[m].mean()")
    unweighted = sum(abs(a - f) for a, f in zip(want["conf"], want["freq"])) / len(want["conf"])
    r.expect(close(got["ece"], want["ece"]), f"ECE = Σ (count/N)·|conf − freq| = {want['ece']:.4f}",
             "weight each bin by its share of the predictions" if close(got["ece"], unweighted)
             else "ECE = Σ_b (n_b / N) · |conf_b − freq_b|")
    return r.done()


# =========================================================================== Part C · pooling
def ref_last_token_index(input_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Hugging Face's rule: the rightmost position whose token is not the pad token, [B]."""
    non_pad = (input_ids != pad_id).long()
    return (torch.arange(input_ids.shape[-1], device=input_ids.device) * non_pad).argmax(-1)


def check_last_token_index(fn, pad_id: int = 151643, strict: bool = False) -> bool:
    if not_attempted(fn):
        return False
    r = Report("last non-pad token", strict)
    P = pad_id
    cases = [
        ("right padding", torch.tensor([[11, 12, 13, P, P], [21, 22, 23, 24, 25], [31, P, P, P, P]]), [2, 4, 0]),
        ("left padding", torch.tensor([[P, P, 11, 12, 13], [21, 22, 23, 24, 25], [P, P, P, P, 31]]), [4, 4, 4]),
    ]
    for label, ids, want in cases:
        ok, got = call(r, fn, ids, P)
        if not ok:
            return r.done()
        got_list = torch.as_tensor(got).flatten().tolist() if got is not None else got
        sumhint = ("mask.sum(-1) − 1 is the *count* of real tokens minus one; that equals the last index "
                   "only under right padding. Take the rightmost column where the mask is 1: "
                   "(arange(T) * mask).argmax(-1)")
        r.expect(got_list == want, f"{label}: last real token at {want}", sumhint if label == "left padding"
                 else f"you returned {got_list}; mark non-pad tokens with ids != pad_id, then take the rightmost one")
    return r.done()


# =========================================================================== Part D · accuracy and loss
def ref_pair_accuracy(s_chosen: torch.Tensor, s_rejected: torch.Tensor):
    """Fraction of pairs with s_chosen > s_rejected (a tie counts 1/2), and its binomial standard error."""
    wins = (s_chosen > s_rejected).double() + 0.5 * (s_chosen == s_rejected).double()
    acc = wins.mean().item()
    se = math.sqrt(acc * (1 - acc) / len(wins))
    return acc, se


def check_pair_accuracy(fn, strict: bool = False) -> bool:
    """fn(s_chosen [N], s_rejected [N]) -> (accuracy, standard error) as floats."""
    if not_attempted(fn):
        return False
    r = Report("pair accuracy", strict)
    sc = torch.tensor([1.0, 2.0, 0.5, 3.0, -1.0, 0.0, 2.0, 4.0])
    sr = torch.tensor([0.0, 2.5, 0.5, 1.0, -2.0, 1.0, 1.0, 4.0])
    want = ref_pair_accuracy(sc, sr)                      # 4 wins + 2 ties → 5/8
    ok, got = call(r, fn, sc, sr)
    if not ok:
        return r.done()
    if not r.expect(isinstance(got, (tuple, list)) and len(got) == 2, "returns (accuracy, standard error)"):
        return r.done()
    acc, se = float(got[0]), float(got[1])
    r.expect(abs(acc - want[0]) < 1e-9, f"8 pairs, 4 wins and 2 ties → accuracy {want[0]:.3f}",
             "ties count as half a win: (s_c > s_r) + 0.5·(s_c == s_r)" if abs(acc - 0.5) < 1e-9
             else "accuracy = mean over pairs of [s_chosen > s_rejected]")
    r.expect(abs(se - want[1]) < 1e-9, f"standard error √(acc(1 − acc)/N) = {want[1]:.4f}",
             "divide by N inside the square root: SE of a mean of N Bernoulli outcomes")
    return r.done()


def ref_rm_pair_loss(scores: torch.Tensor, n: int, loss_fn=ref_bt_loss) -> torch.Tensor:
    """scores [2n]: rows 0..n−1 are chosen answers, rows n..2n−1 the rejected answers to the same prompts."""
    return loss_fn(scores[:n], scores[n:])


def check_rm_pair_loss(fn, strict: bool = False) -> bool:
    if not_attempted(fn):
        return False
    r = Report("pairwise loss on a batch", strict)
    scores = torch.tensor([2.0, 0.5, -1.0, 1.0, 0.0, 1.5, 0.3, -0.4])     # n = 4: chosen block, then rejected
    n = 4
    want = ref_rm_pair_loss(scores, n)
    ok, got = call(r, fn, scores, n)
    if not ok:
        return r.done()
    reversed_ = ref_bt_loss(scores[n:], scores[:n])
    interleaved = ref_bt_loss(scores[0::2], scores[1::2])
    hint = ("you treated the second block as chosen: rows :n are chosen" if close(got, reversed_)
            else "the layout is [chosen block, rejected block], not interleaved pairs" if close(got, interleaved)
            else "split scores into scores[:n] (chosen) and scores[n:] (rejected), then bt_loss")
    r.expect(close(got, want), f"loss on a batch of 4 pairs = {float(want):.4f}", hint)
    return r.done()


# =========================================================================== Part E · length audit
def ref_pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation: centred dot product over the product of centred norms."""
    a, b = a.double() - a.double().mean(), b.double() - b.double().mean()
    return float((a * b).sum() / (a.norm() * b.norm()))


def check_pearson(fn, strict: bool = False) -> bool:
    if not_attempted(fn):
        return False
    r = Report("Pearson correlation", strict)
    g = torch.Generator().manual_seed(5)
    a = torch.randn(200, generator=g, dtype=F64) + 3.0
    b = 0.6 * a + torch.randn(200, generator=g, dtype=F64) + 10.0
    want = float(torch.corrcoef(torch.stack([a, b]))[0, 1])
    ok, got = call(r, fn, a, b)
    if not ok:
        return r.done()
    uncentred = float((a * b).sum() / (a.norm() * b.norm()))
    r.expect(abs(float(got) - want) < 1e-6, f"matches torch.corrcoef on shifted data ({want:.4f})",
             "subtract each mean first: without centring you get the cosine similarity"
             if abs(float(got) - uncentred) < 1e-6 else "ρ = Σ(a−ā)(b−b̄) / (‖a−ā‖·‖b−b̄‖)")
    ok, got2 = call(r, fn, a, -2 * a + 1)
    if ok:
        r.expect(abs(float(got2) + 1) < 1e-6, "a perfectly decreasing line gives −1")
    return r.done()


def ref_length_sliced_accuracy(s_c, s_r, len_c, len_r, acc_fn=ref_pair_accuracy):
    """Pair accuracy on pairs where the chosen answer is longer, and where it is shorter."""
    longer, shorter = len_c > len_r, len_c < len_r
    return {"chosen_longer": acc_fn(s_c[longer], s_r[longer]) + (int(longer.sum()),),
            "chosen_shorter": acc_fn(s_c[shorter], s_r[shorter]) + (int(shorter.sum()),)}


def check_length_sliced_accuracy(fn, strict: bool = False) -> bool:
    """fn(s_c, s_r, len_c, len_r) -> {"chosen_longer": (acc, se, n), "chosen_shorter": (acc, se, n)}."""
    if not_attempted(fn):
        return False
    r = Report("length-sliced accuracy", strict)
    # A scorer that mostly prefers the longer answer (two pairs perturbed so the slices differ from 0 and 1).
    len_c = torch.tensor([30, 50, 10, 80, 20, 40, 60, 60])
    len_r = torch.tensor([10, 20, 40, 30, 70, 40, 15, 90])
    s_c, s_r = len_c.double() / 10, len_r.double() / 10
    s_c[0], s_c[2] = 0.5, 5.0
    want = ref_length_sliced_accuracy(s_c, s_r, len_c, len_r)
    ok, got = call(r, fn, s_c, s_r, len_c, len_r)
    if not ok:
        return r.done()
    if not r.expect(isinstance(got, dict) and {"chosen_longer", "chosen_shorter"} <= set(got),
                    "returns a dict with chosen_longer and chosen_shorter"):
        return r.done()
    for key in ("chosen_longer", "chosen_shorter"):
        g, w = tuple(got[key]), want[key]
        r.expect(len(g) == 3 and abs(g[0] - w[0]) < 1e-9 and abs(g[1] - w[1]) < 1e-9 and g[2] == w[2],
                 f"{key}: (accuracy, SE, n) = ({w[0]:.3f}, {w[1]:.3f}, {w[2]})",
                 "slice with len_c > len_r (longer) and len_c < len_r (shorter); equal lengths belong to "
                 "neither; then call pair_accuracy on the slice and append the slice size")
    return r.done()


# =========================================================================== Part F · outcome vs process
OPS = {"+": operator.add, "-": operator.sub, "*": operator.mul}

# Toy word problems and authored solutions (each step is a, op, b, c meaning "a op b = c").
# Errors are deliberate and labelled in the notebook; this is not model output.
PROBLEMS = {
    "apples": {"text": "A shop has 3 boxes of 12 apples. It sells 15 apples, then receives 8 more. "
                       "How many apples does it have now?", "numbers": [3, 12, 15, 8], "gold": 29},
    "tickets": {"text": "Tickets cost $7. A class of 24 students and 3 teachers goes, and they pay with $200. "
                        "How much change do they get?", "numbers": [7, 24, 3, 200], "gold": 11},
}
SOLUTIONS = [
    ("apples · A", "apples", [(3, "*", 12, 36), (36, "-", 15, 21), (21, "+", 8, 29)]),
    ("apples · B", "apples", [(3, "*", 12, 36), (36, "-", 15, 11), (11, "+", 8, 19)]),
    ("apples · C", "apples", [(3, "*", 12, 36), (36, "-", 15, 22), (22, "+", 8, 29)]),
    ("apples · D", "apples", [(3, "*", 12, 36), (35, "-", 15, 20), (20, "+", 8, 28)]),
    ("tickets · A", "tickets", [(24, "+", 3, 27), (27, "*", 7, 189), (200, "-", 189, 11)]),
    ("tickets · B", "tickets", [(24, "*", 7, 168), (3, "*", 7, 21), (168, "+", 21, 189), (200, "-", 189, 11)]),
    ("tickets · C", "tickets", [(24, "+", 3, 27), (27, "*", 7, 179), (200, "-", 179, 21)]),
]


def ref_outcome_label(steps, gold) -> int:
    """1 if the solution's final result (the last step's right-hand side) equals the gold answer."""
    return int(steps[-1][3] == gold)


def ref_process_labels(numbers, steps) -> list[int]:
    """One label per step: 1 if the arithmetic is right and both operands are already known
    (a number from the problem or an earlier step's result), else 0."""
    known, labels = set(numbers), []
    for a, op, b, c in steps:
        labels.append(int(OPS[op](a, b) == c and a in known and b in known))
        known.add(c)
    return labels


def check_labels(outcome_fn, process_fn, strict: bool = False) -> bool:
    """outcome_fn(steps, gold) -> 0/1; process_fn(numbers, steps) -> list of 0/1 per step."""
    if not_attempted(outcome_fn) or not_attempted(process_fn):
        return False
    r = Report("outcome and process labels", strict)
    cases = [  # problem numbers, steps, gold
        ([2, 3, 4], [(2, "+", 3, 5), (5, "*", 4, 20)], 20),             # correct
        ([2, 3, 4], [(2, "+", 3, 6), (6, "*", 4, 24)], 20),             # slip at step 1
        ([2, 3, 4], [(2, "+", 3, 6), (6, "*", 4, 20)], 20),             # two wrong steps, right answer
        ([2, 3, 4], [(2, "+", 3, 5), (7, "*", 4, 28)], 20),             # step 2 uses an unknown 7
    ]
    names = ["correct", "slip at step 1", "two errors, right final answer", "operand never computed"]
    for (numbers, steps, gold), name in zip(cases, names):
        ok1, out = call(r, outcome_fn, steps, gold)
        ok2, proc = call(r, process_fn, numbers, steps)
        if not (ok1 and ok2):
            return r.done()
        want_o, want_p = ref_outcome_label(steps, gold), ref_process_labels(numbers, steps)
        r.expect(out == want_o, f"{name}: outcome label {want_o}", "outcome = int(last step's result == gold)")
        hint = ("check the operands too: a step that uses a number nobody computed is wrong"
                if name == "operand never computed" else
                "a step is right when OPS[op](a, b) == c (judge each step on its own displayed inputs)")
        r.expect(list(proc) == want_p if proc is not None else False, f"{name}: process labels {want_p}", hint)
    return r.done()


def ref_aggregate(step_probs: torch.Tensor) -> dict:
    """Solution-level scores from per-step correctness probabilities, and the weakest step."""
    return {"min": step_probs.min().item(), "product": step_probs.prod().item(),
            "weakest_step": int(step_probs.argmin())}


def check_aggregate(fn, strict: bool = False) -> bool:
    if not_attempted(fn):
        return False
    r = Report("step-score aggregation", strict)
    probs = torch.tensor([0.9, 0.8, 0.2, 0.95], dtype=F64)
    want = ref_aggregate(probs)
    ok, got = call(r, fn, probs)
    if not ok:
        return r.done()
    if not r.expect(isinstance(got, dict) and {"min", "product", "weakest_step"} <= set(got),
                    "returns a dict with min, product, weakest_step"):
        return r.done()
    r.expect(close(got["min"], want["min"]), f"min of [0.9, 0.8, 0.2, 0.95] = {want['min']:.2f}")
    r.expect(close(got["product"], want["product"]), f"product = {want['product']:.4f}",
             "multiply the step probabilities: P(every step correct) if steps are independent")
    r.expect(int(got["weakest_step"]) == 2, "weakest step is index 2 (0-based)",
             "argmin over the steps, not argmax; indices start at 0")
    return r.done()


# =========================================================================== simulated PRM
def simulated_prm(labels, p_good: float = 0.92, p_bad: float = 0.15) -> torch.Tensor:
    """A declared stand-in for a process reward model: probability p_good for a step labelled 1, p_bad for 0.

    Not a trained model. It lets the notebook show what aggregation does to a perfect-ranking PRM.
    """
    return torch.tensor([p_good if l else p_bad for l in labels], dtype=F64)
