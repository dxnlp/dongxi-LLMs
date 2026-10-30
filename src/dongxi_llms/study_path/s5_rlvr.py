"""Checks and reference implementations for study-path stage S5 (RLVR: verifiers and reasoning tasks).

The notebook ``notebooks/study-path/s5_rlvr.ipynb`` asks the learner to write the key lines of a
verifiable-reward pipeline: answer extraction, exact number normalisation, the verifier reward,
the unbiased pass@k estimator, the zero-signal probability of a 0/1 group, a tabular policy-gradient
loss, the training-pool filter, the reward/advantage lines of a GRPO loop, a paired standard error
and the format/arithmetic decomposition of accuracy.

Each ``check_*`` function receives the learner's object and prints one line per test: ⏳ when the
exercise is not attempted yet, ✅ when it passes and ❌ with a hint when it fails. Checks never raise;
``strict=True`` turns failures into ``AssertionError`` (the reference cells use it). Everything here
runs on CPU without model weights. The synthetic task generator ``make_problem`` is shared by the
notebook and the tests, so the ground truth is computed by a program, never by a model.
"""
from __future__ import annotations

import itertools
import math
import random
import re
from fractions import Fraction
from typing import Callable

import torch

from dongxi_llms.study_path.common import Report, call, close, not_attempted

# --------------------------------------------------------------------------- contract
ANSWER_RE = re.compile(r"^[ \t]*Answer:[ \t]*(.+?)[ \t]*$", re.MULTILINE)
NUMBER_RE = re.compile(r"[+-]?(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?")
INSTRUCTION = ("Work it out in a few short steps, then write the result on a last line as plain text: "
               "Answer: <the number>. For example: Answer: 17")


def ref_extract_answer(text: str):
    """The string after the single 'Answer:' line, or None if there are zero or several such lines."""
    found = ANSWER_RE.findall(text)
    return found[0] if len(found) == 1 else None


def ref_parse_number(s: str):
    """Exact value of a number string in the contract (commas as thousands groups), else None."""
    s = s.strip().removesuffix(".")
    if NUMBER_RE.fullmatch(s) is None:
        return None
    return Fraction(s.replace(",", ""))


def ref_verifier_reward(text: str, gold: int, alpha: float = 0.0, extract=ref_extract_answer,
                        parse=ref_parse_number) -> float:
    """1 if the single Answer line equals gold exactly; alpha if well-formed but wrong; 0 otherwise."""
    answer = extract(text)
    value = None if answer is None else parse(answer)
    formatted = value is not None
    correct = formatted and value == gold
    return 1.0 if correct else alpha * float(formatted)


def substring_reward(text: str, gold: int) -> float:
    """A broken verifier: credit if the gold digits appear anywhere."""
    return float(str(gold) in text)


def last_number_reward(text: str, gold: int) -> float:
    """A common heuristic verifier: compare the last number in the text."""
    found = re.findall(r"-?\d+(?:\.\d+)?", text)
    return float(bool(found) and float(found[-1]) == gold)


# (text, gold, human label: "states one final value equal to gold")
ADVERSARIAL = [
    ("18 + 24 = 42\nAnswer: 42", 42, 1),
    ("1000 + 42 = 1,042\nAnswer: 1,042", 1042, 1),
    ("Half of 84 is 42.\nAnswer: 42.", 42, 1),
    ("**Answer:** 42", 42, 1),
    ("The answer is 42.", 42, 1),
    ("Answer: 42 apples", 42, 1),
    ("We start with 42 and lose 5.\nAnswer: 37", 42, 0),
    ("Answer: 420", 42, 0),
    ("Answer: 42\nAnswer: 24", 42, 0),
    ("Answer: 42 or 43", 42, 0),
    ("First 42 × 3 = 126, then", 42, 0),
    ("6 × 7 = 42, minus 2 is 40.\nAnswer: 40", 42, 0),
    ("Answer: 40\nCheck: 6 × 7 = 42", 42, 0),
]


# --------------------------------------------------------------------------- tasks
NAMES = ["Ana", "Ben", "Chen", "Dara", "Eli", "Femi", "Gita", "Hugo", "Ines", "Jun"]
ITEMS = ["apples", "marbles", "stamps", "books", "coins", "cards", "shells", "pencils"]


def make_problem(rng: random.Random, steps: int) -> dict:
    """A word problem with `steps` operations after the start value; the answer is computed exactly."""
    name, item = rng.choice(NAMES), rng.choice(ITEMS)
    value = rng.randint(10, 60)
    sentences = [f"{name} has {value} {item}."]
    for _ in range(steps):
        kinds = ["add", "sub", "boxes"]
        if value <= 200:
            kinds.append("mul")
        if value >= 20 and any(value % k == 0 for k in (2, 3, 4, 5)):
            kinds.append("div")
        kind = rng.choice(kinds)
        if kind == "add":
            n = rng.randint(5, 60)
            value += n
            sentences.append(f"{name} buys {n} more {item}.")
        elif kind == "sub" and value > 8:
            n = rng.randint(2, value - 3)
            value -= n
            sentences.append(f"{name} gives away {n} {item}.")
        elif kind == "boxes":
            k, m = rng.randint(2, 6), rng.randint(3, 15)
            value += k * m
            sentences.append(f"{name} receives {k} boxes with {m} {item} in each box.")
        elif kind == "mul":
            k = rng.choice([2, 3])
            value *= k
            sentences.append(f"{name}'s number of {item} becomes {['', '', 'twice', 'three times'][k]} as large.")
        elif kind == "div":
            k = rng.choice([k for k in (2, 3, 4, 5) if value % k == 0])
            value //= k
            sentences.append(f"{name} splits the {item} equally into {k} bags and keeps only one bag.")
        else:                                          # sub on a small value: add instead
            n = rng.randint(5, 30)
            value += n
            sentences.append(f"{name} finds {n} more {item}.")
    sentences.append(f"How many {item} does {name} have now?")
    return {"question": " ".join(sentences), "answer": value, "steps": steps}


def make_problems(seed: int, steps: int, n: int) -> list[dict]:
    rng = random.Random(seed * 1000 + steps)
    return [make_problem(rng, steps) for _ in range(n)]


def prompt_text(problem: dict) -> str:
    return problem["question"] + "\n\n" + INSTRUCTION


# --------------------------------------------------------------------------- estimators
def ref_pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from n samples with c correct: 1 − C(n−c, k) / C(n, k)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def naive_pass_at_k(n: int, c: int, k: int) -> float:
    """Plug-in estimate 1 − (1 − c/n)^k: biased for k > 1."""
    return 1.0 - (1.0 - c / n) ** k


def ref_zero_signal_prob(p, G: int):
    """Probability that G independent 0/1 rewards with success rate p are all equal."""
    return p ** G + (1 - p) ** G


def group_advantages(R: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """GRPO advantages within each row of R [B, G]: (R − mean) / (std + eps); constant rows → 0."""
    return (R - R.mean(-1, keepdim=True)) / (R.std(-1, keepdim=True) + eps)


def ref_tabular_pg_loss(theta: torch.Tensor, actions: torch.Tensor, adv: torch.Tensor) -> torch.Tensor:
    """−mean_{i,g} A_ig · log π_θ(a_ig | prompt i) for a tabular softmax policy theta [N, K]."""
    logp = torch.log_softmax(theta, -1).gather(1, actions)
    return -(adv * logp).mean()


def ref_select_pool(successes: torch.Tensor, n: int) -> torch.Tensor:
    """Boolean mask of prompts whose measured success count is strictly between 0 and n."""
    return (successes > 0) & (successes < n)


def ref_rewards_and_advantages(texts, golds, G, verifier=ref_verifier_reward):
    """Verifier reward per response [B·G] and GRPO advantages [B·G] (rows of G share a prompt)."""
    R = torch.tensor([verifier(t, g) for t, g in zip(texts, golds)], dtype=torch.float32)
    A = group_advantages(R.view(-1, G)).flatten()
    return R, A


def ref_paired_se(before: torch.Tensor, after: torch.Tensor) -> tuple[float, float]:
    """Mean and standard error of the per-problem difference after − before."""
    d = (after - before).double()
    return float(d.mean()), float(d.std() / math.sqrt(len(d)))


def ref_decompose(formatted: torch.Tensor, correct: torch.Tensor) -> tuple[float, float]:
    """P(formatted) and P(correct | formatted); their product is the accuracy."""
    p_fmt = formatted.float().mean()
    p_cond = correct[formatted].float().mean() if formatted.any() else torch.tensor(0.0)
    return float(p_fmt), float(p_cond)


# --------------------------------------------------------------------------- checks
def _num(x) -> float:
    """float(x), or NaN when x is not a number (so a check never raises on placeholder output)."""
    try:
        return float(x)
    except Exception:
        return float("nan")


def check_extract_answer(fn: Callable, strict: bool = False) -> bool:
    """fn(text) -> str | None."""
    if not_attempted(fn):
        return False
    r = Report("extract_answer", strict)
    cases = [
        ("6 × 7 = 42\nAnswer: 42", "42", "one Answer line after working", "read the text after 'Answer:' on its line"),
        ("I think 17 is close.\nAnswer: 19", "19", "a number in the reasoning is ignored",
         "only the Answer line counts, not other numbers in the text"),
        ("The answer is 42.", None, "no Answer line → None", "return None when the marker is missing; never guess"),
        ("Answer: 42\nAnswer: 24", None, "two Answer lines → None (ambiguous)",
         "count the matches: exactly one Answer line, otherwise None"),
        ("Answer: 7\nAnswer: 7", None, "two Answer lines that agree → still None",
         "the contract is 'exactly one line', even if both agree"),
        ("  Answer:   1,042  \n", "1,042", "spaces around the value are stripped", "strip the captured text"),
        ("My Answer: 5 is wrong.\nAnswer: 6", "6", "the marker must start its own line",
         "anchor the pattern at the start of a line (re.MULTILINE with ^)"),
    ]
    for text, want, what, hint in cases:
        ok, got = call(r, fn, text)
        if not ok:
            return r.done()
        r.expect(got == want, f"{what}: {want!r}", f"{hint} (you returned {got!r})")
    return r.done()


def check_parse_number(fn: Callable, strict: bool = False) -> bool:
    """fn(s) -> Fraction | None."""
    if not_attempted(fn):
        return False
    r = Report("parse_number", strict)
    cases = [
        ("1000", Fraction(1000), "plain integer", ""),
        ("1,000", Fraction(1000), "'1,000' = 1000", "remove the thousands commas before converting"),
        ("3.50", Fraction(7, 2), "'3.50' = 3.5", "convert to an exact number, not compare strings"),
        ("-0", Fraction(0), "'-0' = 0", "convert to a number; '-0' and '0' are the same value"),
        ("0.1", Fraction(1, 10), "'0.1' is exactly 1/10", "use Fraction(...), not float(...): float(0.1) ≠ 1/10"),
        ("42.", Fraction(42), "one sentence-final period is allowed", "keep the given removesuffix('.') line"),
        ("1,00", None, "'1,00' is not a valid grouping → None", "return None when NUMBER_RE.fullmatch fails"),
        ("12 apples", None, "'12 apples' → None", "return None when NUMBER_RE.fullmatch fails"),
        ("forty-two", None, "words → None", "return None when NUMBER_RE.fullmatch fails"),
    ]
    for s, want, what, hint in cases:
        ok, got = call(r, fn, s)
        if not ok:
            return r.done()
        same = (got is None and want is None) or (got is not None and want is not None
                                                  and isinstance(got, (Fraction, int)) and got == want)
        r.expect(same, f"{what}", f"{hint} (you returned {got!r})")
    return r.done()


def check_verifier_reward(fn: Callable, strict: bool = False) -> bool:
    """fn(text, gold, alpha=0.0) -> float."""
    if not_attempted(fn):
        return False
    r = Report("verifier_reward", strict)
    cases = [
        ("Answer: 42", 42, 0.0, 1.0, "correct → 1", "return 1.0 when the parsed value equals gold"),
        ("Answer: 1,042", 1042, 0.0, 1.0, "'1,042' vs gold 1042 → 1", "compare parsed values, not strings"),
        ("Answer: 42.0", 42, 0.0, 1.0, "'42.0' vs gold 42 → 1", "compare the parsed Fraction with gold"),
        ("Answer: 41", 42, 0.0, 0.0, "wrong, alpha = 0 → 0", "a wrong value gets no correctness credit"),
        ("Answer: 41", 42, 0.1, 0.1, "wrong but formatted, alpha = 0.1 → 0.1",
         "well-formed but wrong earns alpha"),
        ("The answer is 42.", 42, 0.1, 0.0, "no Answer line, alpha = 0.1 → 0",
         "no format credit when extraction returns None"),
        ("Answer: forty-two", 42, 0.1, 0.0, "unparseable value, alpha = 0.1 → 0",
         "formatted means the Answer line also parses as a number"),
        ("Answer: 42", 42, 0.1, 1.0, "correct, alpha = 0.1 → 1 (not 1.1)", "a correct answer gets exactly 1"),
    ]
    for text, gold, alpha, want, what, hint in cases:
        ok, got = call(r, fn, text, gold, alpha)
        if not ok:
            return r.done()
        r.expect(isinstance(got, (int, float)) and abs(_num(got) - want) < 1e-9, what, f"{hint} (you returned {got!r})")
    return r.done()


def check_pass_at_k(fn: Callable, strict: bool = False) -> bool:
    """fn(n, c, k) -> float, the unbiased estimator."""
    if not_attempted(fn):
        return False
    r = Report("pass_at_k", strict)
    for n, c, k in [(8, 2, 1), (8, 2, 4), (8, 0, 3), (8, 8, 2), (10, 3, 5)]:
        subsets = list(itertools.combinations(range(n), k))      # brute force over all k-subsets
        want = sum(any(i < c for i in s) for s in subsets) / len(subsets)
        ok, got = call(r, fn, n, c, k)
        if not ok:
            return r.done()
        r.expect(close(_num(got), want, atol=1e-9),
                 f"n={n}, c={c}, k={k}: {want:.4f} (fraction of k-subsets with ≥1 correct)",
                 f"you returned {got}; use 1 − C(n−c, k)/C(n, k) with math.comb, not 1 − (1 − c/n)^k")
    ok, got = call(r, fn, 8, 6, 3)
    if ok:
        r.expect(got == 1.0, "n − c < k → exactly 1 (every k-subset contains a correct sample)",
                 f"you returned {got}; math.comb(n−c, k) is 0 when n−c < k, so this should be 1")
    return r.done()


def check_zero_signal_prob(fn: Callable, strict: bool = False) -> bool:
    """fn(p, G) -> probability that all G 0/1 rewards are equal."""
    if not_attempted(fn):
        return False
    r = Report("zero_signal_prob", strict)
    for p, G in [(0.0, 8), (1.0, 8), (0.5, 2), (0.5, 8), (0.1, 8), (0.9, 4)]:
        want = p ** G + (1 - p) ** G
        ok, got = call(r, fn, p, G)
        if not ok:
            return r.done()
        r.expect(close(_num(got), want, atol=1e-9), f"p={p}, G={G}: {want:.4f}",
                 f"you returned {got}; a group is constant if all G succeed (p^G) OR all G fail ((1−p)^G)")
    ok, got = call(r, fn, torch.tensor([0.2, 0.5]), 4)
    if ok:
        r.expect(close(got, torch.tensor([0.2 ** 4 + 0.8 ** 4, 2 * 0.5 ** 4])),
                 "works elementwise on a tensor of p values", "use ** so it broadcasts over tensors")
    return r.done()


def check_tabular_pg_loss(fn: Callable, strict: bool = False) -> bool:
    """fn(theta [N,K], actions [N,G] long, adv [N,G]) -> scalar loss."""
    if not_attempted(fn):
        return False
    r = Report("tabular_pg_loss", strict)
    g = torch.Generator().manual_seed(0)
    theta = torch.randn(3, 5, generator=g, dtype=torch.float64)
    actions = torch.randint(0, 5, (3, 4), generator=g)
    adv = torch.randn(3, 4, generator=g, dtype=torch.float64)
    th = theta.clone().requires_grad_(True)
    ok, loss = call(r, fn, th, actions, adv)
    if not ok:
        return r.done()
    want = ref_tabular_pg_loss(theta, actions, adv)
    if not r.expect(torch.is_tensor(loss) and loss.ndim == 0, "returns a scalar tensor",
                    f"you returned {type(loss).__name__}; average over prompts and samples"):
        return r.done()
    r.expect(close(loss, want, atol=1e-8), f"value {float(want):+.5f}",
             "−mean(A · log π(a)); use log_softmax(theta, -1).gather(1, actions), and mind the minus sign")
    try:
        loss.backward()
        pi = torch.softmax(theta, -1)
        onehot = torch.nn.functional.one_hot(actions, 5).double()
        grad = -(adv[..., None] * (onehot - pi[:, None, :])).sum(1) / actions.numel()
        r.expect(close(th.grad, grad, atol=1e-8), "gradient = −(1/NG) Σ_g A_g (e_a − π)",
                 "the loss must depend on theta through log_softmax (not softmax, not detached)")
    except Exception as error:
        r.expect(False, f"backward failed: {error}", "return a tensor connected to theta")
    th2 = theta.clone().requires_grad_(True)
    ok, loss0 = call(r, fn, th2, actions, torch.zeros_like(adv))
    if ok and torch.is_tensor(loss0) and loss0.requires_grad:
        loss0.backward()
        r.expect(th2.grad is not None and float(th2.grad.abs().max()) == 0.0,
                 "all-zero advantages → gradient exactly 0", "multiply log π by the advantage")
    return r.done()


def check_select_pool(fn: Callable, strict: bool = False) -> bool:
    """fn(successes [P] long, n) -> bool mask [P]."""
    if not_attempted(fn):
        return False
    r = Report("select_pool", strict)
    s = torch.tensor([0, 1, 4, 7, 8, 3, 0, 8])
    ok, got = call(r, fn, s, 8)
    if not ok:
        return r.done()
    want = torch.tensor([False, True, True, True, False, True, False, False])
    try:
        got_b = torch.as_tensor(got).bool()
        same = got_b.shape == want.shape and bool((got_b == want).all())
    except Exception:
        same = False
    r.expect(same, "keeps 0 < successes < n (8 samples): [F, T, T, T, F, T, F, F]",
             f"you returned {got}; drop prompts never solved (0) AND always solved (n): both give zero advantage")
    return r.done()


def check_rewards_and_advantages(fn: Callable, verifier: Callable, strict: bool = False) -> bool:
    """fn(texts, golds, G) -> (R [B·G], A [B·G])."""
    if not_attempted(fn):
        return False
    r = Report("rewards_and_advantages", strict)
    texts = ["Answer: 5", "Answer: 6", "Answer: 5", "five",            # prompt 1, gold 5
             "Answer: 9", "Answer: 9", "Answer: 9", "Answer: 9"]       # prompt 2, gold 9: all correct
    golds = [5] * 4 + [9] * 4
    ok, out = call(r, fn, texts, golds, 4)
    if not ok:
        return r.done()
    try:
        R, A = out
        R, A = torch.as_tensor(R).float().flatten(), torch.as_tensor(A).float().flatten()
    except Exception:
        r.expect(False, "returns (R, A)", "return the reward tensor and the advantage tensor")
        return r.done()
    R_want = torch.tensor([verifier(t, g) for t, g in zip(texts, golds)])
    A_want = group_advantages(R_want.view(-1, 4)).flatten()
    r.expect(close(R, R_want), f"rewards {R_want.tolist()}", "call verifier(text, gold) once per response")
    r.expect(close(A[:4].sum(), 0.0, atol=1e-5), "advantages of a mixed group sum to 0",
             "normalise within each group of G (view(-1, G)), not across the whole batch")
    r.expect(close(A[4:], torch.zeros(4), atol=1e-6), "an all-correct group gets advantage 0 on every sample",
             "normalise per prompt: view(-1, G) before group_advantages")
    r.expect(close(A, A_want, atol=1e-4), "advantages match group_advantages(R.view(-1, G)).flatten()",
             "use the given group_advantages on R reshaped to [B, G], then flatten")
    return r.done()


def check_paired_se(fn: Callable, strict: bool = False) -> bool:
    """fn(before [n], after [n]) -> (mean diff, SE of mean diff)."""
    if not_attempted(fn):
        return False
    r = Report("paired_se", strict)
    before = torch.tensor([0.25, 0.5, 0.0, 1.0, 0.75, 0.5])
    after = torch.tensor([0.5, 0.5, 0.25, 1.0, 1.0, 0.25])
    ok, out = call(r, fn, before, after)
    if not ok:
        return r.done()
    m, se = ref_paired_se(before, after)
    try:
        gm, gse = _num(out[0]), _num(out[1])
    except Exception:
        r.expect(False, "returns (mean, se)", "return two numbers")
        return r.done()
    r.expect(close(gm, m, atol=1e-6), f"mean difference {m:+.4f}", "mean of after − before")
    unpaired = math.sqrt(before.var().item() / 6 + after.var().item() / 6)
    r.expect(close(gse, se, atol=1e-6), f"paired SE {se:.4f} = std(d)/√n",
             f"you returned {gse:.4f}" + ("; that is the unpaired SE — take std of the per-problem differences"
                                          if abs(gse - unpaired) < 1e-4 else "; use d.std() / sqrt(n) with d = after − before"))
    return r.done()


def check_decompose(fn: Callable, strict: bool = False) -> bool:
    """fn(formatted [n] bool, correct [n] bool) -> (P(formatted), P(correct | formatted))."""
    if not_attempted(fn):
        return False
    r = Report("decompose", strict)
    formatted = torch.tensor([True, True, False, True, False, True, True, True])
    correct = torch.tensor([True, False, False, True, False, True, False, True])
    ok, out = call(r, fn, formatted, correct)
    if not ok:
        return r.done()
    try:
        pf, pc = _num(out[0]), _num(out[1])
    except Exception:
        r.expect(False, "returns (p_fmt, p_cond)", "return two numbers")
        return r.done()
    r.expect(close(pf, 6 / 8), "P(formatted) = 6/8", "mean of the formatted mask")
    r.expect(close(pc, 4 / 6), "P(correct | formatted) = 4/6",
             "average correct only over the formatted responses: correct[formatted].float().mean()"
             + (" (you divided by all 8)" if close(pc, 4 / 8) else ""))
    r.expect(close(pf * pc, correct.float().mean()), "the product equals the accuracy 4/8", "")
    return r.done()
