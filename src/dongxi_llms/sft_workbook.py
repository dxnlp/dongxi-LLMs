"""Checks and reference implementations for the Day 12 instruction-tuning workbook.

Each ``check_*`` function receives the learner's object (a function, tensor or class) and
prints one line per test: ⏳ when the exercise is not attempted yet, ✅ when it passes and ❌
with a hint when it fails. Learner checks never raise, so an unfinished notebook still runs.
``strict=True`` turns failures into ``AssertionError``; reference solutions use it.
"""
from __future__ import annotations

import math
from typing import Callable, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE = -100


# --------------------------------------------------------------------------- reporting
class _Report:
    def __init__(self, name: str, strict: bool):
        self.name, self.strict, self.failures = name, strict, []

    def expect(self, ok: bool, what: str, hint: str = "") -> bool:
        print(("✅ " if ok else "❌ ") + what + ("" if ok or not hint else f"  — hint: {hint}"))
        if not ok:
            self.failures.append(what)
        return ok

    def done(self) -> bool:
        if self.strict and self.failures:
            raise AssertionError(f"{self.name}: failed {self.failures}")
        return not self.failures


def _not_attempted(obj) -> bool:
    if obj is None or obj is Ellipsis:
        print("⏳ not attempted yet: fill in the exercise cell above, run it, then rerun this check.")
        return True
    return False


def _call(report: _Report, fn: Callable, *args):
    try:
        return True, fn(*args)
    except Exception as error:  # learner code may still contain placeholders
        report.expect(False, f"running your code raised {type(error).__name__}: {error}",
                      "replace every `...` with real code")
        return False, None


# --------------------------------------------------------------------------- Part A · data
def ref_build_labels(ids: list[int], header: list[int], empty_think: list[int], im_end: int) -> list[int]:
    """Grade each assistant answer and its <|im_end|>; -100 elsewhere (empty think block skipped)."""
    labels, i = [IGNORE] * len(ids), 0
    while i < len(ids):
        if ids[i:i + len(header)] == header:
            start = i + len(header)
            if ids[start:start + len(empty_think)] == empty_think:
                start += len(empty_think)
            end = ids.index(im_end, start)
            labels[start:end + 1] = ids[start:end + 1]
            i = end + 1
        else:
            i += 1
    return labels


def check_render(ids, tok, messages, strict: bool = False) -> bool:
    if _not_attempted(ids):
        return False
    r = _Report("render", strict)
    expected = list(tok.apply_chat_template(messages, return_dict=True)["input_ids"])
    ok = r.expect(isinstance(ids, list) and all(isinstance(t, int) for t in ids),
                  "ids is a list of Python ints", "wrap the result in list(...), and take ['input_ids']")
    if ok:
        r.expect(ids == expected, f"ids match Qwen3's own template ({len(expected)} tokens)",
                 "call tok.apply_chat_template(messages, return_dict=True)['input_ids']")
    return r.done()


def check_answer_span(fn, tok, strict: bool = False) -> bool:
    """fn(ids, i) -> (start, end): the graded span after an assistant header at position i."""
    if _not_attempted(fn):
        return False
    r = _Report("answer span", strict)
    header = tok.encode("<|im_start|>assistant\n", add_special_tokens=False)
    think = tok.encode("<think>\n\n</think>\n\n", add_special_tokens=False)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    cases = [
        ([{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello there."},
          {"role": "user", "content": "Again"}, {"role": "assistant", "content": "Hi!"}], "two turns"),
        ([{"role": "user", "content": "Name a color."}, {"role": "assistant", "content": "Blue."}], "one turn, with think block"),
    ]
    for messages, label in cases:
        ids = list(tok.apply_chat_template(messages, return_dict=True)["input_ids"])
        positions = [i for i in range(len(ids)) if ids[i:i + len(header)] == header]
        for i in positions:
            start = i + len(header) + (len(think) if ids[i + len(header):i + len(header) + len(think)] == think else 0)
            want = (start, ids.index(im_end, start))
            ok, got = _call(r, fn, ids, i)
            if not ok:
                return r.done()
            decoded = tok.decode(ids[got[0]:got[1] + 1]) if isinstance(got, tuple) and len(got) == 2 else got
            r.expect(tuple(got) == want if isinstance(got, (tuple, list)) else False,
                     f"{label}: span {want} = {tok.decode(ids[want[0]:want[1] + 1])!r}",
                     f"you returned {got} = {decoded!r}; skip the header (and an empty think block), "
                     "and end at <|im_end|> inclusive")
    return r.done()


def check_build_labels(fn, tok, strict: bool = False) -> bool:
    if _not_attempted(fn):
        return False
    r = _Report("build_labels", strict)
    header = tok.encode("<|im_start|>assistant\n", add_special_tokens=False)
    think = tok.encode("<think>\n\n</think>\n\n", add_special_tokens=False)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    convo = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "One word for happy?"},
             {"role": "assistant", "content": "Elated."}, {"role": "user", "content": "Another?"},
             {"role": "assistant", "content": "Joyful."}]
    ids = list(tok.apply_chat_template(convo, return_dict=True)["input_ids"])
    want = ref_build_labels(ids, header, think, im_end)
    ok, got = _call(r, fn, ids)
    if not ok:
        return r.done()
    if not r.expect(isinstance(got, list) and len(got) == len(ids), "one label per input token",
                    "return a list with the same length as ids"):
        return r.done()
    graded = [tok.decode([t]) for t, l in zip(ids, got) if l != IGNORE]
    r.expect(graded == [tok.decode([t]) for t, l in zip(ids, want) if l != IGNORE],
             "graded tokens are exactly the two answers plus their <|im_end|>", f"you graded {graded}")
    r.expect(got[-1] == IGNORE, "the trailing newline after the last <|im_end|> is -100",
             "stop the span at <|im_end|>, inclusive")
    r.expect(got == want, "every label value matches (token id where graded, -100 elsewhere)",
             "graded positions should hold the token id itself, unshifted")
    return r.done()


def check_shifted_loss(fn, strict: bool = False) -> bool:
    """fn(logits [T, V], labels [T]) -> scalar loss with the one-position shift."""
    if _not_attempted(fn):
        return False
    r = _Report("shifted loss", strict)
    g = torch.Generator().manual_seed(7)
    logits = torch.randn(9, 11, generator=g)
    labels = torch.tensor([IGNORE, IGNORE, IGNORE, 4, 2, 9, IGNORE, 1, IGNORE])
    want = F.cross_entropy(logits[:-1], labels[1:], ignore_index=IGNORE)
    unshifted = F.cross_entropy(logits, labels, ignore_index=IGNORE)
    ok, got = _call(r, fn, logits, labels)
    if not ok:
        return r.done()
    got = torch.as_tensor(got, dtype=want.dtype)
    if r.expect(got.numel() == 1, "returns one scalar", "use the default reduction (mean)"):
        r.expect(torch.allclose(got, want, atol=1e-6),
                 f"equals the shifted, masked mean ({float(want):.4f})",
                 "unshifted? you got the copying-task value" if torch.allclose(got, unshifted, atol=1e-6)
                 else "logit t predicts label t+1: use logits[:-1] and labels[1:], ignore_index=-100")
    return r.done()


# --------------------------------------------------------------------------- Part B · patterns
def check_rank_one(M, strict: bool = False) -> bool:
    if _not_attempted(M):
        return False
    r = _Report("rank-1 pattern", strict)
    want = torch.tensor([[1., 2., 3.], [2., 4., 6.], [3., 6., 9.]])
    M = torch.as_tensor(M, dtype=torch.float32)
    r.expect(tuple(M.shape) == (3, 3), "shape is 3 × 3", "a column (3 × 1) times a row (1 × 3)")
    r.expect(M.shape == want.shape and torch.equal(M, want), "M equals [[1,2,3],[2,4,6],[3,6,9]]",
             "use u @ v or torch.outer(u, v) with u = v = [1, 2, 3]")
    r.expect(int(torch.linalg.matrix_rank(M)) == 1, "rank(M) == 1")
    return r.done()


def ref_top_r(M: torch.Tensor, r: int) -> torch.Tensor:
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    return (U[:, :r] * S[:r]) @ Vh[:r]


def check_top_r(fn, strict: bool = False) -> bool:
    """fn(M, r) -> best rank-r approximation of M."""
    if _not_attempted(fn):
        return False
    rep = _Report("top-r approximation", strict)
    g = torch.Generator().manual_seed(3)
    M = torch.randn(12, 9, generator=g)
    for r in (1, 3, 9):
        ok, got = _call(rep, fn, M, r)
        if not ok:
            return rep.done()
        got = torch.as_tensor(got)
        good = got.shape == M.shape and torch.allclose(got, ref_top_r(M, r), atol=1e-4)
        rep.expect(good, f"r = {r}: matches the SVD truncation" + (" (r = 9 rebuilds M exactly)" if r == 9 else ""),
                   "keep U[:, :r], S[:r], Vh[:r] and multiply back: (U[:, :r] * S[:r]) @ Vh[:r]")
        if good and r < 9:
            rep.expect(int(torch.linalg.matrix_rank(got)) == r, f"r = {r}: result has rank {r}")
    return rep.done()


def check_energy(fn, strict: bool = False) -> bool:
    """fn(M, r) -> share of squared singular values in the top r."""
    if _not_attempted(fn):
        return False
    rep = _Report("top-r energy", strict)
    M = torch.diag(torch.tensor([3., 2., 1., 1.]))
    for r, want in ((1, 9 / 15), (2, 13 / 15), (4, 1.0)):
        ok, got = _call(rep, fn, M, r)
        if not ok:
            return rep.done()
        rep.expect(abs(float(got) - want) < 1e-5, f"diag(3,2,1,1), r = {r}: energy {want:.3f}",
                   f"you returned {float(got):.3f}; square the singular values before summing")
    return rep.done()


def check_lora_param_count(fn, strict: bool = False) -> bool:
    """fn(d_out, d_in, r) -> trainable numbers in one LoRA adapter (A and B)."""
    if _not_attempted(fn):
        return False
    rep = _Report("LoRA parameter count", strict)
    for args, want in (((2048, 1024, 8), 24_576), ((1024, 1024, 8), 16_384), ((10, 4, 2), 28)):
        ok, got = _call(rep, fn, *args)
        if not ok:
            return rep.done()
        rep.expect(got == want, f"d_out={args[0]}, d_in={args[1]}, r={args[2]} -> {want:,}",
                   f"you returned {got}; A is r × d_in and B is d_out × r")
    return rep.done()


# --------------------------------------------------------------------------- Part C · LoRA layer
class RefLoRALinear(nn.Module):
    """Frozen base linear plus trainable (alpha/r)·B·A, with B = 0 at the start."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16):
        super().__init__()
        self.base, self.scale = base, alpha / r
        for p in self.base.parameters():
            p.requires_grad_(False)
        d_out, d_in = base.weight.shape
        device = base.weight.device
        self.A = nn.Parameter(torch.randn(r, d_in, device=device) / math.sqrt(d_in))
        self.B = nn.Parameter(torch.zeros(d_out, r, device=device))

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T @ self.B.T)


def check_lora_linear(cls, strict: bool = False) -> bool:
    """cls(base: nn.Linear, r, alpha) -> module with .A, .B and a LoRA forward."""
    if _not_attempted(cls):
        return False
    rep = _Report("LoRALinear", strict)
    torch.manual_seed(0)
    base = nn.Linear(6, 5, bias=False)
    x = torch.randn(3, 6)
    ok, layer = _call(rep, cls, base, 2, 4)
    if not ok:
        return rep.done()
    if not rep.expect(hasattr(layer, "A") and hasattr(layer, "B"), "the layer has A and B"):
        return rep.done()
    rep.expect(tuple(layer.A.shape) == (2, 6) and tuple(layer.B.shape) == (5, 2), "A is r × d_in, B is d_out × r",
               f"got A {tuple(layer.A.shape)}, B {tuple(layer.B.shape)}")
    rep.expect(bool(torch.all(layer.B == 0)), "B starts at zero", "so the wrapped model starts exactly as pretrained")
    rep.expect(bool(torch.any(layer.A != 0)), "A starts random (not zero)", "both zero would never learn")
    ok, y0 = _call(rep, layer, x)
    if ok:
        rep.expect(torch.allclose(y0, base(x), atol=1e-6), "at step 0 the output equals the base layer's")
    trainable = {n for n, p in layer.named_parameters() if p.requires_grad}
    rep.expect(trainable == {"A", "B"}, "only A and B are trainable", f"trainable: {sorted(trainable)}; freeze the base")
    with torch.no_grad():
        layer.B.copy_(torch.randn(5, 2))
    ok, y1 = _call(rep, layer, x)
    if ok:
        want = base(x) + (4 / 2) * (x @ layer.A.T @ layer.B.T)
        rep.expect(torch.allclose(y1, want, atol=1e-5), "forward = base(x) + (alpha/r)·x·Aᵀ·Bᵀ",
                   "remember the alpha/r scale")
    return rep.done()


def check_trainable_count(n, strict: bool = False) -> bool:
    if _not_attempted(n):
        return False
    rep = _Report("trainable count", strict)
    rep.expect(n == 1_146_880, "1,146,880 trainable parameters (rank 8 on q_proj and v_proj, 28 layers)",
               f"you counted {n:,}; count only parameters with requires_grad=True")
    return rep.done()


class _ToyLM(nn.Module):
    """A tiny stand-in with the same call style as a Hugging Face causal LM."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.emb, self.out = nn.Embedding(12, 8), nn.Linear(8, 12)

    def forward(self, input_ids, labels, attention_mask=None):
        logits = self.out(self.emb(input_ids))
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, 12), labels[:, 1:].reshape(-1), ignore_index=IGNORE)
        return type("Output", (), {"loss": loss, "logits": logits})()


def check_train_step(fn, strict: bool = False) -> bool:
    """fn(model, batch, opt) -> float loss after one forward/backward/step/zero_grad."""
    if _not_attempted(fn):
        return False
    rep = _Report("training step", strict)
    model = _ToyLM()
    batch = {"input_ids": torch.tensor([[1, 2, 3, 4, 5, 6]]), "labels": torch.tensor([[IGNORE, IGNORE, 3, 4, 5, 6]])}
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    before = [p.detach().clone() for p in model.parameters()]
    ok, loss = _call(rep, fn, model, batch, opt)
    if not ok:
        return rep.done()
    rep.expect(isinstance(loss, float), "returns the loss as a Python float", "return float(loss)")
    rep.expect(any(not torch.equal(b, p) for b, p in zip(before, model.parameters())),
               "the weights changed (backward + optimizer step happened)", "call loss.backward() and opt.step()")
    rep.expect(all(p.grad is None or not torch.any(p.grad) for p in model.parameters()),
               "gradients are cleared after the step", "call opt.zero_grad() after opt.step()")
    losses = [loss] + [fn(model, batch, opt) for _ in range(15)] if isinstance(loss, float) else []
    rep.expect(bool(losses) and losses[-1] < losses[0], "repeating the step lowers the loss",
               "forward with labels=..., then backward, step, zero_grad, in that order")
    return rep.done()


# --------------------------------------------------------------------------- Part D · quantization
def ref_quantize(w: torch.Tensor, levels: torch.Tensor, block: int = 64):
    flat = w.float().reshape(-1, block)
    absmax = flat.abs().amax(1, keepdim=True)
    codes = (flat / absmax).unsqueeze(-1).sub(levels).abs().argmin(-1)
    return codes, absmax.squeeze(1)


def ref_dequantize(codes, absmax, levels, shape):
    return (levels[codes] * absmax[:, None]).reshape(shape)


def check_quantize(quantize, dequantize, levels, strict: bool = False) -> bool:
    if _not_attempted(quantize) or _not_attempted(dequantize):
        return False
    rep = _Report("block quantization", strict)
    g = torch.Generator().manual_seed(11)
    w = torch.randn(4, 128, generator=g) * 0.02
    ok, out = _call(rep, quantize, w, levels, 64)
    if not ok:
        return rep.done()
    codes, absmax = out
    want_codes, want_absmax = ref_quantize(w, levels, 64)
    rep.expect(torch.allclose(torch.as_tensor(absmax).float().flatten(), want_absmax, atol=1e-7),
               "one absmax scale per block of 64 (8 scales)", "flat.abs().amax(dim=1) on blocks of 64")
    rep.expect(torch.equal(torch.as_tensor(codes).long().reshape(want_codes.shape), want_codes),
               "each weight gets the index of its nearest level", "divide by absmax, then argmin |value - level|")
    ok, back = _call(rep, dequantize, want_codes, want_absmax, levels, w.shape)
    if ok:
        rep.expect(torch.allclose(back, ref_dequantize(want_codes, want_absmax, levels, w.shape), atol=1e-7),
                   "dequantize = level value × block scale, back in the original shape",
                   "levels[codes] * absmax[:, None], then reshape")
    return rep.done()


def check_pack(pack, unpack, strict: bool = False) -> bool:
    if _not_attempted(pack) or _not_attempted(unpack):
        return False
    rep = _Report("4-bit packing", strict)
    codes = torch.tensor([[0, 15, 7, 8, 3, 12]], dtype=torch.uint8)
    ok, packed = _call(rep, pack, codes)
    if not ok:
        return rep.done()
    packed = torch.as_tensor(packed)
    rep.expect(packed.dtype == torch.uint8 and packed.numel() == 3, "6 codes fit in 3 bytes",
               "two 4-bit codes per byte: (high << 4) | low")
    rep.expect(packed.flatten().tolist() == [15, 120, 60], "byte values are [15, 120, 60]",
               f"you produced {packed.flatten().tolist()}; first code in the high nibble")
    ok, back = _call(rep, unpack, packed.reshape(1, -1))
    if ok:
        rep.expect(torch.equal(torch.as_tensor(back).to(torch.uint8).reshape(codes.shape), codes),
                   "unpack(pack(codes)) gives the codes back", "high = byte >> 4, low = byte & 15")
    return rep.done()


def check_memory(fn, strict: bool = False) -> bool:
    """fn(n_params, method) -> bytes for weights + gradients + AdamW, method in full/lora/qlora (no adapters)."""
    if _not_attempted(fn):
        return False
    rep = _Report("memory accounting", strict)
    for method, want in (("full", 16e9), ("lora", 2e9), ("qlora", 0.5e9)):
        ok, got = _call(rep, fn, 1e9, method)
        if not ok:
            return rep.done()
        rep.expect(abs(float(got) - want) < 1e3, f"1B parameters, {method}: {want / 1e9:g} GB",
                   "full: fp32 weight + grad + 2 AdamW moments; lora: bf16 frozen; qlora: 4 bits")
    return rep.done()
