"""Shared reporting helpers for the study-path workbooks.

Each ``check_*`` function in a stage module receives the learner's object (a function, tensor
or value) and prints one line per test: ⏳ when the exercise is not attempted yet, ✅ when it
passes, and ❌ with a hint when it fails. Learner checks never raise, so an unfinished notebook
still runs top to bottom. ``strict=True`` turns failures into ``AssertionError``; the reference
cells use it.
"""
from __future__ import annotations

from typing import Any, Callable

import torch


class Report:
    def __init__(self, name: str, strict: bool):
        self.name, self.strict, self.failures = name, strict, []

    def expect(self, ok: bool, what: str, hint: str = "") -> bool:
        ok = bool(ok)
        print(("✅ " if ok else "❌ ") + what + ("" if ok or not hint else f"  — hint: {hint}"))
        if not ok:
            self.failures.append(what)
        return ok

    def done(self) -> bool:
        if self.strict and self.failures:
            raise AssertionError(f"{self.name}: failed {self.failures}")
        return not self.failures


def not_attempted(obj: Any) -> bool:
    if obj is None or obj is Ellipsis:
        print("⏳ not attempted yet: fill in the exercise cell above, run it, then rerun this check.")
        return True
    return False


def call(report: Report, fn: Callable, *args, **kwargs):
    """Run learner code; an exception becomes a ❌ line instead of a crash."""
    try:
        return True, fn(*args, **kwargs)
    except Exception as error:  # learner code may still contain placeholders
        report.expect(False, f"running your code raised {type(error).__name__}: {error}",
                      "replace every `...` with real code")
        return False, None


def close(a, b, *, atol: float = 1e-4, rtol: float = 1e-4) -> bool:
    """Shape-and-value comparison that never raises (scalars, lists or tensors)."""
    try:
        ta = a if isinstance(a, torch.Tensor) else torch.as_tensor(a, dtype=torch.float64)
        tb = b if isinstance(b, torch.Tensor) else torch.as_tensor(b, dtype=torch.float64)
        ta, tb = ta.detach().double().cpu(), tb.detach().double().cpu()
        return ta.shape == tb.shape and bool(torch.allclose(ta, tb, atol=atol, rtol=rtol))
    except Exception:
        return False


def mine_or_ref(namespace: dict, name: str, ok: bool, reference):
    """Use the learner's object if its check passed, otherwise the reference."""
    chosen = namespace.get(name) if ok else reference
    print("using", "your" if ok else "the reference", "version of", name)
    return chosen
