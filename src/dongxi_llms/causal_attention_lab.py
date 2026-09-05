"""Inspectable, single-sequence/single-head attention for Chapter 4.

No dropout, padding, positional transformation, cache, or output projection.
The deliberately broken modes are teaching counterexamples, not alternatives.
"""

import math

import torch


def attention_trace(q, k, v, *, mode="causal"):
    """Return intermediates for Q/K [T,d_k] and V [T,d_v].

    Modes: causal, post_softmax (broken), zero_scores (broken).
    Every row permits its own position, so no row is fully masked.
    """
    if q.ndim != 2 or k.shape != q.shape or v.ndim != 2:
        raise ValueError("Expected Q and K [T,d_k], V [T,d_v].")
    if v.shape[0] != q.shape[0] or min(*q.shape, v.shape[1]) < 1:
        raise ValueError("Nonempty tensors must share sequence length T.")
    raw = q @ k.T
    scaled = raw / math.sqrt(q.shape[-1])
    allowed = torch.ones_like(scaled, dtype=torch.bool).tril()
    if mode == "causal":
        masked = scaled.masked_fill(~allowed, -torch.inf)
        weights = torch.softmax(masked, dim=-1)
    elif mode == "post_softmax":
        masked = scaled  # Future scores participate in normalization.
        weights = torch.softmax(scaled, dim=-1).masked_fill(~allowed, 0)
    elif mode == "zero_scores":
        masked = scaled.masked_fill(~allowed, 0)
        weights = torch.softmax(masked, dim=-1)  # exp(0) = 1, not 0.
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return dict(raw=raw, scaled=scaled, allowed=allowed, masked=masked,
                weights=weights, output=weights @ v)


def teaching_inputs():
    """Fixed, untrained feature vectors; rows are positions, not vocabulary IDs."""
    x = torch.tensor([[1., 0., 0.], [0., 1., 0.],
                      [1., 1., 0.], [0., 0., 1.]], dtype=torch.float64)
    wq = torch.tensor([[1., 0.], [0., 1.], [1., -1.]], dtype=x.dtype)
    wk = torch.tensor([[1., 0.], [0., 1.], [1., 1.]], dtype=x.dtype)
    wv = torch.tensor([[2., 0.], [0., 2.], [1., -1.]], dtype=x.dtype)
    return x, wq, wk, wv
