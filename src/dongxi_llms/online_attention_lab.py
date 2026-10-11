"""Original blockwise attention reference with ordinary PyTorch autograd.

This illustrates stable normalization and forward workspace. Python loops and
autograd saves are not a FlashAttention kernel or backward-memory optimization.
Fully masked/invalid query rows return zero output and logsumexp=-inf.
"""
import math
import torch


def _inputs(q, k, v, query_lengths, key_lengths, query_offset):
    if any(not isinstance(x, torch.Tensor) or x.ndim != 4 or not x.is_floating_point() for x in (q, k, v)):
        raise ValueError("Floating Q/K/V tensors [batch,heads,length,width] required")
    if any(min(x.shape) <= 0 or not bool(torch.isfinite(x).all()) for x in (q, k, v)):
        raise ValueError("Positive shapes and finite inputs required")
    if q.shape[:2] != k.shape[:2] or k.shape[:3] != v.shape[:3] or q.shape[-1] != k.shape[-1]:
        raise ValueError("Aligned batches/heads, Q/K width and K/V lengths required")
    if len({(x.dtype, x.device) for x in (q, k, v)}) != 1:
        raise ValueError("Q/K/V need the same dtype and device")
    if type(query_offset) is not int or query_offset < 0:
        raise ValueError("Nonnegative integer query offset required")
    b, _, nq, _ = q.shape; nk = k.shape[-2]
    def lengths(value, limit):
        if value is None:
            return torch.full((b,), limit, dtype=torch.long, device=q.device)
        value = torch.as_tensor(value, device=q.device)
        if value.shape != (b,) or value.dtype not in (torch.int32, torch.int64):
            raise ValueError("One integer valid length per batch row required")
        if bool(((value < 0) | (value > limit)).any()):
            raise ValueError("Valid lengths must fit storage")
        return value
    return lengths(query_lengths, nq), lengths(key_lengths, nk)


def _mask(qi, ki, qlengths, klengths, causal, offset):
    mask = (qi[None, :, None] < qlengths[:, None, None]) & (ki[None, None, :] < klengths[:, None, None])
    if causal:
        mask = mask & (ki[None, None, :] <= qi[None, :, None]+offset)
    return mask[:, None, :, :]


def dense_attention(q, k, v, *, causal=True, query_lengths=None, key_lengths=None, query_offset=0):
    ql, kl = _inputs(q, k, v, query_lengths, key_lengths, query_offset)
    allowed = _mask(torch.arange(q.shape[-2], device=q.device), torch.arange(k.shape[-2], device=q.device), ql, kl, causal, query_offset)
    scores = (q@k.transpose(-1, -2)/math.sqrt(q.shape[-1])).masked_fill(~allowed, -torch.inf)
    live = allowed.any(-1, keepdim=True)
    safe = torch.where(live, scores, torch.zeros_like(scores))
    probabilities = torch.where(live, safe.softmax(-1), torch.zeros_like(safe))
    lse = torch.where(live.squeeze(-1), safe.logsumexp(-1), torch.full_like(safe[..., 0], -torch.inf))
    return probabilities@v, lse


def blockwise_attention(q, k, v, *, query_tile=4, key_tile=4, causal=True,
                        query_lengths=None, key_lengths=None, query_offset=0,
                        trace=False, rescale_old=True):
    """Return output, row logsumexp and optional detached first-head tile trace.

    rescale_old=False is a deliberate broken control: changing the running
    maximum then leaves the previous normalizer/value numerator in wrong units.
    """
    ql, kl = _inputs(q, k, v, query_lengths, key_lengths, query_offset)
    if any(type(t) is not int or t <= 0 for t in (query_tile, key_tile)):
        raise ValueError("Positive integer tile sizes required")
    output, lses, history = [], [], []
    b, h, nq, d = q.shape; nk = k.shape[-2]
    for qs in range(0, nq, query_tile):
        qe = min(qs+query_tile, nq); qq = q[:, :, qs:qe]
        m = q.new_full((b, h, qe-qs, 1), -torch.inf)
        normalizer = torch.zeros_like(m)
        numerator = q.new_zeros((b, h, qe-qs, v.shape[-1]))
        qi = torch.arange(qs, qe, device=q.device)
        for ks in range(0, nk, key_tile):
            ke = min(ks+key_tile, nk); ki = torch.arange(ks, ke, device=q.device)
            allowed = _mask(qi, ki, ql, kl, causal, query_offset)
            scores = (qq@k[:, :, ks:ke].transpose(-1, -2)/math.sqrt(d)).masked_fill(~allowed, -torch.inf)
            new_m = torch.maximum(m, scores.amax(-1, keepdim=True))
            safe_m = torch.where(torch.isfinite(new_m), new_m, torch.zeros_like(new_m))
            scale = torch.exp(m-safe_m) if rescale_old else torch.ones_like(m)
            weights = torch.exp(scores-safe_m)
            normalizer = normalizer*scale + weights.sum(-1, keepdim=True)
            numerator = numerator*scale + weights@v[:, :, ks:ke]
            m = new_m
            if trace:
                maxima = m[0, 0, :, 0].detach().tolist()
                history.append(dict(query_start=qs, query_end=qe, key_start=ks, key_end=ke,
                                    row_max=[x if math.isfinite(x) else None for x in maxima],
                                    normalizer=normalizer[0, 0, :, 0].detach().tolist(),
                                    value_numerator=numerator[0, 0].detach().tolist()))
        live = normalizer > 0
        safe_l = torch.where(live, normalizer, torch.ones_like(normalizer))
        output.append(numerator/safe_l)
        lses.append((m+safe_l.log()).squeeze(-1))
    return torch.cat(output, -2), torch.cat(lses, -1), history


def unstable_attention(q, k, v):
    """Deliberate overflow control, without max subtraction or masks."""
    weights = torch.exp(q@k.transpose(-1, -2)/math.sqrt(q.shape[-1]))
    return (weights/weights.sum(-1, keepdim=True))@v


def forward_workspace(batch, heads, queries, keys, width, *, query_tile=4, key_tile=4, element_bytes=8):
    values = [batch, heads, queries, keys, width, query_tile, key_tile, element_bytes]
    if any(type(v) is not int or v <= 0 for v in values):
        raise ValueError("Positive integer workspace dimensions required")
    bh = batch*heads; tq = min(queries, query_tile); tk = min(keys, key_tile)
    return dict(dense_score_probability_bytes=2*bh*queries*keys*element_bytes,
                block_score_weight_bytes=2*bh*tq*tk*element_bytes,
                block_accumulator_statistics_bytes=bh*tq*(width+3)*element_bytes,
                boundary="Logical forward intermediates only; input/output, masks, temporaries, autograd saves and process/device peak excluded")
