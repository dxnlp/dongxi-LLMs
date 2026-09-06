"""Small, transparent CPU decoder for Chapter 5, not a serving implementation.

Tensor convention: states [B,T,D], Q/K/V [B,H,T,d], weights [B,H,T,S].
No dropout, padding, fused attention, pretrained weights, or hidden global dtype.
The optional QK norm is per-head learned RMSNorm BEFORE RoPE, retaining 1/sqrt(d).
RoPE pairs adjacent coordinates; this is not Qwen's exact checkpoint layout.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


def layer_norm(x, weight, bias, eps=1e-5):
    centered = x - x.mean(dim=-1, keepdim=True)
    return centered * torch.rsqrt(centered.square().mean(-1, keepdim=True) + eps) * weight + bias


def rms_norm(x, weight, eps=1e-6):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight


class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        return rms_norm(x, self.weight, self.eps)


def rope(x, positions, base=10000.):
    """Rotate adjacent coordinate pairs of [B,H,T,d] at absolute positions."""
    d = x.shape[-1]
    if d % 2 or positions.numel() != x.shape[-2]:
        raise ValueError('RoPE needs even head width and one position per time step')
    inv = base ** (-torch.arange(0, d, 2, device=x.device, dtype=x.dtype) / d)
    theta = positions.to(device=x.device, dtype=x.dtype)[:, None] * inv
    c, s = theta.cos(), theta.sin()
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even*c - odd*s, even*s + odd*c), dim=-1).flatten(-2)


def attend(q, k, v, query_positions=None, key_positions=None):
    """Transparent attention; grouped K/V remain compact until score computation.

    The returned weights use Hq heads. The caller's cached K/V use Hkv heads.
    Default positions describe a complete uncached sequence, not a decode step.
    """
    if k.shape[1] != v.shape[1] or q.shape[1] % k.shape[1]:
        raise ValueError('Query heads must be a multiple of KV heads')
    groups = q.shape[1] // k.shape[1]
    kk, vv = k.repeat_interleave(groups, 1), v.repeat_interleave(groups, 1)
    qp = torch.arange(q.shape[-2], device=q.device) if query_positions is None else query_positions
    kp = torch.arange(k.shape[-2], device=k.device) if key_positions is None else key_positions
    allowed = kp[None, :] <= qp[:, None]
    if not allowed.any(-1).all():
        raise ValueError('Every query must have at least one allowed source')
    scores = q @ kk.transpose(-1, -2) / math.sqrt(q.shape[-1])
    weights = scores.masked_fill(~allowed, -torch.inf).softmax(-1)
    return weights @ vv, weights


@dataclass(frozen=True)
class DecoderConfig:
    vocab: int = 16
    width: int = 16
    heads: int = 4
    kv_heads: int = 4
    head_dim: int = 4
    layers: int = 2
    hidden: int = 32
    max_length: int = 32
    modern: bool = False
    qk_norm: bool = False
    tied: bool = True

    def __post_init__(self):
        if min(self.vocab, self.width, self.heads, self.kv_heads, self.head_dim,
               self.layers, self.hidden, self.max_length) <= 0:
            raise ValueError('All dimensions must be positive')
        if self.heads % self.kv_heads:
            raise ValueError('heads must be divisible by kv_heads')
        if self.modern and self.head_dim % 2:
            raise ValueError('RoPE requires even head_dim')


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.q = nn.Linear(cfg.width, cfg.heads*cfg.head_dim, bias=False)
        self.k = nn.Linear(cfg.width, cfg.kv_heads*cfg.head_dim, bias=False)
        self.v = nn.Linear(cfg.width, cfg.kv_heads*cfg.head_dim, bias=False)
        self.out = nn.Linear(cfg.heads*cfg.head_dim, cfg.width, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim) if cfg.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(cfg.head_dim) if cfg.qk_norm else nn.Identity()

    def forward(self, x, cache=None, rope_offset=None):
        b, t, _ = x.shape
        def split(projection, heads):
            return projection(x).reshape(b, t, heads, self.cfg.head_dim).transpose(1, 2)
        q = self.q_norm(split(self.q, self.cfg.heads))
        k = self.k_norm(split(self.k, self.cfg.kv_heads))
        v = split(self.v, self.cfg.kv_heads)
        past = 0 if cache is None else cache[0].shape[-2]
        positions = torch.arange(past, past+t, device=x.device)
        if self.cfg.modern:
            rp = positions if rope_offset is None else torch.arange(rope_offset, rope_offset+t, device=x.device)
            q, k = rope(q, rp), rope(k, rp)
        if cache is not None:
            k, v = torch.cat((cache[0], k), -2), torch.cat((cache[1], v), -2)
        mixed, weights = attend(q, k, v, positions, torch.arange(past+t, device=x.device))
        merged = mixed.transpose(1, 2).contiguous().reshape(b, t, -1)
        return self.out(merged), (k, v), weights


class MLP(nn.Module):
    def __init__(self, width, hidden, gated=False):
        super().__init__()
        self.up = nn.Linear(width, hidden, bias=not gated)
        self.down = nn.Linear(hidden, width, bias=not gated)
        self.gate = nn.Linear(width, hidden, bias=False) if gated else None

    def forward(self, x):
        features = F.gelu(self.up(x)) if self.gate is None else F.silu(self.gate(x))*self.up(x)
        return self.down(features)


class DecoderBlock(nn.Module):
    def __init__(self, cfg, post_norm=False):
        super().__init__()
        norm = RMSNorm if cfg.modern else nn.LayerNorm
        self.norm1, self.norm2 = norm(cfg.width), norm(cfg.width)
        self.attn = MultiHeadAttention(cfg)
        self.mlp = MLP(cfg.width, cfg.hidden, cfg.modern)
        self.post_norm = post_norm

    def forward(self, x, cache=None, rope_offset=None):
        if self.post_norm:
            update, cache, weights = self.attn(x, cache, rope_offset)
            x = self.norm1(x + update)
            x = self.norm2(x + self.mlp(x))
        else:
            update, cache, weights = self.attn(self.norm1(x), cache, rope_offset)
            x = x + update
            x = x + self.mlp(self.norm2(x))
        return x, cache, weights


class TinyDecoder(nn.Module):
    def __init__(self, cfg=DecoderConfig()):
        super().__init__()
        self.cfg = cfg
        self.token = nn.Embedding(cfg.vocab, cfg.width)
        self.position = None if cfg.modern else nn.Embedding(cfg.max_length, cfg.width)
        self.blocks = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.layers)])
        self.final_norm = RMSNorm(cfg.width) if cfg.modern else nn.LayerNorm(cfg.width)
        self.lm_head = nn.Linear(cfg.width, cfg.vocab, bias=False)
        self.apply(self._initialize)
        if cfg.tied:
            self.lm_head.weight = self.token.weight

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0., std=.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, ids, caches=None, return_cache=False, rope_offset=None):
        if ids.ndim != 2 or ids.shape[1] == 0:
            raise ValueError('ids must be nonempty [B,T]')
        if caches is not None and len(caches) != len(self.blocks):
            raise ValueError('One cache pair per block is required')
        past = 0 if caches is None else caches[0][0].shape[-2]
        if past + ids.shape[1] > self.cfg.max_length:
            raise ValueError('Teaching model maximum length exceeded')
        x = self.token(ids)
        if self.position is not None:
            x = x + self.position(torch.arange(past, past+ids.shape[1], device=ids.device))
        updated = []
        for i, block in enumerate(self.blocks):
            x, cache, _ = block(x, None if caches is None else caches[i], rope_offset)
            updated.append(cache)
        logits = self.lm_head(self.final_norm(x))
        return (logits, updated) if return_cache else logits


def parameter_count(module):
    """Count unique stored Parameter objects, so tied embeddings count once."""
    return sum(p.numel() for p in module.parameters())


def analytical_parameters(c):
    attention = 2*c.width*c.head_dim*(c.heads+c.kv_heads)
    qkn = 2*c.head_dim if c.qk_norm else 0
    mlp = 3*c.width*c.hidden if c.modern else 2*c.width*c.hidden+c.hidden+c.width
    norms = (2 if c.modern else 4)*c.width
    embeddings = c.vocab*c.width*(1 if c.tied else 2)
    positions = 0 if c.modern else c.max_length*c.width
    final = c.width if c.modern else 2*c.width
    return embeddings+positions+final+c.layers*(attention+qkn+mlp+norms)


def cost_estimate(c, batch, length, bytes_per_element=4):
    """Dense full-sequence inference matmul FLOPs; one multiply-add = two FLOPs.

    Excludes norms, activation, softmax, embeddings, training backward, allocator
    overhead, and KV expansion temporaries. Causal matrices counted as dense.
    """
    if min(batch, length, bytes_per_element) <= 0:
        raise ValueError('Cost dimensions must be positive')
    projections = 4*batch*length*c.width*c.head_dim*(c.heads+c.kv_heads)
    attention = 4*batch*c.heads*length**2*c.head_dim
    mlp = (6 if c.modern else 4)*batch*length*c.width*c.hidden
    head = 2*batch*length*c.width*c.vocab
    return dict(parameters=analytical_parameters(c),
                dense_forward_matmul_flops=c.layers*(projections+attention+mlp)+head,
                logical_kv_bytes=2*c.layers*batch*c.kv_heads*length*c.head_dim*bytes_per_element)


def teaching_batch():
    """Two fixed sequences; targets are consistent even where prefixes overlap."""
    sequence = torch.tensor([[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14]])
    return sequence[:, :-1], sequence[:, 1:]


def next_token_loss(logits, labels):
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))


def fit_one_batch(steps=160, seed=505, lr=.02):
    """Bounded CPU float32 learning experiment; fixed budget, no best-seed search."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = TinyDecoder()
        ids, labels = teaching_batch()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.)
        initial = float(next_token_loss(model(ids), labels).detach())
        history = []
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = next_token_loss(model(ids), labels)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError('Nonfinite gradient')
            optimizer.step()
            history.append(float(loss.detach()))
        with torch.no_grad():
            logits = model(ids)
            final = float(next_token_loss(logits, labels))
            accuracy = float((logits.argmax(-1) == labels).float().mean())
        return model, dict(seed=seed, steps=steps, lr=lr, initial_loss=initial,
                           final_loss=final, accuracy=accuracy, history=history)
