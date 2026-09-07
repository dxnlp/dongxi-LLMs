"""Read-only CPU architecture checks for Day 7; not a quality benchmark."""
import torch
from torch import nn

from .decoder_lab import DecoderBlock, RMSNorm


def parameter_ledger(model):
    """Group unique Parameter objects; tied output weights count at lookup only."""
    totals = {}
    for name, parameter in model.named_parameters():
        if name.startswith('token.'):
            category = 'Token / tied head' if model.cfg.tied else 'Token lookup'
        elif name.startswith('position.'):
            category = 'Position lookup'
        elif name.startswith('lm_head.'):
            category = 'Untied vocabulary head'
        elif '.attn.' in name:
            category = 'Attention incl. QK norm'
        elif '.mlp.' in name:
            category = 'MLP'
        elif name.startswith('final_norm.'):
            category = 'Final norm'
        else:
            category = 'Block norms'
        totals[category] = totals.get(category, 0) + parameter.numel()
    return totals


def trace_decoder(model, ids):
    """Trace actual module boundaries, removing temporary hooks even on error.

    Projection outputs are before head splitting. Block output means the state,
    not its returned cache. No inputs, parameters, gradients, or RNG are changed.
    """
    rows, handles = [], []

    def capture(name):
        def hook(module, inputs, output):
            state = output[0] if isinstance(output, tuple) else output
            rows.append(dict(module=name, input_shape=list(inputs[0].shape),
                             output_shape=list(state.shape),
                             finite=bool(torch.isfinite(state).all())))
        return hook

    try:
        for name, module in model.named_modules():
            if isinstance(module, (nn.Embedding, nn.Linear, nn.LayerNorm,
                                   RMSNorm, DecoderBlock)):
                handles.append(module.register_forward_hook(capture(name)))
        with torch.no_grad():
            model(ids)
    finally:
        for handle in handles:
            handle.remove()
    return rows


def boundary_report(model, ids, prefix=2, cut=4, rope_offset=None):
    """Compare finite values, future intervention, and cached/full agreement.

    rope_offset applies only to suffix replay: zero is the deliberate failure.
    A passed fixture is not proof for every input or of trained quality.
    """
    if model.training:
        raise ValueError('Use eval mode for this deterministic boundary audit')
    if not (0 < prefix < ids.shape[1] and 0 < cut < ids.shape[1]):
        raise ValueError('prefix and cut must lie inside the sequence')
    with torch.no_grad():
        full = model(ids)
        changed = ids.clone()
        changed[:, cut:] = (changed[:, cut:] + 1) % model.cfg.vocab
        perturbed = model(changed)
        _, cache = model(ids[:, :prefix], return_cache=True)
        suffix = model(ids[:, prefix:], caches=cache, rope_offset=rope_offset)
    return dict(
        finite=all(bool(torch.isfinite(t).all()) for t in (full, perturbed, suffix)),
        causal=torch.allclose(full[:, :cut], perturbed[:, :cut], atol=1e-10, rtol=1e-8),
        cache=torch.allclose(full[:, prefix:], suffix, atol=1e-10, rtol=1e-8),
        future_error=float((full[:, :cut]-perturbed[:, :cut]).abs().max()),
        cache_error=float((full[:, prefix:]-suffix).abs().max()),
    )


class WrongAxisCenter(nn.Module):
    """Deliberate bug: subtract full-sequence time mean, leaking future states."""
    def forward(self, x):
        return x-x.mean(dim=1, keepdim=True)
