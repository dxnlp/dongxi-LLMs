"""Chapter 4 gradient, scaling, and cache microscopes. CPU float64 examples.

The two-layer cache example is a residual attention stack, not a complete
Transformer or trained language model. Inputs are fixed feature vectors.
"""
import math

import torch
import torch.nn.functional as F

from .causal_attention_lab import attention_trace, teaching_inputs


def gradient_evidence(detach=None):
    """One target at position 2; position 3 must have no causal influence."""
    if detach not in (None, 'routing', 'values'):
        raise ValueError('detach must be None, routing, or values')
    x, wq, wk, wv = [t.clone().requires_grad_() for t in teaching_inputs()]
    q, k, v = x @ wq, x @ wk, x @ wv
    trace = attention_trace(q, k, v)
    a = trace['weights']
    for t in (q, k, v, a, trace['scaled']):
        t.retain_grad()
    routing = a.detach() if detach == 'routing' else a
    values = v.detach() if detach == 'values' else v
    o = routing @ values
    o.retain_grad()
    head = x.new_tensor([[1., -.5, .2], [-.3, .7, 1.]])
    logits = o[2] @ head
    loss = F.cross_entropy(logits[None], torch.tensor([1]))
    loss.backward()
    params = dict(X=x, W_Q=wq, W_K=wk, W_V=wv)
    result = dict(loss=float(loss.detach()), logits=logits.detach(),
                  parameters=params, trace=trace,
                  gradient_norms={n: None if p.grad is None else float(p.grad.norm())
                                  for n, p in params.items()})
    if detach is not None:
        return result
    go = o.grad
    ga = go @ v.detach().T
    gr = a.detach() * (ga - (ga * a.detach()).sum(-1, keepdim=True))
    gq = gr @ k.detach() / math.sqrt(q.shape[-1])
    gk = gr.T @ q.detach() / math.sqrt(q.shape[-1])
    gv = a.detach().T @ go
    manual = dict(Q=gq, K=gk, V=gv, A=ga, scores=gr,
                  W_Q=x.detach().T @ gq, W_K=x.detach().T @ gk,
                  W_V=x.detach().T @ gv,
                  X=gq @ wq.detach().T + gk @ wk.detach().T + gv @ wv.detach().T)
    actual = dict(Q=q.grad, K=k.grad, V=v.grad, A=a.grad,
                  scores=trace['scaled'].grad, **{n: p.grad for n, p in params.items()})
    errors = {n: float((manual[n] - actual[n]).abs().max()) for n in manual}
    # Central differences independently check each scalar projection parameter.
    def evaluate(matrices):
        h = attention_trace(x.detach() @ matrices[0], x.detach() @ matrices[1],
                            x.detach() @ matrices[2])['output']
        return F.cross_entropy((h[2] @ head)[None], torch.tensor([1])).item()
    numerical_error = 0.
    base = [wq.detach(), wk.detach(), wv.detach()]
    epsilon = 1e-6
    for m, param in enumerate((wq, wk, wv)):
        for row in range(param.shape[0]):
            for col in range(param.shape[1]):
                plus, minus = [t.clone() for t in base], [t.clone() for t in base]
                plus[m][row, col] += epsilon
                minus[m][row, col] -= epsilon
                fd = (evaluate(plus) - evaluate(minus)) / (2 * epsilon)
                numerical_error = max(numerical_error, abs(fd - param.grad[row, col].item()))
    result.update(manual_errors=errors, finite_difference_error=numerical_error,
                  forbidden_score_gradient=float(trace['scaled'].grad[~trace['allowed']].abs().max()),
                  future_input_gradient=float(x.grad[3].abs().max()),
                  prompt_input_gradient=float(x.grad[:2].norm()))
    return result


def scaling_evidence(widths=(8, 64, 512), rows=256, keys=16, seed=40):
    """IID normal-coordinate simulation, one query with 16 allowed sources per row.

    A softmax-Jacobian trace measures local sensitivity, not a loss gradient.
    Compare scaled and unscaled scores on exactly the same draws at each width.
    """
    generator = torch.Generator().manual_seed(seed)
    results = []
    for width in widths:
        q = torch.randn(rows, width, generator=generator, dtype=torch.float64)
        k = torch.randn(rows, keys, width, generator=generator, dtype=torch.float64)
        raw = torch.einsum('bd,bkd->bk', q, k)
        record = dict(width=width, raw_std=raw.std(unbiased=False).item(),
                      scaled_std=(raw / math.sqrt(width)).std(unbiased=False).item())
        for name, scores in [('raw', raw), ('scaled', raw / math.sqrt(width))]:
            logp = scores.log_softmax(-1)
            p = logp.exp()
            record[name + '_entropy'] = float(-(p * logp).sum(-1).mean())
            record[name + '_jacobian_trace'] = float((1 - p.square().sum(-1)).mean())
        results.append(record)
    return results


def cache_fixture():
    generator = torch.Generator().manual_seed(41)
    def draw(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float64)
    x = draw(6, 4)
    layers = [tuple(draw(4, 4) / 2 for _ in range(3)) for _ in range(2)]
    head = draw(4, 5) / 2
    return x, layers, head


@torch.no_grad()
def full_stack(x, layers, head):
    """Causal residual attention; cache K/V from each layer's incoming states."""
    h, caches = x, []
    for wq, wk, wv in layers:
        q, k, v = h @ wq, h @ wk, h @ wv
        caches.append((k, v))
        h = h + attention_trace(q, k, v)['output']
    return h @ head, caches


@torch.no_grad()
def decode_step(x_new, caches, layers, head):
    """One new position [1,D]. Keys contain only past positions plus self.

    No triangular mask is needed for this last-position query. An upper-left
    1-by-N triangular mask would incorrectly allow only the first cached key.
    """
    if x_new.shape[0] != 1 or len(caches) != len(layers):
        raise ValueError('Supply one new row and one K/V pair per layer')
    h, updated = x_new, []
    for (old_k, old_v), (wq, wk, wv) in zip(caches, layers):
        q = h @ wq
        k = torch.cat((old_k, h @ wk), dim=0)
        v = torch.cat((old_v, h @ wv), dim=0)
        weights = (q @ k.T / math.sqrt(q.shape[-1])).softmax(-1)
        h = h + weights @ v
        updated.append((k, v))
    return h @ head, updated


def cache_evidence(prefill=2):
    x, layers, head = cache_fixture()
    if not 1 <= prefill <= len(x):
        raise ValueError('prefill must be between 1 and sequence length')
    _, caches = full_stack(x[:prefill], layers, head)
    errors, sizes = [], []
    for t in range(prefill, len(x)):
        expected, _ = full_stack(x[:t+1], layers, head)
        actual, caches = decode_step(x[t:t+1], caches, layers, head)
        errors.append(float((actual - expected[-1:]).abs().max()))
        sizes.append(sum(k.numel()*k.element_size() + v.numel()*v.element_size()
                         for k, v in caches))
    # Same second input vector but different first input: deeper-layer KV differs.
    changed = x.clone()
    changed[0] += 3
    _, original_prefix = full_stack(x[:2], layers, head)
    _, changed_prefix = full_stack(changed[:2], layers, head)
    stale_logits, _ = decode_step(changed[2:3], original_prefix, layers, head)
    correct_logits, _ = full_stack(changed[:3], layers, head)
    result = dict(decode_errors=errors, cache_bytes=sizes,
                  final_shapes=[list(k.shape) for k, _ in caches],
                  first_layer_same_token_key_error=float((original_prefix[0][0][1]-changed_prefix[0][0][1]).abs().max()),
                  second_layer_same_token_key_error=float((original_prefix[1][0][1]-changed_prefix[1][0][1]).abs().max()),
                  stale_cache_logit_error=float((stale_logits-correct_logits[-1:]).abs().max()),
                  cached_projected_rows=len(layers)*len(x),
                  uncached_projected_rows=len(layers)*sum(range(prefill, len(x)+1)))
    return result
