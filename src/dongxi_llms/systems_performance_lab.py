"""Original CPU systems microscope; logical costs are not hardware measurements.

Reuse the course decoder and StoriesDecoder checkpoint implementation. Timing
helpers can run on one explicitly selected device; they do not select a device
or launch a benchmark on import.
"""
from dataclasses import asdict, dataclass
import math
import statistics
import time

import torch
from torch.nn import functional as F

from .decoder_lab import DecoderConfig, cost_estimate
from .stories_training import StoriesDecoder


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class MatmulLedger:
    rows: int
    inner: int
    columns: int
    element_bytes: int = 4

    def __post_init__(self):
        for key, value in asdict(self).items():
            _positive_int(value, key)

    def report(self):
        m, k, n, b = self.rows, self.inner, self.columns, self.element_bytes
        flops = 2 * m * k * n
        payload = b * (m*k + k*n + m*n)
        return dict(**asdict(self), matmul_flops=flops,
                    modeled_minimum_io_bytes=payload,
                    arithmetic_intensity=flops/payload,
                    assumption="Read each input once; write output once. No cache/launch/reduction overhead.")


def attention_ledger(batch, heads, queries, keys, width, *, query_tile=16, element_bytes=4):
    """Dense score/value arithmetic and two explicitly modeled traffic policies."""
    for name, value in locals().copy().items():
        _positive_int(value, name)
    bh = batch*heads
    q, kv, out = bh*queries*width, bh*keys*width, bh*queries*width
    scores = bh*queries*keys
    tiles = math.ceil(queries/query_tile)
    return dict(dense_score_value_matmul_flops=4*bh*queries*keys*width,
                dense_score_probability_payload_bytes=2*scores*element_bytes,
                modeled_dense_io_bytes=(q+2*kv+out+4*scores)*element_bytes,
                modeled_blockwise_io_bytes=(q+2*kv*tiles+out+bh*queries)*element_bytes,
                assumptions=["Dense: write/read scores and write/read probabilities once.",
                             "Blockwise: each query block reads all K/V once; row logsumexp written once.",
                             "Inputs/output included; mask, softmax arithmetic and actual cache behavior excluded.",
                             "These are modeled transfers, not measured DRAM/HBM traffic."])


def roofline(intensity, compute_flops_per_second, bandwidth_bytes_per_second):
    values = [intensity, compute_flops_per_second, bandwidth_bytes_per_second]
    if any(isinstance(v, bool) or not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("Roofline inputs must be finite and positive")
    return min(compute_flops_per_second, intensity*bandwidth_bytes_per_second)


def decoder_phase_ledger(config, batch, prefix_length, *, element_bytes=4):
    """Reuse the dense estimator; count a cached new token attending prefix+itself.

    All parameters read once is an explicit decode traffic model, not actual
    cache behavior. KV bytes exclude Python objects and allocator overhead.
    """
    for key, value in dict(batch=batch, prefix_length=prefix_length, element_bytes=element_bytes).items():
        _positive_int(value, key)
    if prefix_length >= config.max_length:
        raise ValueError("Prefix leaves no context room for the new token")
    prefill = cost_estimate(config, batch, prefix_length, bytes_per_element=element_bytes)
    one = cost_estimate(config, batch, 1, bytes_per_element=element_bytes)
    added_attention = 4*config.layers*batch*config.heads*prefix_length*config.head_dim
    cache_read = 2*config.layers*batch*config.kv_heads*prefix_length*config.head_dim*element_bytes
    cache_write = 2*config.layers*batch*config.kv_heads*config.head_dim*element_bytes
    parameter_read = one["parameters"]*element_bytes
    decode_flops = one["dense_forward_matmul_flops"]+added_attention
    return dict(prefill=prefill, cached_new_token_matmul_flops=decode_flops,
                decode_cache_read_bytes=cache_read, decode_cache_append_bytes=cache_write,
                modeled_parameter_read_bytes=parameter_read,
                modeled_decode_intensity=decode_flops/(parameter_read+cache_read+cache_write),
                assumption="New token attends prefix plus itself; projections/MLP/head once; parameter payload read once; excludes softmax, elementwise ops, cache effects, non-KV activations and allocation")


def time_calls(function, *, warmup=3, repeats=7, synchronize=lambda: None):
    if type(warmup) is not int or warmup < 0:
        raise ValueError("warmup must be a nonnegative integer")
    _positive_int(repeats, "repeats")
    for _ in range(warmup):
        function()
    synchronize()
    elapsed = []
    for _ in range(repeats):
        synchronize()
        start = time.perf_counter()
        function()
        synchronize()
        elapsed.append(time.perf_counter()-start)
    return dict(seconds=elapsed, median_seconds=statistics.median(elapsed),
                min_seconds=min(elapsed), max_seconds=max(elapsed),
                population_std_seconds=statistics.pstdev(elapsed),
                warmup=warmup, repeats=repeats)


def benchmark_matmul(rows=64, inner=64, columns=64, *, dtype=torch.float64,
                     device="cpu", warmup=3, repeats=7, seed=6404):
    ledger = MatmulLedger(rows, inner, columns, torch.empty((), dtype=dtype).element_size()).report()
    device = torch.device(device)
    # A local CPU generator avoids changing the learner's global RNG state.
    generator = torch.Generator().manual_seed(seed)
    a = torch.randn(rows, inner, generator=generator, dtype=dtype).to(device)
    b = torch.randn(inner, columns, generator=generator, dtype=dtype).to(device)
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)
    timing = time_calls(lambda: a@b, warmup=warmup, repeats=repeats, synchronize=sync)
    return dict(device=str(device), dtype=str(dtype), ledger=ledger, timing=timing,
                estimated_matmul_flops_per_second=ledger["matmul_flops"]/timing["median_seconds"],
                timed_phase="Preallocated matmul only; input creation/transfer excluded; no utilization claim")


def saved_forward_storage(function):
    """Record tensors saved during this forward, including aliases and parameters.

    The logical sum counts saves, while backing storage is deduplicated within
    the forward. Neither is a liveness trace, process peak or allocator reading.
    Return the function result without adding tensor references to the ledger.
    """
    rows, storages = [], {}
    def pack(tensor):
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr())
        storages[key] = storage.nbytes()
        rows.append(dict(shape=list(tensor.shape), dtype=str(tensor.dtype),
                         logical_bytes=tensor.numel()*tensor.element_size()))
        return tensor
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        result = function()
    return result, dict(saved_tensors=rows,
                        logical_saved_tensor_bytes=sum(r["logical_bytes"] for r in rows),
                        unique_backing_storage_bytes=sum(storages.values()),
                        boundary="Forward saves only; aliases deduplicated for storage; includes saved parameter storage; not peak memory")


def checkpoint_audit(config=None, *, seed=6404, batch=2, length=8):
    config = config or DecoderConfig(vocab=16, width=16, heads=4, kv_heads=4,
                                    head_dim=4, layers=2, hidden=32,
                                    max_length=16, modern=True, qk_norm=False)
    _positive_int(batch, "batch"); _positive_int(length, "length")
    if length > config.max_length:
        raise ValueError("Length exceeds context")
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        models = [StoriesDecoder(config, flag).double().train() for flag in (False, True)]
    models[1].load_state_dict(models[0].state_dict())
    generator = torch.Generator().manual_seed(seed+1)
    ids = torch.randint(config.vocab, (batch, length+1), generator=generator)
    arms, gradients = [], []
    for model in models:
        calls = [0]
        def count(_module, _args):
            calls[0] += 1
        handles = [block.norm1.register_forward_pre_hook(count) for block in model.blocks]
        def forward_loss():
            logits = model(ids[:, :-1])
            return F.cross_entropy(logits.reshape(-1, config.vocab), ids[:, 1:].reshape(-1))
        loss, storage = saved_forward_storage(forward_loss)
        forward_calls = calls[0]
        loss.backward()
        gradients.append([p.grad.detach().clone() for p in model.parameters()])
        arms.append(dict(checkpointing=model.activation_checkpointing, loss=float(loss.detach()),
                         forward_block_entries=forward_calls, total_block_entries=calls[0],
                         recomputed_block_entries=calls[0]-forward_calls, **storage))
        for handle in handles:
            handle.remove()
    difference = max(float((a-b).abs().max()) for a, b in zip(*gradients))
    return dict(config=asdict(config), arms=arms, gradient_max_abs_difference=difference,
                loss_abs_difference=abs(arms[0]["loss"]-arms[1]["loss"]),
                analytical_cost=cost_estimate(config, batch, length, bytes_per_element=8),
                boundary="CPU FP64, ordinary autograd, existing non-reentrant checkpointing, no speedup/peak-memory claim")
