#!/usr/bin/env python3
"""Bounded original CS336 systems/scaling references; no downloads or auto GPU.

CPU: reference | smoke | matrix. The explicit profile mode also accepts cuda;
its caller must establish device/resource authority. JSON files never overwrite.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from dongxi_llms.decoder_lab import DecoderConfig, cost_estimate
from dongxi_llms.stories_training import StoriesDecoder
from dongxi_llms.systems_performance_lab import benchmark_matmul, checkpoint_audit, time_calls
from dongxi_llms.online_attention_lab import blockwise_attention, dense_attention, forward_workspace
from dongxi_llms.scaling_lab import ScalingFitError, evaluate_holdout, fit_isoflops, freeze_prediction, simulated_rows

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False); stream.write('\n')


def environment():
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                torch=torch.__version__, numpy=np.__version__, torch_threads=torch.get_num_threads(),
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), cuda_available=torch.cuda.is_available(),
                source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__), *[ROOT/'src/dongxi_llms'/n for n in ['systems_performance_lab.py', 'online_attention_lab.py', 'scaling_lab.py', 'stories_training.py', 'decoder_lab.py']]]})


def reference():
    g = torch.Generator().manual_seed(6505)
    q, k, v = [torch.randn(2, 2, 7, 4, generator=g, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    dense, dlse = dense_attention(q, k, v, query_lengths=[7, 4], key_lengths=[6, 0])
    online, olse, trace = blockwise_attention(q, k, v, query_tile=3, key_tile=2,
                                            query_lengths=[7, 4], key_lengths=[6, 0], trace=True)
    dg = torch.autograd.grad(dense.square().sum(), (q, k, v), retain_graph=True)
    og = torch.autograd.grad(online.square().sum(), (q, k, v))
    calibration = simulated_rows(budgets=(6e6, 24e6, 96e6))
    frozen = freeze_prediction(fit_isoflops(calibration), 384e6)
    holdout = evaluate_holdout(frozen, simulated_rows(budgets=(384e6,)))
    return dict(checkpoint=checkpoint_audit(), matmul=[benchmark_matmul(n, n, n) for n in [32, 64, 128]],
                attention=dict(output_max_abs_error=float((dense-online).detach().abs().max()),
                               gradient_max_abs_error=max(float((a-b).abs().max()) for a, b in zip(dg, og)),
                               finite_lse_max_abs_error=float((dlse[torch.isfinite(dlse)]-olse[torch.isfinite(dlse)]).detach().abs().max()),
                               trace=trace, workspace=forward_workspace(2, 2, 7, 7, 4, query_tile=3, key_tile=2)),
                simulation=dict(calibration=calibration, frozen_prediction=frozen, holdout=holdout),
                boundary='CPU correctness and timings; all scaling curves are authored simulations')


def corpus():
    # Original finite stochastic transition fixture: P(next=current+1)=.5 + .5/16.
    # Exact duplicate sequences are excluded across all splits before use.
    seen = set(); splits = {}
    for name, count, seed in [('train', 32, 6611), ('dev', 32, 6612), ('test', 64, 6613)]:
        rng = np.random.default_rng(seed); sequences = []
        while len(sequences) < count:
            row = [int(rng.integers(16))]
            for _ in range(8):
                row.append((row[-1]+1)%16 if rng.random() < .5 else int(rng.integers(16)))
            identity = tuple(row)
            if identity not in seen:
                seen.add(identity); sequences.append(row)
        splits[name] = torch.tensor(sequences, dtype=torch.long)
    return splits


def config(width):
    return DecoderConfig(vocab=16, width=width, heads=4, kv_heads=4, head_dim=width//4,
                         layers=2, hidden=2*width, max_length=8, modern=True, qk_norm=False)


def empirical_run(width, budget, seed, data, seconds_limit):
    start = time.monotonic()
    torch.manual_seed(seed)
    model = StoriesDecoder(config(width)).train()
    n = sum(p.numel() for p in model.parameters()); targets_per_update = 32
    updates = max(3, round(budget/(6*n*targets_per_update)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.005, weight_decay=.01)
    losses = []
    for step in range(updates):
        if time.monotonic()-start > seconds_limit:
            raise TimeoutError('Per-run/aggregate runtime gate reached')
        indices = (torch.arange(4)+4*step) % len(data['train']); tokens = data['train'][indices]
        # One warmup update followed by a budget-specific cosine horizon, explicitly recorded.
        lr = .005 if step == 0 else .0005+.0045*.5*(1+math.cos(math.pi*step/(updates-1)))
        for group in optimizer.param_groups: group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(tokens[:, :-1]).reshape(-1, 16), tokens[:, 1:].reshape(-1))
        loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
    model.eval()
    with torch.no_grad():
        dev = F.cross_entropy(model(data['dev'][:, :-1]).reshape(-1, 16), data['dev'][:, 1:].reshape(-1))
        train = F.cross_entropy(model(data['train'][:, :-1]).reshape(-1, 16), data['train'][:, 1:].reshape(-1))
    presented = updates*targets_per_update
    return model, dict(mode='measured-tiny-CPU', architecture='StoriesDecoder-MHA-2-block',
                       data_policy='finite256-target-slots-repeated', width=width, parameters=n,
                       compute_budget=budget, actual_6ND_coordinate=6*n*presented,
                       presented_tokens=presented, unique_available_target_slots=256,
                       source_repetitions=presented/256, seed=seed, updates=updates,
                       loss=float(dev), train_loss=float(train), training_loss_by_update=losses,
                       actual_runtime_seconds=time.monotonic()-start,
                       config=asdict(config(width)), dtype='torch.float32', attention_backend='SDPA actual kernel not inspected',
                       lr_policy='one peak warmup update then cosine to floor, horizon=run updates',
                       counted_forward_matmul=cost_estimate(config(width), 4, 8)['dense_forward_matmul_flops'],
                       loss_boundary='dev NLL over independent exact-deduplicated authored sequences; budget means target 6ND coordinate, quantization reported')


def matrix(spec, output):
    if spec['aggregate_seconds_cap'] > 900 or spec['total_presented_targets_cap'] > 200000:
        raise ValueError('Study cap exceeds implementation bound')
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    data = corpus(); models, rows, failures = [], [], []
    start = time.monotonic()
    data_identity = {key:hashlib.sha256(value.numpy().tobytes()).hexdigest() for key, value in data.items()}
    for budget in spec['calibration_budgets']:
        for width in spec['widths']:
            for seed in spec['seeds']:
                remaining = spec['aggregate_seconds_cap']-(time.monotonic()-start)
                if remaining <= 0: raise TimeoutError('Aggregate study cap reached')
                model, row = empirical_run(width, budget, seed, data, remaining)
                models.append((model, row)); rows.append(row)
                save(output/f'calibration-C{int(budget)}-w{width}-s{seed}.json', row)
    # Freeze a refusal too: a boundary minimum is a real outcome, never retune the grid.
    try:
        law = fit_isoflops(rows, assumed_loss_floor=0.)
        frozen = freeze_prediction(law, spec['heldout_budget'])
    except ScalingFitError as error:
        failures.append(dict(stage='calibration-fit', error=str(error), diagnostics=error.diagnostics))
        frozen = dict(status='fit-refused-before-holdout', error=str(error), diagnostics=error.diagnostics,
                      target_budget=spec['heldout_budget'], calibration_sha256=hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest())
    save(output/'frozen-prediction.json', frozen)
    heldout = []
    for width in spec['widths']:
        for seed in spec['seeds']:
            remaining = spec['aggregate_seconds_cap']-(time.monotonic()-start)
            if remaining <= 0: raise TimeoutError('Aggregate study cap reached')
            model, row = empirical_run(width, spec['heldout_budget'], seed, data, remaining)
            models.append((model, row)); heldout.append(row)
            save(output/f'heldout-C{int(spec["heldout_budget"])}-w{width}-s{seed}.json', row)
    # Independent text test is revealed only after the budget prediction/diagnosis is frozen.
    for model, row in models:
        with torch.no_grad():
            row['test_loss_after_freeze'] = float(F.cross_entropy(model(data['test'][:, :-1]).reshape(-1, 16), data['test'][:, 1:].reshape(-1)))
    if sum(r['presented_tokens'] for r in rows+heldout) > spec['total_presented_targets_cap']:
        raise ValueError('Presented-target cap violated')
    evaluation = evaluate_holdout(frozen, heldout) if frozen['status']=='prediction-frozen-before-holdout' else dict(status='no-identifiable-optimum-prediction', entire_budget_rows=heldout)
    return dict(spec=spec, data_sha256=data_identity, calibration=rows, frozen_prediction=frozen,
                holdout=evaluation, failures=failures, all_runs=rows+heldout,
                total_presented_targets=sum(r['presented_tokens'] for r in rows+heldout),
                aggregate_runtime_seconds=time.monotonic()-start,
                boundary='Local finite repeated corpus; three discrete widths; approximate quantized 6ND; assumed zero floor is a diagnostic fit convention, not known asymptotic loss; no stories/capability/large-model extrapolation')


def profile(device, output):
    device = torch.device(device)
    if device.type=='cuda' and not torch.cuda.is_available(): raise ValueError('CUDA requested but unavailable')
    torch.manual_seed(6404)
    sync = (lambda: torch.cuda.synchronize(device)) if device.type=='cuda' else (lambda:None)
    results = []; started = time.monotonic()
    def gate():
        if time.monotonic()-started > 600:
            raise TimeoutError('Single-device profile 600s cap reached')
        if device.type=='cuda':
            meminfo = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
            available = int(meminfo['MemAvailable'].strip().split()[0])*1024
            if available < 25*2**30:
                raise MemoryError('Host reserve below 25GiB')
            if torch.cuda.memory_allocated(device) > 8*2**30:
                raise MemoryError('Allocator usage exceeds 8GiB')
    # Width128, two blocks. One variable changes from baseline batch1,length128.
    for batch, length in [(1, 128), (2, 128), (1, 256), (1, 512)]:
        gate()
        c = DecoderConfig(vocab=256, width=128, heads=4, kv_heads=4, head_dim=32,
                          layers=2, hidden=256, max_length=512, modern=True, qk_norm=False)
        model = StoriesDecoder(c).to(device).eval(); ids = torch.randint(256, (batch, length), device=device)
        if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
        repetitions = []
        with torch.no_grad():
            for _ in range(3):
                gate()
                repetitions.append(time_calls(lambda:model(ids), warmup=5, repeats=10, synchronize=sync))
        gate()
        cost = cost_estimate(c, batch, length)
        results.append(dict(batch=batch, length=length, config=asdict(c), dtype=str(next(model.parameters()).dtype),
                            timings=repetitions, analytical_cost=cost,
                            estimated_matmul_flops_per_second=[cost['dense_forward_matmul_flops']/r['median_seconds'] for r in repetitions],
                            cuda_max_memory_allocated=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
                            cuda_max_memory_reserved=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None,
                            attention_backend='SDPA; actual fused kernel unknown unless independently profiled',
                            timed_phase='preallocated decoder forward only; no transfer/data preparation/backward',
                            memory_boundary='CUDA allocator bytes exclude host and other-process/unified-memory use; no device-peak utilization inference'))
    return dict(device=str(device), device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else platform.processor(),
                results=results, matmul=benchmark_matmul(512, 512, 512, dtype=torch.float32, device=device),
                resource_gates=dict(seconds_cap=600, host_reserve_gib=25, allocator_limit_gib=8,
                                    cadence='Between shapes and repetitions; external monitor owns intra-call resource enforcement'),
                boundary='Actual caller-selected single device; no multi-GPU evidence, no imported hardware peak or utilization')


def phase_profile(device, output):
    """Small CUDA-only timing supplement; event spans are not per-kernel times."""
    device = torch.device(device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise ValueError('phase-profile requires an explicitly available CUDA device')
    started = time.monotonic()
    def gate():
        if time.monotonic()-started > 120:
            raise TimeoutError('Phase supplement 120s cap reached')
        info = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        if int(info['MemAvailable'].strip().split()[0])*1024 < 25*2**30:
            raise MemoryError('Host reserve below 25GiB')
        if torch.cuda.memory_allocated(device) > 8*2**30:
            raise MemoryError('Allocator exceeds 8GiB')
    gate()
    generator = torch.Generator().manual_seed(6404)
    a = torch.randn(512, 512, generator=generator).to(device)
    b = torch.randn(512, 512, generator=generator).to(device)
    c = DecoderConfig(vocab=256, width=128, heads=4, kv_heads=4, head_dim=32,
                      layers=2, hidden=256, max_length=128, modern=True, qk_norm=False)
    with torch.random.fork_rng():
        torch.manual_seed(6404)
        model = StoriesDecoder(c).to(device).eval()
    ids = torch.randint(256, (1, 128), generator=generator).to(device)
    torch.cuda.reset_peak_memory_stats(device)
    def end_to_end():
        # Same input on each invocation; CPU generation and transfer are timed.
        fresh = torch.randint(256, (1, 128), generator=torch.Generator().manual_seed(6405))
        return model(fresh.to(device))
    phases = []
    with torch.no_grad():
        for name, function, boundary in [
            ('preallocated-matmul', lambda:a@b, '512x512 FP32 inputs already resident; allocation/transfer excluded'),
            ('preallocated-decoder-forward', lambda:model(ids), 'B1,L128,width128,2blocks FP32 resident input; allocation/transfer excluded'),
            ('input-generation-transfer-forward', end_to_end, 'Seeded CPU generator/input creation, blocking transfer, resident-model forward included; model construction/load excluded'),
        ]:
            repetitions = []
            for repetition in range(3):
                gate()
                for _ in range(5): function()
                torch.cuda.synchronize(device)
                host, events = [], []
                for _ in range(10):
                    torch.cuda.synchronize(device)
                    event_start = torch.cuda.Event(enable_timing=True)
                    event_end = torch.cuda.Event(enable_timing=True)
                    timer_start = time.perf_counter()
                    event_start.record()
                    function()
                    event_end.record()
                    event_end.synchronize()
                    host.append(time.perf_counter()-timer_start)
                    events.append(event_start.elapsed_time(event_end)/1000)
                repetitions.append(dict(repetition=repetition, warmup=5, repeats=10,
                                        host_synchronized_seconds=host, cuda_event_span_seconds=events,
                                        host_median_seconds=float(np.median(host)),
                                        device_span_median_seconds=float(np.median(events))))
                gate()
            phases.append(dict(phase=name, boundary=boundary, repetitions=repetitions))
        # Profiling is separate from ordinary timing. CUPTI/kernel availability is observed.
        trace_path = Path(output).parent/'phase-profiler-trace.json'
        if trace_path.exists(): raise FileExistsError(trace_path)
        try:
            gate()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA],
                                        record_shapes=True, profile_memory=True) as profiler:
                model(ids)
                torch.cuda.synchronize(device)
            profiler.export_chrome_trace(str(trace_path))
            trace = json.loads(trace_path.read_text())
            kernels = sorted({event.get('name', '') for event in trace.get('traceEvents', [])
                              if event.get('cat') == 'kernel'})
            profiling = dict(status='kernel-events-observed' if kernels else 'no-kernel-events-observed',
                             trace_path=str(trace_path), trace_sha256=sha(trace_path),
                             actual_kernel_names=kernels, profiled_steps=1,
                             boundary='One preallocated decoder forward; names observed in this trace only; no timing conclusions with profiler overhead')
        except Exception as error:
            profiling = dict(status='profiler-failed', error=f'{type(error).__name__}: {error}',
                             actual_kernel_names=[], boundary='No kernel classification inferred from SDPA API name')
        gate()
    return dict(device=str(device), device_name=torch.cuda.get_device_name(device), dtype='torch.float32',
                config=asdict(c), phases=phases, profiling=profiling,
                float32_matmul_precision=torch.get_float32_matmul_precision(),
                allow_tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                allow_tf32_cudnn=torch.backends.cudnn.allow_tf32,
                cuda_max_memory_allocated=torch.cuda.max_memory_allocated(device),
                cuda_max_memory_reserved=torch.cuda.max_memory_reserved(device),
                resource_gates=dict(seconds_cap=120, host_reserve_gib=25, allocator_limit_gib=8,
                                    cadence='Between repetitions/profile steps; root wrapper enforces total child deadline'),
                event_boundary='CUDA-event stream spans may include host-dispatch idle time, especially CPU creation; not individual kernel durations. Host span includes event-record/synchronization overhead.',
                boundary='Separate actual single-device phase supplement; allocator excludes host/other-process UMA; no peak-utilization, speedup or multi-GPU claim')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['reference', 'smoke', 'matrix', 'profile', 'phase-profile'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--spec', type=Path)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    args = parser.parse_args(); started = time.monotonic(); env = environment()
    if args.output.exists(): raise FileExistsError(args.output)
    if args.mode not in ('profile', 'phase-profile') and args.device != 'cpu': raise ValueError('Only explicit profile modes accept CUDA')
    if args.mode=='reference': result = reference()
    elif args.mode=='smoke':
        _, result = empirical_run(16, 12e6, 6601, corpus(), 30)
        result['smoke_cap_seconds'] = 30
    elif args.mode=='matrix':
        if not args.spec: raise ValueError('Matrix requires predeclared spec')
        spec = json.loads(args.spec.read_text())
        result = matrix(spec, args.output.parent/'matrix-runs'); result['spec_sha256'] = sha(args.spec)
    elif args.mode=='profile': result = profile(args.device, args.output)
    else: result = phase_profile(args.device, args.output)
    receipt = dict(command=sys.argv, environment=env, elapsed_seconds=time.monotonic()-started, result=result)
    save(args.output, receipt)
    print(json.dumps(dict(output=str(args.output), sha256=sha(args.output), elapsed_seconds=receipt['elapsed_seconds'])))


if __name__=='__main__': main()
