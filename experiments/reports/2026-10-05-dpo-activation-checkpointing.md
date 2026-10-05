# DPO CPU activation checkpointing recovery

The CPU activation-checkpointing gate passes through the released DPO runner
functions. Seven controls establish original-equation parity, exact completed
update 3 recovery through update 6 in the same and a fresh Python process, and
wrong-mode contract rejection before state application. This adds CPU FP32 mode
evidence; it does not establish CUDA/BF16 recovery or model-scale memory savings.
The earlier checkpoint-off report and all released source remain unchanged.

## Fixed mode and comparison

The [premeasurement protocol](../specs/2026-10-05-dpo-activation-checkpointing.md)
retains the original random local Qwen3 configuration and authored branch fixture:
seed 1818, vocabulary/width 16, intermediate width 32, one layer, two query heads,
one KV head, head dimension 8, context 32 and zero dropout. Six updates use
accumulation 2, beta 0.2, AdamW learning rate 0.008, weight decay 0.01 and
clipping 1. The CPU weights are FP32; the snapshot envelope is 16 MiB.

Both modes explicitly disable model/reference cache. The policy-on mode calls
`gradient_checkpointing_enable()` exactly as production does. The reference
stays frozen/eval and does not enable checkpointing. Activation checkpointing
recomputes selected forward operations during backward instead of retaining
every intermediate activation. Its differentiated objective should remain the
same when recomputation preserves the forward behavior; this experiment checks
that property rather than assuming it from the API flag.

Before measurement, the off/on comparison fixed absolute and relative tolerances
at 1e-6 for policy/reference tensors, Adam moments and numerical loss/margin/norm
history. Sample indices, work and RNG must match exactly. The observed maximum
absolute difference is 0.0 for every floating category in this tiny CPU fixture.
That is the measured result of this comparison, not a claim that activation
checkpointing always gives bitwise equality across hardware, precision, dropout
or architecture settings.

## Recovery and reference evidence

[The final verification record](2026-10-05-dpo-activation-checkpointing/run-01/verification.json)
has SHA256 `85d9a0321650f9900fd8bf3a8119306b7b9c0e92f95b0071ded3189230f269c5`.
It retains exact commands, raw test output, fixture/package/lock/source identities,
all completed snapshots, numerical records and before/after source hashes.

The focused unittest execution passes all 7 tests in 3.367 s; measured subprocess
runtime including imports and the fresh-process subtest is 5.436989684007131 s,
with actual exit code 0. An exclusive separate collection repeats uninterrupted,
interrupted, restored and fresh-process checkpoint-on trajectories plus the
checkpoint-off comparison. No failed test or collection occurred in this package.
The deliberately injected metric interruption is retained as an expected failure
control, not omitted or reported as an uninterrupted success.

The checkpoint-on policy, reference, Adam moments/steps, sampler/global Torch RNG,
completed cursor, cumulative work and all six numerical history rows are exactly
equal after same-process and fresh-process recovery. The very next row after
update 3 also matches: pair indices `[1,1]`, loss 0.5337749719619751. The reference
still equals the original parent and is different from the trained policy; it
has no trainable parameters or gradients. The original-equation control computes
shifted response likelihoods and the softplus pair loss independently of the
runner loss helper, then compares the actual checkpoint-on optimizer update.

| Observation | CPU result |
| --- | --- |
| Same-mode restored numerical state and history | Exact same-process and fresh-process match |
| Off/on policy, reference, Adam, loss, margin and norm | Maximum absolute difference 0.0; predeclared tolerance passes |
| Selected indices, cumulative work and retained RNG | Exact off/on match |
| Incorrect checkpointing mode with otherwise identical tensor shapes | Expected contract rejected before application |
| EOS, prompt and padding supervision | Real EOS retained; prompt and EOS-valued padding excluded |

The final cumulative work remains 12 sampled pairs, 32 chosen response targets,
30 rejected targets and 110 logical branch tokens. Policy and reference each
make 24 scored training forwards covering 86 input positions. These counters
describe the runner's logical forwards; activation recomputation inside backward
is not counted as additional measured dispatched forward work. No memory-saving,
total compute, recomputation overhead or throughput comparison is inferred from
these counters or the tiny runtimes.

The collector raises a metric-write error immediately after durable update 3.
The snapshot preserves committed rows 1–3 even though only rows 1–2 reached the
metric file. The new invocation restores that boundary and performs updates 4–6
once. Producing invocation identity, parent receipt and output paths differ and
are deliberately excluded from numerical equality; elapsed time is not an
equality target. Initial, intermediate and final completed snapshots remain
immutable.

## Frozen identities and remaining gates

The policy-on scientific snapshot contract SHA256 is
`fcd6e07aebd2b3c1c965305b17049e949af9e5b479a26776b9c949ef083e3ea9`;
the off-mode contract is
`8f63b3f054ba8b368650f9fda21d9bbf728d812e1d91642e614b0401e7459107`.
The observed mode is part of the contract, not merely a descriptive run label.
The final checkpoint-on numerical state digest is
`7ed7c945c62c1fd26ae2edee889eb41362468bbeee9ae26e0061883999aff255`.

The new test/collector SHA256 is
`4179311168ceff0c9b0d0c3d15df4df1ab48f25b0336c182bf6446a7385459e4`;
the protocol SHA256 is
`92dff2c9c9d01c831943104d9a9ca773025f667901c3ba3fa3fdd5d176c3a845`.
The released runner remains
`f0a84e18015bc487e3944f592eb948fc605d9d8455c450cbe93d383d09b18bc7`,
the earlier recovery test remains
`cf31bef76b2b8e7b47d405ccfa03337fe3a6e064afad661eedd02e577506eb35`,
and the shared snapshot module remains
`9cad2851b665b5fe5d640c1327710352396218ea80fab3b0c82f01925e1749a3`.
All recorded before/after hashes agree, including the earlier report. The actual
environment is Python 3.12.14, Torch 2.14.1+cpu and Transformers 5.18.0 with the
retained repository lock.

CUDA/BF16 mode, pretrained checkpoint compatibility, model-scale memory and save
overhead, cross-machine determinism, whole-job containment and language quality
remain separate pending gates. No weights, data or dependencies were acquired;
no GPU, production hardware CLI, service or Git operation was launched. This
package closes the CPU activation-checkpointing gap only.

To reproduce, use a new destination with the existing isolated environment:

```bash
PYTHONPATH=src:tests CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 \
  /tmp/dongxi-course-reproduction.ZfVaEu/venv/bin/python \
  tests/test_dpo_checkpoint_mode.py --collect /tmp/dongxi-dpo-checkpoint-new-output
```

The destination must not exist; prior evidence is never replaced.
