# Recovery tensor bytes with unchanged historical identities

Status: narrow CPU verification passed on 2026-10-05. The
[protocol](../specs/2026-10-05-recovery-tensor-bytes.md) was saved before source
changes or new tests. Tensor byte extraction now supports BF16 without changing
the existing typed v2 identity or hashes for previously supported tensors.
The snapshot size limit and production runners were not changed.

## Exact change and evidence

The original encoder converted each tensor directly to a NumPy numerical dtype.
The independently retained original function raises `TypeError: Got unsupported
ScalarType BFloat16` for the raw BF16 fixture. The new extraction detaches,
copies to CPU, makes data contiguous, flattens it and views its storage as uint8
before byte serialization. Dtype, shape, container tags and length frames remain
unchanged; flattening permits scalar and empty storage without losing shape
metadata. There is no numerical dtype conversion or BF16 neural computation.

The [actual verification](2026-10-05-recovery-tensor-bytes/results.json) records
77 dtype/layout rows across 11 original dtypes and seven layouts. All tensor,
nested-container and contiguous-layout comparisons match both the retained
original encoder and an independent accumulated-byte-stream implementation.
Seven additional BF16 layout rows match independently packed native-endian
uint16 bit patterns, including positive/negative zero, infinity, NaN and a
subnormal. Bit, dtype, shape and container changes remain distinct.

Thirteen preexisting v2 snapshots retain their exact embedded identities: one
generation state and completed, pending and final states for two seeds each of
the tiny DPO/RLVR sessions. Archived generation continuation matches exactly.
All eight completed/pending policy continuations reproduce archived histories,
final parameter hashes and byte-identical policy, reference, Adam, stream,
rollout RNG, global Torch RNG, step, history and pending fields. A retained
pending collection is applied without another collection. These remain the
original float64 CPU task, not pretrained or BF16 training recovery.

## Actual commands and retained failures

Executed with exit 0 using the existing isolated environment:

~~~bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/tmp/dongxi-course-reproduction.ZfVaEu/venv/bin/python \
  experiments/reports/2026-10-05-recovery-tensor-bytes/verify.py
~~~

The script reruns the complete `test_batched_cache_lab.py` suite and records its
command, actual exit and full output. The expanded suite has 30 passing tests;
the complete verification took 3.230943 seconds. No environment installation or
Git inspection was performed. Actual interpreter/package/CPU identity, source
hashes, expected historical file hashes and preservation checks are retained.

The first expanded test panel had one test-setup error: the newly added witness
tamper test omitted the existing required `expected_contract` keyword. It was
corrected before the passing panel and before the archived replay. A prior
read-only metadata projection also assumed a dictionary where the historical
report stores a list. Both tooling errors are recorded, not described as tensor
hash or recovery failures. No measured run or old report was repaired.

Exact original [module](2026-10-05-recovery-tensor-bytes/original-batched_cache_lab.py.txt)
and [test](2026-10-05-recovery-tensor-bytes/original-test_batched_cache_lab.py.txt)
snapshots retain their earlier hashes. Every inspected historical fixture/report
byte and every source/specification byte remained unchanged during verification.
The [acceptance record](2026-10-05-recovery-tensor-bytes-verification.json)
binds old and current identities separately.

## Limits

This establishes dense CPU tensor byte identity readiness, not BF16 optimizer
or neural correctness, CUDA/pretrained recovery, cross-host endianness parity,
production save-size safety or serving performance. Sparse and quantized tensor
formats are not covered. The existing production RLVR runner still lacks exact
completed/pending resume. No pretrained weights, download, GPU, service, shared
environment change, Git operation or course-progress mutation occurred.
