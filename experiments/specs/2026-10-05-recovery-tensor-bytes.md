# Recovery tensor byte compatibility

Status: specified before the authorized source edit and new test measurements
on 2026-10-05. This narrow follow-up fixes the tensor payload in
`batched_cache_lab.digest`; it does not add production runner recovery or change
the existing snapshot size limit. Historical reports and snapshots remain intact.

## Frozen intervention

Keep `dongxi-typed-length-framed-sha256-v2`, tensor dtype/shape metadata, scalar
and container type tags, dictionary ordering and length frames unchanged. Replace
only tensor payload extraction with detached CPU contiguous data, flattened
before a uint8 view and byte serialization. This avoids NumPy's unsupported BF16
conversion without changing payloads that its original conversion supported.
Zero-dimensional, empty and noncontiguous dense strided tensors are included.
No numerical casting or probability/model computation is introduced.

## Fixed CPU acceptance

Use the existing isolated interpreter
`/tmp/dongxi-course-reproduction.ZfVaEu/venv/bin/python`, one CPU thread, CUDA
hidden and offline flags. No new packages, external weights or tokenizers are used.

An independently retained prechange encoder is the parity baseline for bool,
uint8, int8/16/32/64, float16/32/64 and complex64/128, across scalar, empty,
vector, matrix, transpose and slice layouts, including tensor/container nesting.
For BF16, independently packed native-endian uint16 bit patterns provide the
expected payload, including signed zero and nonfinite bit patterns. Dtype,
shape, content and container changes must alter the framed identity; a BF16
storage bit tamper must reject restoration of a declared snapshot. Identical
logical values/layouts must preserve identity despite physical strides.

Run the complete bounded `test_batched_cache_lab.py` suite after adding these
focused controls. Load existing trusted v2 generation and completed/pending
DPO/RLVR snapshots from `fixtures/batched-cache-recovery-hardening/root-review`
with their separately recorded expected file SHA and exact contract. Their
embedded state identities must match the current encoder, and their uninterrupted
remaining updates must equal fresh restoration, including pending reuse without
another collection. Keep all existing fixture/report bytes unchanged.

Record old/current source/test hashes separately, exact commands, actual exit,
test counts and elapsed time, the independent byte matrix, existing snapshot
checks and unchanged historical evidence hashes in new reports. Preserve any
failed control or verification attempt; do not rewrite historical evidence as
current-source data. New source snapshots and verification code are supporting
files only under `experiments/reports/2026-10-05-recovery-tensor-bytes/`.

## Evidence boundary

Passing BF16 byte controls establishes serialization identity readiness only.
The tiny cache/recovery experiment remains CPU float64. No BF16 neural update,
CUDA/pretrained recovery, production save-size budget, checkpoint acquisition,
cross-host bitwise equivalence, installation, service, Git operation or course
progress change is authorized or implied.
