# Independent training snapshot failure checks

This specification is frozen before the independent test panel executes. It
reviews the shared trusted-local snapshot format, not a trained model or a
production recovery claim. The source implementation belongs to the parent
agent; this review owns only its new test module and separate evidence files.
Existing runner and experiment evidence must remain unchanged.

## Interface and assumptions

The reviewed API is `save_snapshot`, `load_snapshot` and `inspect_snapshot` in
`src/dongxi_llms/training_snapshot.py`. A snapshot uses an adjacent
`path + '.commit.json'` marker. Callers retain an independent payload byte
SHA256, exact byte count, stable scientific contract and positive size bound.
The marker is untrusted input, not a substitute for those retained values.

The stable contract binds scientific identity. Invocation identifiers and output
locations may change without changing that identity. A schedule horizon that
changes learning rates is scientific state, not an automatically extendable
runtime budget. The generic loader validates its envelope; a runner callback
must validate cursors, metrics, parameter layouts, original references and
pending collections before applying state.

This panel uses only authored dictionaries and tiny dense CPU tensors. No
pretrained weights, tokenizer, neural fit, GPU, external execution, installation,
service or Git operation is needed. Temporary test directories contain all
constructed payloads, malformed headers and deliberate failed saves.

## Predictions and acceptance checks

The public format should satisfy the following independent checks.

1. A valid data-only snapshot round-trips its contract, state, phase, completed
   count and parent invocation. Inspection verifies actual bytes without
   calling `torch.load`. Loading uses a verified buffer with
   `weights_only=True` and CPU mapping, not a later reopening of the path.
2. Wrong independent byte SHA256, size, contract or bound; absent payload or
   marker; byte mutation; malformed, duplicated, deep, oversized, incomplete or
   unknown marker fields; and unsupported marker schema, phase or counter are
   rejected before deserialization. An independently retained correct byte
   identity cannot be replaced by a self-consistent forged marker.
3. After restricted deserialization, unknown payload fields, schema and
   header/payload progress disagreements are rejected. Invalid state,
   parent identifier, contract or data-only tree is rejected. The per-runner
   callback runs only after generic checks and may refuse cursor semantics.
4. Saving rejects unsupported phases, counters, bounds, invocation identifiers,
   contracts and object trees. Cycles, unsupported tensor layouts and nonfinite
   values must not become a successful commit. A caller mutation after return
   must not alter already written bytes.
5. Existing data or commit paths are never overwritten. Deliberately failed
   writer, guard, data-fsync, header-fsync, publication and directory-fsync paths
   preserve a previous successful snapshot byte-for-byte and retain any newly
   created partial evidence. A failure after marker publication may leave a
   readable marker; it is not silently reported as successful durability.

Tests should report any discrepancy to the source owner rather than silently
editing the shared module. Initial test failures, including mistakes in this
independent test harness, remain in the verification record. The source hash
before and after each measured panel must be captured; a changed source is a
new observation, not a replacement of an earlier result.

## Frozen execution recipe

The panel will run with the existing isolated CPU interpreter:

```text
env PYTHONPATH=src CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/dongxi-course-reproduction.ZfVaEu/venv/bin/python -m unittest tests.test_training_snapshot_failure_paths -v
```

There is no favorable-result selection. All declared tests are included in each
panel. A fixed-source final rerun may follow a parent-owned correction, with its
source hash and the original failure output both retained. Exact panel duration,
exit status, standard output/error, interpreter identity, source/spec/test
hashes and preservation checks belong in the separate report and verification
JSON. Verification artifact paths are new and exclusive.

## Limits and next acceptance boundary

The bounded format checks are not a hostile-pickle allocation sandbox, a total
disk quota, OS containment or CUDA/BF16 replay proof. Inspection verifies file
bytes and the marker's schema but cannot independently prove runner state
semantics without deserializing. Runner-owned SFT/DPO completed-update and RLVR
completed/pending replay remain separate acceptance panels. RLVR pending state
must bind the pre-update policy/reference and post-collection cursor/RNG and be
consumed once without a fresh draw; this generic panel does not manufacture
that integration.
