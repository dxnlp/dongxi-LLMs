# Independent training snapshot failure review

All 31 independent adversarial tests passed against the frozen shared snapshot
implementation on 2026-10-05. The first panel exited 0 and took 1.288490565 seconds
including interpreter startup; unittest reported 0.665 seconds. No shared source
or runner was changed by this review. The
[specification](../specs/2026-10-05-training-snapshot-failure-paths.md),
[test source](../../tests/test_training_snapshot_failure_paths.py),
[first raw panel](2026-10-05-training-snapshot-failure-paths-first.json) and
[final verification](2026-10-05-training-snapshot-failure-paths-verification.json)
record the exact contract, command, source identity and full output.

## What the checks establish

The panel independently rejects wrong retained SHA256, exact size, scientific
identity and byte bound before calling `torch.load`. Scientific identity checks
cover source, lock, data, original parent, schedule horizon, token map, template
and stop IDs. Moving unchanged bytes and changing the parent invocation remain
separate from that identity. Malformed, duplicate, deep, oversized, unknown and
ill-typed marker fields are rejected, as are orphan and symlink files. A forged
self-consistent marker cannot replace the caller's retained byte receipt.

After generic validation, a runner callback can refuse its own cursor invariant.
Unknown payload fields, unsupported schema, header/payload progress disagreement,
invalid state/invocation/contract and unsupported object trees are rejected.
The pending phase is representable without assuming that the generic format
knows RLVR's cursor or collection semantics. Dense BF16 state round-trips, and
subsequent caller mutation does not alter the saved bytes.

Inspection never calls the tensor deserializer. Loading calls
`torch.load(..., weights_only=True, map_location='cpu')` on the verified byte
buffer. One controlled test replaces the filesystem path after verification
but before deserialization; loading still reads the original verified buffer,
while a subsequent inspection rejects the changed path. This addresses that
specific path-reopening race, not all filesystem attacks.

## Failed saves remain observable

Each injected failure preserves a previous successful data file and marker
byte-for-byte. Existing data or marker paths are never overwritten. A writer
failure retains partial bytes; failures in data fsync, temporary-header fsync,
the guard or atomic publication retain their created evidence without claiming
a new commit. A hard serialized-size bound does not publish a successful marker.

Directory-fsync failures occur after marker publication in two tested cases.
The save raises an error, but the marker may already be readable. The tests
explicitly retain and verify that state instead of conflating visibility with
confirmed durability or deleting it. The first directory-fsync failure leaves
the temporary header too; the second occurs after its removal. The prior
checkpoint remains unchanged in both cases.

## Environment and retained observations

Execution used the existing isolated Linux ARM64 CPU environment: Python 3.12.14,
PyTorch 2.14.1+cpu and one CPU thread, with CUDA visibility disabled and Hugging
Face offline flags. No fit, pretrained model, tokenizer, installation, service,
external execution or Git operation occurred. Resource peaks and actual
power-loss behavior were not measured.

The first panel had no test failures. PyTorch emitted a sparse-invariant warning
and a quantized-tensor deprecation warning while constructing rejection
fixtures; both remain in its raw output. The parent's earlier unsupported
pytest command is separately retained in
[its first-command record](2026-10-05-training-snapshot-first-command.json),
not silently reclassified as this panel's result.

The shared source SHA256 remained
`9cad2851b665b5fe5d640c1327710352396218ea80fab3b0c82f01925e1749a3`
before and after the first panel. The final verification binds source, tests,
specification, this report and raw command evidence, and rechecks historical
recovery artifacts against their previously retained byte identities.

## Remaining acceptance boundaries

This is a scoped pass for the trusted-local format and its declared failure
paths. It is not a hostile-pickle allocation sandbox, OS containment, total disk
quota, production crash proof or GPU/BF16 numerical replay. The loader buffers
the bounded serialized file and then restores tensors; file/storage bounds do
not establish a bound against arbitrary hostile tensor metadata.

Inspection validates actual bytes and the marker's schema, but cannot verify
its progress claims against an unparsed payload. Those checks occur during
loading, and runner-owned state semantics still belong to the callback.
Completed-update SFT/DPO replay and RLVR completed/pending replay require their
own actual-loop tests. RLVR must retain the original reference and bind pending
actions to the pre-update policy plus post-collection cursor/RNG, then consume
that collection once without drawing again. This format panel does not claim
that integration or approve any Spark pilot.
