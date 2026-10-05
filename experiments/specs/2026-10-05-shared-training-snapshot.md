# Shared snapshot format: bounded CPU acceptance

This source check precedes SFT/DPO runner replay. It does not authorize model
acquisition, pretrained inference, Spark training, services or Git operations.
Only original tensors and temporary local files are used. Python runs in the
isolated CPU reproduction environment, with CUDA disabled and no network use.

The version1 format freezes a finite primitive/dense-tensor state, records its
scientific contract separately from the invocation, and writes an exclusive
payload followed by an atomic exclusive JSON commit marker. Both files and the
directory are fsynced before successful return. Failed partial files remain
diagnostic evidence. Existing payloads/markers are never overwritten. A failure
after marker publication can leave a readable marker with unconfirmed durability;
the caller records the failure instead of treating it as successful closure.

Read requires a caller-retained independent SHA256, exact byte size, expected
scientific contract and explicit positive file-size envelope. The marker is not
an independent source of those expectations. Header schema/size/contract and
actual payload bytes are checked before restricted `weights_only=True` tensor
loading. Loading uses the same verified in-memory byte buffer, not a re-opened
path. Runner callbacks validate actual counters, cursor, parameter/reference
layout, metric history and pending pools before applying anything.

Acceptance uses scalar, empty, noncontiguous and BF16 tensors, Adam-like integer
keys, CPU RNG and completed/pending envelopes. Check lossless data-only roundtrip,
no mutation of caller tensors, guarded operation, strict schema and independent
contract changes, immutable repeated saves and preload rejection with a loader
spy. A separate reviewer exercises failure paths before integration closes.
Fixture files are capped at16MiB; oversized saves keep uncommitted partial bytes.

This is a trusted-local corruption/replay boundary, not authentication, a hostile
pickle/tensor-allocation sandbox, process containment or an overall disk quota.
Memory includes the declared file buffer and restored tensor storage. A production
caller must explicitly budget both and supply resource checks; the CPU bound is
not a production sizing recommendation. No BF16 training or GPU replay is claimed.
