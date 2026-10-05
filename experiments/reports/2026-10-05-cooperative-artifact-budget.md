# Cooperative artifact reservation verification

A new ledger reserves logical pathname bytes and file entries before cooperative
artifact creation. The optional `save_snapshot` hook now reserves payload,
commit marker and header staging together, including coexistence with older
snapshots. Current CPU checks pass 71 tests, including an actual fresh-process
restore and tensor snapshot. This is a bounded writer mechanism, not a physical
filesystem quota or proof that a complete training job's outputs are bounded.

The [original specification](../specs/2026-10-05-cooperative-artifact-budget.md)
and [FIFO follow-up protocol](../specs/2026-10-05-cooperative-artifact-budget-fifo.md)
precede their measurements. No quota, cgroup, service, model download, pretrained
model, GPU, installation, runner recipe, book tracker or Git operation changed.

## API and protected scope

`ArtifactBudget.create` exclusively creates one private owned root, with fixed
campaign identity, root device/inode, unique ledger ID, maximum bytes/entries and
journal capacity. `ArtifactBudget.restore(root, expected_receipt=...)` requires
an independently retained identity and journal-prefix receipt. It validates and
replays the entire current journal, including activity after that retained
prefix; it does not truncate or reset counters to the prefix.

`reserve_bundle(operation, {name: capacity, ...})` admits the whole bundle or
refuses it before an artifact opens. Its returned reservation supplies bounded
exclusive writers, hard-link publication, sealing and exact redundant-staging
removal. A successful seal accounts actual file length and digest. A failure
keeps conservative capacity, including failed partials and members not yet
created. Explicit absent release cannot free an existing partial file. A removed
staging link is released only after the exact redundant link is unlinked and
absence is verified; retained payloads are never silently deleted.

The root permits direct file names only. Nested and outside paths, traversal,
symlink ancestors/targets, root or journal replacement, unsafe ownership/mode,
external hard links, special files, unexpected entries and changed sealed bytes
fail closed. Nonblocking opens reject FIFOs before reading or waiting for a peer.
An advisory lock excludes another ledger handle, but does not authenticate
hostile same-UID writers. Identity and file-state views are copies, so ordinary
caller mutation cannot change frozen capacities. Live journal-byte checks detect
same-inode changes outside the ledger writer.

The journal is append-only, hash-chained and fsynced before a reserved file opens.
Its entire fixed capacity and one file entry are charged upfront. The writer
refuses journal exhaustion instead of growing metadata beyond the declared
envelope. Incomplete or corrupted tails are retained and refused, not repaired
automatically. A retained receipt detects rollback before its supplied prefix;
it cannot authenticate an independently supplied stale expectation or turn
unrelated new campaign ledgers into a global quota.

## Snapshot integration

`save_snapshot(..., artifact_budget=budget)` is optional and backward compatible.
It reserves `max_bytes` for the payload, `HEADER_LIMIT` for the final marker and
another `HEADER_LIMIT` for staging before the first artifact open. Marker and
staging names count separately during hard-link coexistence even though they
share physical storage. Header/payload schemas and restricted load/inspect
contracts are unchanged. Actual no-hook and hooked tensor saves produce identical
payload bytes and headers in this fixture.

The budget receipt remains separate from the checkpoint header. A future caller
must retain it independently and bind its identity/caps to its scientific
workflow. SFT/DPO/RLVR runners, reports, logs, HF exports and children do not
currently use this hook. They remain unprotected by this writer. Requiring a
physical aggregate-quota backend explicitly refuses because no such backend was
provisioned or verified.

## Measured controls

The final command used the existing isolated CPU interpreter, offline flags,
CUDA hidden and one thread:

```sh
env PYTHONPATH=src CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/dongxi-course-reproduction.ZfVaEu/venv/bin/python tests/test_snapshot_artifact_budget.py --verify-tests experiments/reports/2026-10-05-cooperative-artifact-budget-fifo-final.json
```

The [final command receipt](2026-10-05-cooperative-artifact-budget-fifo-final.json)
records actual exit 0, 71 tests in 4.768 seconds, subprocess-wrapper duration
5.415281 seconds and seven unchanged before/after source/spec/test/lock hashes.
Python was 3.12.14 and Torch 2.14.1+cpu on ARM64 Linux; CUDA was unavailable.
The panel contains 23 ledger tests, nine actual snapshot integration tests and
all 39 existing shared snapshot checks. Its
[raw observations](2026-10-05-cooperative-artifact-budget-fifo-final.observations.json)
retain commands, actual child exits, byte/entry measurements and refusal cases.

| Original CPU control | Observation |
| --- | --- |
| First snapshot's worst-case reservation | 167,936 bytes: 32,768 journal, 4,096 payload and two 65,536 header capacities; four entries |
| Crossing first bundle | Refused at 167,935 bytes or three entries, before serialization or file creation |
| Exact-envelope tensor snapshot | 1,833 payload bytes; after staging removal, 34,865 reserved bytes and three entries |
| Old plus next snapshot and staging reservation | Peak 170,033 reserved bytes and six entries; completed artifacts then retain 36,962 bytes and five entries |
| Serializer failure after an older snapshot | 16 physical partial bytes retained; 170,033 reserved bytes and six entries remain charged; another full snapshot with a new name is refused |
| Fresh-process restore from an earlier receipt | A later failed file still reserves 256 bytes despite containing eight bytes; the child saves and loads an exact new six-value tensor snapshot |

The pure byte control admits a 24-byte bundle and refuses its 25th byte before
creation. A crossing stream writes eight bytes, refuses the next nine against a
16-byte capacity and retains all 16 reserved bytes. Entry caps include empty
files. Buffered `SEEK_END` is checked against the flushed true end before moving.

Additional controls cover changed caps/campaign/root, rollback, hash/JSON/tail
corruption, public-view mutation, simultaneous ledger handles, unsafe artifact
paths and inventory, sealed content changes, absent release, journal capacity,
artifact and journal fsync failures and marker publication failure. Failed fsync
can leave visible bytes that later replay conservatively accounts; this proves
observed retention, not crash durability of the failed fsync itself. Old snapshot
bytes and markers remain unchanged in failure cases. Temporary test directories
are cleaned after checks; raw reports preserve observed receipts, not reusable
checkpoint files at those temporary paths.

Four no-peer FIFO child processes test journal, digest target, snapshot header
and snapshot payload refusal under separate three-second external bounds. They
return explicit errors without deserialization or a timeout. This is input-type
refusal, not arbitrary process containment.

## Historical results and the retained failure

The [first collection](2026-10-05-cooperative-artifact-budget-first.json) passed
67 tests. The [identity and live-journal hardening collection](2026-10-05-cooperative-artifact-budget-final.json)
passed 69. Those source identities and raw observations remain unchanged.

Source review then identified the FIFO open-before-stat gap and saved a new
protocol. The [first FIFO follow-up](2026-10-05-cooperative-artifact-budget-fifo-hardening.json)
ran 71 tests and exited 1: the digest helper refused the FIFO promptly with
`Unsafe or over-cap artifact before hashing`, while the test asserted the word
`regular`. The failure was a diagnostic-wording oracle mismatch, not a timeout
or admitted FIFO. The role-specific assertion was corrected and child receipts
are now captured before assertions. The [separate final FIFO collection](2026-10-05-cooperative-artifact-budget-fifo-final.json)
passes all 71. Neither failed evidence nor earlier source hashes were rewritten.

## Remaining production gates

Acceptance is scoped to one synchronous cooperative ledger owner and outputs
written through its explicit reservation paths. Thread-coordinated writes through
one shared object are not an acceptance claim. A file's reserved logical bytes
are not ext4 allocated blocks, and a hard-linked header is intentionally counted
twice by pathname. Fixed metadata capacity may conservatively block a small
artifact; this is intentional, not evidence that disk space is exhausted.

Physical aggregate quotas, arbitrary export/child containment, complete runner
artifact coverage, model-scale save overhead and an actual pretrained Spark run
remain pending. Creating a different unrelated budget does not enforce a global
per-user cap. No external campaign outcome or learner mastery changed. Exact
current and historical artifact identities are bound in the
[verification receipt](2026-10-05-cooperative-artifact-budget-verification.json).
