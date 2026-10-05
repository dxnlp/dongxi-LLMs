# Budget checkpoint inspection before reading it

The original dependency and acceptance requirements below now have an optional
separate shared I/O hook, an actual Day25 tiny reader microscope and independent
CPU controls. Actual SFT/DPO/RLVR runner/CLI integration is tracked in the
[native-RLVR source record](../experiments/reports/2026-10-05-rlvr-io-readiness.md)
with original-equation fresh-process replay. Native RLVR keeps its historical
pre-semantic saved23 prefix, binds it and the I/O prefix before inspection,
and retains every later failed charge. Day25 Exercise8 compares native phases.
This is not an approved production run or a physical
quota. Runner callbacks still validate semantics after shared generic checks;
their reservations cannot retroactively charge that earlier work.

## Observed reader order

In [the shared reader](../src/dongxi_llms/training_snapshot.py),
`inspect_snapshot` checks the expected contract, bounded marker and every payload
byte without deserializing. `load_snapshot` repeats byte verification, restricts
`torch.load` to weights, checks the payload/contract and scans its state with
`_safe_tree` before calling the runner's semantic validator. `_safe_tree` checks
finite dense tensors and a bounded container tree. The writer separately clones
the state before serialization. These are actual source paths, not estimates of
their elapsed time or GPU memory. Explicit envelopes do not sandbox hostile
tensor metadata; trusted local artifacts remain a requirement.

## The prefix dependency

An existing work journal opens unbound for resume. It cannot reserve until a
trusted snapshot prefix is validated. If that prefix exists only inside the
checkpoint, the reader must already deserialize the checkpoint to obtain it.
A post-read callback therefore cannot be the sole authority for pre-read work.
Do not bypass this safeguard or treat opening a fresh journal as a solution.

The source integration requires a separately retained, bounded work
receipt/prefix for pre-read binding, matching the same physical journal and
scientific contract. Bind it before any full payload hashing or tensor load.
Preserve later failed charges. A payload must still agree with this independently
retained expectation before numerical state application. Copying receipt files
does not establish a cross-machine budget or authenticate an adversarial writer.

## Implementation and verification requirements

Give inspect/load/save distinct declared work units and reserve complete envelopes
before their expensive phases. Keep input byte hashing, restricted load, finite
tree inspection, clone/serialization and runner semantics separately observable;
do not call those units FLOPs, wall time or a hard memory quota. Optional teaching
APIs must not create a production bypass. Schema changes need explicit new
contracts, not an automatic extension of old caps or refill.

Test exhausted allowance before byte hashing, `torch.load`, finite scans and
cloning; malformed or changed independent prefix; a copied/replaced journal;
later failed work; repeated inspect/load calls; save failures with retained bytes;
and actual runner CLI ordering before model allocation. Use original local tiny
models and fresh-process numerical replay, keeping objective/masks/RNG/history
unchanged. Match declared stage schedules to the actual calls. Keep the existing
reader's identity, restricted-type, no-follow, finite and durable-publication
checks intact.

## Separate source contract

The shared hook uses explicit `dongxi-snapshot-io-work-v1` and nine dimensions,
not a silent extension of SFT16/DPO19/RLVR23. Its whole-operation counters span all
contract/state visits; tensor elements and bytes are checked before finite scans
and cloning. An independently retained <=64KiB receipt uses regular no-follow
bootstrap reads, pins both prefixes and requires explicit accounted payloadv2.
Unhooked teaching references remainv1; changing a marker's schema does not
migrate its payload. The public inspected receipt/contract/science/journal binding
cannot be mutated into a new allowance.

Save completion is later journal history. The checkpoint and retained receipt
carry the exact pre-save I/O prefix, and recovery keeps all later completed,
failed and uncertain reservations. The optional artifact writer also reserves
the external work-receipt entry. An I/O hook does not charge caller state capture,
artifact inventories or generic identity reads; runners must remove the latter's
pre-admission payload path and prove the actual ordering.

The separate pure [schedule calculator](../src/dongxi_llms/snapshot_io_schedule.py)
keeps commit cursors, diagnostic loads and complete-attempt capacity explicit.
It grants neither runtime admission nor retry authority. Metadata/journal
processing, application, deserializer internals, other outputs and physical
resources remain named boundaries rather than inferred coverage.

Native RLVR has a different publication schedule from that completed-only
calculator: initial, every pending pool and every completed update. Chapter14
answer33 derives fresh/completed-resume/pending-resume save counts and keeps
them separate from the manual one-boundary lesson's actual operations. Exact
recovery of the original tiny0/2 held-out result does not establish learning.
Story persistent model accounting is now CPU verified in its separate original
layout by the [current acceptance](../experiments/reports/2026-10-05-story-work-readiness.md),
not a shared I/O9 migration. Next safe source work is story byte-I/O/validation/
output admission, live/other-stage adapters,
SFT/RLVR snapshot artifact routing and broader logs/exports/scratch reservation.

This source task is within the active improvement goal. Actual physical quotas,
containment, hostile-input isolation, pretrained/CUDA/BF16 cost profiles, Mac/CI
execution and model campaigns require their own authority and evidence. No such
action is authorized or completed by this plan. Follow
[the supervisor plan](PRODUCTION_SUPERVISOR_PLAN.md) for the remaining output and
production boundaries.
