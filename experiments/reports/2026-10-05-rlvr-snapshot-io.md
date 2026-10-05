# RLVR shared snapshot-I/O admission: bounded CPU evidence

The frozen original local-random Qwen RLVR loop now supports the separate
shared snapshot-I/O hook. The reference panel passed; completed and pending
restarts preserve their exact original numerical trajectories and retain later
failed spending. This is a CPU source/recovery result, not arithmetic capability,
pretrained recovery, physical containment or production authorization.

The protocol was declared before the new measurements:
[implementation specification](../specs/2026-10-05-rlvr-snapshot-io-admission.md),
[CPU control protocol](2026-10-05-rlvr-snapshot-io-cpu-protocol.md).
The original actor and old numerical helpers were archived before implementation
started; [their manifest](2026-10-05-rlvr-snapshot-io-preintegration/manifest.json)
pins original actor SHA `819cd8bbfc1701416d050b14888cbc11d4128e143cb7c80266da9678b3f27df1`.
All older reports remain unchanged.

## Current measured result

[Exclusive run-01](2026-10-05-rlvr-snapshot-io/run-01/verification.json) passed in
72.763 seconds. Its SHA-256 is
`f16126a598247adf53a931469dedddd824b406fd98b61369a0c3897a6dfc613a`.
It binds 27 unchanged before/after source/input hashes, corresponding actual-byte
archives, and 440 raw artifact paths/hashes. Independent read-back found no
issue in those 440 artifact hashes, 27 current sources, 27 source archives,
88 committed-marker/payload byte bindings and 88 external-receipt/marker
bindings. A failed staging header is not counted as a committed marker.

Actual component footers, not inferred test counts:

| Panel | Actual tests | Unittest seconds | Actual exit |
| --- | ---: | ---: | ---: |
| [New RLVR I/O controls](2026-10-05-rlvr-snapshot-io/run-01/new-focused.log) | 21 | 23.535 | 0 |
| [Existing RLVR/shared regressions](2026-10-05-rlvr-snapshot-io/run-01/existing-shared.log) | 102 | 20.564 | 0 |
| [Reader-lesson controls](2026-10-05-rlvr-snapshot-io/run-01/reader-lesson.log) | 6 | 4.342 | 0 |

These components overlap the full course suite; do not add them to the global
test count. This collector did not execute a notebook. The separately measured
[reader-lesson run-02](2026-10-05-rlvr-reader-lesson/run-02/verification.json)
owns its fresh-kernel/figure evidence.

Current executable identities are:

- Actor `qwen_rlvr_lab.py`: `f32f41c77c7efe14b675d9a2e8371398e368e0ccdc1c376f787a40f5602636d7`.
- New test module: `bcc9e26fff274e86995c52b848b3910a43fb2e1ac486f169e1ff3fb0d9d4e84a`.
- Exclusive collector: `33fcc6cd749bd3f5360ca141363891c41623b558cc6ddd83ad0e4f3dea6ecf85`.

Observed environment: Python 3.12.14, Linux aarch64, Torch 2.14.1+cpu,
Transformers 5.18.0, tokenizers 0.23.2, PEFT 0.20.0; one CPU thread, CUDA hidden,
HF offline. The minimum of the recorded before/after child host-memory samples
was 116.264 GiB, above the declared 25 GiB reserve. These samples are not
continuous monitoring or an enforced physical memory quota.

## Original science and exact replay

No favorable seed, response, checkpoint or cap was selected. Both original
seeds 2323/2324 retain the one-layer random Qwen3, vocabulary/width 16,
intermediate width 32, query/KV heads 2/1, head dimension 8, context 32,
dropout 0, untied embeddings, float32/SDPA, EOS/pad 0 and stop IDs 0/7.
The original three authored token records and integer decoder are unchanged.
Group size 3, response cap 4, four updates, temperature 1/full support,
population group advantages, beta 0.02 and AdamW learning rate 0.008/weight
decay 0 stay unchanged. The actor still uses its original uncached operations.

All 23 original logical/semantic work caps remain 100000 each, with the original
1 MiB journal. The separate strict I/O contract has nine dimensions and the
explicit microscope limits in the CPU protocol; no original cap is expanded
or silently migrated. Numerical parity compares the hooked current loop,
unhooked current teaching loop and archived original actor, retaining
policy/reference/Adam, all RNG, cursor, sampled IDs/stops/rewards/advantages,
complete history and numerical counters. Only resource receipts are excluded
from that numerical comparison; their histories intentionally differ.

Each fresh child first reopens the same two physical journals from the
independent receipt and inspects the payload, then constructs the original
random model, compares its full current science contract and restores. The
actual native uninterrupted lifecycle commits nine boundaries: completed0,
then pending/completed for every update. Interruption is injected only after
completed2 or pending2 is durably delivered. Original baseline/final evaluation
and their RNG boundaries remain in the native lifecycle.

| Seed / retained phase | Fresh exit | New collection cursors | Fresh save boundaries | Final numerical SHA-256 |
| --- | ---: | --- | ---: | --- |
| [2323 / completed2](2026-10-05-rlvr-snapshot-io/run-01/2323-completed/arm.json) | 0 | 2, 3 | 5 | `8c54b97225b6a137a5161af0c1d4ddcff6701a7b901c2af5c9d11f21144f0fa1` |
| [2323 / pending2](2026-10-05-rlvr-snapshot-io/run-01/2323-pending/arm.json) | 0 | 3 | 4 | `8c54b97225b6a137a5161af0c1d4ddcff6701a7b901c2af5c9d11f21144f0fa1` |
| [2324 / completed2](2026-10-05-rlvr-snapshot-io/run-01/2324-completed/arm.json) | 0 | 2, 3 | 5 | `6c02bf9f251ef12445156c3b0632fad2f35b83c1acc2635fd23b28abaf62b804` |
| [2324 / pending2](2026-10-05-rlvr-snapshot-io/run-01/2324-pending/arm.json) | 0 | 3 | 4 | `6c02bf9f251ef12445156c3b0632fad2f35b83c1acc2635fd23b28abaf62b804` |

All four exactly match uninterrupted final state and complete two-update tail.
Seed 2323 tail SHA is `cf21286dd2710028879d6c27ddb1a223f92efb11480a0e53fc258c0200f7d05d`;
seed 2324 tail SHA is `5dce10495dafbdf8463b2967e5ea75ee0696c400b24d2845e9a412ca776bdbca`.
A pending restart first recommits its retained pool, applies it without
collection, then collects only at cursor 3. Raw fresh child exit/stdout/stderr,
first action, original pending pool digest, full history, evaluations, snapshots
and journals are retained in each arm, not only the final digests.

The quality failures remain visible. Final held-out greedy accuracy is 0/2 for
both seeds, as is the initial evaluation. Seed 2323 has zero reward in all four
training groups. Seed 2324 has zero reward in its first three groups and one
positive response in the final group; the other two final responses are wrong.
Original EOS/invalid-text/token-limit outcomes are preserved. Exact recovery
does not establish learning to solve arithmetic.

## Two carried prefixes, no refunded suffix

Keep the original state-capture convention. The saved runner23 prefix precedes
save's semantic validation; the external receipt copies that exact older prefix.
The saved I/O prefix precedes shared-save reservation. Neither is refreshed to
erase, compress or mislabel later work. Shared load checks both carried prefixes
against the independent receipt before the one native RLVR semantic callback.

Each arm injects an admitted restricted-load failure and a separately admitted
semantic RNG-layout failure after its retained boundary. For completed2, the
carried runner/I/O sequences are 18/8 and the post-failure physical sequences
are 22/12. For pending2 they are 22/10 and 26/14. After native restore and finish,
all four physical sequences are runner 46 / I/O 26, with the original failed
tickets still present. Completed arms retain runner ticket 21 / I/O ticket 11;
pending arms retain runner ticket 25 / I/O ticket 13. Reopening from the old
receipt does not refill either allowance. Copied journals and changed declared
caps/science are separately refused.

## Admission, bootstrap and publication controls

The new controls check omission/mismatch before caller state capture, semantic
validation, shared save/load and lifecycle snapshot-directory creation; strict
missing/extra/boolean caps; independently changed expected bytes/hash; copied
physical journals; and v1 refusal in explicit accounted mode. The explicit
unhooked schema-v1 teaching API remains usable without a fabricated receipt.
Zero shared-save allowance blocks shared tree/finite/clone/serialization, but
the original separately accounted caller-semantic validation has already run.
This is not a blanket claim that no caller work occurs before shared admission.

Ten retained actual native CLI controls use mocked platform/parent/interface
providers, an authored parent-config placeholder and a model-loader sentinel:

- Exhausted inspect: two generic identity passes, two journal opens, zero
  payload-verifier calls and zero model-provider calls.
- Admitted inspect: two identity passes, real journals, one actual payload
  verification and its measured identity, then one sentinel provider call;
  no model weights are loaded. Expectations and measured header are separate.
- Unknown phase, cursor above update4, or terminal pending4: zero identity,
  tokenizer, journal, payload-verifier and model-provider calls; no output.
- Direct and hardlink template/environment-lock aliases plus a parent-artifact
  hardlink: the same zero-call bootstrap refusal. Missing payload metadata is
  not silently hashed by the alias guard.

Both generic identity passes exclude payload/marker/contracts/work receipts/cap
files; bounded bootstrap identities are separate from the admitted payload
hash. Core metadata FIFO/duplicate/size controls are covered by the shared
regression panel, not claimed as additional unmocked production CLI trials.

[Partial serialization](2026-10-05-rlvr-snapshot-io/run-01/partial-publication/partial-publication.json)
retains the actual 8-byte partial file, unchanged old committed boundary, no
marker/receipt, known partial work and failed ticket 3. The separate
[native atomic-publication failure](2026-10-05-rlvr-snapshot-io/run-01/native-publication-failure/publication-control.json)
allows completed0 to commit, then refuses pending0's header link. Its fully
serialized 80308-byte payload and fsynced staging header remain; no pending
marker, work receipt or `on_commit` delivery occurs. The old payload/header/
receipt/pointer and hook's last successful receipt remain unchanged. Both the
original pending collection and failed shared-save reservation remain charged.
These intentionally injected failures are not reported as production incidents.

## Preserved development records

- [Development-01](2026-10-05-rlvr-snapshot-io-development-01.json): 15 tests,
  exit 1, four test-setup errors (wrong private verifier spy name and wrong
  I/O-prefix key). The actual errors are retained, not presented as a pass.
- [Development-02](2026-10-05-rlvr-snapshot-io-development-02.json): 17 tests,
  exit 1. One fresh contract correctly refused concurrent actor source change;
  two new CLI mocks lacked the real parent path and failed identity validation.
  The source drift and errors remain; no comparison gate was waived.
- [Development-03](2026-10-05-rlvr-snapshot-io-development-03.json): 20 tests,
  exit 0, stable actor/test hashes. This is historical before the additional
  publication control. Peer review identified that partial serialization alone
  did not prove publication failure; the protocol clarification preceded its
  new targeted test and final 21-test collection, without changing science/caps.

The actor owner's [implementation notes](2026-10-05-rlvr-snapshot-io-implementation-notes.md)
separately retain mistaken legacy test selectors and old scaffold diagnostics.
Those older selector/scaffold outcomes are not current alias-guard proof.

## Reproduction and remaining boundaries

Use a new exclusive directory, never overwrite this run:

```bash
env PYTHONPATH=src:tests CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 /tmp/dongxi-course-reproduction.ZfVaEu/venv/bin/python \
  experiments/reports/collect_rlvr_snapshot_io.py \
  experiments/reports/2026-10-05-rlvr-snapshot-io/run-02
```

The nine shared units count declared visits/bytes and whole-phase reservations,
not all CPU instructions, wall time, RAM/storage/GPU work or hostile-payload
sandboxing. Caller state capture/history copying, later state application,
metadata/identity/journal operations, artifact inventory/export/log/scratch and
deserializer internals remain outside these units. This panel does not exercise
RLVR cooperative artifact routing, production output containment, external
watchdogs, actual model suppliers or authenticated launch approval.

No pretrained/Spark GPU/BF16/CUDA, actual Mac/hosted/cross-machine recovery,
acquisition, installation or model-scale job was run. All 45 external campaign
outcomes remain null, with no new checkpoint genealogy; learner progress stays
Day 9 and the existing bounded-package count stays 13 of 18. Root's separate
full-course test/kernel/math/route panel is the integration gate, not an
additional arithmetic or production-readiness claim.
