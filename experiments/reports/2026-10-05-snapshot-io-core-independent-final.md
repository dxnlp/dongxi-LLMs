# Final independent checkpoint IO review

The final trusted-local shared core passed **58 focused tests**: 24 IO-core
controls, eight legacy snapshot controls and 26 pure schedule controls. All six
independently reproduced defects now refuse before the relevant read, finite
scan, tree walk, deserialization or new reservation. No substantive issue
remains within this declared source/CPU scope.

The actual final command was:

```bash
env PYTHONPATH=src:tests CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 /tmp/dongxi-course-reproduction.ZfVaEu/venv/bin/python -m unittest tests.test_snapshot_io_budget tests.test_training_snapshot tests.test_snapshot_io_schedule -v
```

It exited 0. The retained [test log](2026-10-05-snapshot-io-core-independent/run-04/tests.json)
reports 58 tests in 1.205 seconds; actual child duration was 1.812304093 seconds.
Eight source/protocol hashes were unchanged throughout this final collection.
The [six targeted rechecks](2026-10-05-snapshot-io-core-independent/run-04/diagnostics.json)
separately confirm both tensor-byte gates, receipt bool-mutation refusal,
header/payload/mode consistency, immutable scientific identity and retained
physical journal identity.

The owner repaired the source; this reviewer edited no shared implementation,
runner, canonical book or learner tracker. Initial failures remain in
[run-01](2026-10-05-snapshot-io-core-independent/run-01/diagnostics.json),
[run-02](2026-10-05-snapshot-io-core-independent/run-02/schema-diagnostic.json)
and [run-03](2026-10-05-snapshot-io-core-independent/run-03/identity-mutation-diagnostics.json).
The malformed load control explicitly patches deserializer output; it is not
evidence that the recorded real checkpoint was contaminated.

## Correct the reviewer count without rewriting evidence

The collector initially assumed the suite size instead of parsing the retained
unittest footer. Run-03 metadata says 58, but its actual log says 56. Run-04
metadata and the earlier review narrative/receipt say 60, but its actual log
says 58. Those mistaken records remain unchanged historical inspector evidence.
The [explicit count correction](2026-10-05-snapshot-io-core-independent-count-correction.json)
binds both actual log hashes and current method counts. This final report and
its separately named receipt supersede only the count interpretation; source,
raw results, control outcomes and hashes are unchanged. The correction required
no source change, fit, rerun or favorable selection.

## What the review establishes

Separate inspect/load/save reservations are admitted before shared contract/state
walking, payload processing, finite scans, cloning or serialization. Whole-op
counters span repeated phases. Exact byte receipts bind the scientific contract,
IO contract, pre-save prefix and same physical journal. Later failed spending is
retained, not refunded on restore or moved to a replacement journal. Explicit
unhooked version-1 references and accounted version-2 artifacts remain distinct.
The separate pure schedule preserves the nineteen DPO dimensions; its additive
one-million-node alignment passed 26 controls and 960 independent schedules.

These are cooperative logical visit units, not CPU instructions, FLOPs, hard
allocation/storage quotas or elapsed-time measurements. Payload hash bytes do
not include every metadata/journal hash. Bootstrap parsing, deserializer internals,
caller state capture, runner semantic application and arbitrary child/output
work remain separately scoped. The loader is still trusted-local, not a hostile
checkpoint sandbox or authentication mechanism.

Actual runner CLI ordering, fresh-process numerical model continuation,
pretrained/GPU/Mac/hosted CI execution and physical containment are not established
by these tests. Day 9 learner position and external execution status are unchanged.
