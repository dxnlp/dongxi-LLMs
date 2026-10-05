# Checkpoint node bound verification

The pure IO supplier now refuses node envelopes above one million, matching the
shared consumer's declared ceiling. The [additive protocol](../specs/2026-10-05-snapshot-io-schedule-node-bound.md)
was saved before this change. The original protocol, 25-test measurement, report
and acceptance receipt remain unchanged historical evidence.

The new exclusive collection used the existing isolated CPU interpreter with
hidden CUDA, offline flags and one thread:

```bash
env PYTHONPATH=src:tests CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 /tmp/dongxi-course-reproduction.ZfVaEu/venv/bin/python tests/test_snapshot_io_schedule.py --collect experiments/reports/2026-10-05-snapshot-io-schedule/run-02
```

All 26 focused tests passed, including the original 960 independently enumerated
schedules and the new one-million/one-million-plus-one boundary. Actual child
exit was 0; unittest reported 0.211 seconds and the child duration was
0.261964774 seconds. The
[raw verification](2026-10-05-snapshot-io-schedule/run-02/verification.json) has
SHA256 `4278c0bbe2cf1e7c9f3c047164fe978917b953787c69ed715d48d504cd76d0dc`.

The current module is `6a1988d02651673d96f31407d7a088546a1c9bd435a111b22fe138f7fd94f1f8`
and tests are `6f7fa649c5cec8b6f9c9307fb2d8e482abe1114f92ac609f86c81f1b61a0cfcc`.
Both protocol files and both executable sources were unchanged during collection.
All four separately retained DPO v2 source/protocol/raw hashes were also unchanged.
The original primary and cadence-one requirement vectors are numerically
identical: this hardening rejects an impossible supplier envelope rather than
adding training capacity or changing a scientific recipe.

This remains a pure allowance calculation, not actual shared-reader admission,
numerical model replay, supplier authentication, a launch decision or physical
containment. The independent core review is separate; its initial failing
diagnostics are not hidden by this passing arithmetic result.
