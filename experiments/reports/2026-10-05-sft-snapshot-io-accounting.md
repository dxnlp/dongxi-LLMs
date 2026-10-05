# Budgeted SFT checkpoint inspection and recovery

The actual SFT consumer now binds independently retained work receipts before
checkpoint inspection or model allocation. Separate snapshot I/O reservations
cover the shared payload hash, restricted-load/tree checks, finite scans,
cloning and serialization visits. The existing sixteen-dimensional SFT work
contract and all numerical equations remain unchanged. The current bounded CPU
reference passes 83 controls and exact replay in four fresh processes; this is
not pretrained, CUDA/BF16 or whole-job physical-containment evidence.

The [premeasurement specification](../specs/2026-10-05-sft-snapshot-io-accounting.md)
declares the recipe, allowances, independent receipts and excluded work. The
[exclusive verification record](2026-10-05-sft-snapshot-io-accounting/run-01/verification.json)
has SHA256 `6a75c7768247152de30bc5242890b7f743e683c0b07a6925ba7c093ac05029e4`.

## Two budgets and one selected checkpoint

The sixteen original SFT training, evaluation, generation and semantic-validation
caps stay exactly as declared in the previous CPU reference. The new
`dongxi-snapshot-io-work-v1` contract uses a separate physical journal with nine
dimensions: inspect/load/save operations, hash bytes, tree nodes, tensor
elements, primitive bytes, clone bytes and serialization bytes. Explicit
per-operation envelopes and cumulative caps are scientifically bound; no default
allowance, journal migration or old-checkpoint upgrade is implemented.

The production CLI requires `--snapshot-io-limits` containing the complete strict
contract and `--snapshot-io-ledger` naming its separate journal. Resume also
requires `--resume-io-receipt`, in addition to the retained scientific contract,
payload SHA/bytes and original logical-work journal. Bootstrap JSON is bounded
to 64KiB through nonblocking no-follow regular-file reads. Its exact parsed byte
identity is rechecked before later stages.

Both generic identity-hashing passes now exclude the resume payload, marker,
contract, receipt and cap metadata. Payload identity begins as an independently
declared expectation, not a measurement. After the current scientific contract
matches the retained contract, both journals open with the independently retained
prefixes, preserving all later charges. Only then may the shared accounted reader
inspect bytes. Successful inspection records an actual verified header separately
from the earlier expectation; it does not claim numerical semantics were loaded.

Shared load verifies the exact embedded I/O and logical-work prefixes against
the independent receipt before the SFT semantic callback or state application.
Each initial, periodic and final save publishes an exclusive `.work.json`
receipt beside its payload/marker. The carried I/O prefix precedes that save's
reservation; the live journal still contains the subsequent save charge. No
circular rewrite tries to place future save completion into the older payload.
Declared I/O science requires its matching hook on direct actual-loop APIs too;
only teaching references with no declared I/O contract may use legacy schema one.

## Fixed CPU recipe and exact replay

The fixture remains the original locally constructed random one-layer Qwen3:
vocabulary 32, width 16, intermediate width 32, two query and KV heads, head
dimension eight, context 64, tied embeddings and SDPA attention dropout 0.1.
Seeds 1212 and 1213 run full training and rank 2 LoRA on query/value projections.
Four updates, microbatch one, accumulation two, learning rate 0.003, zero weight
decay, clip one, actual activation checkpointing and CPU FP32 are unchanged.
Ragged assistant/end masks, one causal shift, EOS/padding alias, Adam, selector,
RNG and cached generation remain unchanged. No external model bytes are used.

The separate per-operation envelope is 16MiB payload, 100000 nodes, 1000000
tensor elements, 8MiB tensor bytes and 1MiB primitive bytes. Cumulative fixture
capacities are 16 inspections, 16 loads, 32 saves, 1GiB hash bytes, 6000000 nodes,
48000000 tensor elements, 64MiB primitive bytes, 256MiB clone bytes and 512MiB
serialization bytes, with a 2MiB journal. These are authored microscope
allowances, not estimates or approval for a pretrained job.

The focused panel exited zero: 83 tests in 40.611 seconds reported by unittest,
41.428 seconds observed by the parent subprocess. It contains 19 new controls
and 64 earlier recovery, semantic-budget, work-budget and CLI contract controls.
Execution used Linux ARM64, Python 3.12.14, Torch 2.14.1+cpu, Transformers 5.18.0,
Tokenizers 0.23.2 and PEFT 0.20.0, offline with CUDA hidden and one CPU thread.

Each arm commits update two and then encounters an authored restricted-load
failure after its complete accounted hash. A new process opens the same two
journals, inspects and restores the selected checkpoint, and finishes four in
a new output directory. All four child exits are zero. Policy including frozen
LoRA base weights, Adam, cursor/order, Python/Torch RNG and history match
uninterrupted execution exactly; separate original-equation controls also pass.
Numerical digests match the earlier semantic-only CPU reference, not just a
new self-comparison.

| Seed | Mode | Exact fresh replay | Reserved loads | Successful loads | Retained failed ticket | Known partial hash bytes from failed load |
| --- | --- | --- | --- | --- | --- | --- |
| 1212 | full | yes | 2 | 1 | 5 | 76161 |
| 1212 | LoRA | yes | 2 | 1 | 5 | 47302 |
| 1213 | full | yes | 2 | 1 | 5 | 76225 |
| 1213 | LoRA | yes | 2 | 1 | 5 | 47366 |

Every resumed arm retains one inspection and four successful saves, including
the repeated restored-boundary save. Its logical journal records five successful
semantic validations separately. Conservative unused reservation is not
refunded. A failed load records known successful hash/tree visits and an entered
but unsuccessful load, rather than pretending the load completed. A separate
early guard failure records no new hash visits while retaining full admission.

## Negative controls and actual CLI ordering

Zero inspect/load/save allowance refuses before shared contract walks, payload
reads, `torch.load`, tree/finite inspection or clone/serialization. Save's earlier
caller capture and separately charged semantic validation are not mislabeled as
part of that shared hook. Repeated actual inspect/load calls charge separately.
Copied journals cannot reset physical receipt identity. A different known logical
prefix fails exact payload/receipt comparison before semantics/application.
Serializer failure retains a partial file and spending while preserving the prior
snapshot and receipt; baseline-observer failure preserves the initial commit.

Four persisted [CLI controls](2026-10-05-sft-snapshot-io-accounting/run-01/cli-exhausted/control.json)
exercise actual source order with explicitly mocked hardware/parent resolution,
original local CPU tokenizer/model fixtures and no pretrained/GPU allocation.
Exhausted inspection opens two already-bound journals, reaches neither payload
reading nor model loading, and excludes payload/bootstrap paths from both
generic identity passes. Successful shared inspection records the verified
header, then reaches one mocked model-loader sentinel without allocating a
model. Over-horizon receipt and postparse no-peer FIFO metadata controls call
neither tokenizer nor model nor journal open. Their original metadata, raw
errors, journal summaries and identity records remain retained.

Generation remains negative where previously negative. For seed 1212 full mode,
development NLL changes from 3.497850 to 3.355714, but both final four-token
outputs remain capped, fail exact match and repeat `assistant`. No story-quality
or general-capability improvement follows from this accounting result.

## Identity and remaining boundaries

Current runner SHA256 is
`f8b51cda52e205c457cef34001e3165ec9f7ad381cb84d8e76a4dd1467357c40`;
new test/collector SHA256 is
`55debbb3367c0fd9df14e124d7d22e69dc21c33568f78c8a9e8c19eee6f0cc97`;
specification SHA256 is
`48b59788da5861424d5b1d0be016358576e6d643dbf145f2bcb178f6202089c8`.
All fourteen exact before/after source bindings matched. A separate read-only
post-collection check verified all fourteen current sources and 201 retained
artifact hashes with zero mismatches. Actual subprocess commands/exits, all
metrics/generated IDs, both journals, independent receipts, failed loads and
numerical digests are retained under the verification directory. Root-owned
legacy CLI scaffolding follows its
[predeclared compatibility protocol](../specs/2026-10-05-snapshot-io-cli-fixture-compatibility.md).
Earlier reports and their raw bytes remain historical and unchanged.

The shared hook counts declared visits, not every CPU instruction, wall time,
physical peak memory or hostile-pickle isolation. Bounded bootstrap metadata,
journal integrity, inventory and current contract construction, caller history
capture, state application and general library overhead remain explicitly
excluded. Semantic validation is charged separately by the existing sixteen
dimensions. Snapshot I/O accounting is not SFT artifact-path reservation or a
hard disk quota; logs, HF export and scratch routing remain separate gates.
Trusted-local artifacts, live production cap mapping/authority/backend checks,
physical containment, story work and pretrained/CUDA/BF16 profiling still need
their own evidence. No installation, acquisition, service, Git mutation,
publication or learner advancement occurred.

Reproduce only in the existing isolated offline CPU environment, using a new
exclusive destination:

```bash
PYTHONPATH=src:tests CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 /tmp/dongxi-course-reproduction.ZfVaEu/venv/bin/python tests/test_sft_snapshot_io_accounting.py --collect experiments/reports/2026-10-05-sft-snapshot-io-accounting/run-02
```

The displayed command is a CPU evidence reproduction, not production permission.
