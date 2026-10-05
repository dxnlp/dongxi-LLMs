# Independent review: native-campaign archive-size/gzip correction

Review date: 2026-10-05. Scope is the frozen CPU archiver source, precise ignore
rule, author audit and focused authored controls. Native model jobs were active
elsewhere; this review did not run the actual campaign assembler.

## Finding

**CLEAR at the source/authored-CPU-control boundary.** No blocking discrepancy
was found in the independent archive ceiling, exclusive bounded raw/gzip
writers, byte-closing bindings, deterministic-compression qualification,
retained-failure behavior or narrow ignore rule. All45 focused controls passed
independently with actual exit0.

This does **not** establish that the real campaign snapshot fits under1GiB,
compresses to a Git-host-compatible size, has stable final inputs, or has passed
actual closing verification. No real archive was produced by this review.
The author audit's847 paths/352,827,895bytes and missing-input count are prior
stat-only observations, not a new final size measurement or a formal lower
bound repeated by this reviewer.

## Frozen inputs and independently observed controls

The author audit is
[the archive-size audit](2026-10-05-native-campaign-archive-size-audit.md).
The independently reviewed source/operational SHA-256 values match its frozen
panel and remained identical after the focused run:

| File | SHA-256 |
| --- | --- |
| `scripts/assemble_native_campaign_evidence.py` | `4b7bd4188c8d0d8612b815fe44fa4b26c6e5ad8c16cb3dd22903d7c53926c55b` |
| `tests/test_native_campaign_evidence.py` | `7b7530572baeae504fed756fb1b2ba190828910370bbfae00f8a04f4249b4f51` |
| `src/dongxi_llms/staged_campaign.py` | `4744cc5f399dea11a2dc05a61f29378e80e777d9a4b189848f5f8c3b636873f1` |
| `src/dongxi_llms/run_identity.py` | `a98d244efbfcbecf13ba97e28f512b2743642ac7d6e2ac8fad3a36eb5bcb4999` |
| `src/dongxi_llms/native_profile_supervisor.py` | `88100a0fbe33fa8de34155838485e6c31fb5c3fd436d1dc27725fe720ad4f9b2` |
| `src/dongxi_llms/campaign_supervisor.py` | `d2959a7856bc0cc4fb6b6623387d45bbc0d7193607a52c29c59bb8823b97d253` |
| `.gitignore` | `f459dc5711831cc5336dd739591cb9fe37a730dc404377e0273304094bd4a6c6` |

The independent command was:

```sh
env CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=src:tests \
  /tmp/dongxi-course-reproduction.ZfVaEu/venv/bin/python \
  -m unittest discover -s tests -p test_native_campaign_evidence.py
```

Actual exit was0; the actual unittest footer was:

```text
.............................................
----------------------------------------------------------------------
Ran 45 tests in 0.357s

OK
```

The actual interpreter reports Python3.12.14, Linux
`Linux-6.17.0-1031-nvidia-aarch64-with-glibc2.39`, and compile/runtime zlib1.3.2.
CUDA visibility was empty; both offline flags were1; all three thread-count
environment settings were1. No environment install or synchronization occurred.
These are45 authored CPU controls, including the33 prior controls and12 added
methods, not45 new model jobs or independent replications of author results.
The45 original planned campaign rows are a different count, preserved by the
controls rather than executed by them.

## Independent source checks

The changed output path is isolated from producer/input limits:

| Boundary | Frozen value and behavior |
| --- | --- |
| Individual interpreted JSON input |64MiB, unchanged. |
| Journal/general evidence input hashing |256MiB, unchanged. |
| Individual export file hashing |4GiB, unchanged; no model deserialization. |
| Closed evidence/directory count |6000, unchanged. |
| Raw assembled snapshot output |1GiB, separate from the input caps. |
| Compressed snapshot output |1GiB, separately enforced while writing. |
| Raw bytes admitted to gzip |1GiB, checked before output and while reading. |
| Decompressed snapshot |1GiB declared in closing provenance for consumers; no automatic extraction is performed. |

`_archive_limit` admits only literal integers from1 through the1GiB ceiling,
rejecting booleans, floats, strings, zero, negative or enlarged ceilings before
opening outputs (`scripts/assemble_native_campaign_evidence.py:60`).
`retain_snapshot` streams the original indented, UTF-8, nonfinite-refusing JSON
representation with its final newline included. Its binary exclusive writer
admits each chunk against the remaining cap **before writing**; incomplete
writes fail, and previously written prefixes are not removed or replaced
(`:68–89`). This is an output-byte envelope, not a RAM quota or claim that
every JSON encoder chunk has bounded allocation.

`compress_snapshot` uses an `O_RDONLY|O_NOFOLLOW|O_NONBLOCK` source descriptor,
requires a bounded regular source before creating the exclusive gzip target,
streams1MiB reads under the raw cap and wraps all compressed writes in the same
bounded writer. It closes the gzip stream, checks the count and source
descriptor/path identity, size, mtime and ctime, then flushes/fsyncs
(`:91–116`). Symlinks, directories, FIFOs, oversized raw input, source mutation
and same-byte replacement by another inode are refused in the focused controls.
The compressed ceiling counts headers/trailer too; an incompressible small
fixture can exceed the compressed cap even though its raw bytes fit.

Compression uses level9, `mtime=0` and an empty original filename. Authored
controls compare byte-identical compression under different target names,
check the filename-free/time-zero gzip header and decompress only tiny fixtures
for exact equality to the original raw bytes. The closing record retains actual
Python version plus compile/runtime zlib versions and qualifies determinism as
identical input in that recorded environment, **not across versions**. A new
assembly contains its own timestamps/elapsed values, so the source does not
promise that two independently assembled reports have identical bytes.

The1GiB decompression field is an explicit consumer ceiling, not an implemented
automatic extractor or physical decompression-memory guarantee. This archiver
creates gzip from an actually capped raw stream and never extracts a production
archive. A later consumer must enforce the declared raw/decompressed limit and
verify the closing bindings; the presence of a gzip file alone is not authority
to perform unbounded decompression.

## Closing and failure boundaries

The final assembly path creates a new private exclusive run directory, retains
all original45 stage rows and failure/missing evidence without compacting away
duplicate historical receipts, verifies input/source/interpreter stability,
streams the raw file, streams gzip and verifies inputs/sources again before
publishing closing metadata (`scripts/assemble_native_campaign_evidence.py:687–745`).

`closing-bindings.json` includes both actual complete stream bindings:
`snapshot_binding` and `snapshot_gzip_binding`, each with byte count and
SHA-256 under the1GiB ceiling. It also records all three output/decompression
ceilings, compression provenance, unchanged input/source bindings, directory
layouts and missing evidence. The authored small complete-assembly control
independently compares each closing binding to its actual raw/gzip bytes and
round-trips gzip to the raw stream. These fixture checks do not bind the not-yet
executed real archive.

Writes are exclusive at the run-directory, raw-file, gzip-file and receipt
levels. Existing output and failed prefixes are not overwritten. A raw limit,
nonfinite JSON value or gzip limit/read/identity failure preserves entered
prefixes. The small assembly compression-read fault retains the complete raw
snapshot with all45 original rows, the gzip prefix and `failure.json`, and
does not publish a success closing receipt. Earlier failures may have no raw
snapshot at all. A gzip prefix or malformed/partial JSON is therefore not
promoted to completed evidence. On an assembly failure, the source attempts an
exclusive failure receipt and re-raises; it neither repairs nor removes
producer/history files.

The exact new ignore rule is:

```text
/experiments/reports/native-campaign-evidence-*/snapshot.json
```

Full-file inspection and the focused literal-rule control find no broad
campaign-directory or `*.gz` rule. The specific raw snapshot is ignored;
`snapshot.json.gz`, `closing-bindings.json` and `failure.json` remain
versionable. No Git mutation/publication or hosting-size claim was made.

## What remains actual work

This correction changes only CPU archive-output handling and its scoped tests
and ignore rule. It does not alter original native deadlines, work/token
ceilings, reserve, snapshot-I/O or scientific contracts, weaken exact body
hashing, replace failed DPO01/02 with replay03, complete optional/maximal stage
rows, change original eighteen-package acceptance/dependencies, or advance the
learner's Day9/mastery state.

The actual assembler must still wait for owned model jobs to finish and final
sources/inputs to be frozen. Its real raw/gzip fit, complete receipt closure,
all-input stability and any subsequent publication compatibility must be
observed separately. Successful authored controls are not a real native
campaign archive or a new native experiment.
