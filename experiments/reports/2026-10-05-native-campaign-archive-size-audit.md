# Native campaign archive-output size audit

This audit concerns the CPU evidence assembler, not the native experiment's
scientific, memory, token, work, disk, time or snapshot-I/O limits. No actual
campaign assembly, model loading or GPU work was performed for this audit.
The original receipts, failures and 45 planned stage rows remain inputs, not
objects to compact or discard to make an archive fit.

## Observed size risk before the change

A stat-only enumeration of the assembler's fixed catalog and declared output
directories found 847 distinct available JSON paths totalling **352,827,895
bytes (336.483 MiB)**. Nineteen catalog entries were missing at that point,
including later reasoning and RLVR jobs. No enumerated individual JSON input
exceeded the existing 64 MiB reader ceiling. This enumeration did not parse
child command lines to discover additional output roots, hash model or corpus
bodies, or run the assembler.

The aggregate exceeds the former 268,435,456-byte (256 MiB) *closing snapshot*
bound. It is not a measured final JSON serialization size or a formal lower
bound: parsed documents are re-indented, and the snapshot also repeats selected
documents in scientific views. It nevertheless establishes a real risk rather
than a projected model-training requirement. Missing inputs and future outputs
mean that a final fit under a new ceiling is not yet demonstrated.

A separate bounded read of the DPO100 receipts confirmed structural duplication:

| Retained receipt | File bytes | Observation rows | In-memory event rows |
| --- | ---: | ---: | ---: |
| `returned-supervision-pilot.json` | 8,070,062 | 9,587 | 9,592 |
| `supervision-pilot/result.json` | 8,069,696 | 9,587 | 9,592 |
| `acceptance.json` | 9,767,386 | The returned receipt is embedded unchanged | The returned receipt is embedded unchanged |

The returned and terminal receipts have equal observation and event arrays;
their other fields are not asserted interchangeable. The acceptance's embedded
pilot result equals the returned receipt. Counting a stand-alone, indented
serialization of the two arrays gives 2,947,050 and 4,562,659 bytes respectively.
Those arrays were neither printed nor rewritten. Path-level deduplication cannot
remove these distinct historical receipt documents without changing the archive.

## Approved implementation protocol, declared before focused checks

The old assembler source was
`e9262b247415cc5a55d030038ad92588caf9aa640f42ee23d0e85d89819cad0a`;
the old focused test source was
`f8dfce45b24d375e9704e3a68f49ec1649e8a84f3737c31cea3bc86eed50e371`.

1. Keep the JSON-input ceiling at 64 MiB, journal/input-hash ceiling at
   256 MiB, individual export ceiling at 4 GiB, and all native producer limits
   unchanged. Introduce a separate **1 GiB archive-output ceiling** for the
   assembled snapshot. Enforce it while writing, not only after serialization.
2. Preserve the exclusive uncompressed `snapshot.json` locally. Stream an
   exclusive `snapshot.json.gz` companion at compression level 9, with gzip
   `mtime=0` and an empty original filename. Bound both output files by 1 GiB.
   A failed write keeps its partial prefix; no extraction, overwrite or cleanup
   is performed automatically.
3. Bind both actual byte streams in `closing-bindings.json`, together with the
   explicit 1 GiB decompression ceiling. Deterministic compression is claimed
   for identical input in the same recorded Python/zlib environment, not across
   every implementation or version. The gzip companion does not authorize
   unbounded decompression or replace a closing receipt.
4. Ignore only `/experiments/reports/native-campaign-evidence-*/snapshot.json`.
   The compressed companion, closing receipts and failures remain versionable.
   This is preparation for a later separately authorized Git publication; no
   commit, push or publication is performed here.
5. Verify only authored small CPU fixtures: unchanged original controls, separate
   bounds, exclusive writes, deterministic gzip headers/bytes, round-trip
   identity, source mutation/type refusal, retained partial failures, and both
   closing bindings. Do not run the actual assembler until the parent observes
   GPU quiet and freezes the final sources.

## Focused verification

The [assembler](../../scripts/assemble_native_campaign_evidence.py) and its
[authored controls](../../tests/test_native_campaign_evidence.py) are frozen for
independent review. The first development panel passed 45 tests, actual exit 0,
with a 0.392-second unittest footer. A second source-bound panel passed the same
45 tests, actual exit 0, with a 0.347-second footer and 0.417363 seconds measured
parent wall time. These overlapping passes are **not 90 distinct tests**.
There were no failed focused runs in this change.

The final command was:

```sh
env CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=src:tests \
  /tmp/dongxi-course-reproduction.ZfVaEu/venv/bin/python \
  -m unittest discover -s tests -p test_native_campaign_evidence.py
```

Actual stdout was empty; actual stderr was:

```text
.............................................
----------------------------------------------------------------------
Ran 45 tests in 0.347s

OK
```

All seven source/operational hashes were identical before and after that panel:

| File | SHA-256 before and after |
| --- | --- |
| `scripts/assemble_native_campaign_evidence.py` | `4b7bd4188c8d0d8612b815fe44fa4b26c6e5ad8c16cb3dd22903d7c53926c55b` |
| `tests/test_native_campaign_evidence.py` | `7b7530572baeae504fed756fb1b2ba190828910370bbfae00f8a04f4249b4f51` |
| `src/dongxi_llms/staged_campaign.py` | `4744cc5f399dea11a2dc05a61f29378e80e777d9a4b189848f5f8c3b636873f1` |
| `src/dongxi_llms/run_identity.py` | `a98d244efbfcbecf13ba97e28f512b2743642ac7d6e2ac8fad3a36eb5bcb4999` |
| `src/dongxi_llms/native_profile_supervisor.py` | `88100a0fbe33fa8de34155838485e6c31fb5c3fd436d1dc27725fe720ad4f9b2` |
| `src/dongxi_llms/campaign_supervisor.py` | `d2959a7856bc0cc4fb6b6623387d45bbc0d7193607a52c29c59bb8823b97d253` |
| `.gitignore` | `f459dc5711831cc5336dd739591cb9fe37a730dc404377e0273304094bd4a6c6` |

The actual interpreter was Python 3.12.14 in
`/tmp/dongxi-course-reproduction.ZfVaEu/venv`, on
`Linux-6.17.0-1031-nvidia-aarch64-with-glibc2.39`. Compile-time and runtime zlib
versions were both 1.3.2. CUDA was hidden, both offline flags were 1, and all
three declared thread counts were 1.

The 33 original controls remain; 12 additional methods verify the separate
ceiling, original JSON byte representation, newline-inclusive admission,
invalid bounds, nonfinite-prefix retention, deterministic filename-free gzip,
raw/compressed limits, exclusive output, changed source bytes/inode, and the
precise ignore rule. The original small complete-assembly control additionally
checks both closing byte bindings and a real compression-read fault: all 45
original rows survive in the retained raw snapshot, while no closing receipt is
published. All assembly calls in this panel used authored temporary fixtures,
not the actual campaign catalog.

`git diff --check` on the scoped files exited 0. A read-only
`git check-ignore -v --no-index` probe matched only the exact uncompressed
snapshot among authored `snapshot.json`, `snapshot.json.gz`,
`closing-bindings.json` and `failure.json` paths. It exited 0; the other three
paths had no ignore match.

No actual campaign snapshot has yet been produced with this version. The eventual
raw and gzip sizes, all-input stability, Git hosting size compatibility, and
final full-course verification remain unmeasured here. A fit below 1 GiB will
still be checked rather than assumed. This CPU archive change does not complete
any original native acceptance criterion or advance learner mastery.
