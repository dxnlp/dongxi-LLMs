# A finite training path can still fail recovery acceptance

The first native DPO replay attempt used the selected full400 policy, the
original two-update location recipe and a600-second external guard per child.
It did not pass the recovery gate.

| Actual child | External seconds | Observed outcome |
|---|---:|---|
| Independent clean2 |380.162608|Completed, exit0|
| Source2, retaining completed1 for resume |379.245998|Completed, exit0|
| Fresh completed1→2 resume |600.683331|External deadline; native exit unknown|

The [clean](native-dpo-replay-20261005-run-01/returned-supervision-clean.json),
[source](native-dpo-replay-20261005-run-01/returned-supervision-source.json) and
[resume](native-dpo-replay-20261005-run-01/returned-supervision-resumed.json)
receipts are retained with the
[adapter failure](native-dpo-replay-20261005-run-01/failure.json).
The outer adapter returned1; this is not the unknown native child's exit code.
The resume receipt also records `Owned leader reap timed out`. A later read-only
process check found its retained leader PID275819 absent; that later observation
does not replace the missing original exit or erase the cleanup error.

The minimum externally sampled available memory for the failed child was
84,718,571,520 bytes, above the original25GiB reserve. The declared stop reason
is the deadline, not memory exhaustion or an observed nonfinite loss. The work
and I/O journals show recovery-state validation and saving in progress. An
updated committed payload in this recipe is about10.782GB. Its policy,
reference and optimizer state must not be mistaken for the size of an inference
export. A state being computed or partially written is not an accepted committed
resume artifact. No final three-way comparison or recovery acceptance was
produced for this attempt.

The three supervised child intervals total1,360.092 seconds. This includes
failed work, not only the useful final optimizer trajectory. It excludes
parent preparation, hashing and offline review; it is not a total compute or
energy measurement. Clone, serialization and hashing counters overlap the same
payloads and cannot simply be added as unique disk traffic. The original
journals, failure and uncommitted artifact layout remain retained.

## First controlled implementation response: run02

Keep the600-second deadline, all validation, saves, generation diagnostics,
training arguments and work/I/O/artifact ceilings. The separately reviewed
[CPU8 child entry](../../scripts/run_native_dpo_cpu8_child.py) sets and prints
actual Torch intra-op8/inter-op1 and OMP8 before invoking the unchanged native
runner. The initial command requested OMP1; its original receipt did not
independently observe these thread getters. Do not rewrite that older evidence
as if it contained the new witness.

At this stage the [closed stage adapter](../../scripts/run_native_dpo_stages.py)
selected a fresh `run-02` replay and required actual thread witnesses from every
child and the accounted CPU comparison before admitting pilot100. Its26 authored CPU controls
and independent source/preflight review pass. Those checks establish readiness,
not actual replay success. The new attempt's outcome must come from its own
receipts below. Neither a deadline failure nor a later technical fix establishes
preference learning or general assistant quality.

## Run 02: witnessed CPU8 still fails the unchanged deadline

The second attempt witnessed Torch intra-op 8, inter-op 1 and OMP 8 in every
native child. It preserved the 600-second external deadline, original two-update
recipe, seed 1818, learning rate 5e-7, beta 0.1, accumulation 4, length 512,
generation cap 64, checkpoint cadence 1 and all work/I/O/artifact ceilings.
It also failed the whole-child recovery gate.

| Actual child | External seconds | Minimum sampled MemAvailable bytes | Observed outcome |
|---|---:|---:|---|
| Independent clean2 |398.695142|96,012,181,504|Completed, exit 0|
| Source2, retaining completed1 |379.741676|96,063,782,912|Completed, exit 0|
| Fresh completed1→2 resume |600.682756|85,373,120,512|External deadline; native exit unknown|

The actual [clean](native-dpo-replay-20261005-run-02/returned-supervision-clean.json),
[source](native-dpo-replay-20261005-run-02/returned-supervision-source.json),
[resume](native-dpo-replay-20261005-run-02/returned-supervision-resumed.json)
and [adapter failure](native-dpo-replay-20261005-run-02/failure.json) remain
retained. The resume receipt again has `actual_exit_code=null` and
`Owned leader reap timed out`; an outer adapter exit is not a native exit code.
A later read-only process check at 10:56:23 UTC found leader 284898, helpers
284895/284894 and queue leader 282500 absent, with no GPU processes reported.
That later observation does not recover the missing original exit or erase
the cleanup error. The declared stop remains the deadline, not sampled memory
exhaustion. The three child intervals sum to 1,379.119574 seconds, including
failed work but excluding preparation, parent-side hashing and offline review.

This failure is narrower than “the resumed update never completed.” The
retained [model work journal](../../outputs/native-dpo-replay-20261005-run-02-shared-journals/work.jsonl)
and [I/O journal](../../outputs/native-dpo-replay-20261005-run-02-shared-journals/io.jsonl)
record completed resumed-boundary and update-2 saves: I/O tickets 11 and 13
complete at sequences 12 and 14. The resumed metric records update 2 with
`last_committed_update=2`; both generation panels, validation and an exported
policy are present. Its invocation journal's last stage is
`rehashing-actual-source-input-lock-and-parent-files`, while status remains
running and `result.json` is absent. These are retained intermediate outcomes,
not a successful final recovery acceptance or an independently completed
three-way numerical comparison.

## Read-only diagnosis: logical reads are not physical disk traffic

During the resumed child's initial recommit, two `/proc/284898/io` point probes
showed:

| Observation, UTC | `rchar`, bytes | `read_bytes`, bytes |
|---|---:|---:|
|10:49:06|372,768,668,254|54,501,376|
|10:49:49.906|472,136,095,602|54,501,376|

The second probe found an open descriptor for a retained source completed-1
payload. These are cumulative kernel counters from the running leader, not
per-phase timings or sampled Python stacks. The large logical-read increase
with unchanged physical-read counter is consistent with repeated cached
integrity reads. It is not evidence of 472 GB of physical storage reads, and
neither counter is a complete energy or compute measure.

The source supplies a concrete mechanism. `ArtifactBudget._inventory` reads
and hashes every sealed/removing file in full. Receipt validation, reservation,
writer opening, sealing, publication linking, staging removal and final
receipt/usage inspection all invoke that inventory. At the initial resumed
save, the retained three source payloads total about 27.577 GB; the newly
written recommit brings the payload set to about 38.360 GB per inventory.
The native per-node snapshot guard only checks available memory and time—it
does not hash the payload at each tree node. Repeated full inventories are
therefore a code-supported explanation for the observed logical-read scale,
not a claim of directly profiled call-stack time.

The final stage label alone cannot distinguish identity rehashing from the
subsequent artifact `receipt()`/`usage()` calls: the runner computes both before
writing its final result. Each again invokes a full inventory. The completed
saves and exports do not prove those last integrity checks finished before the
watchdog deadline.

The implementation-only follow-up parallelizes independent file digests
within each inventory, keeping the serial default and explicitly bounding
native workers. It must retain every existing full-byte hash, no-follow/private
file and hardlink check, deterministic first-error handling, all save/fsync
boundaries and the unchanged scientific recipe, ceilings and 600-second
deadline. Digest caching based only on metadata or omitted repeated validation
would be a different, weaker contract. No speedup or successful replay is
claimed from this diagnosis. Artifact inventory reads remain outside the
declared shared snapshot I/O counters; their exclusion must stay explicit
rather than relabelling all logical reads as charged snapshot hash bytes.

## Independent byte-only review of the hashing implementation

The reviewed [artifact module](../../src/dongxi_llms/artifact_budget.py) accepts
an exact integer `hash_workers` from 1 to 4; default 1 retains the original
direct serial digest path. The opt-in path keeps at most four outstanding
complete-file digest futures and corresponding digest file descriptors/1-MiB
buffers, consumes errors in the original directory iteration order and waits
for every submitted worker before returning or raising. It does not mutate
the ledger concurrently, cache checksums, deduplicate hardlinked pathnames or
remove any inventory call. Existing root/journal, no-follow regular-file,
ownership, permission, byte/entry, complete-SHA, missing-file and hardlink
checks remain present. The execution setting does not change the immutable
ledger identity or receipt.

Parallel lookahead can read a later file before an earlier metadata refusal.
Thus preserving checks and reported first-error precedence does not mean
identical I/O timing or atomic protection from hostile same-UID mutation. The
executor drain and descriptor `finally` cleanup are required even on failures.
Those boundaries are documented in the module rather than promoted into a
physical quota or arbitrary concurrent-writer guarantee.

Independent CPU verification used the existing isolated interpreter, with
`PYTHONPATH=src:tests`, hidden CUDA, offline Hugging Face flags and
OMP/OpenBLAS/MKL threads set to 1:

```bash
/tmp/dongxi-course-reproduction.ZfVaEu/venv/bin/python -m unittest discover -s tests -p test_artifact_budget_parallel.py -v
/tmp/dongxi-course-reproduction.ZfVaEu/venv/bin/python -m unittest discover -s tests -p test_artifact_budget.py -v
```

| Suite | Actual test executions | Unittest-reported seconds | Exit |
|---|---:|---:|---:|
| New scheduling controls plus original controls under opt-in factories |39|1.871|0|
| Original default-serial controls |23|1.012|0|

These are 62 executions, comprising 16 new scheduling controls and the same
23 original controls exercised in two modes—not 62 new independent tests or
a model-scale timing benchmark. They cover full-file/chunk-tail hashing,
ordered metadata/digest failures, maximum outstanding tasks and actual overlap,
worker/FD draining, symlink/FIFO/permission/growth faults, retained missing-file
and hardlink behavior, no checksum caching and unchanged receipts/charges.

The following bytes were identical before and after this independent panel:

- Artifact module: `af1c1fcc4349393e50f5603e85017f851128c92c34dbcb82f75aa4694efdacc0`.
- New test file: `646fe4533f5125c37e2639ba661ed90720675b70dcbfe66d45a5f44630bb3ee7`.
- Original test file: `b4ad9ecfc1b3421e6aeae9317212d9ce8c6dcce4869ede23edff46da741c1a4b`.

That review established the scoped byte-only implementation controls, not the
native runner's complete wiring, a measured speedup or a successful replay
within 600 seconds. Both failed native attempts remain failed regardless of
these CPU test outcomes. The later replay below has its own actual receipts.

## Run 03: whole-child recovery and comparison pass

The fresh selected `run-03` replay passes its
[actual acceptance](native-dpo-replay-20261005-run-03/acceptance.json).
It includes independent clean2, source2 retaining completed1, fresh
completed1→2 resume and a separately supervised CPU three-way comparison. All
four returned receipts report completed status, actual exit 0, no stop reason
and empty cleanup errors. This is not inferred from a committed payload,
export directory or update2 metric alone.

| Actual invocation | `child_seconds` | Supervisor `seconds` | Minimum sampled MemAvailable bytes |
|---|---:|---:|---:|
| [Clean2](native-dpo-replay-20261005-run-03/returned-supervision-clean.json) |319.7739916170249|319.8190489980043|95,734,423,552|
| [Source2](native-dpo-replay-20261005-run-03/returned-supervision-source.json) |296.33726876898436|296.38268336304463|96,150,581,248|
| [Fresh completed1→2](native-dpo-replay-20261005-run-03/returned-supervision-resumed.json) |392.90805816999637|392.95715733000543|86,374,494,208|
| [CPU comparison](native-dpo-replay-20261005-run-03/returned-supervision-comparison.json) |127.59657517704181|127.6439409229788|101,931,466,752|

These are the retained receipt fields, not estimates. Every invocation keeps
the original 600-second external guard and 26,843,545,600-byte (25 GiB) sampled
reserve. The three native child intervals sum to 1,009.0193185560056 seconds;
their supervisor intervals sum to 1,009.1588896910544 seconds. Including the
CPU comparison gives 1,136.6158937330474 child seconds and 1,136.8028306140332
supervisor seconds. None is a single invocation or total compute/energy measure.
These timing boundaries overlap and must not be added to one another.

The separately retained [outer-launcher observation](native-dpo-replay-20261005-run-03/outer-launcher-observation.json)
records outer exit 0 and parent monotonic interval 1,156.8203529980383 seconds.
It is an owner transcription of actual unified-exec session 64405 stdout,
without a retained stdout digest—not a native producer receipt or pre-launch-
bound terminal archive. Its interval includes preparation, all four supervised
children and acceptance work. Do not label it native-child time or apply one
child's 600-second deadline to that whole serial interval. Point samples above
reserve do not prove continuous memory safety.

### Unchanged science and actual execution settings

The [preparation](native-dpo-replay-20261005-run-03/preparation.json) preserves
the selected `native-sft-full-pilot400-20261005-run-01/policy` parent, original
location 8/4/4 train/validation/evaluation fixtures, seed 1818, learning rate 5e-7,
DPO beta 0.1, accumulation 4, maximum length 512, generation cap 64 and
checkpoint cadence 1. Policy and frozen reference weights remain FP32 with BF16
CUDA autocast; neither the objective nor the model arithmetic recipe was
replaced to obtain acceptance. Resume restores the saved RNG/sampler state
rather than drawing a new trajectory from the starting seed. New source bindings
freeze the reviewed implementation; fixed input/parent identities, the scientific
contract, encoded IDs/masks and numerical method retain the original recipe.
All 19-dimensional work, 9-dimensional I/O, artifact and time ceilings are
unchanged from run02.

Actual [clean](native-dpo-replay-20261005-run-03/cpu-thread-witness-clean.json),
[source](native-dpo-replay-20261005-run-03/cpu-thread-witness-source.json),
[resumed](native-dpo-replay-20261005-run-03/cpu-thread-witness-resumed.json) and
[CPU comparison](native-dpo-replay-20261005-run-03/cpu-thread-witness-cpu-comparison.json)
witnesses bind their actual stdout bytes and report OMP=8, Torch intra-op 8 and
inter-op 1. The corresponding complete-file hash witnesses report worker count 4
in each native child and the ordered clean/source/resumed ledger restorations in
the [CPU comparison](native-dpo-replay-20261005-run-03/snapshot-hash-witness-cpu-comparison.json).
They are actual runtime observations, not assumed command-line settings.

The reviewed execution-only option leaves `hash_workers=1` as the serial
default. Explicit worker count 4 uses at most four outstanding independent-file
futures and digest file descriptors with 1-MiB buffers. Every inventory invocation, body-byte
SHA, no-follow/type/ownership/permission check, byte/entry cap, missing-file,
hardlink and race check remains. Original ordered first-error handling and
drained worker/FD cleanup remain required. It does not cache by metadata, omit
revalidation, change ledger/receipt identity, reset capacity or constitute a
sandbox. Its [independent runtime source/control review](2026-10-05-native-dpo-parallel-hash-runtime-review.md)
is separate from the actual execution evidence here.

### Numerical equality preserves unequal spending

The [CPU comparison](native-dpo-replay-20261005-run-03/comparison.json) uses
admitted shared inspect/load operations on all three final completed2 payloads.
Its ten typed component fingerprints agree exactly: completed cursor, counters,
CUDA RNG, history, Adam optimizer, policy, frozen reference, sampler RNG, schema
and Torch RNG. The reference fingerprint equals the original frozen reference;
the three scientific-contract identities match. The acceptance also verifies
the exact numerical metric tail: clean/source two-update histories match, and
the resumed row matches their original update2. Numerical comparison excludes
the declared operational fields `resume_parent`, `work_ledger` and
`snapshot_artifact_ledger`; it does not require legitimate cost prefixes to be
numerically identical.

The independent clean journal retains two actual optimizer updates. The shared
source/resume journal retains three: source updates 1 and 2, then the repeated
update 2 after restoring completed1. Its final reserved and completed
`train_updates` are both 3, even though every compared numerical cursor is 2.
The CPU comparison preserves later physical spending: the clean I/O history
includes one inspect, one load and three saves; the final shared history includes
three inspects, three loads and five saves. All work/I/O open and failed ticket
lists are empty at this accepted gate. These are routed logical operations, not
FLOPs or physical disk bytes. Artifact inventory hashes and independent typed
component hashing remain explicitly outside work19/I/O9; the latter is covered
by its own external 600-second comparison bound and sampled reserve, not a claim
that the ledgers count all physical CPU work.

The [independent actual replay review](2026-10-05-native-dpo-replay03-independent-review.md)
checks retained metadata, witnesses, comparisons and returned receipts without
rehashing state bodies or rerunning models. It is not a second model execution.
Its verdict is CLEAR at this declared two-update recovery boundary. All retained
before/after four-item location panels remain 0/4 strict exact answers and 4/4
declared natural stops. Those diagnostics do not become useful preference or
general assistant quality merely because technical replay passed.

Run03 establishes bounded local completed1→2 recovery and whole-child/
comparison acceptance. It does not isolate a causal performance improvement:
the two earlier capped failures and this complete attempt are not a matched
hashing benchmark. Both `run-01` and `run-02` remain failed with null native
resume exits, retained reap faults, spent work and intermediate artifacts;
run02's durable update2/export but missing final result remains part of the
lesson. The replay gate admits the separate fixed 100-update pilot, but supplies
no pilot100 result, independent preference/assistant quality, Mac recovery,
continuous resource safety, physical quota or hostile-process containment.
