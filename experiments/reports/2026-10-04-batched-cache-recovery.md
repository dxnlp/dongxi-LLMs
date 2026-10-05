# Ragged cache and exact recovery on a tiny CPU decoder

The declared correctness and recovery checks pass. They do not establish a
serving speedup or learned stopping: all natural greedy rows reached the cap,
and cached batch was slightly slower than full-prefix batch on this tiny CPU
workload. Preserving those results is part of the lesson.

The [specification](../specs/2026-10-04-batched-cache-recovery.md) was written
before measurements. Original integer prompts, fixed random weights, seeds,
supports, budgets and tolerances were not changed after observing the outcomes.
The [initial ledger](2026-10-04-batched-cache-recovery.json) and
[first verification](2026-10-04-batched-cache-recovery-verification.json) are
preserved as historical v1 evidence. The
[current-source hardening verification](2026-10-04-batched-cache-recovery-hardening.json)
adds unambiguous typed identities, an explicit migration audit against actual
numerical/state evidence, fresh notebook/check results and new snapshot files.
The original reports, tensor files and headers were not rewritten.

## Actual model and boundary

The isolated wrapper reuses the existing course decoder's learned blocks, not
a downloaded implementation or checkpoint. It adds explicit ragged attention
key masks, row-relative positions and real compact KV tensors. Existing
`decoder_lab.py`, `dpo_lab.py`, `grpo_lab.py` and pretraining recovery lessons
remain unchanged.

The model has 5,072 parameters: vocabulary 12, width 16, two modern blocks, four
query heads, two KV heads, head width 4, hidden 32, RMSNorm/RoPE/SwiGLU and an
untied vocabulary head. No dropout is present. Three authored prefixes have
lengths 2/4/6. EOS and PAD share ID 0; mask semantics determine validity. The
primary behavior is full-vocabulary temperature 1, cap 4. Seeds 2501/2510 define
weights and request-keyed sampled generators respectively.

Actual execution was Python 3.12.14, Torch 2.14.1+cpu, Linux/aarch64, float64 CPU,
one Torch thread, in the isolated locked course environment. No CUDA, Mac,
pretrained model, API, service or new installation was used. CPU identity
checking and transparent repeated KV expansion are part of the implementation,
not an optimized serving benchmark.

## Generation comparisons retain natural caps

Every single/batch cached/uncached greedy path produced:

| Row | Prompt length | Generated IDs | Stop |
|---|---:|---|---|
| short | 2 | 2,5,4,11 | cap |
| medium | 4 | 3,2,5,4 | cap |
| long | 6 | 2,5,4,11 | cap |

Selected IDs, stop reasons and raw/behavior selected log probabilities match.
Maximum selected-score error was zero in all three collections. The predeclared
acceptance remains absolute/relative 1e-10; this observation is not a guarantee
of bitwise batch-layout equivalence on other backends or near argmax ties.
Independent tests compare complete unpadded wrapper logits and parameter
gradients with the original decoder, plus complete cached next-token logits
with single-row full recomputation.

The sampled short row emitted `[8,0]` and stopped naturally after draw 2.
Medium emitted `[5,9,5,9]` and long `[7,9,4,2]`; both remained capped. Independent
request-keyed generators preserve sampled IDs and selected scores across
single/batch and reversed-row order. This is the declared sampling protocol,
not a promise that a single shared global generator is reorder-invariant.

The forced stopping control is separate: short emits EOS immediately, long
emits token 3 then EOS, and medium emits token 3 four times then caps. These are
conditional support controls, not a trained termination result. Its active
row order goes from three rows to medium/long, then medium alone. The short
row's EOS behavior log probability is zero, while its raw model log probability
is negative. Both values are retained rather than silently using the wrong
distribution for a likelihood or objective.

## Actual work and retained storage

All primary greedy paths produce twelve useful output tokens. Forwarded
positions count actual rectangular input slots supplied to the model; valid
positions exclude padding, but include repeated real-prefix work.

| Path | Prefill calls | Decode calls | Full-prefix continuation calls | Forwarded positions | Valid positions | Padding | Peak retained KV bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| single / uncached | 3 | 0 | 9 | 66 | 66 | 0 | 0 |
| batch / uncached | 1 | 0 | 3 | 90 | 66 | 24 | 0 |
| single / cached | 3 | 9 | 0 | 21 | 21 | 0 | 2,304 |
| batch / cached | 1 | 3 | 0 | 27 | 21 | 6 | 6,912 |

Each event retains active request IDs, real position/mask tensors, input shape,
actual valid/forwarded counts and compact cache shapes. Zero uncached retained
bytes means no cache survives between calls, not that attention creates no
temporary K/V. Single paths run sequentially, so their peak is the largest
individual request cache rather than the sum of simultaneous requests.

Real cache payload follows `2 × layers × active rows × KV heads × physical
retained columns × head width × element bytes`. It includes masked pad columns.
It excludes parameters, attention/KV expansion temporaries, snapshots, Python,
allocator and whole-process memory. Active compaction removes finished rows,
not every earlier padding column.

All five runtime repetitions per path are retained. Current-source medians
are descriptive measurements, not selected best timings:

| Path | Median seconds |
|---|---:|
| single / uncached | 0.009989336 |
| batch / uncached | 0.004216790 |
| single / cached | 0.010604439 |
| batch / cached | 0.004383510 |

Both historical collections similarly measured cached batch slightly slower
than uncached batch. Their raw timings remain separate; the table above uses
only the new v2 collection. No warm-up or allocation boundary was invented afterward.
These small CPU times include Python/identity work and cannot predict vLLM,
Spark throughput, distributed serving or large-model memory.

## Generation state and safe local reload

The interrupted sampled state follows two global draws. Short has finished;
medium and long are active at two consumed draws each. Their retained cache
is 4,096 bytes. The actual snapshot keeps prefixes, active order, K/V, next
logits, per-row RNG states, sampler cursors, histories and work counters.

Saved-cache reload reproduces the exact next draw and complete remaining
trajectory. The independent test also compares the whole returned work/record
state. Explicit full-prefix rebuilding is a distinct intervention checked to
1e-10. It happened to give zero selected-score error and the same IDs here;
it is not advertised as universally bitwise.

Policy parameter bytes/version, prefix/order, token/template/position/support,
dtype and environment are bound before reuse. Runtime cache/next-logit content
and prefix checks catch unintended changes. The report retains rejected stale
weights, changed template, wrong external byte digest, cursor corruption and
RNG corruption. Tests add changed versions, tensor/logit corruption and
independently resealed inconsistent cursor/pending states.

Trusted local tensor states are loaded with `weights_only=True` only after
checking an independently expected contract/environment, external SHA-256 and
bounded file size. The tests confirm rejected identities never call the tensor
loader. Headers supplied alongside an untrusted checkpoint are not independent
authentication. The 2 MB file bound is not an adversarial tensor-allocation
sandbox, and arbitrary external checkpoint loading is outside this lesson.

The historical collections' original snapshot files/headers are preserved under
`fixtures/batched-cache-recovery/` and
`fixtures/batched-cache-recovery-verification/`. The report supplies their
external expected identities. New v2 files are under
`fixtures/batched-cache-recovery-hardening/`: the generation interruption,
eight completed/pending training states and four final training states.
Current-source metadata does not rewrite historical byte provenance or make
the old identity encoding acceptable for resume.

### A real structural identity bug and its migration

Before closure, independent review found that the original digest serialized
dictionary entries without container boundaries. It assigned the same identity
to `{'a': {'b': 1}, 'c': 2}` and `{'a': {'b': 1, 'c': 2}}`. This was a
serialization collision, not a SHA-256 cryptographic collision. It invalidated
the original encoding as a general structural state identity even though the
observed trajectories remained reproducible.

The new `dongxi-typed-length-framed-sha256-v2` encoding marks types, container
item counts and byte lengths, with explicit tensor dtype/shape/payload. Sorted
typed primitive keys make dictionary order irrelevant without confusing key
types. Generation and training contracts are v2. Current reload rejects an
obsolete identity algorithm before tensor deserialization; old v1 snapshots
are historical evidence, not supported current resume inputs.

The migration audit does not compare new identity strings with old strings.
It rechecks all four frozen recipes, exact token/action/score/work records and
all loss/gradient histories. Historical and new completed/pending files are
read only after checking their independently recorded external byte hashes and
sizes, then their actual policy, reference, Adam and data-stream tensor bytes
are compared directly. Rollout RNG bytes and pending sampled actions also
match. A narrowly labelled diagnostic recreates the old flat-parameter hash to
bridge the historical final hash, because the original ledger did not archive
final tensors. New final tensors are now archived. This diagnostic is neither a
generic legacy identity validator nor a legacy resume interface.

All migration checks passed. Global Torch RNG is saved/restored within each
run; independently initialized process-wide RNG states are not asserted equal
between different collection processes. No numerical recipe, coefficient,
seed, support, tolerance or budget changed during this identity migration.

## Actual DPO and RLVR interruption recovery

Each objective uses seeds 2521/2522 and six primary updates. DPO scores correct
chosen versus alternative rejected atom+EOS completions, sum log likelihood,
beta 0.5, and a fixed initial reference. The sampled RLVR task uses support
`{EOS,3,4}`, temperature 1, group 4, cap 3, population-normalized detached group
advantages, response-mean clipping 0.2 and exact conditional KL beta 0.02. Reward
is one only for the independently fixed correct atom followed immediately by
emitted EOS. Model/reference bytes, shuffle/data stream, Adam, rollout and
Torch RNGs, completed versions and history are actual saved state.

Each original run is interrupted after three completed updates and after
collecting the fourth batch. The post-collection interruption raises explicitly
and retains that pending batch. Resume applies it without resampling; its
old-policy byte identity/version and saved likelihoods are validated. Both
boundaries reproduce next data/rollout, complete loss histories, final model
bytes, optimizer state, data cursor and rollout RNG exactly for every seed.
The hardened verification's actual primary histories and snapshot states match
the initial collection without retuning. Derived v2 hashes intentionally differ;
the explicit migration diagnostic records the old final-hash bridge separately.

| Objective | Seed | First loss | Sixth loss | Completed replay | Pending replay |
|---|---:|---:|---:|---|---|
| DPO | 2521 | 0.693147181 | 0.332527579 | exact | exact |
| DPO | 2522 | 0.693147181 | 0.384949600 | exact | exact |
| RLVR | 2521 | approximately 0 | 0.000780386 | exact | exact |
| RLVR | 2522 | approximately 0 | 0.000167256 | exact | exact |

RLVR loss values are not quality scores. Seed 2521 has reward vectors
`[1,0,0,0]` at updates 1/4 and all zeros otherwise. Seed 2522 has all-zero groups
at updates 1/2/3/5, `[1,0,1,1]` at update 4 and `[0,0,0,1]` at update 6. Near-zero
early gradients and zero relative signal remain visible. All four final policy
hashes differ from initialization, but no held-out quality improvement or
reasoning transfer is claimed.

Every applicable omission control changes history and final model bytes.
Missing Adam changes updates despite the same initial future data schedule;
missing the data cursor changes batches; missing rollout RNG changes RLVR
samples. DPO consumes no rollout RNG, so that branch is explicitly N/A and
unchanged. Every broken branch's actual history is retained in verification,
not just a success/failure flag. These are intentionally broken illustrative
branches, not supported resume modes or evidence of an unrun production failure.

Actual collection work is 24 primary optimizer updates, 24 exact-replay updates
and 36 broken-branch updates: 84 total calls, not 24. Tests/notebook mechanism
updates are separate verification work. Six updates per main run were fixed
before observation; no successful seed or checkpoint was selected.

## Verification and retained failed checks

The current-source hardening verification passed in 3.903 seconds including
collection and its children. Twenty-one source/dependency/lesson/figure
identities were unchanged. Twenty-six focused tests passed in 1.260 seconds,
including independent
logits/gradients, ragged masks/positions, real compact GQA cache storage,
stopping/reordering, safe loading, old-policy support alignment and complete
or pending policy-update recovery. Five new regressions cover nested container
collisions, typed identities, tensor payload identity, honest migration
projection and obsolete-algorithm rejection before loading. The book math
check recorded 54 Markdown files, 1,213 expressions and zero issues at collection
time. This does not hash every file covered by a future shared-suite pass.

The initial raw report SHA-256 remains
`76bdec79bf00f66e37d21447b0419107b42f380a881d41179ac1fa82f890af5d`.
The first verification remains
`294edab76c7ba0e7e254ff9b18aa7efe302f1d43fc7ef8713c122abfcd75e9b8`.
The current hardening report is
`cb223bbb8bbc8c05c3334422581f0df442b53f981d02ee249a2cde708d69129f`;
it explicitly identifies both current sources and the historical comparison.

Before v2 hardening, an additional shared-suite invocation passed 443 tests in 22.736 seconds on
the concurrently evolving course tree. This is a time-scoped diagnostic,
not evidence that the 21 package hashes identify every sibling test/source
used by that invocation. The root's stable course-wide panel remains separate.

The [new notebook](../../notebooks/day-25/02_ragged_kv_cache_and_exact_recovery.ipynb)
passed a fresh isolated CPU kernel against the hardened source: seven complete
cells, no skipped exercises and five images, in 2.425 seconds. Its manifest is
`/tmp/dongxi-course-check-1cfaouu9/manifest.json` with SHA-256
`8689cc1b68f4f71f1795e14dcd5f691958065a5c5062ec084d2253ad0d992557`.
The JSON records actual kernel prefix/packages and all preview hashes. All five
images remain byte-identical to the previously visually inspected previews.
Schematics are labeled structural; plots use actual model tensors, work and
retained histories. Source notebook/learner cells are not overwritten.

Three substantive development findings are explicitly retained in this account. The first
focused run executed 20 tests with one failure: a test expected 2,304 bytes for
the forced three-row prefill while the measured payload was 4,608. The assertion
omitted the factor two for K and V; independent tensor geometry already passed.
The expectation was corrected, not the measured tensors or model recipe.
Subsequent state checks brought the suite to 21 tests; structural identity
hardening then brought it to 26. The nested-dictionary collision above was a
real source defect found after those earlier passing checks, not a failed
measurement silently replaced by a favorable run.

The first notebook preflight exited before launching a kernel with
`CourseManifestError: Notebook day/chapter/path mismatch`. Root had registered
the Day 25 extension as Chapter 5 despite the fixed day-to-chapter 14 mapping.
Root corrected the registry and the notebook metadata/preview route was aligned
to Chapter 14; Chapter 5 retains the conceptual bridge. The validator was not
loosened. A temporary navigation check also reported unfinished report links;
this report resolves this package's links while sibling work remains independently
owned. No failed kernel execution or favorable result substitution is hidden.

## Reproduce without changing the shared GPU environment

Use new report/workspace paths for collection:

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
  /tmp/dongxi-course-reproduction.ZfVaEu/venv/bin/python -m dongxi_llms.batched_cache_lab \
  --report /tmp/NEW-cache-report.json --workspace /tmp/NEW-cache-states
```

For current-source acceptance, first create a fresh notebook manifest using
the isolated kernel, then supply its actual new path:

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  JUPYTER_PATH=/tmp/dongxi-course-reproduction.ZfVaEu/kernel/share/jupyter PYTHONPATH=src \
  /tmp/dongxi-course-reproduction.ZfVaEu/venv/bin/python scripts/verify_course_notebooks.py \
  --notebooks notebooks/day-25/02_ragged_kv_cache_and_exact_recovery.ipynb \
  --kernel dongxi-course-clean --expected-prefix /tmp/dongxi-course-reproduction.ZfVaEu/venv \
  --timeout 120 --export-figures

CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
  /tmp/dongxi-course-reproduction.ZfVaEu/venv/bin/python -m dongxi_llms.batched_cache_lab \
  --report /tmp/NEW-cache-verification.json --workspace /tmp/NEW-cache-verification-states \
  --notebook-manifest /tmp/ACTUAL-NEW-manifest/manifest.json --export-previews
```

To reproduce the explicit v1-to-v2 migration diagnostic, use a new workspace
inside `fixtures/batched-cache-recovery-hardening/` and add
`--historical-reference experiments/reports/2026-10-04-batched-cache-recovery.json`.
The diagnostic restricts historical tensor reads to this package's owned
fixture roots and does not rewrite or support resuming obsolete snapshots.

These commands preserve prior outputs. The temporary interpreter/kernel paths
identify actual execution here, not a portable environment already present on
Mac. The locked CPU reproduction appendix describes building its equivalent.

## Remaining limits

The evidence establishes actual tiny ragged caching and same-environment DPO/RLVR
recovery. It does not retrofit recovery into the optional pretrained Spark
SFT/DPO/RLVR runners, validate production multi-epoch PPO, infer cross-device
bitwise continuation, certify adversarial checkpoint safety, or establish
language/reasoning skill. Model-scale recovery, optimized-engine comparison,
Spark throughput/peak memory, Mac execution and learner practice remain separate
pending evidence. The learner's live position stays Day 9.
