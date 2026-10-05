# Story training target budget verification

The story runner now accepts an optional cumulative valid-target limit. It stops
before an optimizer update would cross that limit, preserving the rejected
update's data-stream state. Nineteen new CPU checks and twelve existing story
pipeline checks pass. This verifies an instrument needed by the proposed Day 9
comparison; no Spark profile, new story run or trained comparison was executed.

The [specification](../specs/2026-10-05-story-valid-target-budget.md) was saved
before these measurements. The [raw report](2026-10-05-story-valid-target-budget.json)
records accepted update metrics, refusals, state identities, test commands and
full test output, actual exits, environment and ten unchanged source/input
hashes. Historical September results and the October campaign specification,
preparation matrix and supervisor reports remain unchanged.

## Update boundary and observed outcomes

A training label unequal to `IGNORE` consumes one valid-target presentation.
Real EOS labels count; EOS-valued positional padding with an ignored label does
not. The runner assembles the entire next accumulation group and counts its
labels before forward computation. If the group would cross the cap, it restores
the preceding stream permutation, cursor, epoch and RNG and refuses the whole
update. It does not truncate targets, change the batch or reweight the loss.

The authored CPU fixture has valid window lengths 3, 6, 1 and 7. With two
microbatches per update, its first update has 9 valid targets and 16 physical
input positions; its second has 8 and 16.

| Cumulative target cap | Accepted updates | Completed valid targets | Completed physical positions | Refused next targets | Unused allowance |
|---|---:|---:|---:|---:|---:|
| 8 | 0 | 0 | 0 | 9 | 8 |
| 9 | 1 | 9 | 16 | 8 | 0 |
| 16 | 1 | 9 | 16 | 8 | 7 |
| 17 | 2 | 17 | 32 | 8 | 0 |

All four refusals made zero forward calls. Model weights, optimizer state,
retained gradients, model training/evaluation mode, completed counters, stream
state and CPU RNG matched their pre-refusal state bit-for-bit. The positive
remainder at cap 16 is intentional: spending all 16 targets would require a
different batch or objective. An epoch-crossing test separately verifies that a
refused collection does not consume the next shuffle or its RNG state.

There are seventeen actual tiny forward/backward/AdamW reference updates in the
raw measurement panel, in addition to the bounded test-suite updates. The
accepted updates have finite loss and gradients; their timings are retained as
small CPU observations, not a Spark throughput prediction.

## Default behavior and checkpoint recovery

`valid_target_budget=None` remains the default. The optimizer recipe, AdamW
settings, target-weighted loss, update schedule, initial weights and training
order do not change. A separate test compares all five uncapped updates with an
independent implementation of the original update equations, including exact
loss, learning rate, gradient norm, parameter/optimizer state and stream/RNG
state. A sufficiently large cap also matches the five-update uncapped path
bit-for-bit. Both paths finish with 43 valid target presentations and 80 physical
positions in the collected panel. Wall-clock measurements need not match.

An active cap is stored in the compatibility contract without becoming an
optimizer hyperparameter. Adding, removing or changing it rejects resume before
model-state loading. Matching-cap resume restores previously completed targets
and physical positions rather than resetting the allowance. The recorded
completed-boundary replay loads update 1 with 9 targets and 16 positions; its next
update exactly matches uninterrupted update 2 with cumulative totals 17 and 32.
The full resulting state identity is
`cd1f57a9f93e2cca48f9c37dcdd5a3c9c15ba0e1f54b9343d002977be919632f`.

A checkpoint saved after cap rejection reconstructs the same pending
microbatches and rejects them again under the same cap. Invalid negative,
noninteger, boolean, inconsistent or over-cap counters are rejected before
model loading. Current checkpoints save cumulative physical positions measured
from actual input tensor sizes. A deliberately constructed, same-source
checkpoint without that new field is marked
`derived-legacy-fixed-geometry-plus-measured-continuation`; its initial position
count is a derivation, not an independent historical measurement. Existing
implementation-hash compatibility still rejects older source revisions. No
September checkpoint was loaded, migrated or certified by these tests.

The run metadata records the cap, and a refused update produces
`target-budget-stop.json`. Completion records distinguish target-budget
stopping from time-budget stopping, requested-update completion and schedule
completion. A clean target stop saves the last completed state, including the
untrained state if the first update cannot fit. The full-run test substitutes
only the tiny decoder/data and mocks observation helpers, tokenizer loading and
memory monitoring; it verifies clean target completion and a matching-cap
resumption, not a large baseline launch or an actual memory guard.

## Verification and reproduction

The isolated interpreter is Python 3.12.14 with Torch 2.14.1+cpu on Linux aarch64;
CUDA is unavailable. The report collector exited 0 in 3.591850 seconds,
including its reference controls and the two test processes. The focused suite
ran 19 tests in 0.755 seconds and the unchanged pipeline suite ran 12 in 0.554
seconds; both actual exits were 0. The separate initial focused invocation also
passed 19 tests. No failing initial attempt occurred in this task.

From the repository root, the recorded collector invocation is:

```bash
env CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
  /tmp/dongxi-course-reproduction.ZfVaEu/venv/bin/python \
  tests/test_stories_valid_target_budget.py \
  --report experiments/reports/2026-10-05-story-valid-target-budget.json
```

The output path must not exist. A new reproduction must use a new report path;
the collector refuses to overwrite earlier evidence. It runs only the authored
CPU fixture and fixed focused test commands. The normal training CLI now exposes
`--valid-target-budget`; inspecting or setting that option is not approval to
launch training.

| Artifact | SHA256 at collection |
|---|---|
| `src/dongxi_llms/stories_training.py` | `84efb975690045c85f02f327887ffca7e044ebc60f89125d40de44a990b9fefa` |
| `scripts/train_stories.py` | `59774cb658a9f84c4723a9c9c019f6d3ee462f2db79e3cba4391443e559faa35` |
| `tests/test_stories_valid_target_budget.py` | `74bac735e5948fadc209a614c6226629d9393776cf1614db3de84f9e83ef2fe5` |
| Frozen specification | `48f698263e67cf873666e796fb3be1d6212e6a2e8bd5c237464d58884a09b2b2` |
| Raw report | `97c287c1b1d096d941bf88d37b18a836d6a8640b4777b9d8656ca9561a594302` |

The JSON retains the complete ten-file before/after hash scope, including decoder,
data/stream sources, the original pipeline tests, the frozen campaign spec and
CPU lock. These source identities intentionally differ from historical training
identities; old evidence remains valid for its own frozen source, not this revision.

## Remaining limits

This change protects a valid-target cap at completed optimizer-update boundaries
for the existing deterministic single-process document loader. It does not make
a partially executed optimizer update transactional, recover asynchronous
prefetch or stochastic transforms, authenticate arbitrary checkpoint edits, or
provide production launch-tree containment. Temporary test checkpoint bytes are
not archived; the raw report retains replay identities and outcomes.

Physical-position telemetry covers completed training updates only, not
validation, sampling, rejected collection overhead or every attempted operation
before a crash. It is not a separately enforced physical-position cap. Actual
Spark behavior, safe profiling, disk/resource guards, interface and recovery
checks, explicit stage approvals and controlled story-quality ratings remain
pending. All model-scale campaign cells remain null. The learner remains on
Day 9; no service, installation, model acquisition, GPU launch, Git write or
global course tracker change occurred here.
