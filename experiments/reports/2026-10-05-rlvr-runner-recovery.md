# RLVR runner recovery verification

The actual `qwen_rlvr_lab.py` runner now separates collection from application
and publishes trusted-local snapshots at two durable boundaries. Bounded tests
with original randomly initialized HF Qwen3 models on CPU reproduce the entire
remaining training state from both boundaries, including a fresh process. This
establishes the runner mechanism for the fixed local recipe, not pretrained
Qwen capability, CUDA reproducibility or a completed Spark training campaign.

The [premeasurement specification](../specs/2026-10-05-rlvr-runner-recovery.md)
freezes seeds 2323 and 2324, four updates, group size 3, generation cap 4,
FP32, context 32, AdamW learning rate 0.008, zero weight decay, KL coefficient
0.02 and gradient clipping at 1. Models have vocabulary 16, hidden width 16,
intermediate width 32, one layer, two query heads and one KV head. Three authored
integer-token prompts cycle deterministically. No model or tokenizer was
downloaded; no GPU, installation, service or Git operation was used.

## Durable states and retained data

| Boundary | Policy and Adam | Source cursor and rollout RNG | Next action |
| --- | --- | --- | --- |
| Completed after update k | After k applications | After k complete collections | Collect the next pool |
| Pending for update k plus 1 | After k applications | After k plus 1 complete collections | Apply the saved pool without sampling |

Both states include the original frozen reference, policy, ordered Adam moments,
Python and global Torch RNG, rollout-generator state, applicable CUDA RNG schema,
full committed numerical and raw history, cumulative work and parent invocation.
The complete pending pool includes source and prompt IDs, response IDs, first-stop
inclusive masks, detached behavior log probabilities, decoded texts, rewards,
population advantages, stop IDs and policy/reference typed hashes. Reference
logits are rescored from the retained original reference, not from the updated
policy. Restoring a pending pool does not refresh the reference or draw tokens.

Resume requires independently retained scientific contract, payload SHA256,
exact byte count and an explicit size envelope. The CLI compares observable
source, lock, parent, role-input, tokenizer/template, encoded data, environment
and recipe identity before model allocation. It then compares effective model
configuration, selected attention backend, class, layout, tied aliases, ordered
optimizer bindings and original reference identity before applying saved state.
Only the documented operational model load path is excluded from configuration
identity. Actual source/input/lock/parent bytes are rehashed at closure.

Each invocation uses a new output directory. Initial, pending and completed
snapshots are exclusive publications before durable metrics; final training
state is committed before evaluation and HF export. The status journal separates
the latest durable completed count and phase from records written by the current
invocation. Thus pending at k reports k completed updates, not k plus 1 or zero
simply because a resumed invocation has not yet written a new metric.

## Actual verification commands

All owned collectors used the isolated CPU interpreter with `PYTHONPATH=src`,
CUDA hidden, HF and Transformers offline flags and one CPU thread:

```sh
env PYTHONPATH=src CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/dongxi-course-reproduction.ZfVaEu/venv/bin/python tests/test_rlvr_runner_recovery.py --verify-tests experiments/reports/2026-10-05-rlvr-runner-recovery-final-guards.json
```

The final owned collector exited 0: 19 tests passed in 7.024 seconds; the
subprocess wrapper measured 7.910143 seconds. Python was 3.12.14, Torch
2.14.1+cpu on ARM64 Linux, and CUDA was unavailable. The
[final command receipt](2026-10-05-rlvr-runner-recovery-final-guards.json)
retains the actual subprocess arguments, stdout, stderr, exit code, environment
and unchanged before/after hashes. Its
[raw observations](2026-10-05-rlvr-runner-recovery-final-guards.observations.json)
retain all collected pools, histories, fresh-process output and authored failure
controls. Snapshot payloads created inside temporary test directories are cleaned
up by the tests; the report retains their observed byte receipts and raw data,
not reusable checkpoint files from those temporary paths.

Earlier collections are historical evidence, not overwritten results:

- [First collection](2026-10-05-rlvr-runner-recovery-first.json): 13 tests passed.
- [Semantic and evaluation hardening](2026-10-05-rlvr-runner-recovery-hardening.json): 17 tests passed.
- [Final observer and durable status](2026-10-05-rlvr-runner-recovery-final.json): 18 tests passed.
- [Scoring guards and final source](2026-10-05-rlvr-runner-recovery-final-guards.json): 19 tests passed.

All four owned collections exited 0. Each binds its own measured source/test
identities; earlier identities are not relabeled as current. The original
monolithic runner source remains
[archived byte for byte](2026-10-05-rlvr-runner-recovery-original.py.txt).

## Numerical replay and sampled outcomes

The preserved `update_model` wrapper matches the archived original equations
exactly for both seeds across all four updates: returned numerical/raw values,
policy bytes, Adam state and rollout RNG agree after every update. Completed
and pending restarts agree with uninterrupted tails and full final state for
both seeds. The two fresh-process restarts use seed 2323 and compare the entire
tail and state; the pending test installs a collection spy that would fail if
the first restored application sampled again.

| Seed | Mean rewards by update | Valid response tokens by update | Generated slots including EOS fill | Generation forward positions | Old policy score positions |
| --- | --- | --- | --- | --- | --- |
| 2323 | 0, 0, 0, 0 | 10, 12, 7, 12 | 48 | 204 | 69 |
| 2324 | 0, 0, 0, 1/3 | 6, 12, 10, 10 | 45 | 189 | 66 |

Each seed has four policy-score and four reference-score application forwards;
each role processes 69 positions for seed 2323 and 66 for seed 2324. Generation
calls are 16 and 15 respectively. Counts include actual rectangular sampling
and EOS fill, while the validity mask excludes fill from the objective. Pending
and completed arms are deterministic replay checks of these same seed
trajectories, not four independent statistical samples.

Seed 2324 naturally sampled a positive answer at the fourth update, with
pre-clipping gradient norm 2.06782865524292. No sampled answer was repaired.
Seed 2323 never received positive task reward, but its floating-point exact-KL
gradient was not identically zero: the first norm was approximately 1.61e-9,
and later norms reached approximately 0.00286. The archived equations reproduce
these values. Zero task reward therefore does not imply bitwise unchanged Adam
training in this implementation, and these numerical effects are not evidence
of useful reasoning learning.

An additional explicitly authored positive application control has reward 1/3
and gradient norm 2.4231176376342773, changes policy parameters and leaves the
original reference unchanged and without gradients. It is a separate mechanism
control, not a naturally sampled training outcome. Authored first-stop, cap and
zero-signal controls likewise remain explicitly labeled. Comparing the first
and final raw reports shows all four sampled histories, pools and cumulative
work are unchanged by hardening; final full-state hashes change when the bound
loop-contract schema gains identity fields.

## Failures and refusal controls

Byte/digest/schema checks and semantic callbacks reject malformed snapshots
before `load_state_dict`. Controls include wrong payload SHA, byte tampering,
boolean counters and IDs, missing history, wrong source/version/cursor, shifted
mask, altered old likelihood, resealed reward/advantage/RNG/work data, original
reference reset or byte changes, broken policy-hash lineage, incompatible
optimizer order and invalid RNG layout. A private generator verifies rectangular
sampling consumption. Pending behavior likelihood validation uses temporarily
bound saved weights without applying them to the live policy or Adam; this is
separate recovery work, not included in training-budget counters.

A second-generation-call failure retains the first sampled IDs `[8, 3, 14]`,
one successful generation forward over six positions and three actual draws.
It leaves no resumable pending pool and poisons the live loop. Pre-old-policy
and pre-reference guard refusals retain successful work already observed and
zero scoring calls for the refused stage. Interrupted optimizer mutation also
poisons the live state; restoring the earlier durable pending checkpoint then
reproduces the clean next update without collection.

Injected baseline, metric, final evaluation and export failures preserve durable
updates 0, 1, 4 and 4 respectively. Failed new saves and existing-output refusal
preserve prior receipts. Evaluation retains legacy pre-stop tokens plus raw
stop-inclusive tokens, final stop ID, truncation, generated-token and successful
forward counts. A partial row is published before an evaluation error is raised;
both initial and final CLI observers retain it. Failed forward, backward or
optimizer internal costs are explicitly unavailable rather than fabricated.

The parent independently preserved the original inadequate eight-word tokenizer
as an actual
[encoded train and evaluation collision refusal](2026-10-05-rlvr-identity-encoded-collision.json).
It also ran a separately authored adequate 21-token random-model CLI fixture;
the encoded-disjointness gate was not weakened. The
[candidate independent panel](2026-10-05-rlvr-runner-recovery-root/acceptance.json)
passed 17 runner and 15 identity tests on its earlier frozen source. That panel
is historical; current-source independent acceptance is recorded separately.

The [final independent panel](2026-10-05-rlvr-runner-recovery-root-final/acceptance.json)
passed 19 runner and 15 identity tests with actual exit codes 0 and all 16 scope
hashes unchanged. It measured 10.529404 seconds and a minimum observed
before/after `MemAvailable` of 125,807,996,928 bytes, above the 25 GiB reserve.
These are sampled command-boundary memory observations, not a continuously
measured minimum during model execution. The panel includes the real positive
CLI fixture and retained encoded-collision negative control.

## Limits and remaining execution gates

This is same-environment, bounded FP32 CPU verification of the actual runner
functions and trusted-local recovery format. It is not CUDA/BF16 numerical
parity, cross-machine bitwise reproducibility, pretrained reward improvement,
large-checkpoint save performance, total disk/work containment or an OS memory
sandbox. Size envelopes bound each declared snapshot file, not all allocations
or total campaign disk usage. Mid-generation continuation is intentionally
excluded: an incomplete attempt must replay from its last durable boundary.

Pretrained Spark execution and model-scale resource measurements remain pending.
No missing campaign measurement is filled using these tiny CPU controls. Exact
source identities and the current independent panel are bound in the separate
[final verification receipt](2026-10-05-rlvr-runner-recovery-verification.json).
