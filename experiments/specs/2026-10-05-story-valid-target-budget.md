# Story training valid target budget and resume contract

This specification precedes the new bounded CPU measurements. It implements
one unresolved source gate from the staged Spark campaign: a cumulative cap on
valid training targets. It does not authorize that campaign, load historical
weights or prepared TinyStories data, establish GPU behavior, or revise any
September result or October campaign report.

## Counting and stopping

A valid target is a training label unequal to `IGNORE`, including a genuine EOS
label and excluding positional padding. Count all accumulation microbatches in
the next proposed optimizer update. Accept the whole update only when its valid
target count plus the previously completed count does not exceed the positive
integer cap. Equality is allowed. Otherwise reject the entire update before
forward computation, gradient computation, learning-rate mutation or optimizer
mutation. Do not shorten a batch, erase labels, change the objective denominator,
or count rejected targets as completed training exposure.

Reject zero, negative, boolean, fractional, nonfinite and other noninteger caps;
`None` is the default and retains the existing uncapped recipe and numerical
path. The cap is separate from the fixed optimizer schedule. An active cap is
part of the checkpoint compatibility contract; changing or removing it requires
a different declared experiment, not silent continuation.

Batch collection advances the single-process shuffle stream, possibly across an
epoch boundary. Restore its pre-collection permutation, cursor, epoch and RNG on
a rejected update. The deterministic document-window loader can then reconstruct
the same next microbatches; a tensor queue is unnecessary. This guarantee does
not cover asynchronous workers, prefetch, stochastic data transforms or an
interrupted partially executed optimizer update.

Checkpoint cumulative valid targets and actual physical input positions
separately. Resume starts from these saved cumulative counters, not zero. The
next update must still fit the same cap. Validate counters and the cap before
loading model or optimizer state. A same-source older checkpoint without the
new physical-position field may derive its count from completed updates and the
historical fixed geometry, explicitly identifying that derivation. Existing
implementation hashes still reject historical source revisions; this change
does not claim compatibility with the September checkpoint files.

The run identity and completion record must identify the requested cap, unused
valid-target allowance, rejected next-update count, and target-budget stopping.
Time stopping, schedule completion and requested-update stopping remain distinct.
An ordinary cap stop is a clean bounded outcome, not optimization failure or
evidence that the model learned coherent stories. Physical positions consumed
by completed training updates remain separate from validation/generation work.

## Frozen CPU recipe and controls

Use the already approved isolated CPU interpreter with CUDA hidden, offline HF
flags and one Torch thread. Construct a fresh original decoder only: vocabulary
16, width 8, two query and KV heads, head dimension 4, one layer, hidden width 16,
context 8, modern MHA without QK normalization, tied input/output weights.
Use four deterministic authored ID windows with valid lengths 3, 6, 1 and 7,
EOS ID 15, positional `IGNORE` padding, seed 909, AdamW peak 0.003/floor 0.0003,
five-update schedule, one warmup update and no activation checkpointing. Force
the first shuffle permutation to `[0,1,2,3]` only for the explicitly identified
boundary controls; normal seeded shuffling is a separate path.

With microbatch 1 and accumulation 2, the first update has 9 valid targets and
16 physical positions; the next has 8 and 16. Freeze caps 8, 9, 16 and 17 to test
rejection before the first step, exact equality, a positive but unusable remainder
and two exactly fitting steps. Also test accumulation across an epoch boundary,
one-microbatch collection, legitimate EOS versus masked pad IDs, empty-target
rejection, and uncapped versus sufficiently capped numerical equivalence.

Save and restore before and after cap rejection; require bitwise equality of
model and optimizer tensors, schedule counters, stream state and CPU RNG for the
compatible replay. Test changed/missing cap rejection and malformed cumulative
counters before any model load. CLI tests must use mocked runner dispatch;
orchestrator tests may substitute this tiny model/data and stub observation and
memory monitoring. They must not invoke the full-size baseline, a tokenizer
download, actual story inference or an external service.

## Evidence and limits

Retain every control outcome, test exit, actual invocation, Python/Torch/host
identity, source/spec/test hashes and bounded elapsed time in a new report.
Preserve any initial failure rather than replacing it with a passing account.
The fixture uses actual tiny decoder forward/backward/AdamW updates where an
update is accepted, and asserts no forward call when the cap rejects it. This
verifies the local update-boundary instrument, not throughput, model quality,
production supervisor containment, disk limits, profile fit or a Spark pilot.
All external campaign rows remain null and pending.
