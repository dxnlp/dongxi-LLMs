# Critique, revision and acceptance: frozen CPU replay protocol

Frozen before the first campaign. This is an original, offline mechanism experiment;
it does not fit or invoke a model, API, external tool or neural critic. Its purpose
is to inspect how a critique, revision rule and acceptance rule can change the
delivered answer. A correct draft may become wrong. An invalid draft may become
well formed without becoming correct.

## Hypothesis and evidence boundary

We expect a format-only acceptance rule to admit both beneficial and harmful
well-formed changes, and to reject invalid proposals. Two contrarian rounds can
undo each other. More emitted tokens need not improve the delivered answer.
These are mechanism hypotheses, not a prediction of pretrained self-refinement
quality. All counts and descriptive comparisons will include negative results.

The motivating reference is the pinned
[upstream self-refinement loop](https://github.com/rasbt/reasoning-from-scratch/blob/a788466dc85cfe8617b6c0b809ed4ab9084485de/reasoning_from_scratch/ch05.py).
It separates critique, revision and scorer-based acceptance, with accepted score
ties. Our independently written programmatic microscope is not its neural-loop
reproduction. No upstream code, prose, data or weights are copied.

## Inputs, panels and frozen recipe

The immutable JSON protocol is
[protocol.json](../../fixtures/critique-revision/protocol.json). It names SHA-256
digests for the original DXI05 item file and all 864 actual response records.
Existing response bytes, measured timings, failures and likelihoods stay intact.
The original item schema and encoded/source split gate are checked before replay;
its authored references are then used only by the separate evaluator. No fit,
selection or critique callback receives reference, grade or rubric metadata.

Three panels remain separate in every summary:

1. Twelve original adversarial fixtures demonstrate beneficial and harmful
   proposals, rejection, empty/capped outputs and critique/revision exceptions.
   They are chosen tests, not an empirical accuracy benchmark.
2. An exact serialized-token control uses all 18 DXI05 items and three fixed
   seeds: 26061, 26062, 26063. Eight independent binary-plus-EOS programmatic
   candidates per item/seed are drawn by a hash-derived coordinate seed. The
   generator never reads the reference or arithmetic operands. Each candidate
   emits two tokens. Critiques emit an action symbol plus EOS: also two tokens.
   Valid revisions emit two tokens. This makes spent output tokens exactly
   match an integer number of independent attempts. These are programmatic
   symbols, not tokens emitted by an LLM.
3. All 108 DXI05 model/item pools (six frozen checkpoint identities, 18 items,
   eight candidates each) are replayed without additional model forwards. Draft
   candidate zero is unchanged. Critiques and revisions are programmatic;
   independent controls consume the recorded candidate prefix. Historical model
   work is reported separately, never as work performed by this replay.

For each non-adversarial panel run three fixed modes: `identity` asks KEEP;
`repair` asks KEEP for a valid binary/EOS draft and REPAIR otherwise;
`contrarian` asks FLIP for a valid draft and REPAIR otherwise. REPAIR takes the
first observed binary token, defaulting to zero when none exists. It does not
solve the task. FLIP exchanges zero and one. No settings or seeds change after
observing the outcomes.

## State machine and controls

At most two rounds. Each round retains the input draft, critique emission,
proposed revision, callback exception/validation failure, acceptance decision
and delivered state. The acceptance gate requires exactly one binary token then
EOS, no error, and natural EOS termination. A valid proposal is accepted even
when its value differs from the current valid answer. Invalid/error proposals
are rejected. Stop only if an accepted token path is identical to the previous
path, or at round two. Correctness never stops or steers the loop.

No-critique delivers candidate zero and spends only its recorded output tokens.
The independent-attempt arm uses stable majority over eligible binary/EOS
responses; ties choose the earliest eligible answer, not a reference. Its ceiling
is the revision arm's draft-plus-critique-plus-revision serialized output count.
For the exact programmatic panel this ceiling is fully spent. In the actual pool,
a whole candidate exceeding the remaining ceiling is retained and charged but
excluded; underfill, overrun, exhausted pools and exact matches are reported.
Do not call this actual-pool comparison exactly matched or compute matched.

Critique/revision callback elapsed time and attempted/completed calls are newly
measured CPU quantities. The symbolic output count includes EOS and all emitted
invalid tokens. Actual-pool candidate generation/rescoring costs retain their
historical source attribution. Prefix reuse across modes does not mean that the
source model was run repeatedly. No model-token cost is attributed to a
programmatic operation.

## Acceptance and failure retention

Tests cover gold-stripped callback views, gold-injection invariance, bound/type
checks, invalid and empty emissions, capped paths, critique and revision failures,
accepted ties, unchanged stopping, wrong→right, right→wrong and double-flip paths,
exact serialized matching, actual-pool overshoot, immutable source hashes and
safe new-output journaling. Every attempted proposal and every delivered
transition is graded separately after decisions. Source/task slices are reported
descriptively; related source siblings are not an IID population or a confidence
interval. No favorable subset or scorer is selected after the campaign.

The fresh notebook must run in the clean CPU kernel with bounded time, show an
architecture/state diagram and data-backed transition/budget/quality plots, and
include worked explanations adjacent to experiments. CLI failure or interruption
must retain fsynced partial round records and a failure event. Resume/exactly-once
execution is not claimed for this package.

## What this will not prove

Neural self-critique, factual reasoning, faithful natural-language explanations,
pretrained improvement, equal FLOPs/latency/billing, production safety, general
repair success or transferable statistical significance. Optional same-model
generation remains unexecuted and requires a separate frozen protocol, actual
checkpoint/interface identity and measured raw continuations.
