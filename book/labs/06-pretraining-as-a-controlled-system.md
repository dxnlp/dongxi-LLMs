# Lab — Specify, Perturb, and Recover a Training Run

Use the [Day 8 notebook index](../../notebooks/day-08/README.md) for execution and
the [chapter](../chapters/06-pretraining-as-a-controlled-system.md) for the argument.
This is the Day 8 foundation of Chapter 6, not the later Day 9 scaling campaign.

## Three connected sessions

Begin with the data notebook. Explain which targets are counted, inspect the
input/label grids, then change context length. Compare gradients of the intended
mean objective against the deliberately broken reduction. Do not move on merely
because an assertion passes: explain what assumption the assertion isolates.

Next trace a real decoder gradient through AdamW. Compare its update with an
SGD sketch, inspect the schedule clock, and separate clipping, finite-value
checks, representable range and memory accounting. No single safeguard is a
substitute for the others.

Finally train the bounded fixture, inspect validation's denominator, and restore
three checkpoint variants. Complete restoration is the positive control;
missing moments and missing data position isolate two different failure paths.
The observations should explain why model weights alone are not training state.

## Evidence to retain

Record source/data identity, seed, dtype, microbatch and accumulation, valid
target count, update count, actual gradient/parameter differences, validation
contract, device and runtime. Separate schematic process maps, estimated memory
ledgers and measured curves. The verification command saves executed copies and
a manifest outside the source notebooks; important values are preserved in the
[material verification report](../../experiments/reports/2026-09-09-day8-material-verification.md).

## Discussion-first use

The learner need not complete every cell before discussing the mechanism. Start
with a concrete dilemma: “the curve looks good, but the two microbatches have
different valid lengths,” or “the checkpoint loads, but continuation changes.”
Predict the distinguishing observation, run the minimum test, and explain its
limits. Use figures to reveal the data flow; avoid isolated arithmetic quizzes.

Small CPU sessions can run on a verified Mac environment or Spark. The current
learner explicitly chooses Spark. Heavy GPU work requires its own bounded
specification and approval. Animation candidates are recorded separately and
remain Mac Studio production tasks, not authorized renders.
