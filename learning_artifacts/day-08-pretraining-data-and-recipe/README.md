# Day 8 — Pretraining data and recipe

- Opened for orientation and learning-design planning: 2026-09-09.
- State: outline introduced; interactive implementation and guided study pending.
- Book placement: Chapter 6, *Pretraining as a Controlled System* (Days 8–9).
- Required result: a bounded pretraining specification, not an unbounded run.

## Introduced outline

Corpus provenance and held-out splits; tokenization/document boundaries and
shuffling; sequence length, microbatches, gradient accumulation and token budgets;
AdamW, learning rate, warmup/decay and clipping; BF16 and separate memory
categories; validation and checkpoint recovery. Day 9 executes and diagnoses the
approved recipe. No Day 8 exercises or experiments have been completed here.

## Learning experience requested

The learner explicitly requested more than notebooks: real-time interventions,
rich explanatory visuals, and discussion-driven experiments. They proposed Mac
Studio for interactive CPU work and Spark for heavy GPU work, and requested
reliable departure/arrival phrases with commit/push and safe synchronization.

Accepted design: [one journey, two environments](../../docs/LEARNING_WORKFLOW.md).
The chapter supplies coherent explanations; interactive tools expose immediate
causal effects; notebooks expose reproducible implementation and worked answers.
This is a workflow preference, not demonstrated mastery of the planned topics.

First proposed tool: Mac-local Optimizer Playground with loss landscape, actual
gradient/update arrows, adjustable learning rate/momentum, paired trajectories,
play/pause/step/reset, and faithful saved state. Distinguish toy-objective behavior
from measured decoder training. No UI/library is selected or implemented yet.

## Progress and portable continuation

[Current Mac handoff](../../docs/handoffs/CURRENT.md): `LEARN-MAC-001`.
Use `Switch to Mac` on departure and `Continue on Mac` after arriving. Reverse
with `Switch to Spark` / `Continue on Spark`. Synchronize durable course Git
state at the departure/arrival gates; runtime state and credentials are not
implicitly transferred.

Default unfinished learning remains Day 6 norm comparison. The learner may
choose the Day 8 pilot first without retrospectively completing Day 4/5 practice
or Day 7 defense. Save predictions, interventions, explanations, and evidence
as they occur, rather than merely marking content "read."

Departure update, 2026-09-09: learner invoked `Switch to Mac`. The Spark task
prepared the scoped records for commit/push and refreshed `LEARN-MAC-001`.
On the Mac, use `Continue on Mac` in a locally executing course task; arrival
must verify repository synchronization and the platform runtime. No additional
learning, interactive implementation, or GPU run occurred during departure.
