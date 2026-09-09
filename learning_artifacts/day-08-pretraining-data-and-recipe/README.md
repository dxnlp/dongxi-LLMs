# Day 8 — Pretraining data and recipe

- Opened for orientation and learning-design planning: 2026-09-09.
- State: complete requested Day 8 materials prepared; guided study and mastery pending.
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

The learner's intended continuation is Day 8. The prior Day 6 resume instruction
was an agent mistake: unfinished earlier practice must not override current
intent. Preserve Day 4–7 gaps as optional review. Save predictions, interventions,
explanations, and evidence as they occur, rather than marking content "read."

Departure update, 2026-09-09: learner invoked `Switch to Mac`. The Spark task
prepared the scoped records for commit/push and refreshed `LEARN-MAC-001`.
On the Mac, use `Continue on Mac` in a locally executing course task; arrival
must verify repository synchronization and the platform runtime. No additional
learning, interactive implementation, or GPU run occurred during departure.

Return/correction, 2026-09-09: user reported successful Mac repository sync but
dependency friction, rejected the Day 6 instruction, and requested continuation
on Spark. Confirmed actual execution on `spark-aa66` and safely pulled the repo.
User clarified that rich visuals should live directly in the conversation.
First inline demonstration: one SGD update on the toy objective L(w)=w squared,
starting w=2 with gradient=4. Slider changes learning rate and recomputes the
next parameter/loss; a discussion action carries selected values back to chat.
This is a limited SGD example, not the full planned SGD/momentum/AdamW lab or
evidence of training a language model. Learner interaction/understanding is not
yet assessed. Source is outside the repository in this task's visualization
directory (`day-eight-learning-rate.html`); no cross-machine Git export claimed.

Validation: headless Chromium exercised learning rates 0.20, 0.50, and 1.20,
giving next losses 1.44, 0.00, and 7.84. Checked 736px/360px in light/dark themes:
no JavaScript errors, label overlap, or horizontal overflow. Inspected wide-light
and narrow-dark screenshots. Discussion-button fallback was tested outside the
host; actual host follow-up confirmation is not claimed as exercised. Browser
testing used isolated cached tooling; no course Python environment was changed.

## Complete material request — 2026-09-09

The learner found the isolated quadratic slider boring, requested the Day 8
outline again, and then explicitly asked to build the notebooks and coherent
chapter directly. Prefer concrete LLM-system questions over generic numerical
controls. This feedback does not cancel the preference for rich live visuals.

Prepared canonical pathway:

- [Chapter 6](../../book/chapters/06-pretraining-as-a-controlled-system.md):
  complete Day 8 data, objective, optimizer, stability, validation and recovery
  narrative; Day 9's larger run evidence remains pending.
- [Three worked notebooks](../../notebooks/day-08/README.md): data/budgets and
  accumulation; AdamW/schedules/clipping/precision; validation/checkpoint recovery.
- [Twelve conceptual solutions](../../book/solutions/06-pretraining-as-a-controlled-system.md)
  and [bounded specification](../../experiments/specs/2026-09-09-day8-bounded-pretraining.md).
- [Verification report](../../experiments/reports/2026-09-09-day8-material-verification.md):
  measured fixture results and explicit CPU-only limits.

Refined mechanisms recorded for upcoming discussion: targets vs processed
positions; same token count vs same context; global target weighting vs mean of
means; AdamW moment state vs raw gradient; schedule update clock vs token clock;
clipping norm vs final update; BF16 range vs precision; complete process state
vs saved weights. These are authored explanations, not claims that the learner
has independently explained or defended them.

Next: remain on Spark, start with Notebook 1's data/recipe dilemma in conversation,
and record the learner's prediction and explanation. Older Days 4–7 gaps remain
optional review. No GPU campaign, animation production, server start or machine
switch was authorized or performed by this content request.
