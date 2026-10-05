# Dongxi LLMs

**From equations to experiments: architecture, supervised fine-tuning, and reinforcement learning for language models.**

`Dongxi_LLMs` is an English-first, experiment-driven learning course for technically capable Python users with a foundation in deep learning. Its first learner is Dongxi; its public purpose is to help others move beyond calling LLM APIs and understand how models are designed, trained, evaluated, and diagnosed.

## Current status

The teaching draft covers **15 chapters, 28 days and 76 visual notebooks**, with
worked solutions, original bounded CPU experiments and optional Spark
SFT/DPO/RLVR runners. Start with the [reader's guide](book/README.md) and
[day-by-day route](docs/COURSE_SEQUENCE.md). Prepared material is not completed
learner practice or proof of pretrained-model performance.

The bounded reference-audit campaign and final Linux verification have completed.
The [signed original-criteria review](experiments/reports/2026-10-05-final-original-criteria-signoff.md)
closes **18 of 18 packages against all86 unchanged criteria and their dependencies**.
This is bounded course-production acceptance, not learner mastery. The
[pinned repository audit](docs/REFERENCE_REPOSITORY_AUDIT.md),
[improvement plan](docs/COURSE_IMPROVEMENT_PLAN.md) and
[acceptance review](docs/UPGRADE_ACCEPTANCE_REVIEW.md) distinguish implemented
mechanisms from scoped model and platform evidence.
The [JSON ledger](docs/course_improvements.json) is the cross-session record.
The [current empirical defense](experiments/reports/2026-10-05-final-native-campaign-defense.md)
now joins the actual first400 story, assistant400 and preference100 comparisons,
including432 common preference responses and680/800 initial reasoning responses.
Both RLVR groups have exact recovery and completed16 policies; four fresh
post-evaluations and both matched32/128 comparisons are complete. G4's failed
supervision is retained separately from its explicit export-consistency closure.
Held-out cap128 sampled correct counts are13/44,11/44,13/44 for unchanged
Instruct/G4/G8, with greedy7/11 unchanged—not a general method ranking.
Missing responses and failed attempts remain visible rather than becoming zero
scores or successful runs.

Final [current-source Linux verification](experiments/reports/2026-10-05-final-course-verification.md) passes
[1,533 CPU tests](experiments/reports/2026-10-05-final-goal-cpu/run-01/tests.log)
and [all76 fresh notebook references](experiments/reports/2026-10-05-final-goal-notebooks/run-01/manifest.json):
539 code cells,535 executed references,210 images and four preserved unfinished
learner exercises. The [campaign archive closing receipt](experiments/reports/native-campaign-evidence-20261005-run-01/closing-bindings.json)
binds45 original stage rows and74 retained model-directed terminal receipts, including
failures. It is a model-free evidence archive, not portable weights or a raw
corpus; its preclosure15/18 status documents remain immutable.
The [native recovery report](experiments/reports/2026-10-05-native-sft-replay.md)
and [actual response replay](experiments/reports/2026-10-05-pretrained-evaluation-replay.md)
keep exact recovery separate from negative answering/stopping behavior.
Historical source revisions and failed
checks remain in [the progress ledger](PROGRESS.md) and
[experiment matrix](docs/EXPERIMENT_MATRIX.md). Mac execution, hosted CI and actual
inline-host learner delivery remain unobserved; none is inferred from Linux or
local browser tests. Original platform criteria require these distinctions,
not added mandatory successful Mac/hosted runs. Apply the learner's current-gap
rule before further engineering: check upstream fixes and the installed stack,
reuse adequate solutions, and do not revive resolved memory-hang monitoring work.

The learner is on **Day 9 on Spark**. The
[first TinyStories run](experiments/reports/2026-09-14-tinystories-learning-result.md)
completed 14,000 updates and 48.84M valid target presentations, reaching
fixed-development NLL 1.6743 without establishing reliable coherence. The new
paired first400 comparison is separate; its later14k-schedule/publication cells
remain unrun. [Chapter 6](book/chapters/06-pretraining-as-a-controlled-system.md)
and its [evidence-reading lab](book/labs/06-reading-a-pretraining-run.md) are
the current companions; earlier practice is a review backlog, not a forced rewind.

The release target is a `v0.1` public beta; publication is a separate decision.
Mac Studio is the default live-learning, visual and media machine; DGX Spark
handles approved GPU experiments. The learner currently chooses Spark.
The primary pretrained model family is Qwen3.

## Start here

1. Read [`BOOK.md`](BOOK.md) for the reader-facing book architecture.
2. Read [`ROADMAP.md`](ROADMAP.md) for the authoritative four-week production plan.
3. Follow [the daily course route](docs/COURSE_SEQUENCE.md), then read
   [`PROGRESS.md`](PROGRESS.md) for the current position and next action.
4. Read [`LEARNING_MEMORY.md`](LEARNING_MEMORY.md) for the artifact index, content ideas, and cross-machine task packets.
5. Read [`learning_artifacts/`](learning_artifacts/) for deep discussions organized by day and topic.
6. Use [`notebooks/`](notebooks/) for the book's interactive mechanism lessons.
7. Contributors and coding agents must read [`AGENTS.md`](AGENTS.md) before changing the project.

## Continue across machines

Say **Switch to Mac** or **Switch to Spark** before leaving: the agent saves
progress and commits/pushes scoped course work. On the destination, open the
course project and say **Continue on Mac** or **Continue on Spark**: verify the
execution host, safely synchronize, and resume the recorded next step.
These are course conventions, not application slash commands. Full contract:
[learning workflow](docs/LEARNING_WORKFLOW.md); latest
[handoff packet](docs/handoffs/CURRENT.md).

## Learning promise

At the end of the course, the learner should be able to:

- design and defend an LLM architecture under explicit constraints;
- determine what data is needed for a target capability;
- design and justify pretraining, SFT, preference, and RL recipes;
- measure model capability with a frozen evaluation contract;
- monitor training health and diagnose common failure modes;
- explain the mathematics, implementation, evidence, and trade-offs clearly.

Expert answers should follow this structure:

> Choice → rationale → evidence → trade-off → failure risk → next experiment

## Project principles

1. Predict before running an experiment.
2. Evaluation is designed before training begins.
3. Every public empirical claim links to reproducible evidence.
4. Notebooks explain experiments; reusable logic lives in importable modules.
5. Failed experiments and negative results are retained and explained.
6. English is canonical. Chinese is used selectively for social publishing. Swedish reasoning research lives in the sibling `Dongxi_LLMs_Swedish` project.
7. Existing repositories are references, not material to concatenate or paraphrase.

## Local reference projects

- `../LLMs-from-scratch`: architecture and from-scratch implementation reference
- `../reasoning-from-scratch`: Qwen3-0.6B reasoning, evaluation, GRPO, and distillation reference
- `../rlhf-book`: broad post-training and instrumentation reference
- `/home/dongxi/dgx-spark-dongxi`: validated DGX Spark platform and memory-safety layer

## Planned model ladder

| Tier | Model | Role |
|---|---|---|
| Numerical microscope | 1–10M parameters | Derivations, gradients, and mechanism tests |
| From-scratch model | `DongxiGPT`, approximately 50–150M | Architecture design and brief pretraining |
| Glass-box model | Qwen3-0.6B | Frequent SFT, DPO, RM, and GRPO experiments |
| Flagship model | Qwen3-1.7B | Validate and publish important post-training results |
| Scale-transfer model | Qwen3-4B/8B | Selected LoRA and transfer experiments |
| Teacher/judge | Approximately 14–32B | Inference, synthetic data, and evaluation assistance |

## Version scope

The four-week target is a high-quality public beta, not the end of the mastery journey. Advanced architecture variants, broad algorithm surveys, large-model sweeps, full translations, and extensive distributed training are candidates for later releases.
