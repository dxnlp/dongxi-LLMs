# Complete Course Design — 15 Chapters, 28 Learning Days

Authorized2026-10-04: prepare the complete course now, including future chapters,
visual notebooks, worked solutions, reusable code and experiments. This supersedes
the usual incremental-material rule for this build. The learning position remains
Day 9 until the learner resumes; writing future material does not complete study.

## One continuous argument

The book follows a model from its input boundary to a defended development result.
Evidence standards come first because every later chapter makes empirical claims.
Tokenization fixes the units; the decoder specifies conditional distributions;
the objective and optimizer change those distributions; evaluation decides which
changes matter. Demonstrations and preferences shape behavior, then policy rewards
introduce another objective and additional opportunities for failure.

The capstone compares checkpoints using an independent contract. It returns to
Chapter 1's question with a substantially richer system: can the evidence support
the desired model-development claim?

## Teaching structure

Every chapter supplies a motivation, explicit assumptions and tensor conventions,
mechanism explanation, derivation where needed, worked example, executable
companion, meaningful perturbation, evidence boundary, deep exercises and complete
worked answers. Several short notebooks carry the mechanism/failure/integration
path. Original procedural fixtures make the CPU lessons runnable offline and
their intended answers inspectable.

The question comes before the answer, while the runnable reference is adjacent.
Readers can spend their attention on the mechanism rather than external syntax
searches. Plots use the actual tensors or stored measurements. Architecture and
process diagrams identify themselves as schematics. Neither a decorative chart
nor a generic optimizer slider substitutes for an LLM-relevant intervention.

## Chapter questions and progression

| Chapter | Main question | Necessary evidence |
|---:|---|---|
| 1 | What does a successful run prove? | Identity, declared criteria, actual report |
| 2 | What does the model receive when we write text? | Frozen encoding, Unicode round trips, embedding paths |
| 3 | How does one observed token train a distribution? | Stable probabilities, independent gradient check |
| 4 | What information may a prediction use? | Causal invariants, Q/K/V gradients, cache equivalence |
| 5 | How do decoder components work together? | Shapes, gradients, design accounting, integration |
| 6 | What makes training a reproducible process? | Target/update clocks, state recovery, measured run |
| 7 | When is one model better on the intended task? | Frozen metric, paired errors, uncertainty, leakage controls |
| 8 | How do messages become supervised positions? | Serialization, completion labels, visibility boundaries |
| 9 | How does a base model learn assistant behavior? | Correct SFT updates, adaptation comparison, independent evaluation |
| 10 | What can pairwise preference identify? | Reward margins, fitting, bias and calibration checks |
| 11 | How do preferences update a policy directly? | Reference-relative sequence ratios and independent behavior |
| 12 | How does a sampled completion receive credit? | Exact-versus-sampled gradients, baseline and clipping controls |
| 13 | What does a group of rollouts contribute? | Verifier correctness, group advantages, detached policy roles |
| 14 | Why can reward improve as behavior deteriorates? | Proxy/true metric divergence, telemetry, freshness checks |
| 15 | Can we defend the final model and its cost? | Distillation/selection comparisons, genealogy, release evidence |

Chapters2–5 establish the architectural prerequisites for SFT and sequence-level
preferences. Chapter 7 precedes all claims about post-training gains. Chapter 12's
estimator and graph boundaries are prerequisites for Chapter 13's grouped
surrogate. Chapter 14 supplies failure categories that Chapter 15 must include in
its final model card.

## Evidence at two scales

CPU demonstrations test exact objectives, gradients, masks, estimators, verifier
logic and deliberate failures. A tiny model can also show a measurable learning
signal on original fixtures. Its capability claim stays within that fixture.

Spark extensions supply bounded model-scale protocols and executable paths where
implemented. They require explicit identities, frozen evaluation, memory/time
limits and actual reports. The completed TinyStories run is existing measured
GPU evidence. New Qwen SFT/DPO/RLVR protocols remain unexecuted until their own
measurements exist. A model checkpoint's presence does not itself prove a gain.

Material readiness, experiment execution and learner mastery are tracked as
three independent states. The full-course verifier checks references and links;
it cannot grade the learner's defense or invent model outcomes.

## Machines and public assets

Use Mac for reading, small CPU mechanisms, notebook plots, writing and approved
animation production. Use Spark for actual model-scale training and inference.
CPU lessons are also supported on Spark when the learner chooses it. Switch
phrases preserve the durable course state through Git and the handoff.

Animation candidates are grouped by mechanism: text-to-loss, causal retrieval,
assistant-only supervision, preference margins, policy clipping, grouped
advantages and failure/selection dynamics. The course includes precise production
briefs; existing approved media remain usable. Building future lesson materials
does not commission new rendered films or publish articles externally.

## Completion checks for the material build

All 15 chapters, matched worked solutions, introductory material and4appendices
must be navigable. Every one of the 28 days has a concrete notebook route and
defined output. All executable references run in fresh CPU kernels; independent
tests cover important invariants. Specs precede new demonstrations and reports
carry actual results. Source and figure hashes, environment versions, exceptions
and limitations are retained. Markdown mathematics and local links are checked.

The [sequence](COURSE_SEQUENCE.md), [notebook map](NOTEBOOK_CURRICULUM.md) and
build-verification report are the concrete inventory. Public publication and
model-scale campaigns remain separately recorded decisions.

The [experiment matrix](EXPERIMENT_MATRIX.md) connects every chapter's question
to an executable route, measured or unexecuted status, and precise evidence
boundary. The [animation storyboard catalog](../visuals/animations/COURSE_STORYBOARDS.md)
provides portable mechanism briefs without commissioning production.
