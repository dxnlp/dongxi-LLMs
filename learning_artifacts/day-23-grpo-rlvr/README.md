# Day 23 — A full autoregressive RLVR update

Status: full material prepared on 2026-10-04; live learner demonstration remains
unassessed. This records the curriculum's mechanism and reusable prompts, not a
claim that the learner has studied or mastered the chapter.

## Book placement and executable route

[Chapter 13](../../book/chapters/13-group-relative-policy-optimization.md),
[worked solutions](../../book/solutions/13-group-relative-policy-optimization.md),
[notebook route](../../notebooks/day-23/README.md).

## Learning arc

Trace one TinyDecoder rollout through EOS, strict verification, group statistics and a real backward/update; compare G 4/G 8 with unequal cost recorded.

Begin with the conceptual tension, record a prediction, inspect the smallest
transparent implementation, change one assumption, and explain the result and
its boundary. Exercises use mathematics to support an argument rather than
arithmetic as the main learning activity.

## Mechanism to preserve

The course CPU mechanism uses original addition tokens, four frozen held-out pairs and fresh synchronous behavior versions. A strict final-answer checker does not certify reasoning. The optional local Qwen adapter is implementation readiness, not a measured model result.

The separately accepted native inference row is now captured in
[generated answers and grader interfaces](generated-answers-and-grader-interfaces.md)
and the [partial actual report](../../experiments/reports/2026-10-05-native-reasoning-rlvr-evidence.md).
Its Instruct/thinking-off/cap32 child completes all100 planned records, but declared
whole-output grades remain negative. The focused artifact separates unsupported
parsing, permissive format, natural stops and caps without silently rescoring a
visible `5 + 6 = 11` equation. This is one actual baseline, not native RLVR training,
a completed interface comparison or learner mastery. Later rows remain pending
until their own actual receipts are incorporated.

## Demonstration still required

Reproduce response alignment, prove old/reference immutability, inspect every held-out row and identify an exploitable substring reward.

Notebook readiness and agent verification are course-production evidence. They
must not be promoted to demonstrated learner understanding. Capture the
learner's actual initial explanation, correction and unresolved edge here during
live study.

## Experiment evidence and limits

[Bounded CPU specification](../../experiments/specs/2026-10-04-grpo-diagnostics-distillation.md)
and [actual report](../../experiments/reports/2026-10-04-grpo-diagnostics-distillation.md).
The CPU runs use original tiny fixtures. Exact logits/gradients and trained toy
policies are measured; selection draws are labeled simulations and system timing
budgets are projections. No Qwen/CUDA result is inferred.

The [cross-day recovery artifact](../day-12-sft-mechanics/durable-training-boundaries.md)
adds Chapter13§13.7.1's completed-versus-pending distinction: a fully collected
group is an observation to preserve, not a chance to resample a better group.
It extends CAND-ANIM-019 only as a Mac production proposal. See the source plan
for actual runner acceptance and separate pretrained/Spark gates; this addition
does not record a learner demonstration.

The native reader extension adds an independent external receipt for both
saved model-work and shared-I/O prefixes. Chapter13 keeps the historical
before-semantic-validation model-work capture; the physical journal retains
later validation charges. [Day25 Exercise8](../../notebooks/day-25/02_ragged_kv_cache_and_exact_recovery.ipynb)
compares completed/pending first actions using the same tiny random-Qwen recipe.
Its [predeclared microscope](../../experiments/specs/2026-10-05-rlvr-reader-lesson.md)
is a material-production control, not a newly completed learner session, model
quality result or external run. Full native publication geometry also belongs
in Chapter14 rather than being inferred from that manual one-boundary trace.

## Article and animation opportunities

prompt → rollout group → rewards → advantages → masked gradients, linked to real tensor shapes; Mac production deferred.

Potential article: a full autoregressive rlvr update explained through the chapter's concrete
failure and evidence. Source: roadmap/agent synthesis. Both are proposed content,
not approved production; link the canonical derivation and report before wording
or rendering. Mac handles any later animation production.

The October5 focused artifact also suggests “What did your reasoning benchmark
actually grade?” and extends existing CAND-ANIM-022 with separate response,
stopping and grading lanes. See `X-EVAL-001` in the central content queue and
the [animation proposal](../../visuals/animations/PROPOSALS.md). Both are suggestions
only: no drafting, rendering, publication or grading intervention is authorized
by their capture.

## Next live-learning action

Open the first session and ask the learner to defend the prediction in words,
then reveal the adjacent reference answer and inspect the plot together. Keep
machine progress and study status in the central trackers; do not automatically
launch a long experiment or public publishing operation.
