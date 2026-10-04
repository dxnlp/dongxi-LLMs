# Day 26 — Selection and distillation

Status: full material prepared on 2026-10-04; live learner demonstration remains
unassessed. This records the curriculum's mechanism and reusable prompts, not a
claim that the learner has studied or mastered the chapter.

## Book placement and executable route

[Chapter 15](../../book/chapters/15-distill-evaluate-and-defend.md),
[worked solutions](../../book/solutions/15-distill-evaluate-and-defend.md),
[notebook route](../../notebooks/day-26/README.md).

## Learning arc

Derive T-scaled soft-target gradients, fit a student distribution, and compare candidate availability with actual selection.

Begin with the conceptual tension, record a prediction, inspect the smallest
transparent implementation, change one assumption, and explain the result and
its boundary. Exercises use mathematics to support an argument rather than
arithmetic as the main learning activity.

## Mechanism to preserve

T² scaling changes gradient strength; temperature used for training need not be used at inference. Best-of-N can amplify scorer mistakes. Rejection changes the data mixture.

## Demonstration still required

Verify teacher detach and aligned vocabulary; identify winner's curse and cost/selection boundaries.

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

## Article and animation opportunities

teacher/student bars plus T² derivative; oracle availability versus imperfect winner selection, deferred Mac candidates.

Potential article: selection and distillation explained through the chapter's concrete
failure and evidence. Source: roadmap/agent synthesis. Both are proposed content,
not approved production; link the canonical derivation and report before wording
or rendering. Mac handles any later animation production.

## Next live-learning action

Open the first session and ask the learner to defend the prediction in words,
then reveal the adjacent reference answer and inspect the plot together. Keep
machine progress and study status in the central trackers; do not automatically
launch a long experiment or public publishing operation.
