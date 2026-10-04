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

## Article and animation opportunities

prompt → rollout group → rewards → advantages → masked gradients, linked to real tensor shapes; Mac production deferred.

Potential article: a full autoregressive rlvr update explained through the chapter's concrete
failure and evidence. Source: roadmap/agent synthesis. Both are proposed content,
not approved production; link the canonical derivation and report before wording
or rendering. Mac handles any later animation production.

## Next live-learning action

Open the first session and ask the learner to defend the prediction in words,
then reveal the adjacent reference answer and inspect the plot together. Keep
machine progress and study status in the central trackers; do not automatically
launch a long experiment or public publishing operation.
