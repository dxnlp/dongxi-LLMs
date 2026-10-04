# Day 22 — GRPO: relative rewards, ratios and KL

Status: full material prepared on 2026-10-04; live learner demonstration remains
unassessed. This records the curriculum's mechanism and reusable prompts, not a
claim that the learner has studied or mastered the chapter.

## Book placement and executable route

[Chapter 13](../../book/chapters/13-group-relative-policy-optimization.md),
[worked solutions](../../book/solutions/13-group-relative-policy-optimization.md),
[notebook route](../../notebooks/day-22/README.md).

## Learning arc

Predict what constant groups can teach; derive population-standardized advantages and both clipping branches; compare exact KL gradients against autograd.

Begin with the conceptual tension, record a prediction, inspect the smallest
transparent implementation, change one assumption, and explain the result and
its boundary. Exercises use mathematics to support an argument rather than
arithmetic as the main learning activity.

## Mechanism to preserve

A group baseline compares sampled alternatives within one prompt. Centering, standardization and length reduction are separate choices. Old log probabilities and reward-derived advantages detach; the current logits remain differentiable.

## Demonstration still required

Explain a zero-value/nonzero-gradient first update and why a sampled KL value identity does not specify its complete gradient.

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

group-advantage bars → signed ratios → clipping plateau; production candidate, Mac only.

Potential article: grpo: relative rewards, ratios and kl explained through the chapter's concrete
failure and evidence. Source: roadmap/agent synthesis. Both are proposed content,
not approved production; link the canonical derivation and report before wording
or rendering. Mac handles any later animation production.

## Next live-learning action

Open the first session and ask the learner to defend the prediction in words,
then reveal the adjacent reference answer and inspect the plot together. Keep
machine progress and study status in the central trackers; do not automatically
launch a long experiment or public publishing operation.
