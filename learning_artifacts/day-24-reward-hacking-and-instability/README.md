# Day 24 — Objective failure, entropy and length

Status: full material prepared on 2026-10-04; live learner demonstration remains
unassessed. This records the curriculum's mechanism and reusable prompts, not a
claim that the learner has studied or mastered the chapter.

## Book placement and executable route

[Chapter 14](../../book/chapters/14-when-optimization-goes-wrong.md),
[worked solutions](../../book/solutions/14-when-optimization-goes-wrong.md),
[notebook route](../../notebooks/day-24/README.md).

## Learning arc

Predict the winner under the broken proxy, observe actual optimization, test a strict-reward restart, and compare length denominators.

Begin with the conceptual tension, record a prediction, inspect the smallest
transparent implementation, change one assumption, and explain the result and
its boundary. Exercises use mathematics to support an argument rather than
arithmetic as the main learning activity.

## Mechanism to preserve

Rising proxy reward can coincide with declining strict accuracy. Low entropy can reflect correct concentration or failure collapse. Response/token means weight gradients differently.

## Demonstration still required

Distinguish measurement from cause; derive expected-reward gradient and state what the paired repair does and does not test.

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

proxy/true curves diverge; identical-entropy policies have different accuracy; animation candidates only.

Potential article: objective failure, entropy and length explained through the chapter's concrete
failure and evidence. Source: roadmap/agent synthesis. Both are proposed content,
not approved production; link the canonical derivation and report before wording
or rendering. Mac handles any later animation production.

## Next live-learning action

Open the first session and ask the learner to defend the prediction in words,
then reveal the adjacent reference answer and inspect the plot together. Keep
machine progress and study status in the central trackers; do not automatically
launch a long experiment or public publishing operation.
