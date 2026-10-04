# Chapter 15 lab route — selection, distillation and release evidence

Prerequisites: Chapter 7 evaluation, Chapter 10 reward error and Chapter 13 policies.
CPU on Mac or Spark. No model or dataset downloads, external publishing or
animation rendering are part of these notebooks.

| Day | Session | Main question | Intervention |
|---|---|---|---|
|26|[Temperature distillation](../../notebooks/day-26/01_temperature_distillation.ipynb)|What does a softened teacher teach?|T² scaling and analytical/autograd gradients|
|26|[Selection and sampling](../../notebooks/day-26/02_selection_and_sampling.ipynb)|Does more sampling improve the selected answer?|Majority, imperfect ranker and oracle ceiling|
|27|[Genealogy and panel](../../notebooks/day-27/01_genealogy_and_frozen_panel.ipynb)|Which comparison does a model score identify?|Cycle/unknown parent and changed evaluation hash|
|28|[Defense and release](../../notebooks/day-28/01_defense_and_release_gate.ipynb)|What evidence supports the release claim?|Missing evidence, bounded claim and acceptance gate|

The three-logit student is trained; inference candidates are explicitly simulated.
The genealogy is a synthetic structural fixture and must be replaced by actual
checkpoint reports for a model card. The release gate assesses preparedness and
does not publish anything. Its booleans must come from recorded evidence,
not optimistic defaults.

Final task: defend one complete comparison using target, choice, rationale,
measured evidence, cost, failure and next experiment. Trace every number to a
report and every checkpoint to a parent. State unexecuted pathways explicitly.
[Worked answers](../solutions/15-distill-evaluate-and-defend.md) explain both
the mechanisms and the limits of each check.
