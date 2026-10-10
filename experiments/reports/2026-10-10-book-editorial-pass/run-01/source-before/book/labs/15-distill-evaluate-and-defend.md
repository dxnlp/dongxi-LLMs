# Chapter 15 lab route — selection, distillation and release evidence

Prerequisites: Chapter 7 evaluation, Chapter 10 reward error and Chapter 13 policies.
CPU on Mac or Spark. No model or dataset downloads, external publishing or
animation rendering are part of these notebooks.

| Day | Session | Main question | Intervention |
|---|---|---|---|
|10 bridge|[Actual candidates and selection](../../notebooks/day-10/05_actual_candidates_and_answer_selection.ipynb)|Can a correct available answer lose the vote?|Frozen real candidate pools, gold-blind rankers, errors and all attempted costs|
|11 bridge|[Teacher attempts and matched rejection SFT](../../notebooks/day-11/04_teacher_attempts_and_matched_rejection_sft.ipynb)|Does a better demonstration pool guarantee transfer?|Immutable attempt journal, format versus correctness, per-prompt/global coverage and matched exposure|
|26|[Temperature distillation](../../notebooks/day-26/01_temperature_distillation.ipynb)|What does a softened teacher teach?|T² scaling and analytical/autograd gradients|
|26|[Selection and sampling](../../notebooks/day-26/02_selection_and_sampling.ipynb)|Does more sampling improve the selected answer?|Majority, imperfect ranker and oracle ceiling|
|26|[Actual response distillation](../../notebooks/day-26/03_response_level_distillation.ipynb)|What transfers from a realized teacher trace?|Actual larger-tiny teacher, same-parent complete/answer-only students, step/final contradictions and unequal exposure|
|26 extension|[Critique, revision and acceptance](../../notebooks/day-26/04_critique_revision_and_acceptance.ipynb)|Which state reaches the user, and can revision harm it?|Programmatic state loop, unchanged actual-candidate replay, natural-stop gate and serialized-budget boundaries|
|26 extension|[Student prefix and teacher context](../../notebooks/day-26/05_student_prefix_and_teacher_context.ipynb)|What history receives which teacher target?|Actual tiny students, authored finite teacher, independent prefix/KL/context factors, exact tail detail and occupancy boundary|
|27|[Genealogy and panel](../../notebooks/day-27/01_genealogy_and_frozen_panel.ipynb)|Which comparison does a model score identify?|Cycle/unknown parent and changed evaluation hash|
|28|[Defense and release](../../notebooks/day-28/01_defense_and_release_gate.ipynb)|What evidence supports the release claim?|Missing evidence, bounded claim and acceptance gate|

The three-logit student is trained; the original Day26 inference candidates are
explicitly simulated. The Day10 bridge separately supplies actual tiny-decoder
candidates and measured CPU work; it is not a pretrained reasoning benchmark.
The Day11 bridge actually trains tiny sequence students from an explicitly
programmatic teacher; its failures are evidence, not a pretrained teacher claim.
The actual neural response-distillation session separately executes sequence
NLL; it does not replace the soft-distribution KL/T² microscope. Compare
teacher/student costs and final versus printed-step validity independently.
The refinement session is programmatic, not neural self-critique. Its accepted
format ties can harm a correct answer; historical candidate costs are replayed,
not new model calls. The student-prefix session actually fits neural students
but uses an authored finite teacher and training-only hints. It differentiates
conditional KL at fixed visited states, not state occupancy, and reports all
negative held-out outcomes. Neither extension establishes pretrained quality.
The genealogy is a synthetic structural fixture and must be replaced by actual
checkpoint reports for a model card. The release gate assesses preparedness and
does not publish anything. Its booleans must come from recorded evidence,
not optimistic defaults.

Final task: defend one complete comparison using target, choice, rationale,
measured evidence, cost, failure and next experiment. Trace every number to a
report and every checkpoint to a parent. State unexecuted pathways explicitly.
[Worked answers](../solutions/15-distill-evaluate-and-defend.md) explain both
the mechanisms and the limits of each check.

## Read an actual incomplete lineage

Before replacing the synthetic genealogy, inspect the
[current model-defense tables](../../experiments/reports/2026-10-05-current-model-defense.md).
Trace the disposable profile, fresh replay references and FP32 merge to their
actual parent identities. Explain why0/8 generated correctness does not negate
exact recovery, and why a failed BF16 merge is not repaired by a different
precision's pass. Reconcile455 numerical targets with684 physical training
presentations and1,440 development presentations per replay pair.

Then identify which original campaign rows are still unrun. No model loading,
new generation or Git publication is needed for this evidence-reading exercise.
That historical defense describes its short-run revision, not every later
artifact. The bounded final comparison needs actual results and source-bound
review, not successful execution of every optional or maximum-horizon row.
Solution10 shows how to preserve missing cells without discarding useful evidence.

## Compare two actual training lineages without conflating them

Read the [fresh story evidence](../../experiments/reports/2026-10-05-native-story-first400-comparison.md)
beside the [full/LoRA400 instruction evidence](../../experiments/reports/2026-10-05-native-sft400-comparison.md).
The story models start randomly; the instruction policies share a pinned
pretrained Base. Name each parent, objective, target exposure, held-out population
and decoding contract before comparing a number. Neither branch supplies the
other's missing descendants or unrun cells.

Predict whether 45 natural story endings should imply 45 satisfactory endings,
then check the separately submitted AI rating means and disagreements. Separate
instances do not establish independent human or model populations. Compare that boundary
with full400's 120 exact instruction answers within three familiar templates:
neither result establishes general assistant competence. Chapter6's model-free
archive exercise recomputes the retained joins; this lab requires no new model
loads, generations, notebooks, downloads or publication. The final controlled
reasoning defense still needs its own actual evidence. The
[accepted preference comparison](../../experiments/reports/2026-10-05-native-preference-comparison.md)
already supplies a further actual lineage: original full400→fresh chosen100
and DPO100, not Instruct→RLVR. Trace the common likelihood and greedy64
contracts, explain why a larger DPO margin accompanies fewer strict location
answers, and check the original instruction retention and per-item reasoning
changes. Shared chosen labels do not mean equal supervision or compute.
The [new final defense](../../experiments/reports/2026-10-05-final-native-campaign-defense.md)
labels the remaining reasoning/RLVR/closing verification incomplete rather
than filling it with the preference branch's results.
