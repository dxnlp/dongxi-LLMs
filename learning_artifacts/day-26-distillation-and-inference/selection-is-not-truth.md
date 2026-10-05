# Selection is not truth, and a trace is not a distribution

Recorded during2026-10-04 course development, not an independently completed
learner session. Canonical teaching prose belongs in Chapter15; this artifact
preserves the conceptual thread and portable production opportunities.

The decisive distinction is between having an answer, selecting it and learning
from it. An actual tiny-model pool can contain three correct answers and still
deliver the five-vote wrong answer. A format gate can admit a well-formed wrong
demonstration. A student can fit selected demonstrations but fail a polite-prefix
control. A valid printed intermediate total can coexist with a wrong final,
and a correct final can coexist with an invalid printed total.

These observations are now exercised in the actual candidate, teacher-data
and neural response-distillation notebooks. Gold-blind decisions and independent
grading stay separate. Responses, rejected attempts, EOS/caps, valid-target
exposure and generation/scoring costs remain available; speculative natural-
language or internal-faithfulness explanations do not replace the evidence.

Canonical routes: [Chapter15](../../book/chapters/15-distill-evaluate-and-defend.md),
[actual candidates](../../notebooks/day-10/05_actual_candidates_and_answer_selection.ipynb),
[teacher-data SFT](../../notebooks/day-11/04_teacher_attempts_and_matched_rejection_sft.ipynb),
[response distillation](../../notebooks/day-26/03_response_level_distillation.ipynb),
and the [live selection pilot](../../visuals/interactive/README.md).

Two later bridges make the distinction sharper. In
[critique/revision](../../notebooks/day-26/04_critique_revision_and_acceptance.ipynb),
accepted format ties can turn a correct answer wrong, then undo it; identical
final accuracy hides that trajectory. In
[student-prefix distillation](../../notebooks/day-26/05_student_prefix_and_teacher_context.ipynb),
prefix source, conditional-KL direction and teacher-private context are separate
factors. Sampling current student states does not supply a state-occupancy
derivative, and conserving total tail mass does not conserve within-tail detail.
The teacher is explicitly finite/authored; the students actually train.

Possible future X article: “A correct answer in the pool is not a correct
answer to the user.” Explain oracle availability versus votes/likelihood,
then show how selection changes the future student's dataset. A second article
could contrast realized teacher traces with full probability-vector targets,
using printed-step/final contradictions without claiming hidden reasoning.
Both are proposed only; no publication approval is implied.
Further proposed angles: “Revision is a proposal, not an improvement” and
“Where the teacher meets the student's mistakes.” Both should preserve the
actual negative results and distinguish finite controls from pretrained claims.

Animation opportunity: keep one source identity visible from candidate attempts
through format gate, selection, masked response/EOS NLL and independent held-out
generation. Compare target-count tiles, not just number of demonstrations.
Separate teacher and student compute; split final correctness from printed-step
validity. Detailed anchors and caveats are in the
[animation inbox](../../visuals/animations/PROPOSALS.md). All production remains
a separately approved Mac task; no media was rendered here.
