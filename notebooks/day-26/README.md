# Day 26 — Selection and distillation

Material status: ready for study; source notebooks are CPU companions, with adjacent
reference solutions and explanatory plots. Notebook execution is recorded separately
from learner mastery. No GPU/model download/server is required.

Chapter: [15](../../book/chapters/15-distill-evaluate-and-defend.md).
Lab route: [guide](../../book/labs/15-distill-evaluate-and-defend.md).
Solutions: [worked answers](../../book/solutions/15-distill-evaluate-and-defend.md).

1. [temperature distillation](01_temperature_distillation.ipynb)
2. [selection and sampling](02_selection_and_sampling.ipynb)
3. [complete-response distillation](03_response_level_distillation.ipynb)
4. [critique, revision and acceptance](04_critique_revision_and_acceptance.ipynb)
5. [student prefixes and teacher context](05_student_prefix_and_teacher_context.ipynb)

Learning outcome: Derive T-scaled soft-target gradients, fit a student distribution, and compare candidate availability with actual selection.

Acceptance: Verify teacher detach and aligned vocabulary; identify winner's curse and cost/selection boundaries.

## Complete-response distillation

The third notebook extends Chapter 15 with the [Chapter 9 SFT objective](../../book/chapters/09-supervised-fine-tuning.md) bridge: actual larger tiny teacher responses, a genuinely generating smaller student, response/EOS masks, source/template provenance, first format-eligible selection and independent printed-step/final-answer grading. Three fixed CPU seeds and all negative slices are retained in the [report](../../experiments/reports/2026-10-04-response-distillation.md). Adjacent answers, live fixed-seed replay and six reference figures distinguish emitted text correctness from internal reasoning faithfulness. No pretrained/Spark transfer is executed.

## Critique, revision and acceptance

The fourth notebook is a Chapter 15 control-and-replay extension: original
programmatic critiques/revisions, every attempted and delivered round, twelve
authored failure cases, exact serialized-token independent-attempt controls and
replay of 864 actual tiny-model responses from Day 10. Six data-backed/reference
figures and adjacent worked explanations separate format gates from correctness.
The [frozen spec](../../experiments/specs/2026-10-04-critique-revision.md) and
[report](../../experiments/reports/2026-10-04-critique-revision.md) retain harmful
changes, invalid outputs, exceptions and budget overrun. This is not neural
self-critique or a new model run; historical model work is distinct from newly
measured callback/replay CPU time.

## Student prefixes and teacher context

The fifth notebook is a selected Chapter 15 extension. An actual tiny neural
student learns from an explicitly authored finite teacher on either fixed offline
prefixes or its freshly sampled histories. A three-factor comparison separates
prefix source, forward/reverse KL and training-only teacher hints; evaluation
never receives those hints. Exact top-k-plus-tail and zero-support controls expose
mass conservation, lost token detail and gradient boundaries. All 24 fitted arms
and negative held-outs remain in the [report](../../experiments/reports/2026-10-04-student-prefix-distillation.md).
It is not a pretrained teacher, state-occupancy gradient or full SDPO reproduction.

Machine: Mac or Spark CPU. Model-scale execution requires its own frozen config,
resource guard and measured report. Animation production remains Mac-only and
requires its own instruction.
