# Day 23 — A full autoregressive RLVR update

Material status: ready for study; source notebooks are CPU companions, with adjacent
reference solutions and explanatory plots. Notebook execution is recorded separately
from learner mastery. No GPU/model download/server is required.

Chapter: [13](../../book/chapters/13-group-relative-policy-optimization.md).
Lab route: [guide](../../book/labs/13-group-relative-policy-optimization.md).
Solutions: [worked answers](../../book/solutions/13-group-relative-policy-optimization.md).

1. [decoder rlvr](01_decoder_rlvr.ipynb)
2. [verifier contract](02_verifier_contract.ipynb)
3. [reasoning tasks and positive controls](03_reasoning_tasks_and_positive_controls.ipynb)

Learning outcome: Trace one TinyDecoder rollout through EOS, strict verification, group statistics and a real backward/update; compare G 4/G 8 with unequal cost recorded.

Acceptance: Reproduce response alignment, prove old/reference immutability, inspect every held-out row and identify an exploitable substring reward.

The third session is an additive DXI-04 control, not a replacement for the
earlier negative arithmetic results. Three predeclared tiny-decoder seeds receive
actual sampled autoregressive updates on a known-solvable, initially unsaturated
predicate. EOS is sampled and learned. Compare paired frozen policies, an exact
lookup counterexample and an independent oracle; inspect new-source/template/
family failures, full-vocabulary decoding and cap1. Five original figures expose
learning, stopping and limitations. English arithmetic/algebra/two-step fixtures
validate evaluation coverage, not this symbolic decoder's language capability.
References follow each exercise; the [specification](../../experiments/specs/2026-10-04-reasoning-controls.md)
and [measured report](../../experiments/reports/2026-10-04-reasoning-controls.md)
keep CPU learnability separate from the unexecuted pretrained baseline.

Machine: Mac or Spark CPU. Model-scale execution requires its own frozen config,
resource guard and measured report. Animation production remains Mac-only and
requires its own instruction.
