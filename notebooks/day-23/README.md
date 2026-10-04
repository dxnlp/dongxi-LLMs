# Day 23 — A full autoregressive RLVR update

Material status: ready for study; source notebooks are CPU companions, with adjacent
reference solutions and explanatory plots. Notebook execution is recorded separately
from learner mastery. No GPU/model download/server is required.

Chapter: [13](../../book/chapters/13-group-relative-policy-optimization.md).
Lab route: [guide](../../book/labs/13-group-relative-policy-optimization.md).
Solutions: [worked answers](../../book/solutions/13-group-relative-policy-optimization.md).

1. [decoder rlvr](01_decoder_rlvr.ipynb)
2. [verifier contract](02_verifier_contract.ipynb)

Learning outcome: Trace one TinyDecoder rollout through EOS, strict verification, group statistics and a real backward/update; compare G 4/G 8 with unequal cost recorded.

Acceptance: Reproduce response alignment, prove old/reference immutability, inspect every held-out row and identify an exploitable substring reward.

Machine: Mac or Spark CPU. Model-scale execution requires its own frozen config,
resource guard and measured report. Animation production remains Mac-only and
requires its own instruction.
