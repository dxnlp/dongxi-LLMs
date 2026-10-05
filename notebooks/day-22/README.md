# Day 22 — GRPO: relative rewards, ratios and KL

Material status: ready for study; source notebooks are CPU companions, with adjacent
reference solutions and explanatory plots. Notebook execution is recorded separately
from learner mastery. No GPU/model download/server is required.

Chapter: [13](../../book/chapters/13-group-relative-policy-optimization.md).
Lab route: [guide](../../book/labs/13-group-relative-policy-optimization.md).
Solutions: [worked answers](../../book/solutions/13-group-relative-policy-optimization.md).

1. [group advantages](01_group_advantages.ipynb)
2. [token ratios kl](02_token_ratios_kl.ipynb)
3. [Same rollouts, different objectives](03_objective_weighting_and_filtering.ipynb):
   reduction/scaling gradients, asymmetric clipping and real rejected-group costs.
   This optional extension computes gradients without training a model.

Learning outcome: Predict what constant groups can teach; derive population-standardized advantages and both clipping branches; compare exact KL gradients against autograd.

Acceptance: Explain a zero-value/nonzero-gradient first update and why a sampled KL value identity does not specify its complete gradient.

Machine: Mac or Spark CPU. Model-scale execution requires its own frozen config,
resource guard and measured report. Animation production remains Mac-only and
requires its own instruction.
