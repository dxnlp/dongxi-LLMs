# Day 20 — Chapter 12 Companion Sessions

What changes the expectation, variance, and update restraint?.

- [01 baselines and rloo](01_baselines_and_rloo.ipynb)
- [02 ppo clipping and kl](02_ppo_clipping_and_kl.ipynb)
- [03 behavior probabilities and support](03_behavior_probabilities_and_support.ipynb)
- [04 learned critics and frozen text rewards](04_learned_critics_and_frozen_text_rewards.ipynb)

The third session is the DXI-13 probability bridge into Chapter14: raw model,
transformed behavior and declared target need not agree. It deliberately breaks
the likelihood denominator, rejects missing target support, distinguishes KL
values from gradients and tests EOS/padding/truncation boundaries. Read its
[integrated explanation](../../book/chapters/14-when-optimization-goes-wrong.md#143-entropy-collapse-certainty-can-mean-several-things)
and [CPU evidence](../../experiments/reports/2026-10-04-sampling-support.md).

The fourth session adds a real actor/neural value head, returns/TD/GAE,
EOS versus cap bootstrap masks and separate gradient contracts. It reloads
actual fitted character rewards, then traces proxy scores versus independent
quality across every fixed arm. Negative outcomes are central, not failed
demonstrations to skip. [Chapter12 sections12.10–12.14](../../book/chapters/12-language-generation-as-a-policy.md#1210-a-critic-predicts-the-return-before-an-action)
and the [new report](../../experiments/reports/2026-10-04-critic-policy.md) define
its finite-grammar limits; no pretrained model or API is used.

Read [the chapter](../../book/chapters/12-language-generation-as-a-policy.md),
[worked solutions](../../book/solutions/12-language-generation-as-a-policy.md), and
[lab guide](../../book/labs/12-language-generation-as-a-policy.md) together.
All code is complete, bounded, CPU, and uses original synthetic fixtures.
Each prediction is followed by an adjacent runnable reference and explained
plots. Change a declared variable and rerun the plot; saved PNG previews are
fixed reference outputs, not live controls. No model/data download is required.

Use an existing Python environment with PyTorch and Matplotlib, and a selected
Jupyter kernel when working interactively. Spark reference execution does not
prove Mac dependencies are configured. Verify fresh temporary copies with:

```bash
python scripts/verify_course_notebooks.py --days 20 --kernel dgx-spark-native --export-figures
```

Executed source notebooks remain unmodified; figures can be regenerated. The
[day artifact](../../learning_artifacts/day-20-baselines-rloo-and-ppo/README.md) separates content readiness,
reference evidence, and future learner practice.
