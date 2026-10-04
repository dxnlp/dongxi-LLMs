# Day 20 — Chapter 12 Companion Sessions

What changes the expectation, variance, and update restraint?.

- [01 baselines and rloo](01_baselines_and_rloo.ipynb)
- [02 ppo clipping and kl](02_ppo_clipping_and_kl.ipynb)

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
