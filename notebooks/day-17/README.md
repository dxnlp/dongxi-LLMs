# Day 17 — Chapter 11 Companion Sessions

From a KL optimum to response log-probability ratios.

- [01 kl regularized optimum](01_kl_regularized_optimum.ipynb)
- [02 sequence likelihood and masks](02_sequence_likelihood_and_masks.ipynb)

Read [the chapter](../../book/chapters/11-direct-preference-optimization.md),
[worked solutions](../../book/solutions/11-direct-preference-optimization.md), and
[lab guide](../../book/labs/11-direct-preference-optimization.md) together.
All code is complete, bounded, CPU, and uses original synthetic fixtures.
Each prediction is followed by an adjacent runnable reference and explained
plots. Change a declared variable and rerun the plot; saved PNG previews are
fixed reference outputs, not live controls. No model/data download is required.

Use an existing Python environment with PyTorch and Matplotlib, and a selected
Jupyter kernel when working interactively. Spark reference execution does not
prove Mac dependencies are configured. Verify fresh temporary copies with:

```bash
python scripts/verify_course_notebooks.py --days 17 --kernel dgx-spark-native --export-figures
```

Executed source notebooks remain unmodified; figures can be regenerated. The
[day artifact](../../learning_artifacts/day-17-direct-preference-optimization/README.md) separates content readiness,
reference evidence, and future learner practice.
