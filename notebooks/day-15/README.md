# Day 15 — Chapter 10 Companion Sessions

Preferences need a rubric, a likelihood, and a noise model.

- [01 bradley terry margin](01_bradley_terry_margin.ipynb)
- [02 disagreement and calibration](02_disagreement_and_calibration.ipynb)

Read [the chapter](../../book/chapters/10-preferences-and-reward-models.md),
[worked solutions](../../book/solutions/10-preferences-and-reward-models.md), and
[lab guide](../../book/labs/10-preferences-and-reward-models.md) together.
All code is complete, bounded, CPU, and uses original synthetic fixtures.
Each prediction is followed by an adjacent runnable reference and explained
plots. Change a declared variable and rerun the plot; saved PNG previews are
fixed reference outputs, not live controls. No model/data download is required.

Use an existing Python environment with PyTorch and Matplotlib, and a selected
Jupyter kernel when working interactively. Spark reference execution does not
prove Mac dependencies are configured. Verify fresh temporary copies with:

```bash
python scripts/verify_course_notebooks.py --days 15 --kernel dgx-spark-native --export-figures
```

Executed source notebooks remain unmodified; figures can be regenerated. The
[day artifact](../../learning_artifacts/day-15-preference-data-and-bradley-terry/README.md) separates content readiness,
reference evidence, and future learner practice.
