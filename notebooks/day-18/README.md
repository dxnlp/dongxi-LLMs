# Day 18 — Chapter 11 Companion Sessions

A favorable preference margin is a diagnostic, not the final task score.

- [01 dpo controlled comparison](01_dpo_controlled_comparison.ipynb)
- [02 retention and preference controls](02_dpo_retention_and_preference_controls.ipynb): shared SFT-started decoder, chosen NLL/rehearsal, fixed noisy/length interventions and full generated-response evaluation.

Read [the chapter](../../book/chapters/11-direct-preference-optimization.md),
[worked solutions](../../book/solutions/11-direct-preference-optimization.md), and
[lab guide](../../book/labs/11-direct-preference-optimization.md) together.
All code is complete, bounded, CPU, and uses original synthetic fixtures.
Each prediction is followed by an adjacent runnable reference and explained
plots. Change a declared variable and rerun the plot; saved PNG previews are
fixed reference outputs, not live controls. No model/data download is required.

Session02's [specification](../../experiments/specs/2026-10-04-dpo-retention.md)
and [original fixture](../../fixtures/dpo-retention/README.md) freeze three seeds,
four data conditions and four arms before fitting. Read its
[retained evidence](../../experiments/reports/2026-10-04-dpo-retention.md) alongside
the earlier negative comparison. The new reference executes all 48 small fits;
warm initial states, final generations and all failures remain visible. The
locked CPU environment and temporary `dongxi-course-clean` kernel can execute
it without modifying the shared Spark GPU environment. Kernel identity/device,
not the host name, determine its evidence scope.

Use an existing Python environment with PyTorch and Matplotlib, and a selected
Jupyter kernel when working interactively. Spark reference execution does not
prove Mac dependencies are configured. Verify fresh temporary copies with:

```bash
python scripts/verify_course_notebooks.py --days 18 --kernel dgx-spark-native --export-figures
```

Executed source notebooks remain unmodified; figures can be regenerated. The
[day artifact](../../learning_artifacts/day-18-dpo-experiment/README.md) separates content readiness,
reference evidence, and future learner practice.
