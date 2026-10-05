# Day 16 — Chapter 10 Companion Sessions

Which feature does the reward actually reward?.

- [01 reward model bias audit](01_reward_model_bias_audit.ipynb)
- [03 text reward and process labels](03_text_reward_and_process_labels.ipynb)

The text lesson adds an actual tiny causal decoder/scalar head, left/right
padding and EOS checks, multi-seed held-out nuisance results, explicit terminal
and step-boundary labels, and a strictly reloaded frozen reward. Its original five
figures include the model/branch map, endpoint grids and measured outcomes.
The [report](../../experiments/reports/2026-10-04-text-reward.md) retains unknown-
token arithmetic failures and calibration/test encoding collisions rather
than claiming general reasoning or independent held-out calibration. Authored
labels are not human feedback; no pretrained or live judge campaign ran.

The separately [specified character follow-up](../../experiments/specs/2026-10-04-text-reward-vocabulary-intervention.md)
adds three visuals for exact-input independence, fitted training versus failed
heldout transfer and counted calibration. The [new report](../../experiments/reports/2026-10-04-text-reward-character.md)
retains every declared seed and negative process/outcome prediction. The fixed
ASCII alphabet removes OOV collisions, not the need to learn a general rule.
The extension now has twelve complete code cells/eight previews, with a frozen
seed 1611 export and explicit unsupported-input failures. Historical word
reports/export/preview bytes remain intact; its first cross-version retraining
assertion failure is retained beside the passing same-state reload check.

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
python scripts/verify_course_notebooks.py --days 16 --kernel dgx-spark-native --export-figures
```

Executed source notebooks remain unmodified; figures can be regenerated. The
[day artifact](../../learning_artifacts/day-16-reward-models/README.md) separates content readiness,
reference evidence, and future learner practice.
