# Day 15 — Chapter 10 Companion Sessions

Preferences need a rubric, a likelihood, and a noise model.

- [01 bradley terry margin](01_bradley_terry_margin.ipynb)
- [02 disagreement and calibration](02_disagreement_and_calibration.ipynb)
- [03 preference collection and judge audits](03_preference_collection_and_judges.ipynb)
- [04 preference alignment end to end · optional, Spark GPU](04_preference_alignment_end_to_end.ipynb)

The collection notebook introduces original authored text comparisons and
review labels, blind display ordering, strict verdict parsing, ordinal
rating/ranking conversions, and controlled **simulated** order, repetition,
verbosity and injection failures. Its three measured figures expose nuisance
sensitivity, retained nondecisive outcomes and source-balanced weighting.
It is not a human annotation study or a live AI-judge evaluation. The
[audit report](../../experiments/reports/2026-10-04-preference-audit.md) preserves
every raw observation and parser/transport failure.
The later [parser hardening report](../../experiments/reports/2026-10-04-preference-audit-hardening.md)
adds retained failures for adversarial nesting and oversized verdicts; the
original 484-observation experiment and its historical source identity remain
unchanged.

Read [the chapter](../../book/chapters/10-preferences-and-reward-models.md),
[worked solutions](../../book/solutions/10-preferences-and-reward-models.md), and
[lab guide](../../book/labs/10-preferences-and-reward-models.md) together.
Notebooks 01–03 are complete, bounded, CPU, and use original synthetic fixtures.
Each prediction is followed by an adjacent runnable reference and explained
plots. Change a declared variable and rerun the plot; saved PNG previews are
fixed reference outputs, not live controls. No model/data download is required.

Notebook 04 is GPU first (`spark-gpu`). It gathers the SLP3 §8.3–8.4 discussions of the
[learning note](../../learning_artifacts/day-15-preference-data-and-bradley-terry/learning-from-preferences.md)
into one flow on real Qwen3-0.6B and public preference data (HH-RLHF `helpful-base`,
UltraFeedback binarized `test_prefs`, both MIT): what preference data looks like for chat and for
agent trajectories, Bradley–Terry, training a reward model, phase-2 prompts, PPO component by
component (four networks, per-token rewards, value model, GAE, clipping), GRPO group advantages,
and DPO on the same pairs. Nineteen predictions, each with its reference. It needs CUDA and the
cached Qwen3-0.6B, downloads two small dataset files on first run, and takes about 20–25 minutes.
It shows mechanics, not an aligned model.

Use an existing Python environment with PyTorch and Matplotlib, and a selected
Jupyter kernel when working interactively. Verify fresh temporary copies with:

```bash
python scripts/verify_course_notebooks.py --days 15 --kernel dgx-spark-native --export-figures
```

Executed source notebooks remain unmodified; figures can be regenerated. The
[day artifact](../../learning_artifacts/day-15-preference-data-and-bradley-terry/README.md) separates content readiness,
reference evidence, and future learner practice.
