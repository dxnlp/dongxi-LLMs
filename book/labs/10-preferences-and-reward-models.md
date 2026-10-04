# Chapter 10 — Preference and Reward Laboratories

Use a Python environment with PyTorch and Matplotlib, plus Jupyter if opening
the notebooks. Mac Studio is the ordinary CPU learning lane; these checks also
run on Spark CPU. No model/data download, credential, or inference server is
needed. Source: `reward_model_lab.py`; all fixture features are synthetic.

| Day | Session | Prediction and controlled change |
|---|---|---|
| 15 | [Bradley–Terry margin](../../notebooks/day-15/01_bradley_terry_margin.ipynb) | Predict signs; change score offsets, vote fraction, and margin |
| 15 | [Disagreement and calibration](../../notebooks/day-15/02_disagreement_and_calibration.ipynb) | Predict a noisy optimum; stretch margins without changing ranks |
| 16 | [Reward shortcut audit](../../notebooks/day-16/01_reward_model_bias_audit.ipynb) | Compare confounded versus balanced nuisance features on an independent population |

The first notebook connects scalar gradients to the shared reward model. The
second distinguishes ranking from probability quality. The third measures a
distribution intervention and retains the adversarial failure. Every exercise
has an adjacent runnable reference, labeled plots, and an interpretation. Saved
figures are reference outputs; rerunning a cell regenerates the current values.

Reproduce from the course root:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 python scripts/run_preference_policy_cpu.py
python scripts/verify_course_notebooks.py --days 15 16 --kernel dgx-spark-native --export-figures
```

The runner prints a deterministic JSON evidence object; it does not overwrite
the historical report. The [predeclared specification](../../experiments/specs/2026-10-04-preference-policy-cpu.md)
and [report](../../experiments/reports/2026-10-04-preference-policy-cpu.md) describe
the actual fixture and measurement. Reading or running these material checks
does not assess learner mastery. A neural reward-model Spark extension should
start with a small independently labeled pair set, frozen source groups,
endpoint-mask tests, and an explicit rubric; no such model-scale run is claimed.
