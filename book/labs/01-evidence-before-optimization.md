# Lab 1 — Evidence Before Optimization

Machine: CPU on Mac or Spark, offline. Use the declared Torch/Matplotlib course
kernel. Read [the chapter](../chapters/01-evidence-before-optimization.md) and attempt
[the worked exercises](../solutions/01-evidence-before-optimization.md) alongside this route.

Use [run identity](../../notebooks/day-01/01_experiment_identity.ipynb) to predict
whether changing JSON key order and changing a corpus revision have the same
effect. Execute canonical fingerprinting and change one declared control.

Next inspect [smoke criteria](../../notebooks/day-01/02_smoke_criteria_and_claims.ipynb).
A zero exit code, finite state and safe sampled reserve are separate requirements.
Break one deliberately and identify which conclusion changes. Never replace a
failed safety criterion with an average over successful ones.

Finally [read the real run](../../notebooks/day-01/03_read_a_real_run.ipynb).
Separate tested operation compatibility from usefulness, quality or general support
for every model/size/recipe. This saved case uses the completed TinyStories run
as a forward-looking evidence exercise; Chapter1's original Qwen smoke remains
a distinct historical report. This lab does not rerun either GPU experiment.

Deliver a run card: hypothesis, identity, controls, acceptance/failure criteria,
actual observations, interpretations and what the experiment cannot prove.
Implementation: [course_foundations.py](../../src/dongxi_llms/course_foundations.py).
Checks: `PYTHONPATH=src python -m unittest discover -s tests -p test_course_foundations.py`.

Fresh reference execution from the repository root:

```bash
python scripts/verify_course_notebooks.py --days 1 --kernel dgx-spark-native --export-figures
```

A passing reference verifies declared mechanism properties, not broad model
capability or independently assessed learner mastery.
