# Lab 3 — Learning the Next Token

Machine: CPU on Mac or Spark, offline. Use the declared Torch/Matplotlib course
kernel. Read [the chapter](../chapters/03-learning-the-next-token.md) and attempt
[the worked exercises](../solutions/03-learning-the-next-token.md) alongside this route.

The [Day3 route](../../notebooks/day-03/README.md) links stable probabilities,
causal supervision and trainable conditional distributions. Predict the effect
of a shared logit shift. Complete the scaffold if desired, then execute the
adjacent reference and inspect probability and signed p−q plots.

Trace input/next-target alignment and separate attention visibility from loss
inclusion. Change an answer mask and explain why prompt positions can still learn.
Do not apply the causal shift twice or average unequal microbatch means.

Optimize the two-logit model on the declared 70/30 population. Read probabilities,
loss and gradients together. A nonzero converged loss can indicate successful
learning of irreducible uncertainty, not optimization failure.

Deliver a defense of how one-hot observations teach a conditional distribution.
Read the [original measured report](../../experiments/reports/2026-09-03-next-token-distribution.md).
Verification skips only explicitly tagged unfinished exercise scaffolds; complete
references and completed attempts execute without erasing learner code.
Checks: `PYTHONPATH=src python -m unittest discover -s tests -p test_next_token_distribution_lab.py`.

Fresh reference execution from the repository root:

```bash
python scripts/verify_course_notebooks.py --days 3 --kernel dgx-spark-native --export-figures
```

A passing reference verifies declared mechanism properties, not broad model
capability or independently assessed learner mastery.
