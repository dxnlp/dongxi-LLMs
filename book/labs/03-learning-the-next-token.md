# Lab 3 — Learning the Next Token

Machine: isolated CPU course kernel on Mac or Spark, offline.
Expected time: 25–45 minutes per notebook; longer routes span several sessions (planning estimate).
Prerequisites: Chapter 2 token IDs/embedding paths; elementary probability and derivatives.
Deliverable: A defended alignment diagram and explanation of nonzero optimum NLL.

Read the [chapter](../chapters/03-learning-the-next-token.md) and use the
[worked solutions](../solutions/03-learning-the-next-token.md) after attempting each exercise.
Write each prediction before executing the reference. Numerical checkpoints use
the notebook’s declared fixture, seed, dtype and tolerance; changing inputs may
change the result. References and interpretations remain adjacent to the attempt.

| Notebook | Question / prediction before reveal | Checkpoint on the declared fixture | One intervention | Evidence boundary |
|---|---|---|---|---|
| [Logits, surprise and gradient](../../notebooks/day-03/01_logits_softmax_nll.ipynb) | Predict: Will a shared logit shift change probabilities? | Probability sum is one; the declared target NLL is about 1.3490122; logit gradients sum to zero. | Raise one competing logit instead. | A normalized distribution does not establish calibration. |
| [Shift and loss mask](../../notebooks/day-03/02_causal_shift_and_masks.ipynb) | Predict: Will copying current tokens solve next-token prediction? | Shifted logits are [1,4,5]; labels are [1,2,3,4]; copy loss is small only under the wrong alignment. | Remove one valid target and compare mean denominators. | Loss masks do not erase causal prompt influence. |
| [Learning a distribution](../../notebooks/day-03/03_distribution_learning.ipynb) | Predict: Should the 70/30 optimum have zero loss? | NLL approaches 0.6108643, probabilities 0.7/0.3 and logit gap 0.8473. | Change target frequencies before fitting the copied fixture. | Two logits demonstrate estimation, not language modeling. |

Keep predictions and learner attempts before revealing the adjacent references. The [70/30 report](../../experiments/reports/2026-09-03-next-token-distribution.md) supplies the fixed numerical checkpoint; interventions change its expected optimum.

Retain a short explanation of the intervention, observed change and claim it
does not establish. The deliverable should connect these explanations into one
defensible argument, with source/report links rather than an execution-only checklist.

<details>
<summary>Fresh CPU verification and operational reference</summary>

From the repository root, use the isolated environment/kernel described in
[Appendix D](../appendices/d-reproduction-and-environments.md):

```bash
.venv-course/bin/python scripts/verify_course_notebooks.py --days 3 --kernel dongxi-course --expected-prefix .venv-course
```

The verifier retains executed copies and identities in a new directory. Source
notebooks and learner attempts remain intact. Exact runner/replay/export commands
and their evidence boundaries are in the [runbook](../../docs/runbooks/evaluation_tools.md).

</details>
