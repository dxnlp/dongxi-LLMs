# Lab 11 — Direct Preference Optimization

Machine: isolated CPU course kernel on Mac or Spark, offline.
Expected time: 25–45 minutes per notebook; longer routes span several sessions (planning estimate).
Prerequisites: Chapters 9–10 sequence likelihood and learned reward; Chapter 7 evaluation.
Deliverable: A DPO comparison defense separating pair ratio, absolute likelihood, retention and work.

Read the [chapter](../chapters/11-direct-preference-optimization.md) and use the
[worked solutions](../solutions/11-direct-preference-optimization.md) after attempting each exercise.
Write each prediction before executing the reference. Numerical checkpoints use
the notebook’s declared fixture, seed, dtype and tolerance; changing inputs may
change the result. References and interpretations remain adjacent to the attempt.

| Notebook | Question / prediction before reveal | Checkpoint on the declared fixture | One intervention | Evidence boundary |
|---|---|---|---|---|
| [Finite KL optimum](../../notebooks/day-17/01_kl_regularized_optimum.ipynb) | Predict: Will a common reward offset move the optimal policy? | Stationarity matches across coordinates; adding 50 to rewards leaves the policy unchanged. | Vary beta before examining the exact tilt. | Finite fixed rewards do not model a learned reward landscape. |
| [Sequence likelihood](../../notebooks/day-17/02_sequence_likelihood_and_masks.ipynb) | Predict: Will equal token likelihoods yield equal response sums? | Unscored direct logit gradients are zero; longer responses have a larger-magnitude sum while token means agree. | Replace summed response likelihood with a mean. | That replacement changes the preference objective. |
| [DPO and controls](../../notebooks/day-18/01_dpo_controlled_comparison.ipynb) | Predict: Must a rising chosen/rejected ratio raise chosen probability? | Constructed chosen mass falls 0.2→0.1 while its ratio rises 2→10; frozen reference gets no gradient. | Flip preferences and compare the separate SFT control. | Different losses are not comparable quality scales. |
| [Retention controls · extension](../../notebooks/day-18/02_dpo_retention_and_preference_controls.ipynb) | Predict: Will extra chosen NLL recover plain DPO at coefficient zero? | Zero coefficient matches plain loss/gradient; all 48 rows retain unchanged references and identical caches. | Compare chosen NLL/rehearsal with noisy and length controls. | Extra supervision and unequal cost must remain visible. |

Read all 48 [retention-control rows](../../experiments/reports/2026-10-04-dpo-retention.md). Then inspect the [actual matched chosen/DPO comparison](../../experiments/reports/2026-10-05-native-preference-comparison.md): common chosen exposure does not match rejected/reference work. Unchanged/chosen/DPO location answers are 0/4, 4/4 and 1/4 under the common contract; original instruction retention is a separate panel. Exercises 14–15 retain the original-reference and sampler/reduction arguments; detailed source/recovery reading stays in the runbook.

Retain a short explanation of the intervention, observed change and claim it
does not establish. The deliverable should connect these explanations into one
defensible argument, with source/report links rather than an execution-only checklist.

<details>
<summary>Fresh CPU verification and operational reference</summary>

From the repository root, use the isolated environment/kernel described in
[Appendix D](../appendices/d-reproduction-and-environments.md):

```bash
.venv-course/bin/python scripts/verify_course_notebooks.py --days 17 18 --kernel dongxi-course --expected-prefix .venv-course
```

The verifier retains executed copies and identities in a new directory. Source
notebooks and learner attempts remain intact. Exact runner/replay/export commands
and their evidence boundaries are in the [runbook](../../docs/runbooks/dpo_spark_runner.md).

</details>
