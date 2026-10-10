# Lab 10 — Preferences and Reward Models

Machine: isolated CPU course kernel on Mac or Spark, offline.
Expected time: 25–45 minutes per notebook; longer routes span several sessions (planning estimate).
Prerequisites: Chapter 7 instrument identity; sigmoid, likelihood and source groups.
Deliverable: A preference/reward audit separating judgment direction, calibration, shortcuts and transfer.

Read the [chapter](../chapters/10-preferences-and-reward-models.md) and use the
[worked solutions](../solutions/10-preferences-and-reward-models.md) after attempting each exercise.
Write each prediction before executing the reference. Numerical checkpoints use
the notebook’s declared fixture, seed, dtype and tolerance; changing inputs may
change the result. References and interpretations remain adjacent to the attempt.

| Notebook | Question / prediction before reveal | Checkpoint on the declared fixture | One intervention | Evidence boundary |
|---|---|---|---|---|
| [Preference margin](../../notebooks/day-15/01_bradley_terry_margin.ipynb) | Predict: Will adding a common reward offset change pair preference? | Probability is sigmoid(chosen−rejected); loss gradient is sigmoid(margin)−1; common offsets cancel. | Change vote fraction or relative margin. | Relative scores do not identify an absolute reward level. |
| [Disagreement and calibration](../../notebooks/day-15/02_disagreement_and_calibration.ipynb) | Predict: Can correct ordering coexist with miscalibrated probabilities? | Positive score stretching preserves the 0.5 decision boundary; inverse scaling restores this fixture calibration. | Stretch margins while keeping rankings fixed. | Synthetic vote probabilities are not human consensus. |
| [Collection audit · extension](../../notebooks/day-15/03_preference_collection_and_judges.ipynb) | Predict: Will swapping a blind display change the chosen candidate identity? | Retain 24 pairs, 484 judgments and four failures; swaps map sides while candidate identity survives. | Perturb verbosity/injection and preserve ties/abstentions. | Simulated judges are not independent human annotations. |
| [Reward shortcuts](../../notebooks/day-16/01_reward_model_bias_audit.ipynb) | Predict: Can a low training objective hide nuisance dependence? | Worse-answer win probability is about 0.962504 confounded versus 0.136947 balanced on the declared adversary. | Vary length with quality fixed. | Features are authored; this is not a text-understanding benchmark. |
| [Text reward and process labels · extension](../../notebooks/day-16/03_text_reward_and_process_labels.ipynb) | Predict: Will distinct text always receive distinct encoded input? | Original word encoding has four held-out collisions; character encoding separates them; frozen reload is exact. | Move padding/EOS or compare the separately specified character arm. | Closing collisions does not establish robust reward transfer. |

Inspect all seeds and the [collection ledger](../../experiments/reports/2026-10-04-preference-audit.md). The [word-reward](../../experiments/reports/2026-10-04-text-reward.md) and [character intervention](../../experiments/reports/2026-10-04-text-reward-character.md) remain separate measured arms. Character distinctions remove an encoding collision; their retained failures prevent a reward-transfer claim.

Retain a short explanation of the intervention, observed change and claim it
does not establish. The deliverable should connect these explanations into one
defensible argument, with source/report links rather than an execution-only checklist.

<details>
<summary>Fresh CPU verification and operational reference</summary>

From the repository root, use the isolated environment/kernel described in
[Appendix D](../appendices/d-reproduction-and-environments.md):

```bash
.venv-course/bin/python scripts/verify_course_notebooks.py --days 15 16 --kernel dongxi-course --expected-prefix .venv-course
```

The verifier retains executed copies and identities in a new directory. Source
notebooks and learner attempts remain intact. Exact runner/replay/export commands
and their evidence boundaries are in the [runbook](../../docs/runbooks/evaluation_tools.md).

</details>
