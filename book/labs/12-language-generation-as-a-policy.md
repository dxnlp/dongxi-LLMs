# Lab 12 — Language Generation as a Policy

Machine: isolated CPU course kernel on Mac or Spark, offline.
Expected time: 25–45 minutes per notebook; longer routes span several sessions (planning estimate).
Prerequisites: Chapters 3 and 11 log probabilities/gradients/KL; finite probability sums.
Deliverable: A signed estimator derivation and comparison card with rollout and gradient-work clocks.

Read the [chapter](../chapters/12-language-generation-as-a-policy.md) and use the
[worked solutions](../solutions/12-language-generation-as-a-policy.md) after attempting each exercise.
Write each prediction before executing the reference. Numerical checkpoints use
the notebook’s declared fixture, seed, dtype and tolerance; changing inputs may
change the result. References and interpretations remain adjacent to the attempt.

| Notebook | Question / prediction before reveal | Checkpoint on the declared fixture | One intervention | Evidence boundary |
|---|---|---|---|---|
| [Exact REINFORCE](../../notebooks/day-19/01_exact_reinforce_gradient.ipynb) | Predict: Must an action with positive reward gain probability? | Exact/autograd and weighted sampled gradients agree; a below-average rewarded action can lose mass. | Add the same constant to every reward. | One sampled gradient need not point in the expectation direction. |
| [Baselines and RLOO](../../notebooks/day-20/01_baselines_and_rloo.ipynb) | Predict: Will including an action in its own baseline preserve expectation? | Leave-one-out mean matches the exact gradient; the inclusive size-three mean is scaled by 2/3. | Use equal rewards or change baseline conditioning. | Variance reduction depends on estimator assumptions. |
| [PPO and KL](../../notebooks/day-20/02_ppo_clipping_and_kl.ipynb) | Predict: Will clipping stop every ratio outside 0.8–1.2? | The active flat branch depends on advantage sign; sampled KL estimators agree in expectation under named support. | Use rare-action probabilities or multiply token ratios. | Clipping is not a hard policy-distance bound. |
| [Behavior support · extension](../../notebooks/day-20/03_behavior_probabilities_and_support.ipynb) | Predict: Will identical raw logits give ratio one after filtering? | Matched behavior ratios are one on retained support; wrong denominators differ; old/reference/advantage have no gradient. | Remove support or substitute raw-policy probabilities. | Conditional support changes the estimand. |
| [Critics and rewards · extension](../../notebooks/day-20/04_learned_critics_and_frozen_text_rewards.ipynb) | Predict: Will treating a cap as termination change bootstrapped targets? | Correct cap-row targets are [0.972,1.08]; all twelve runs survive; frozen reward parameters remain unchanged. | Break the cap mask or substitute oracle/noisy values. | A fitted critic can amplify a mistaken reward prediction. |
| [Two work clocks](../../notebooks/day-21/01_policy_algorithm_comparison.ipynb) | Predict: Will equal rollouts imply equal gradient work? | Each run samples 3,840 completions; PPO takes 480 passes, the other arms 160. | Compare sample and gradient-work axes separately. | Separate context logits test optimization, not unseen arithmetic. |

The [finite-policy report](../../experiments/reports/2026-10-04-preference-policy-cpu.md) retains three seeds and both work clocks. PPO uses an exact detached value in that control. The separate [critic report](../../experiments/reports/2026-10-04-critic-policy.md) fits a neural value head and uses actual frozen text rewards; terminal reward and cap bootstrapping have different boundaries.

Retain a short explanation of the intervention, observed change and claim it
does not establish. The deliverable should connect these explanations into one
defensible argument, with source/report links rather than an execution-only checklist.

<details>
<summary>Fresh CPU verification and operational reference</summary>

From the repository root, use the isolated environment/kernel described in
[Appendix D](../appendices/d-reproduction-and-environments.md):

```bash
.venv-course/bin/python scripts/verify_course_notebooks.py --days 19 20 21 --kernel dongxi-course --expected-prefix .venv-course
```

The verifier retains executed copies and identities in a new directory. Source
notebooks and learner attempts remain intact. Exact runner/replay/export commands
and their evidence boundaries are in the [runbook](../../docs/runbooks/evaluation_tools.md).

</details>
