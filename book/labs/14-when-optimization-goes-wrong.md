# Lab 14 — When Optimization Goes Wrong

Machine: isolated CPU course kernel on Mac or Spark, offline.
Expected time: 25–45 minutes per notebook; longer routes span several sessions (planning estimate).
Prerequisites: Chapter 13 objective, strict verifier and rollout identities.
Deliverable: An incident record: observation, first broken invariant, competing causes, intervention and remaining uncertainty.

Read the [chapter](../chapters/14-when-optimization-goes-wrong.md) and use the
[worked solutions](../solutions/14-when-optimization-goes-wrong.md) after attempting each exercise.
Write each prediction before executing the reference. Numerical checkpoints use
the notebook’s declared fixture, seed, dtype and tolerance; changing inputs may
change the result. References and interpretations remain adjacent to the attempt.

| Notebook | Question / prediction before reveal | Checkpoint on the declared fixture | One intervention | Evidence boundary |
|---|---|---|---|---|
| [Reward misspecification](../../notebooks/day-24/01_reward_hacking.ipynb) | Predict: Can proxy reward rise while strict correctness falls? | Exact gradient matches probability×(reward−mean); the favored wrong action illustrates objective misspecification. | Repair proxy rewards from equal initial logits. | Repair after a hacked checkpoint is a different intervention. |
| [Length and entropy](../../notebooks/day-24/02_length_entropy_diagnostics.ipynb) | Predict: Will equal response weight imply equal token weight? | Lengths two/eight get response weights 0.5/0.5 versus token weights 0.2/0.8; relabelling preserves entropy. | Favor the wrong action at the same probability mass. | Low entropy alone does not identify success or failure. |
| [Versions and work](../../notebooks/day-25/01_rollout_versions_and_budget.ipynb) | Predict: Will a metadata pass certify the actual sampling engine? | Version two accepts at two; default version-three reuse refuses; explicitly allowed lag retains behavior identity. | Change tokenizer identity or the projected generation rate. | The budget plot is an analytical scenario, not Spark throughput. |
| [Ragged cache and recovery · extension](../../notebooks/day-25/02_ragged_kv_cache_and_exact_recovery.ipynb) | Predict: Will restoring weights permit recollecting a pending update? | Four generation paths agree within 1e-10; saved next draw/tail replay exactly; pending tokens apply without recollection. | Use wrong row positions or remove pending/RNG state. | Forced stopping, logical work and physical runtime remain separate. |

Optional bridges remain in their original routes: [text reward](10-preferences-and-reward-models.md), [DPO retention](11-direct-preference-optimization.md), [behavior support/critics](12-language-generation-as-a-policy.md) and [fixed-pool objective controls](13-group-relative-policy-optimization.md). Use one negative outcome to distinguish two causes; do not change reward, sampling and reduction together. [Detailed operational exercises](../solutions/14-when-optimization-goes-wrong.md#evidence-reading-extensions) retain their original mappings.

Retain a short explanation of the intervention, observed change and claim it
does not establish. The deliverable should connect these explanations into one
defensible argument, with source/report links rather than an execution-only checklist.

<details>
<summary>Fresh CPU verification and operational reference</summary>

From the repository root, use the isolated environment/kernel described in
[Appendix D](../appendices/d-reproduction-and-environments.md):

```bash
.venv-course/bin/python scripts/verify_course_notebooks.py --days 24 25 --kernel dongxi-course --expected-prefix .venv-course
```

The verifier retains executed copies and identities in a new directory. Source
notebooks and learner attempts remain intact. Exact runner/replay/export commands
and their evidence boundaries are in the [runbook](../../docs/runbooks/rlvr_runner.md).

</details>
