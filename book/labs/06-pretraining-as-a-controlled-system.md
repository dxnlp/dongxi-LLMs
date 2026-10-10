# Lab 6 companion — Specify, Perturb and Recover a Training Run

Machine: isolated CPU course kernel on Mac or Spark, offline.
Expected time: 25–45 minutes per notebook (planning estimate).
Prerequisites: Chapters 3–5 objective, decoder and gradients.
Deliverable: a training-state diagram and a run card identifying the changed causal path.

This is the Day 8 mechanism route for [Chapter 6](../chapters/06-pretraining-as-a-controlled-system.md).
Predict before executing each reference; retain learner attempts. Use the
[worked solutions](../solutions/06-pretraining-as-a-controlled-system.md) after
the checkpoint, then continue with [Lab 6’s completed-run reading](06-reading-a-pretraining-run.md).

| Notebook | Question / prediction before reveal | Checkpoint on the declared fixture | One intervention | Evidence boundary |
|---|---|---|---|---|
| [Documents, batches and targets](../../notebooks/day-08/01_data_batches_and_token_budget.ipynb) | Predict: Will shorter windows change counted targets or their context? | Targets retain their document ownership; correct weighted accumulation error is below 1e-10. | Shorten the window or average unequal batch means. | An authored corpus does not eliminate every leakage route. |
| [Optimizer and schedule](../../notebooks/day-08/02_adamw_schedule_and_stability.ipynb) | Predict: Must identical current gradients produce identical updates? | Manual/PyTorch AdamW error is below 1e-12; update-three rate is 0.01 and final rate 0.001. | Change the second gradient or clip microbatches separately. | Precision range, clipping and finite state are distinct checks. |
| [Development and recovery](../../notebooks/day-08/03_validation_and_checkpoint_recovery.ipynb) | Predict: Will weights alone reproduce continuation? | Complete restore has zero parameter error and identical history; missing moments/cursor fail different controls. | Omit only optimizer state or only data position. | This CPU continuation does not certify cross-device recovery. |

Record which predictions count, which clock advances, and which state a
checkpoint restores. A loss curve cannot diagnose all three. Separate the
schematic data flow, calculated memory ledger and measured numerical trajectory.
Use the [Day 8 verification report](../../experiments/reports/2026-09-09-day8-material-verification.md)
as existing material evidence, with its actual platform boundary.

<details>
<summary>Fresh CPU verification and operational reference</summary>

From the repository root, use the isolated environment/kernel described in
[Appendix D](../appendices/d-reproduction-and-environments.md):

```bash
.venv-course/bin/python scripts/verify_course_notebooks.py --days 8 --kernel dongxi-course --expected-prefix .venv-course
```

The verifier retains executed copies and identities in a new directory. Source
notebooks and learner attempts remain intact. Exact runner/replay/export commands
and their evidence boundaries are in the [runbook](../../docs/runbooks/evaluation_tools.md).

</details>
