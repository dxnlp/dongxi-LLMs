# Lab 6 — Read a Pretraining Run

Machine: isolated CPU course kernel on Mac or Spark, offline.
Expected time: 25–45 minutes per notebook; longer routes span several sessions (planning estimate).
Prerequisites: Chapters 3–5; the companion Day 8 foundation route below.
Deliverable: A run card joining target exposure, time boundary, development NLL and complete-story evidence.

Read the [chapter](../chapters/06-pretraining-as-a-controlled-system.md) and use the
[worked solutions](../solutions/06-pretraining-as-a-controlled-system.md) after attempting each exercise.
Write each prediction before executing the reference. Numerical checkpoints use
the notebook’s declared fixture, seed, dtype and tolerance; changing inputs may
change the result. References and interpretations remain adjacent to the attempt.

Begin with [the training-system foundation route](06-pretraining-as-a-controlled-system.md), then read the completed-run evidence below.

| Notebook | Question / prediction before reveal | Checkpoint on the declared fixture | One intervention | Evidence boundary |
|---|---|---|---|---|
| [Documents, batches and targets](../../notebooks/day-08/01_data_batches_and_token_budget.ipynb) | Predict: Will shorter windows change counted targets or their context? | Targets retain their document ownership; correct weighted accumulation error is below 1e-10. | Shorten the window or average unequal batch means. | An authored corpus does not eliminate every leakage route. |
| [Optimizer and schedule](../../notebooks/day-08/02_adamw_schedule_and_stability.ipynb) | Predict: Must identical current gradients produce identical updates? | Manual/PyTorch AdamW error is below 1e-12; update-three rate is 0.01 and final rate 0.001. | Change the second gradient or clip microbatches separately. | Precision range, clipping and finite state are distinct checks. |
| [Development and recovery](../../notebooks/day-08/03_validation_and_checkpoint_recovery.ipynb) | Predict: Will weights alone reproduce continuation? | Complete restore has zero parameter error and identical history; missing moments/cursor fail different controls. | Omit only optimizer state or only data position. | This CPU continuation does not certify cross-device recovery. |
| [Read training clocks](../../notebooks/day-09/01_read_training_clocks.ipynb) | Predict: Will restored numerical exposure erase failed spending? | Historical run ends at 14,000 updates; retry control has 17 successful targets and 25 reservations. | Compare the lower-cap refusal with zero new forward calls. | Reservations and completed work are distinct from physical quotas. |
| [Document visibility](../../notebooks/day-09/02_masking_and_padding_control.ipynb) | Predict: Will ordinary causal packing isolate separate documents? | The isolated mask blocks position 3 from source 0; changing the first document leaves the second isolated output unchanged. | Use ordinary cross-document causal visibility. | A valid loss mask alone does not enforce attention isolation. |
| [Complete stories](../../notebooks/day-09/03_story_quality_and_decoding.ipynb) | Predict: Will natural EOS imply a satisfactory ending? | Read both final archived completions: EOS and plot continuity are separate observations. | Compare greedy and sampled stories under retained settings. | These selected stories do not estimate general story quality. |

The historical September run and fresh October control/half-LR arms have different initializations and budgets. Read the [historical result](../../experiments/reports/2026-09-14-tinystories-learning-result.json) and [fresh matched comparison](../../experiments/reports/2026-10-05-native-story-first400-comparison.md) separately. For the fresh arms, reconcile 13,132 development targets with exposure/schedule differences; EOS, satisfactory endings and reviewer disagreement are separate evidence. The [five-axis rating route](07-evaluation-is-a-contract.md) owns the detailed rubric.

Retain a short explanation of the intervention, observed change and claim it
does not establish. The deliverable should connect these explanations into one
defensible argument, with source/report links rather than an execution-only checklist.

<details>
<summary>Fresh CPU verification and operational reference</summary>

From the repository root, use the isolated environment/kernel described in
[Appendix D](../appendices/d-reproduction-and-environments.md):

```bash
.venv-course/bin/python scripts/verify_course_notebooks.py --days 8 9 --kernel dongxi-course --expected-prefix .venv-course
```

The verifier retains executed copies and identities in a new directory. Source
notebooks and learner attempts remain intact. Exact runner/replay/export commands
and their evidence boundaries are in the [runbook](../../docs/runbooks/evaluation_tools.md).

</details>
