# Lab 8 — Instruction Data as an Interface

Machine: isolated CPU course kernel on Mac or Spark, offline.
Expected time: 25–45 minutes per notebook; longer routes span several sessions (planning estimate).
Prerequisites: Chapters 2–4 token semantics, single shift and causal visibility.
Deliverable: A data card and annotated ownership/attention/loss grids for one conversation.

Read the [chapter](../chapters/08-instruction-data-as-an-interface.md) and use the
[worked solutions](../solutions/08-instruction-data-as-an-interface.md) after attempting each exercise.
Write each prediction before executing the reference. Numerical checkpoints use
the notebook’s declared fixture, seed, dtype and tolerance; changing inputs may
change the result. References and interpretations remain adjacent to the attempt.

| Notebook | Question / prediction before reveal | Checkpoint on the declared fixture | One intervention | Evidence boundary |
|---|---|---|---|---|
| [Role ownership](../../notebooks/day-11/01_roles_templates_and_masks.ipynb) | Predict: Will masking prompt loss erase prompt influence? | The shown serialized fixture has two scored targets; assistant content/ending remain labelled. | Change role ownership in a copied serialization. | Ownership and visibility are different boundaries. |
| [Padding and packing](../../notebooks/day-11/02_padding_packing_boundaries.ipynb) | Predict: Will ordinary causality isolate packed conversations? | Ordinary mask permits position 8→0 while isolated mask blocks it; padding labels are ignored. | Replace document isolation with ordinary causality. | Packing alters available context unless contracts are matched. |
| [Mixture exposure](../../notebooks/day-11/03_mixtures_and_data_cards.ipynb) | Predict: Will equal example weight imply equal token weight? | Fixture token shares are 0.1/0.9; adjusted exposure is 0.5/0.5; source-group sizes are one and two. | Change length or mixture unit independently. | Balanced exposure does not establish balanced capability. |
| [Teacher attempts · extension](../../notebooks/day-11/04_teacher_attempts_and_matched_rejection_sft.ipynb) | Predict: Will a better selected dataset guarantee student transfer? | All nine arms survive; EOS labels cover twelve examples; padding is ignored. | Compare top/random/within-stratum selection and 42/46 target exposure. | The programmatic teacher is not a pretrained rationale model. |

Keep scientific controls visible: one shift, assistant-owned content and ending targets, document isolation, frozen source groups and a declared example/token mixture unit. The [teacher-attempt report](../../experiments/reports/2026-10-04-teacher-data.md) retains rejected attempts and negative student outcomes; format eligibility does not mean correctness.

Retain a short explanation of the intervention, observed change and claim it
does not establish. The deliverable should connect these explanations into one
defensible argument, with source/report links rather than an execution-only checklist.

<details>
<summary>Fresh CPU verification and operational reference</summary>

From the repository root, use the isolated environment/kernel described in
[Appendix D](../appendices/d-reproduction-and-environments.md):

```bash
.venv-course/bin/python scripts/verify_course_notebooks.py --days 11 --kernel dongxi-course --expected-prefix .venv-course
```

The verifier retains executed copies and identities in a new directory. Source
notebooks and learner attempts remain intact. Exact runner/replay/export commands
and their evidence boundaries are in the [runbook](../../docs/runbooks/evaluation_tools.md).

</details>
