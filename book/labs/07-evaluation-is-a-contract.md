# Lab 7 — Evaluation Is a Contract

Machine: isolated CPU course kernel on Mac or Spark, offline.
Expected time: 25–45 minutes per notebook; longer routes span several sessions (planning estimate).
Prerequisites: Chapter 6 evidence boundaries; fractions, grouping and uncertainty.
Deliverable: An evaluation card with frozen instrument, denominators, source pairing, slices and limits.

Read the [chapter](../chapters/07-evaluation-is-a-contract.md) and use the
[worked solutions](../solutions/07-evaluation-is-a-contract.md) after attempting each exercise.
Write each prediction before executing the reference. Numerical checkpoints use
the notebook’s declared fixture, seed, dtype and tolerance; changing inputs may
change the result. References and interpretations remain adjacent to the attempt.

| Notebook | Question / prediction before reveal | Checkpoint on the declared fixture | One intervention | Evidence boundary |
|---|---|---|---|---|
| [Metrics and identity](../../notebooks/day-10/01_metrics_and_contracts.ipynb) | Predict: Will three allowed attempts equal three independent successes? | For n=10,c=2,k=3, pass@3 is 0.5333; changing the cap changes contract identity. | Change normalization under a new named contract. | Oracle availability is not a deployed selector. |
| [Pairing and uncertainty](../../notebooks/day-10/02_paired_uncertainty.ipynb) | Predict: Will duplicated rows create new information? | Duplicating rows narrows the naive interval without adding independent sources. | Pair responses by source rather than treating them independently. | An interval inherits its sampling/population assumptions. |
| [Slices and overlap](../../notebooks/day-10/03_slices_and_contamination.ipynb) | Predict: Can mean improvement hide a rare failure? | One planted source overlap is found; removing it clears that fixture overlap. | Change a rare slice while keeping the aggregate visible. | Exact source checks are not semantic deduplication. |
| [Grading and replay · extension](../../notebooks/day-10/04_mathematical_grading_and_response_replay.ipynb) | Predict: Can an extracted correct value fail the task contract? | Two conflicting boxes are AMBIGUOUS; comparisons use 14 source groups; correctness and format remain separate. | Change a copied response while freezing parser and coverage. | Authored responses test the instrument, not a model. |
| [Actual candidates · extension](../../notebooks/day-10/05_actual_candidates_and_answer_selection.ipynb) | Predict: Can a correct available response lose selection? | Oracle availability is nondecreasing over pool prefixes; gold-blind selector decisions ignore injected grades. | Change the selector on the same retained candidate pool. | Tiny decoder candidates are not pretrained reasoning evidence. |

For actual evidence, compare the [short development replay](../../experiments/reports/2026-10-05-pretrained-evaluation-replay.md), [120-item instruction panel](../../experiments/reports/2026-10-05-native-sft400-comparison.md) and [blind story comparison](../../experiments/reports/2026-10-05-native-story-rating-independent.md) under their own populations. Retain correctness, format, stopping and reviewer disagreement separately; forty lexical groups within three templates are not 120 independent tasks.

Retain a short explanation of the intervention, observed change and claim it
does not establish. The deliverable should connect these explanations into one
defensible argument, with source/report links rather than an execution-only checklist.

<details>
<summary>Fresh CPU verification and operational reference</summary>

From the repository root, use the isolated environment/kernel described in
[Appendix D](../appendices/d-reproduction-and-environments.md):

```bash
.venv-course/bin/python scripts/verify_course_notebooks.py --days 10 --kernel dongxi-course --expected-prefix .venv-course
```

The verifier retains executed copies and identities in a new directory. Source
notebooks and learner attempts remain intact. Exact runner/replay/export commands
and their evidence boundaries are in the [runbook](../../docs/runbooks/evaluation_tools.md).

</details>
