# Lab 1 — Evidence Before Optimization

Machine: isolated CPU course kernel on Mac or Spark, offline.
Expected time: 25–45 minutes per notebook; longer routes span several sessions (planning estimate).
Prerequisites: Read the chapter opening; basic Python dictionaries and conjunctions.
Deliverable: An evidence card separating intended controls, actual observations and warranted claims.

Read the [chapter](../chapters/01-evidence-before-optimization.md) and use the
[worked solutions](../solutions/01-evidence-before-optimization.md) after attempting each exercise.
Write each prediction before executing the reference. Numerical checkpoints use
the notebook’s declared fixture, seed, dtype and tolerance; changing inputs may
change the result. References and interpretations remain adjacent to the attempt.

| Notebook | Question / prediction before reveal | Checkpoint on the declared fixture | One intervention | Evidence boundary |
|---|---|---|---|---|
| [Run identity](../../notebooks/day-01/01_experiment_identity.ipynb) | Predict: Will reordered JSON change identity, and will a changed context? | Reordering preserves the fingerprint; changing context changes it. | Change one tokenizer/context field. | Identity sensitivity does not measure capability. |
| [Smoke acceptance](../../notebooks/day-01/02_smoke_criteria_and_claims.ipynb) | Predict: Can a zero exit compensate for failed reserve or nonfinite state? | Acceptance is the conjunction; each planted failure refuses. | Remove a required observation. | Sampled reserve does not observe every instant. |
| [Read an actual run](../../notebooks/day-01/03_read_a_real_run.ipynb) | Predict: Does declining NLL establish a coherent plot? | Read the final whole story and EOS separately from NLL. | Compare both retained decoding modes at the same checkpoint. | Selected archive examples do not estimate a story population. |
| [Checkpoint meanings · extension](../../notebooks/day-01/04_checkpoint_interface.ipynb) | Predict: Will an eight-ID permutation load yet change the prediction problem? | Equal vocabulary size is eight; swapped mapping is incompatible; unchanged local reload passes. | Keep IDs fixed and change normalization. | These local interfaces do not establish pretrained quality. |

The eight-ID checkpoint extension constructs and serializes local random interfaces. It downloads nothing; loading success and semantic compatibility are separate checks.

Retain a short explanation of the intervention, observed change and claim it
does not establish. The deliverable should connect these explanations into one
defensible argument, with source/report links rather than an execution-only checklist.

<details>
<summary>Fresh CPU verification and operational reference</summary>

From the repository root, use the isolated environment/kernel described in
[Appendix D](../appendices/d-reproduction-and-environments.md):

```bash
.venv-course/bin/python scripts/verify_course_notebooks.py --days 1 --kernel dongxi-course --expected-prefix .venv-course
```

The verifier retains executed copies and identities in a new directory. Source
notebooks and learner attempts remain intact. Exact runner/replay/export commands
and their evidence boundaries are in the [runbook](../../docs/runbooks/evaluation_tools.md).

</details>

### Assignment F: connect the mechanism to a comparison

Before opening the [companion route](../../docs/COURSE_ASSIGNMENTS.md), predict your answer to this question: Two arms have the same update count but one discards half its collected responses. What must the defense disclose?

Use the [worked reasoning](../solutions/01-evidence-before-optimization.md) after recording the prediction. Then change one declared condition and explain whether the result changes the mechanism, the measured population or the supported claim. Retain the failed variant alongside the successful reference. The [assignment brief](../../docs/COURSE_ASSIGNMENTS.md) specifies the combined deliverable.
