# Lab 9 — From Answer Gradients to a Defended Recipe

Machine: isolated CPU course kernel on Mac or Spark, offline.
Expected time: 25–45 minutes per notebook; longer routes span several sessions (planning estimate).
Prerequisites: Chapter 8 serialization, target ownership, padding and source groups.
Deliverable: A matched full/LoRA recipe card with target exposure, generation outcomes and retained limitations.

Read the [chapter](../chapters/09-supervised-fine-tuning.md) and use the
[worked solutions](../solutions/09-supervised-fine-tuning.md) after attempting each exercise.
Write each prediction before executing the reference. Numerical checkpoints use
the notebook’s declared fixture, seed, dtype and tolerance; changing inputs may
change the result. References and interpretations remain adjacent to the attempt.

| Notebook | Question / prediction before reveal | Checkpoint on the declared fixture | One intervention | Evidence boundary |
|---|---|---|---|---|
| [SFT gradient paths](../../notebooks/day-12/01_sft_objective_and_gradient_paths.ipynb) | Predict: Can ignored prompt logits coexist with nonzero prompt-embedding gradients? | Ignored direct logit gradients are zero; inspected user and unused tied-output rows receive gradient. | Detach a prompt path rather than changing its labels. | Direct supervision and downstream gradient paths differ. |
| [Token-weighted accumulation](../../notebooks/day-12/02_accumulation_and_checkpoint_identity.ipynb) | Predict: Will a mean of unequal microbatch means reproduce the intended objective? | Correct gradient difference is below 2e-6; the broken reduction exceeds 1e-5; complete CPU restore has zero error. | Replace total-target weighting with batch averaging. | Fixture recovery does not certify every saved state/interface. |
| [Tiny assistant](../../notebooks/day-13/01_tiny_assistant_training.ipynb) | Predict: Will lower answer NLL establish successful free generation? | Final fixture NLL is below initial NLL; evaluate complete responses and endings separately. | Change one response target or compare the frozen policy. | Four seen symbolic requests are not assistant generalization. |
| [Full versus LoRA](../../notebooks/day-14/01_full_sft_lora_and_recipe_defense.ipynb) | Predict: Will both low-rank factors receive gradient at zero-update initialization? | Initial LoRA equals its base; A gradient is zero, B nonzero, base gradient absent; merged output agrees. | Change rank while retaining target exposure. | A rank/recipe comparison does not establish universal LoRA quality. |

Read the [actual full/LoRA comparison](../../experiments/reports/2026-10-05-native-sft400-comparison.md): same Base and 9,321 training labels, different trainable spaces, whole-answer outcomes and natural endings. Keep the failed BF16 merge distinct from the separately verified FP32 export. [Response-level distillation](15-distill-evaluate-and-defend.md) has its learning home in Chapter 15; its supervised-objective bridge uses this chapter’s single-shift/mask reasoning.

Retain a short explanation of the intervention, observed change and claim it
does not establish. The deliverable should connect these explanations into one
defensible argument, with source/report links rather than an execution-only checklist.

<details>
<summary>Fresh CPU verification and operational reference</summary>

From the repository root, use the isolated environment/kernel described in
[Appendix D](../appendices/d-reproduction-and-environments.md):

```bash
.venv-course/bin/python scripts/verify_course_notebooks.py --days 12 13 14 --kernel dongxi-course --expected-prefix .venv-course
```

The verifier retains executed copies and identities in a new directory. Source
notebooks and learner attempts remain intact. Exact runner/replay/export commands
and their evidence boundaries are in the [runbook](../../docs/runbooks/sft_spark_runner.md).

</details>
