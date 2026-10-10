# Lab 5 — Building a Modern Decoder

Machine: isolated CPU course kernel on Mac or Spark, offline.
Expected time: 25–45 minutes per notebook; longer routes span several sessions (planning estimate).
Prerequisites: Chapter 4 attention and cache identity; tensor shapes and autograd.
Deliverable: A whole-model shape map, parameter/KV ledger and one diagnosed architectural failure.

Read the [chapter](../chapters/05-building-a-modern-decoder.md) and use the
[worked solutions](../solutions/05-decoder-notebook-solutions.md) after attempting each exercise.
Write each prediction before executing the reference. Numerical checkpoints use
the notebook’s declared fixture, seed, dtype and tolerance; changing inputs may
change the result. References and interpretations remain adjacent to the attempt.

| Notebook | Question / prediction before reveal | Checkpoint on the declared fixture | One intervention | Evidence boundary |
|---|---|---|---|---|
| [Token and position lookup](../../notebooks/day-05/01_embeddings_and_positions.ipynb) | Predict: Will repeated IDs remain identical after position addition? | The same embedding row is reused; position-aware states differ. | Remove learned positions in a copied forward pass. | Random rows are not learned semantic geometry. |
| [Multi-head views](../../notebooks/day-05/02_multi_head_attention.ipynb) | Predict: Will head concatenation equal averaging? | Loop and vectorized outputs agree; a head ablation changes output; future weights remain zero. | Ablate one retrieved head. | One ablation does not identify an interpretable head role. |
| [Residual stream](../../notebooks/day-05/03_residual_stream.ipynb) | Predict: Can a branch cancel a skip-path contribution? | Zero branch leaves the stream unchanged; direct and branch gradients add. | Use the cancellation counterexample. | A skip path does not guarantee information preservation. |
| [Normalization placement](../../notebooks/day-05/04_layernorm_and_placement.ipynb) | Predict: Can correct attention coexist with future leakage? | Feature normalization matches the reference; wrong time normalization changes an earlier state. | Normalize over time rather than features. | Tiny checks do not rank pre-norm and post-norm training. |
| [Positionwise MLP](../../notebooks/day-05/05_positionwise_mlp.ipynb) | Predict: Can two affine maps replace a nonlinear branch? | A changed position does not directly change another MLP position; nonlinear output differs from the collapsed affine map. | Remove the activation. | Context can already reside in the MLP input. |
| [Whole decoder](../../notebooks/day-05/06_assemble_decoder.ipynb) | Predict: Are equal-valued weights necessarily one shared parameter? | Logits are [2,6,16]; tied input/output weights are the same object. | Use an untied equal-valued copy. | Architecture compatibility with pretrained weights is not tested. |
| [One-batch learning](../../notebooks/day-05/07_one_batch_learning.ipynb) | Predict: Does fitting one sequence imply transfer? | Declared fixture reaches loss below 0.05 and accuracy one; frozen control loss is unchanged. | Change the target alignment. | This is memorization of a fixed sequence. |
| [RMSNorm and SwiGLU](../../notebooks/day-06/01_rmsnorm_and_swiglu.ipynb) | Predict: Does RMS scaling remove an additive offset? | It does not; equal hidden width gives unequal two/three-projection parameter budgets. | Shift input and match parameter budgets separately. | Finite local outputs do not establish a method ranking. |
| [RoPE geometry](../../notebooks/day-06/02_rotary_positions.ipynb) | Predict: Will the right causal mask repair a wrong decode offset? | Rotations preserve pair norms; correct replay matches; the wrong offset produces an error. | Reset the cached query position to zero. | RoPE is inside Q/K attention, unlike learned input positions. |
| [KV sharing and cost](../../notebooks/day-06/03_gqa_qknorm_and_costs.ipynb) | Predict: Does reducing KV heads remove query-head views? | Four/two/one KV-head payloads have ratios 4:2:1; actual tensor bytes match the ledger. | Toggle the defined QK normalization independently. | Transparent KV expansion is not an optimized serving kernel. |
| [Architecture defense · core](../../notebooks/day-07/02_architecture_defense.ipynb) | Predict: Will a causal model necessarily have a correct cache? | Fixture ledger: 4,960 parameters, 3,072 logical KV bytes; wrong offset remains causal but breaks replay. | Diagnose wrong offset and time mixing independently. | FLOP estimates do not establish serving speed. |
| [Recurrent depth · optional](../../notebooks/day-07/01_recurrent_depth.ipynb) | Predict: Will shared weights produce equal activations at both depths? | Independent copies store twice the block parameters; the shared gradient equals their summed paths. | Compare per-application K/V before reusing a cache. | Fixed sharing does not implement adaptive depth or measure quality. |

Study the core architecture defense before the optional recurrent-depth notebook. Each session pairs a whole-model map with the component close-up. Learned input position embeddings and RoPE rotations inside attention are distinct mechanisms. The [ragged-cache extension](14-when-optimization-goes-wrong.md) belongs to Chapter 14; its operations remain in the runbook.

Retain a short explanation of the intervention, observed change and claim it
does not establish. The deliverable should connect these explanations into one
defensible argument, with source/report links rather than an execution-only checklist.

<details>
<summary>Fresh CPU verification and operational reference</summary>

From the repository root, use the isolated environment/kernel described in
[Appendix D](../appendices/d-reproduction-and-environments.md):

```bash
.venv-course/bin/python scripts/verify_course_notebooks.py --days 5 6 7 --kernel dongxi-course --expected-prefix .venv-course
```

The verifier retains executed copies and identities in a new directory. Source
notebooks and learner attempts remain intact. Exact runner/replay/export commands
and their evidence boundaries are in the [runbook](../../docs/runbooks/evaluation_tools.md).

</details>
