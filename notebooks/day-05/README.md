# Chapter 5 — Layer-by-Layer Notebook Pathway

Designed on 2026-09-06 at the learner's request. Chapter 5 spans Days 5–7;
these are focused sessions, not a requirement to finish ten notebooks in one day.
This replaces the earlier three-session plan. All **eleven notebooks are built
and verified** in fresh kernels, including the optional extension. Each includes
adjacent runnable solutions and explanations. Learner sessions begin with 01;
reference verification does not mark learner completion.

See the [companion lab](../../book/labs/05-building-a-modern-decoder.md),
[solution guide](../../book/solutions/05-decoder-notebook-solutions.md), and
[verification report](../../experiments/reports/2026-09-06-decoder-notebooks.md).
On Spark choose **Python (DGX Spark Native)**. On another machine choose a local
Python kernel with Torch installed; no GPU or downloads are required. Source
outputs are cleared intentionally. Keep notebooks inside this checkout so their
startup cells can locate the reusable code.

## Baseline decoder — Day 5

Seven notebooks under `notebooks/day-05/` build one consistent tiny decoder.
Sessions 01–05 isolate components; 06 connects them; 07 tests learning.

| Session / notebook | Central question and learner implementation | Controlled change and acceptance evidence |
|---|---|---|
| 01 — [01_embeddings_and_positions.ipynb](01_embeddings_and_positions.ipynb) | How do categorical IDs become position-aware states? Implement lookup and learned absolute position addition; inspect repeated IDs at different positions and embedding gradients. | Disable or offset positions; verify lookup equality before position addition, shape preservation, and which embedding rows receive lookup-path gradients. Do not claim a causal model without position embeddings is wholly order-blind: its mask still provides structure. |
| 02 — [02_multi_head_attention.ipynb](02_multi_head_attention.ipynb) | What do several retrieval mixtures preserve? Implement per-head projections, split/merge, causal attention, concatenation, and the attention output projection. | Compare looped-head and vectorized implementations using identical weights; verify outputs and gradients within declared tolerances. Zero one head; inspect changed outputs without assigning heads guaranteed linguistic roles. Test causal prefix invariance. |
| 03 — [03_residual_stream.ipynb](03_residual_stream.ipynb) | How can a layer modify a state while retaining a direct path? Implement a residual wrapper and inspect branch versus skip-path gradients. | Set the branch output to zero; check forward identity and the gradient sum. Contrast replacing the stream with updating it. A direct path is not a guarantee against cancellation or unstable training. |
| 04 — [04_layernorm_and_placement.ipynb](04_layernorm_and_placement.ipynb) | What does normalization normalize, and where does it belong? Implement per-position LayerNorm with population variance, epsilon, learned scale and bias; compare pre-norm and post-norm blocks. | Match library forward/backward results; deliberately normalize the wrong axis; inspect constant inputs and zero-branch behavior. Compare fixed-seed activation/gradient traces, without treating one toy trace as proof of universal pre-norm superiority. |
| 05 — [05_positionwise_mlp.ipynb](05_positionwise_mlp.ipynb) | What does an MLP add after attention has mixed tokens? Implement expansion, GELU, and contraction with weights shared across positions. | Remove the nonlinearity and verify equivalence to one affine map, including biases. Perturb one position at the isolated MLP input and verify other positions do not change. Trace gradients through both projections. |
| 06 — [06_assemble_decoder.ipynb](06_assemble_decoder.ipynb) | Can we explain every tensor from IDs to vocabulary logits? Assemble embeddings, positions, pre-norm blocks, final norm, and vocabulary head; inspect initialization and optional embedding/output tying. | Check shapes, finite activations/loss/gradients, full-model causal invariance, tied versus untied parameter counts, and selected parameter gradients. Distinguish the attention output projection from the vocabulary head. Include a deliberately broken label shift. |
| 07 — [07_one_batch_learning.ipynb](07_one_batch_learning.ipynb) | Does the assembled network actually learn the declared next-token task? Implement a bounded training loop with correctly shifted labels and inspect predictions and parameter changes. | Specify seed, consistent targets, batch, optimizer, step budget, and success threshold before running. Compare frozen and trained models; retain failures. Fit the batch and probe changed contexts, explicitly separating memorization from generalization. |

## Modern architecture — Days 6–7

Built ahead of live study at the learner's explicit request on 2026-09-06.
Use the verified baseline as the reference and change one mechanism at a time.

| Session / notebook | Central question and implementation | Controlled change and acceptance evidence |
|---|---|---|
| 08 — [01_rmsnorm_and_swiglu.ipynb](../day-06/01_rmsnorm_and_swiglu.ipynb) | How do modern normalization and gated feature transforms differ from the baseline? Implement RMSNorm and SwiGLU in two short sections. | Contrast centering with RMS scaling; inspect gate behavior; verify forward/backward references and compare MLP widths at a declared parameter budget. Test each replacement separately before combining them. |
| 09 — [02_rotary_positions.ipynb](../day-06/02_rotary_positions.ipynb) | How can position change query–key compatibility? Implement pairwise RoPE rotations with explicit position indices. | Check pairwise norm preservation, relative-position dot-product identity, and cached versus full-prefix outputs with correct offsets. Introduce incorrect decode offsets. Numerical equivalence is not evidence of length extrapolation quality. |
| 10 — [03_gqa_qknorm_and_costs.ipynb](../day-06/03_gqa_qknorm_and_costs.ipynb) | Which projections and cached states are shared, and what does that save? Implement grouped-query attention and a separately toggled, explicitly specified QK-normalization variant. | Match grouped and explicitly repeated K/V references; check MHA/MQA endpoints, cache shapes, parameter counts, and logical cache bytes. State multiply-add conventions for FLOP estimates; distinguish estimates from measured latency and allocator memory. Integrate tested variants into a tiny DongxiGPT and map components to a pinned Qwen3 configuration. |

Day 7 uses the assembled model for an architecture defense: explain each shape,
component, gradient path, and cost; diagnose a withheld broken variant; reconcile
the implementation with its configuration. Candidate 50M/100M/150M designs are
accounting exercises until a separate profiled run is authorized.

Optional [session 11](../day-07/01_recurrent_depth.ipynb)
implements fixed shared-block reuse after the baseline works. Compare stored
parameters, block applications, and compute under explicitly separate matching
conditions. Adaptive routing is a later extension, not an implied implementation
of an entire paper. Reuse `ARCH-LOOP-001`, `X-LOOP-001`, and `CAND-ANIM-011`;
refresh primary sources when this frontier session becomes active.

## Common notebook design

- Start with a conceptual question and prediction, followed by a small learner
  implementation, an inspection, and a controlled intervention.
- Place a clearly labeled runnable reference solution and explanation directly
  after every exercise. No external syntax search or arithmetic quiz is required.
- Use a common CPU-first toy configuration initially: batch 2, length 6,
  vocabulary 16, model width 16, four heads of width 4, two blocks, MLP width 32.
  These are the declared baseline teaching dimensions. Use float64
  for derivative checks and a declared training dtype; disable dropout for
  deterministic equivalence checks.
- Reuse components in `src/dongxi_llms/`; the notebook should expose, not hide,
  the small implementation and intermediate tensors. Do not begin from an opaque
  complete Transformer API. No large model downloads or GPU requirement.
- Verify reference paths in a fresh kernel, declare numerical tolerances, add
  regression tests for important identities, and link empirical training claims
  to a pre-run specification and post-run report.
- Link completed notebooks from Chapter 5 and its worked solutions. Keep
  material readiness, reference execution, and learner completion separate.
- Preserve learner modifications. CPU fixtures should remain portable to Mac
  and Spark with the environment recorded; execution on one does not prove the
  other environment has been tested.

## Production order and animation reuse

All sessions are built; study 01 first, then progress through the baseline in
order. Modern variants depend on the integrated baseline, not just isolated
layer tests. The bounded CPU training reference has been executed; no notebook
server or persistent training job was started.

Multi-head and residual examples now supply verified evidence for `CAND-ANIM-012` and
`CAND-ANIM-013`. Check for additional mathematical animation opportunities when
each derivation is developed; do not pre-authorize a film per notebook. All
animation production remains on the Mac Studio after explicit approval.
