# Chapter 5 — Notebook Worked-Solution Guide

The [companion lab](../labs/05-building-a-modern-decoder.md) provides the
sequential route. Every exercise has its runnable solution and explanation
immediately below it in the notebook. This guide summarizes what a sound
interpretation must preserve; it is not a replacement for running the cells.

| Session | Core reasoning in the worked solution |
|---|---|
| [1. Lookup and positions](../../notebooks/day-05/01_embeddings_and_positions.ipynb) | Repeated IDs select the same embedding row; different positions add different vectors. Lookup gradients sum over occurrences. Output tying can add gradients to rows absent from the input. Removing position embeddings does not remove the causal graph. |
| [2. Multiple heads](../../notebooks/day-05/02_multi_head_attention.ipynb) | Split and merge preserve time as its own axis. Loop and vectorized results and gradients agree under identical weights. Zeroing a head changes its value contribution, not the definition of causal visibility. |
| [3. Residuals](../../notebooks/day-05/03_residual_stream.ipynb) | For Y=X+XW, the incoming gradient is G+GW transpose in row-vector notation. A zero branch yields identity; a branch equal to -X cancels both state and gradient. |
| [4. LayerNorm](../../notebooks/day-05/04_layernorm_and_placement.ipynb) | Normalize over features with population variance and epsilon, then apply scale/bias. Time-axis statistics can leak future information. Zero branches make a pre-norm block identity, but post-norm still normalizes. |
| [5. MLP](../../notebooks/day-05/05_positionwise_mlp.ipynb) | Without activation, the combined weight is W_down W_up and combined bias is W_down b_up+b_down. Nonlinearity prevents this general collapse. Positionwise computation still acts on contextual inputs. |
| [6. Assembly](../../notebooks/day-05/06_assemble_decoder.ipynb) | Blocks preserve residual width, whereas the vocabulary head outputs one logit per candidate token. Tying is Parameter identity, not copying. A copy oracle fails the next-token objective despite near-zero incorrectly unshifted loss. |
| [7. Learning](../../notebooks/day-05/07_one_batch_learning.ipynb) | The declared batch is consistent and shifted once. Optimization changes parameters and reduces the correct loss. The frozen control stays unchanged. No gold labels are declared for the reversed-context probe, so its outputs cannot be scored as a generalization result. |
| [8. Modern norm/MLP](../../notebooks/day-06/01_rmsnorm_and_swiglu.ipynb) | RMS scaling does not center features. A SiLU gate is not a probability. Three gated projections require a different width for approximately matched parameters; biases must be accounted for. |
| [9. RoPE](../../notebooks/day-06/02_rotary_positions.ipynb) | Rotations preserve coordinate-pair norms; R_m transpose R_n gives R_(n-m). Cached keys keep their original positions. Restarting new query/key offsets breaks equivalence even with a correct causal mask. |
| [10. GQA/costs](../../notebooks/day-06/03_gqa_qknorm_and_costs.ipynb) | Fewer KV heads reduce compact cache payload and KV projection parameters, not the number of query distributions. QK normalization is explicitly specified. Logical bytes and dense arithmetic estimates are not measured latency or total runtime memory. |
| [11. Recurrence](../../notebooks/day-07/01_recurrent_depth.ipynb) | Shared parameters receive the sum of gradients through their uses. Weight sharing preserves stored size while additional applications spend computation. Different application states generally require different K/V; no adaptive router or quality gain is established. |

## Architecture defense

Use the modern tiny model after session 10. Explain, without relying on its
class names alone, what each operation does to the token, head, and feature
axes; why it is causal; which parameters it trains; and which invariants survive
its controlled intervention. An answer is strong when it identifies the relevant
test and its evidence boundary, not merely when it repeats a layer name.

Notebook readiness and the recorded reference experiment are complete material
deliverables. Learner completion and the full Chapter 5 narrative synthesis
remain separate tasks.
