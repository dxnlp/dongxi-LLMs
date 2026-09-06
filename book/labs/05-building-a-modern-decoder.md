# Chapter 5 Companion Lab — Building a Modern Decoder

Chapter 4 isolated attention: a position asks a learned query, compares legal
keys, and retrieves a value mixture. A language model needs more than this one
operation. It must turn IDs into states, preserve and transform those states
through depth, represent position, and return vocabulary-wide next-token scores.
This lab follows that complete path through eleven executable lessons.

This is the Chapter 5 **companion lab**, not a claim that the full Days 5–7
narrative chapter has been synthesized. Integrate its evidence into that chapter
during Day 7. All sessions already contain adjacent worked solutions; doing them
one by one remains separate from their reference verification.

## One model, inspected at several levels

Begin with a small baseline: two sequences, six input positions, vocabulary 16,
model width 16, four query heads of width four, two blocks, and MLP width 32.
The small dimensions expose the same tensor roles without requiring a GPU or
pretrained weights. A notebook may shrink a fixture further to inspect one
identity. Random vectors are not presented as learned semantic representations.

| Boundary | Tensor shape | Meaning |
|---|---|---|
| Input | `[B,T]` | Categorical token IDs |
| Token + learned position lookup | `[B,T,D]` | Initial position-aware states |
| Query projection and split | `[B,Hq,T,d]` | One query view per head and position |
| Key/value projection and split | `[B,Hkv,T,d]` | Source-address and source-content views |
| Attention distribution | `[B,Hq,T,S]` | Weights on allowed sources for each receiver |
| Head concatenation | `[B,T,Hq*d]` | Separate retrieved feature mixtures |
| Attention output projection | `[B,T,D]` | A same-width update for the stream |
| MLP expansion | `[B,T,F]` | Nonlinear feature transformation at each position |
| Vocabulary head | `[B,T,V]` | Next-token logits |

During a full forward pass S=T. During decoding, S includes cached positions.
Hq*d need not equal D; the projections connect these spaces. In the teaching
baseline they happen to be equal. Do not confuse the vocabulary size V with
the attention value tensor, also conventionally named V.

## Establish the baseline before changing it

1. [Embeddings and positions](../../notebooks/day-05/01_embeddings_and_positions.ipynb)
   distinguishes repeated lookup rows from different position-aware states and
   follows gradients into shared embedding rows.
2. [Multi-head attention](../../notebooks/day-05/02_multi_head_attention.ipynb)
   compares a loop over heads with a vectorized computation and tests a head
   ablation and future-token intervention.
3. [Residual connections](../../notebooks/day-05/03_residual_stream.ipynb)
   makes the direct and branch gradient contributions visible, including a
   cancellation counterexample.
4. [LayerNorm and placement](../../notebooks/day-05/04_layernorm_and_placement.ipynb)
   implements per-position feature normalization, then exposes how normalizing
   over time can violate causality even with a correct attention mask.
5. [The positionwise MLP](../../notebooks/day-05/05_positionwise_mlp.ipynb)
   examines nonlinear feature transformation and shows why removing the
   activation collapses two affine projections into one.
6. [Decoder assembly](../../notebooks/day-05/06_assemble_decoder.ipynb)
   joins the pieces, follows gradients into their parameters, tests causality
   and cache replay, and distinguishes tied weights from equal-valued copies.
7. [One-batch learning](../../notebooks/day-05/07_one_batch_learning.ipynb)
   tests a declared optimization claim under fixed data, seed, and budget.

For the pre-norm baseline, one block is:

$$
U=X+\operatorname{MHA}(\operatorname{LN}_1(X)),\qquad
Y=U+\operatorname{MLP}(\operatorname{LN}_2(U)).
$$

The normalization modules transform the branch inputs; the skip paths carry X
and U directly. Zero branch output therefore leaves the stream unchanged.
This does not imply perfect information preservation: a branch can write an
opposing update. For y=x+f(x), backward adds a direct contribution and a
Jacobian-transformed contribution; those contributions can cancel.

The MLP shares parameters across positions but does not directly mix their
states. Its input can already contain contextual information from attention.
Final normalization and the vocabulary head connect the last state to the
next-token loss from Chapter 3. Tying the output head to the input embedding
table accumulates both paths into one stored Parameter.

## Change one mechanism at a time

8. [RMSNorm and SwiGLU](../../notebooks/day-06/01_rmsnorm_and_swiglu.ipynb)
   separates normalization from gated feature transformation. RMSNorm scales
   without centering; SwiGLU uses three projection matrices, so equal hidden
   width is not an equal-parameter comparison with a two-projection GELU MLP.
9. [Rotary positions](../../notebooks/day-06/02_rotary_positions.ipynb)
   verifies pairwise rotation geometry and relative-position compatibility, then
   exposes incorrect decode offsets while holding causal visibility correct.
10. [GQA, QK normalization, and costs](../../notebooks/day-06/03_gqa_qknorm_and_costs.ipynb)
    reduces KV heads while retaining query-head distributions, toggles an
    explicitly defined QK-normalization variant, and reconciles parameter counts
    and logical cache payload with actual tensors.

The modern tiny decoder combines RMSNorm, SwiGLU, adjacent-pair RoPE, GQA, and
optional per-head RMS normalization of Q/K before RoPE. It is a teaching model,
not an implementation compatible with Qwen checkpoint tensors. The configuration
mapping in session 10 cites a pinned primary source and identifies this boundary.

Cache accounting asks a different question from latency measurement. For L
layers, batch B, retained length S, Hkv KV heads, head width d, and b bytes per
element, logical cache payload is:

$$
2LBH_{kv}Sdb.
$$

The factor two counts keys and values. This excludes allocator overhead and
temporary copies. Our transparent attention expands grouped K/V for arithmetic;
it is not an optimized grouped-attention kernel. Dense forward matrix-multiply
FLOP estimates count a multiply-add as two operations and do not measure speed.

## Optional: share depth without hiding the accounting

11. [Recurrent depth](../../notebooks/day-07/01_recurrent_depth.ipynb) applies
    one block twice and compares it with two equal-valued independent copies.
    Outputs initially agree, while stored parameters differ. The shared weight
    gradient equals the sum of the corresponding independent-copy gradients.

Equal weights do not imply equal activations: the second application reads a
different hidden state, so its K/V generally differ. The lab does not implement
a recurrent inference cache, adaptive early exit, or a trained quality
comparison. The notebook links the two requested papers as further reading;
its fixed-sharing mechanism is deliberately narrower than their methods.

## What the reference runs establish

The [verification report](../../experiments/reports/2026-09-06-decoder-notebooks.md)
records fresh-kernel execution of all eleven sessions and the bounded training
result. This demonstrates that the reference paths run and satisfy their checks
in the recorded environment. It does not establish learner mastery, general
language capability, long-context quality, Mac compatibility, or serving speed.

After each session, explain what changed in the controlled intervention and
why. After session 10, defend every shape and component in the complete decoder.
Use the [solution guide](../solutions/05-decoder-notebook-solutions.md) to resolve
conceptual gaps; exact runnable answers remain next to their exercises.
