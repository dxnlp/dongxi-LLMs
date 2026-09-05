# Day 5 — The Decoder-Only Transformer

- Opened: 2026-09-05
- Status: in progress; outline introduced, implementation not yet started
- Book destination: Chapter 5, Building a Modern Decoder (Days 5–7)
- Central question: how do attention, feature transformations, and a persistent
  representation combine into a trainable next-token model?

## Learning outline

1. Reconnect token IDs, embeddings, causal states, and next-token logits.
2. Multi-head attention: separate learned retrieval patterns, split/merge shapes,
   concatenation, and the attention output projection.
3. Residual stream: preserve a representation while sublayers contribute
   updates; distinguish feature flow from gradient flow.
4. Normalization: LayerNorm over features at each position, learned scale/bias,
   numerical epsilon, and pre-norm versus post-norm placement.
5. Feed-forward network: positionwise expansion, nonlinear activation, and
   contraction; why token mixing and feature transformation are complementary.
6. Position information: why a practical decoder includes position handling;
   use learned absolute positions for the first baseline. RoPE belongs to Day 6.
7. Assemble a pre-norm decoder block, stack blocks, apply final normalization,
   then project to vocabulary logits. Distinguish this output head from the
   attention head-merging projection. Revisit optional embedding/output tying.
8. Initialization and a minimal model: inspect shapes, finite activations, causal
   invariance, and the flow from shifted labels through loss to parameters.
9. Fit one small, consistent batch, inspect loss and predictions, and explain
   what memorization verifies and what it cannot establish about generalization.

## Planned companion sessions

See `notebooks/day-05/README.md`: multiple heads, residual/normalization/MLP
interventions, and end-to-end tiny-decoder integration. Each notebook will
include adjacent runnable solutions and explanations. Live notebook practice
is optional for the current discussion; complete material remains part of the
course deliverable.

## Boundaries and completion evidence

The baseline uses ordinary multi-head attention, LayerNorm, a GELU MLP, learned
absolute positions, and causal next-token loss. RMSNorm, RoPE, SwiGLU, GQA,
QK normalization, detailed cost comparisons, and recurrent depth are reserved
for Days 6–7 after the baseline is understood.

Required material: reader-facing Chapter 5 sections, solutions, executable
notebooks, importable decoder implementation, tests, and a specified one-batch
experiment/report. Define its dataset, target consistency, training budget, and
success criteria before running. Do not infer generalization from fitting it.

Learner mastery requires explaining each component's role and the complete
tensor path. Day 4 practice remains deferred; it is not silently marked complete.

## First discussion

Why could one weighted mixture be insufficient when a position needs several
different kinds of information from its prefix? Begin with this motivation for
multiple heads without assigning guaranteed linguistic roles to individual heads.
