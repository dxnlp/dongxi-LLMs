# Day 5 — The Decoder-Only Transformer

- Opened: 2026-09-05
- Status: in progress; live discussions reached Notebook 07; Day 5 narrative
  foundation and worked answers written; independent mastery not yet assessed
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

On 2026-09-06 the learner explicitly requested hands-on sessions for each
important architecture component. The expanded canonical design is
`notebooks/day-05/README.md`: seven baseline sessions, three modern-architecture
sessions across Days 6–7, and one optional recurrent-depth extension. Residuals,
LayerNorm, and MLPs now each have their own baseline notebook. Each includes an
implementation, controlled intervention, adjacent runnable solution, explanation,
and acceptance checks. At the learner's subsequent explicit request, all eleven
were built and executed in fresh kernels on 2026-09-06. See
`experiments/reports/2026-09-06-decoder-notebooks.md` and
`book/labs/05-building-a-modern-decoder.md`.

The Chapter 4 conceptual review reached projection-matrix gradients before this
transition. Its hands-on practice remains deferred; no additional mastery claim
is inferred. The Day 5 discussions have reached batched next-token learning.
Next: review the chapter/Notebook 07 evidence boundary and explain the whole
decoder path before moving to the Day 6 variants.

## Boundaries and completion evidence

Implementation record:
[`notebook-pathway-and-evidence.md`](notebook-pathway-and-evidence.md).

Reader-facing foundation:
[`Chapter 5`](../../book/chapters/05-building-a-modern-decoder.md), with
[twelve worked conceptual answers](../../book/solutions/05-decoder-notebook-solutions.md#day-5-foundation--worked-conceptual-solutions).
Discussion refinements:
[`normalization-mlp-and-batched-learning.md`](normalization-mlp-and-batched-learning.md).
Days 6–7 will extend the same chapter rather than replace this foundation.

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

Opened topic:
[`multiple-heads-and-output-projection.md`](multiple-heads-and-output-projection.md).
Introduced during the Day 4 review; this does not mark Day 4 mastery complete.

Next topic introduced:
[`residual-stream-and-attention-updates.md`](residual-stream-and-attention-updates.md).

Why could one weighted mixture be insufficient when a position needs several
different kinds of information from its prefix? Begin with this motivation for
multiple heads without assigning guaranteed linguistic roles to individual heads.
