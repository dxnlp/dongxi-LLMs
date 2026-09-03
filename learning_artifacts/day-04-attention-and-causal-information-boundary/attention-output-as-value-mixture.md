# Attention Output as a Learned Value Mixture

- Day: 04
- Date opened: 2026-09-03
- Status: demonstrated
- Book destination: Chapter 4, normalized weights and value retrieval
- Related evidence: planned Day 4 attention notebook and gradient inspection
- Related production tasks: `CAND-ANIM-008`

## Questions that drove the discussion

- Does attention select one earlier token or construct a new representation?
- What exactly is mixed after the attention weights are normalized?

## Learner's initial model

The learner correctly explained that attention constructs something new by
blending multiple value vectors.

## Refined mental model

For one receiving query position $i$, causal masked softmax produces nonnegative
weights over allowed source positions:

$$
a_{ij}\ge 0,\qquad \sum_{j\ \mathrm{allowed}}a_{ij}=1.
$$

The attention output is

$$
o_i=\sum_j a_{ij}v_j.
$$

Thus one head performs soft retrieval: it creates a weighted mixture of learned
value vectors associated with positions. It does not normally emit a token ID,
copy an embedding row, or select exactly one source. A very sharp distribution
can approximate selection, but the operation remains a weighted sum.

For one isolated head before later projections, the output lies in the convex
hull of the available value vectors. This geometric statement must not be
overextended to the complete Transformer update: multi-head concatenation,
output projection, residual addition, normalization, and later nonlinear layers
can move the residual stream far beyond that single-head convex mixture.

The mixed objects are values, not original token strings. Each
$v_j=x_jW_V$ is a learned layer-specific message derived from a position's
incoming state. The same attention weights applied to different values produce a
different output; conversely, distinct weight patterns can sometimes produce the
same output if their weighted value sums coincide.

## Concrete examples and derivations

For three allowed sources with weights

$$
[0.2,0.7,0.1],
$$

the result is

$$
o_i=0.2v_1+0.7v_2+0.1v_3.
$$

Source 2 contributes most strongly, but sources 1 and 3 still affect every
output feature through their value vectors. The weights are shared across the
features of a value vector within one head; attention does not choose a separate
source distribution for every feature coordinate.

The operation for all query positions is

$$
O=AV,
$$

with shapes

$$
A:[T,T],\qquad V:[T,d_v],\qquad O:[T,d_v].
$$

## Demonstrated understanding

The learner stated that attention constructs a new representation by blending
multiple value vectors. This satisfies the central output-mechanism checkpoint.

## Evidence and limitations

The mixture and shape identities are analytical. Attention weights alone do not
identify the causal contribution or human-interpretable reason for a model
decision. Values, output projections, residual pathways, later layers, and
possible non-unique mixtures all limit a direct “attention is explanation”
interpretation.

## Open edges

- Trace gradients separately through the value path and the query/key routing
  path.
- Show two distinct attention distributions that yield the same output under
  redundant value vectors.
- Add output projection and residual flow when assembling the decoder block.
- Connect soft mixtures to near-hard retrieval under low-entropy attention.

## Reuse opportunities

- Chapter 4 transition from attention distributions to contextual states.
- Notebook visualization preserving source colors as their value vectors mix.
- `CAND-ANIM-008`: animate probability mass carrying value-vector components
  into a newly constructed receiving-position representation.
