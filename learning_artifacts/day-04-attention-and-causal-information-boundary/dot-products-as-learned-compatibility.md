# Dot Products as Learned Compatibility

- Day: 04
- Date opened: 2026-09-03
- Status: demonstrated
- Book destination: Chapter 4, query-key compatibility and score matrices
- Related evidence: planned Day 4 attention notebook and reference implementation
- Related production tasks: `CAND-ANIM-008`

## Questions that drove the discussion

- Does a large query-key dot product mean two tokens are intrinsically
  semantically similar?
- Why can a dot product serve as a retrieval score at all?

## Learner's initial model

The learner correctly rejected intrinsic semantic similarity and proposed that a
large score instead indicates a relationship in the current context.

## Refined mental model

The contextual intuition is correct but should be stated more precisely. A score

$$
s_{ij}=q_i\cdot k_j
$$

measures whether source position $j$, represented through its learned key, is
compatible with what receiving position $i$ asks for through its learned query.
The compatibility is specific to a layer, attention head, model state, and input.
It is not a universal semantic similarity between the two token strings.

The projections make the comparison learnable:

$$
s_{ij}=(x_iW_Q)(x_jW_K)^\top
=x_iW_QW_K^\top x_j^\top.
$$

Thus dot-product attention is a learned bilinear relation over incoming states.
The dot product itself contains no linguistic knowledge. Training adjusts
$W_Q$ and $W_K$ so that retrieving useful value vectors becomes favorable under
the downstream loss.

## Concrete examples and derivations

Because

$$
q_i\cdot k_j=\lVert q_i\rVert\lVert k_j\rVert\cos\theta,
$$

the raw score reflects both directional alignment and vector magnitude in the
learned comparison space. A high score can represent syntactic dependency,
coreference, positional routing, copying, delimiter matching, or another useful
relation. Semantically related tokens need not receive a large score, and
semantically different tokens may be strongly compatible for a particular
retrieval job.

Stacking all positionwise queries and keys produces

$$
S=QK^\top,
$$

with shape $[T,T]$ for one sequence and one head. Row $i$ contains one receiving
query's scores over all candidate source positions; column $j$ shows how that
source is scored by the different receiving queries. These are unnormalized
scores, not probabilities.

## Demonstrated understanding

The learner explained that a high attention score does not necessarily establish
semantic similarity and instead depends on the relationship created in context.
The refined checkpoint is to call that relationship query-specific learned
compatibility rather than generic relatedness.

## Evidence and limitations

The geometrical and tensor identities are analytical. No particular linguistic
role should be attributed to a real model head without controlled evidence.
Attention weights derived from these scores are internal routing coefficients,
not automatically faithful explanations of model decisions.

## Open edges

- Explain why raw dot-product spread grows with key/query dimension.
- Derive the $1/\sqrt{d_k}$ scale from a variance argument.
- Apply causal masking and row-wise softmax to turn allowed scores into weights.
- Verify hand-computed scores against an executable implementation.

## Reuse opportunities

- Chapter 4 distinction between semantic similarity and learned compatibility.
- Exercise using the bilinear form to explain directed relations.
- `CAND-ANIM-008`: keep token positions fixed while projected query/key vectors
  generate the full score matrix.
