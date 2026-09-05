# Solutions — Chapter 4: Attention and the Causal Information Boundary

These answers accompany the complete [chapter](../chapters/04-attention-and-the-causal-information-boundary.md).
The notebooks contain the same mechanisms as inspectable code with adjacent
solutions. A convincing answer explains the causal and computational assumptions.

## Exercise 1 — Shapes and meaning

| Object | Shape |
|---|---|
| X | [5,8] |
| W_Q, W_K | [8,4] |
| W_V | [8,3] |
| Q, K | [5,4] |
| V | [5,3] |
| QKᵀ, mask, A | [5,5] |
| O=AV | [5,3] |

None of these axes indexes the output vocabulary. Rows are receiving positions
and A's columns are source positions. The later output head maps final hidden
features to vocabulary logits. Q/K widths must match; V's width is independent.

## Exercise 2 — Directed retrieval

The score is x_i W_Q W_Kᵀ x_jᵀ. Swapping i and j transposes the effective
bilinear relationship. Since W_QW_Kᵀ need not be symmetric, the scores need not
agree. Separate projections let a receiver match a source under a different
role from the one that source uses when receiving information itself.

If W_Q=W_K, the unmasked dot-product matrix is symmetric. Causal masking and
row-wise normalization can still make the final attention matrix asymmetric.

## Exercise 3 — Scaling

With independent zero-mean coordinates,

$$
\operatorname{Var}(q_m k_m)=\sigma_q^2\sigma_k^2,
\qquad
\operatorname{Var}(q\cdot k)=d_k\sigma_q^2\sigma_k^2.
$$

Dividing by sqrt(d_k) removes the dimension factor but retains
σ_q²σ_k². With correlated coordinate products, covariance terms appear in the
variance of the sum. Trained magnitudes and correlations can therefore create
wide score gaps after standard scaling.

The paired simulation in [notebook 2](../../notebooks/day-04/02_attention_gradients_and_failures.ipynb)
tests the unit-variance IID case. It does not establish a universal trained-model
distribution.

## Exercise 4 — Mask order

Pre-softmax masking gives [0.5,0.5,0]. Naive post-softmax zeroing leaves
approximately [0.0000454,0.0000454,0], because exp(10) was present in the
normalization denominator. Replacing the forbidden score with zero gives
[1/3,1/3,1/3], because exp(0)=1.

Dividing the post-masked weights by their remaining sum cancels the original
denominator in exact arithmetic. In finite precision, a sufficiently large
forbidden maximum may have already made every allowed probability zero. The
repair then yields 0/0. Masking before stable normalization avoids that problem.

If a query has no allowed keys, the desired distribution is itself undefined.
A separate padding/query policy is necessary; simply choosing a huge negative
number does not solve the semantic problem.

## Exercise 5 — Future intervention

“Crossed” must have the same output: all its allowed incoming representations,
weights, and positions remain identical. The fourth output may change because
the fourth input itself changes. The state at “crossed” predicts a continuation
without observing that continuation.

Assumptions include causal upstream states, fixed parameters, compatible
position handling, and disabled stochastic effects such as dropout. In floating
point, different kernels or shapes may introduce small rounding differences;
equality checks should declare tolerance.

A single unchanged output under one perturbation does not prove causality.
The chapter's naive post-mask fixture leaves one earlier output unchanged while
changing others. Use a general graph argument and multiple well-chosen tests.

## Exercise 6 — Backward credit

Since dO=(dA)V+A(dV), Frobenius inner products with G_O give

$$
G_A=G_OV^\top,\qquad G_V=A^\top G_O.
$$

The softmax derivative supplies

$$
G_R=A\odot\left(G_A-\operatorname{rowsum}(A\odot G_A)\right).
$$

Then

$$
G_Q=G_RK/\sqrt{d_k},\quad G_K=G_R^\top Q/\sqrt{d_k},
$$

$$
G_X=G_QW_Q^\top+G_KW_K^\top+G_VW_V^\top.
$$

For each projection, G_W=XᵀG_projected. Earlier prompt rows can receive credit
through K and V even with no direct loss at those rows. A forbidden future row
has no path to the isolated earlier target. Shared projection parameters still
receive gradients from other valid paths.

The [gradient implementation](../../src/dongxi_llms/attention_evidence.py)
checks all these identities against autograd and independently checks projection
weights with central differences.

## Exercise 7 — Detach and saturation

Detaching A removes the W_Q/W_K paths while preserving the value path. Detaching
V removes W_V's path while routing can still learn from fixed messages. The
forward loss stays the same because detach changes graph connectivity, not the
tensor's numerical value. These are statements about this isolated graph with
independent projection parameters.

Softmax's Jacobian approaches zero near a one-hot row. For vocabulary
cross-entropy, the target's negative-log derivative compensates, giving p−q.
For attention, the incoming derivative comes through values and subsequent
computation; no such compensation is guaranteed. A small softmax Jacobian can
therefore weaken its routing signal, without implying every full-network
gradient vanishes.

## Exercise 8 — Interpretation

For values [2,0], [0,2], [1,1], both [.4,.4,.2] and [.1,.1,.8] yield [1,1].
All weights are positive and normalized; the ambiguity does not rely on exact
hard selection.

An attention map establishes routing coefficients for that head and execution.
It does not, by itself, identify a final prediction's unique cause. Changing
values, heads, or inputs and measuring downstream effects can provide stronger
evidence, but intervention design and distribution shift also require care.

## Exercise 9 — Cache correctness

A past position cannot read an appended future position. Its representation at
each causal layer remains the same, so that layer's projected K/V remain valid.
The new query uses those K/V to retrieve; it does not use earlier queries.

Identical first-layer incoming states may give identical first-layer keys.
Their deeper states can differ after attention to different earlier prefixes.
The cache replay demonstrates both behaviors.

Cache validity requires an unchanged prefix, compatible parameters and
position treatment, and compatible execution state. If part of the prefix
changes, recompute the affected state; retaining the same token string at one
position is insufficient.

## Exercise 10 — Cache memory and masking

There are two tensors, two layers, six positions, four features, and eight bytes
per float64 scalar:

$$
2\times2\times6\times4\times8=768\ \text{bytes}.
$$

This counts logical tensor contents only, not Python objects, concatenation
temporaries, or allocator memory.

For a single final-position query, the source positions already consist solely
of past positions and self. It may attend to all of them. A [1,N] lower triangle
without an offset allows only key zero, because it treats the query as global
position zero. For a multi-token chunk after P cached positions, new query i
can read keys j≤P+i.

## Exercise 11 — Experimental claims

The checks establish numerical derivative agreement and full-prefix/cached
agreement for declared small fixtures. The operation count demonstrates how
many layer-position rows receive K or V projections in the chosen replay
schedule.

They do not establish trained-model quality, end-to-end generation correctness
for every architecture, or a speedup equal to the operation-count ratio.
Latency claims need controlled timing, warmup, synchronization, batching, and
hardware/backend identity. Cache quantization or changed positional handling
requires new correctness checks. Quality claims need frozen evaluation.

## Exercise 12 — Implementation defense

- Wrong softmax axis: inspect row sums with a nonsymmetric score fixture, verify
  receiver/source orientation, and compare with a reference. Symmetric special
  cases can conceal this bug.
- Future leakage: intervene on later inputs and compare earlier outputs. Also
  inspect forbidden weights, since a finite perturbation may accidentally have
  no effect.
- Zero-score replacement: inspect forbidden probabilities; exp(0) is nonzero.
- All-masked row: detect rows with no allowed sources before normalization,
  define their policy, and assert finite outputs for evaluated rows.

The [forward notebook](../../notebooks/day-04/01_causal_attention_forward.ipynb)
contains mask and invariance checks. The
[cache notebook](../../notebooks/day-04/03_kv_cache_equivalence.ipynb)
adds a stale-prefix counterexample. Together with the analytical argument, these
give evidence at the mechanism level without substituting for broader model
evaluation.
