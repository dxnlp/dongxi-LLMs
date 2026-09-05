# Day 4 — Attention and the Causal Information Boundary

- Date opened: 2026-09-03
- Day status: material complete; live practice deferred, mastery not fully assessed
- Book destination: `book/chapters/04-attention-and-the-causal-information-boundary.md`
- Required build: derive and implement single-head scaled dot-product attention
  without `nn.MultiheadAttention`, inspect attention weights and gradients, and
  deliberately break scaling and causal masking

## Central question

How can the representation at each token position retrieve useful information
from other positions while preserving the rule that next-token prediction may
not inspect the future?

Attention will be treated as learned, content-addressed communication—not as a
vague synonym for human focus and not as an explanation by itself.

## Learning arc

1. Begin with retrieval: a position asks a question, candidate positions
   advertise what they can match, and their content is mixed according to the
   resulting compatibility weights.
2. Derive queries, keys, and values as distinct learned projections of the same
   input states.
3. Track every tensor shape from contextual states through the full `[T,T]`
   score and attention matrices.
4. Explain dot-product similarity, division by `sqrt(d_k)`, row-wise softmax,
   and the weighted sum of values.
5. Apply the causal mask before softmax and connect it to Day 3's distinction
   between attention masks, loss masks, and backward gradient flow.
6. Inspect how gradients train the query, key, and value projections while
   preserving the limits of interpreting attention weights.
7. Build the mechanism directly, compare it with a trusted reference, then
   remove scaling or masking and explain the observed failures.
8. Derive the systems consequence: causal immutability makes past per-layer keys
   and values reusable, while each new query is transient; connect this to
   request-local KV caching, prefill, decoding, and cache release.

The central equation is

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V,
$$

where each softmax row describes one query position's distribution over allowed
key positions and $M$ encodes the causal information boundary.

## Conceptual boundaries for Day 4

Day 4 covers one attention head and the mechanism beneath convenient APIs. It
does not yet attempt to assemble an entire decoder block. Multi-head attention,
residual streams, normalization, feed-forward layers, and the complete
decoder-only Transformer belong to Day 5.

Day 4 does establish why KV caching is mathematically valid and how it follows
from Q/K/V roles plus causality. Detailed cache memory equations, grouped-query
attention, quantization, offloading, eviction, and serving benchmarks remain
forward links to the modern architecture and inference-systems material.

The important distinctions are:

- query versus key versus value;
- compatibility score versus normalized attention weight;
- token position versus feature dimension;
- causal visibility in the forward pass versus credit flow in backpropagation;
- an attention distribution versus a faithful explanation of model behavior.

## Planned interactive evidence

The detailed three-session plan is
[`../../notebooks/day-04/README.md`](../../notebooks/day-04/README.md): causal
attention forward mechanics, gradient and broken-variant diagnosis, and KV-cache
equivalence. The first session begins with the current mask-ordering question.

All three reference notebooks are now available. The complete
[chapter](../../book/chapters/04-attention-and-the-causal-information-boundary.md),
[twelve worked solutions](../../book/solutions/04-attention-and-the-causal-information-boundary.md),
and [coverage/evidence map](chapter-coverage-and-evidence.md) address every topic
in the Day 4 outline. Historical topic entries below preserve earlier discussion
states; use the coverage map for current verification and remaining learning gaps.

The first-class Day 4 notebook should let the learner:

- predict which keys a hand-designed query should retrieve;
- implement the score, scaling, mask, softmax, and value-mixing steps;
- check row sums and forbidden future probabilities;
- compare the manual result with PyTorch primitives;
- inspect gradients through $Q$, $K$, $V$, and their projection matrices;
- remove `1/sqrt(d_k)` and observe softmax saturation as dimensionality grows;
- remove or misplace the causal mask and expose future-token leakage.

Reusable empirical claims must move from notebook exploration into `src/`,
tests, a precommitted experiment specification, and a report before they are
called verified.

## Completion contract

Day 4 is complete only when the learner can:

1. explain the different jobs of queries, keys, and values without relying only
   on the words “search” and “content”;
2. derive the scaled dot-product attention equation and every tensor shape;
3. construct and interpret the causal mask for a short sequence;
4. distinguish score, weight, and attention output;
5. explain why scaling matters statistically rather than treating it as a magic
   constant;
6. trace a gradient path through the attention computation;
7. implement the mechanism without `nn.MultiheadAttention` and diagnose broken
   scaling and broken masking;
8. explain why KV retention is an optional inference optimization, why past
   keys and values can be reused within an unchanged prefix, and when a cache is
   released;
9. preserve the derivation, executable evidence, worked solutions, and coherent
   Chapter 4 contribution.

## Topic artifacts

Focused artifacts will be created as the discussion reaches them rather than in
advance. Expected topics are:

1. queries, keys, values, and content-addressed retrieval;
2. scaled dot products and row-wise attention distributions;
3. causal masking and information boundaries;
4. gradient flow, failure modes, and interpretability limits.

Opened topics:

1. [`queries-keys-values-and-retrieval.md`](queries-keys-values-and-retrieval.md)
2. [`why-cache-keys-and-values.md`](why-cache-keys-and-values.md)
3. [`dot-products-as-learned-compatibility.md`](dot-products-as-learned-compatibility.md)
4. [`why-scale-dot-products.md`](why-scale-dot-products.md)
5. [`causal-mask-before-softmax.md`](causal-mask-before-softmax.md)
6. [`attention-output-as-value-mixture.md`](attention-output-as-value-mixture.md)
7. [`attention-weights-are-not-explanations.md`](attention-weights-are-not-explanations.md)
8. [`how-loss-trains-qkv.md`](how-loss-trains-qkv.md)
9. [`future-recurrent-depth-and-looped-transformers.md`](future-recurrent-depth-and-looped-transformers.md)
10. [`chapter-coverage-and-evidence.md`](chapter-coverage-and-evidence.md)

The ninth artifact preserves a learner-requested frontier topic for Days 6–7.
It is a forward link, not an expansion of the Day 4 completion contract.

## Animation opportunity

`CAND-ANIM-008` records the full stable mechanism: input states become $Q$, $K$,
and $V$; query-key dot products form a score matrix; scaling controls its spread;
the causal mask removes forbidden edges before row-wise softmax; normalized
weights transport value vectors into new contextual states; and backward credit
splits into value/content and query-key/routing branches. The learner approved
the concept and it is promoted into `ANIM-ATTN-001` with a complete task packet
in `LEARNING_MEMORY.md`. Chapter and executable evidence are now complete;
production belongs exclusively on the Mac Studio.
