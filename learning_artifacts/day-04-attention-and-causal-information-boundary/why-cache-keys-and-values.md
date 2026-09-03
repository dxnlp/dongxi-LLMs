# Why Autoregressive Decoding Caches Keys and Values

- Day: 04
- Date opened: 2026-09-03
- Status: demonstrated
- Book destination: Chapter 4 attention consequence; later KV-cache treatment in
  the modern decoder chapter
- Related evidence: planned Day 4 attention notebook; future decoding benchmark
- Related production tasks: `CAND-ANIM-009`

## Questions that drove the discussion

- Why is the mechanism called a KV cache rather than a QKV cache?
- Which projected states remain useful after one autoregressive decoding step?

## Learner's initial model

After distinguishing query, key, and value roles, the learner correctly inferred
that their asymmetry explains the name “KV cache”: past positions must remain
available as sources, whereas a past query has already completed its retrieval.

The learner then identified an important apparent contradiction: the same token
can have different query, key, and value vectors in different contexts, so how
can a cached key or value remain reusable? This exposed the missing distinction
between a token-level global cache and a request-local prefix cache.

## Refined mental model

At decoding step $t$, the new position creates $q_t$, $k_t$, and $v_t$. Its query
is consumed immediately to retrieve from all keys and values available through
the current position:

$$
o_t=\operatorname{softmax}\!\left(
\frac{q_tK_{\le t}^{\top}}{\sqrt{d_k}}
\right)V_{\le t}.
$$

Then $q_t$ has finished its job. A future position will form a new query of its
own; it will not normally reuse $q_t$. By contrast, $k_t$ must remain available
so future queries can decide whether to retrieve from position $t$, and $v_t$
must remain available as the message they would retrieve. The decoder therefore
appends $k_t$ and $v_t$ to a cache for every attention layer.

The cache is specific to one exact sequence prefix and one model execution. It
does not store a universal key and value for a vocabulary token. Two occurrences
of `bank` in different sentences may produce different projected states and
belong to different caches. Reuse occurs only across successive decoding steps
that preserve the same prefix.

During prompt prefill, the model computes queries, keys, and values for all prompt
positions in parallel. It retains the prompt keys and values and can discard the
queries after their current outputs are produced. During token-by-token decode,
only the new position's projections are calculated; its key and value are
appended, and its one query reads the growing cache.

## Concrete examples and derivations

```text
prefill:  prompt states -> Q_prompt, K_prompt, V_prompt
                           use Q once  keep K and V

step t:   new state -> q_t, k_t, v_t
                     q_t reads [K_cache; k_t], [V_cache; v_t]
                     discard q_t; append k_t and v_t
```

The causal boundary is implicit during ordinary one-token decoding because the
new query receives only past cached positions plus itself; no future positions
exist yet. The cache is maintained separately for every layer, because each
layer produces different projected keys and values.

For a fixed earlier position $j$ in a causal model, its layer input depends only
on the prefix through $j$. Appending a later token $x_t$ cannot change that past
state, because position $j$ is forbidden from attending to $t$. Therefore its
already context-specific projections remain unchanged:

$$
k_j^{(\ell)}=h_j^{(\ell-1)}W_K^{(\ell)},\qquad
v_j^{(\ell)}=h_j^{(\ell-1)}W_V^{(\ell)}.
$$

The same token in a different prefix can yield different $h_j^{(\ell-1)}$ and
therefore different keys and values. That fact does not undermine caching; it
explains why caches cannot normally be shared merely because two requests
contain the same token.

Without a KV cache, generating token $t$ would recompute keys and values for the
entire prefix even though causal attention guarantees that future tokens cannot
change those past representations at the same layer. Caching exchanges growing
memory use for substantially less repeated computation.

## Demonstrated understanding

The learner independently connected the Q/K/V role separation to the established
term “KV caching” and correctly noticed why query caching is absent from the
standard name.

The remaining checkpoint is to explain back that the reuse unit is an exact past
position within an unchanged prefix, not a token type across arbitrary contexts.

## Evidence and limitations

This is a derived architectural consequence, not yet a benchmarked performance
claim in this course. Ordinary full-sequence training usually computes attention
in parallel rather than using an inference-style KV cache. Specialized decoding,
attention variants, beam management, or systems implementations may retain or
rearrange additional state; “queries are never cached” would therefore be too
absolute. The standard decoder cache specifically preserves the reusable source
keys and values.

Some serving systems can reuse a cache across requests when the requests share
an exact token prefix under compatible model and execution settings. This is
prefix caching, not reuse based on token identity alone.

## Open edges

- Measure cached versus uncached decoding work and memory growth.
- Explain cache shapes across layers, heads, grouped-query attention, batch, and
  sequence dimensions.
- Connect grouped-query attention to sharing fewer key/value heads.
- Distinguish self-attention KV caching from encoder-side cross-attention caches.

## Reuse opportunities

- A Chapter 4 consequence box immediately after Q/K/V roles.
- A forward reference to the modern architecture chapter's KV-cache accounting.
- `CAND-ANIM-009`: show each transient query reading a persistent, growing pair
  of key/value tracks during autoregressive generation.
