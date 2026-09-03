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
- Is caching part of the Transformer definition or an additional runtime
  mechanism?
- When does a request's cache cease to be useful, and when is it released?

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

### Architecture computation versus runtime caching

Computing keys and values is intrinsic to ordinary self-attention. Retaining
them for a later decoding call is not: KV caching is an optional inference-time
optimization implemented by the model runtime or serving system. A conventional
full-sequence forward pass can calculate $Q$, $K$, $V$, use them, and release
them without ever exposing a persistent cache. Autoregressive inference engines
normally enable caching because recomputing the unchanged prefix would be
wasteful. Full-sequence training normally does not use an inference-style cache;
it processes all positions in parallel and retains whatever activations backward
requires.

### Request-local lifecycle

For a prompt that will generate a continuation, processing the prompt is the
prefill phase rather than the end of the useful sequence. The cache grows while
new tokens are decoded. When generation reaches an end token, length limit,
cancellation, or other terminal condition, no future query in that request needs
the stored keys and values, so the logical request cache can be released.

A memory allocator may keep the released device pages in a reusable pool rather
than returning them immediately to the operating system; this does not mean the
old request's semantic cache remains active. Serving systems may deliberately
retain selected exact-prefix caches for later requests and evict them under a
cache policy. If a conversation continues after its cache was discarded, the
system must reconstruct the state from the retained token history or another
saved representation.

Without a KV cache, generating token $t$ would recompute keys and values for the
entire prefix even though causal attention guarantees that future tokens cannot
change those past representations at the same layer. Caching exchanges growing
memory use for substantially less repeated computation.

## Demonstrated understanding

The learner independently connected the Q/K/V role separation to the established
term “KV caching” and correctly noticed why query caching is absent from the
standard name.

The learner then explained back that KV caching happens while extending the same
input prefix: calculating a later hidden state does not require recalculating
unchanged earlier states, and those states are not reused merely because another
input sentence contains the same tokens. This satisfies the central conceptual
checkpoint. Exact-prefix reuse across requests remains an explicitly named
systems-level exception rather than token-level reuse.

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

Whether a specific library enables caching by default, returns it to the caller,
retains it between conversation turns, or immediately frees it is an
implementation contract rather than a universal Transformer guarantee.

### Concrete implementations checked on 2026-09-03

- Hugging Face Transformers exposes cache objects to ordinary model code. Its
  documented default is a growing `DynamicCache`; it also provides static,
  offloaded, and quantized strategies, and permits disabling generation caching
  with `use_cache=False`: <https://huggingface.co/docs/transformers/kv_cache>.
- vLLM is a serving engine with block-oriented KV-cache management and optional
  hash-based automatic prefix caching across requests that share an exact
  prefix: <https://docs.vllm.ai/en/latest/design/prefix_caching/>.
- NVIDIA TensorRT-LLM manages KV state as runtime block pools and supports
  request reuse, offloading, eviction policies, variable attention windows,
  MQA, and GQA: <https://nvidia.github.io/TensorRT-LLM/features/kvcache.html>.
- `llama.cpp` is a local inference runtime whose command-line interface exposes
  KV offloading and independent key/value cache storage types:
  <https://github.com/ggml-org/llama.cpp/blob/master/tools/completion/README.md>.

These systems do not introduce the mathematical $K$ and $V$ operations; they
decide how the already-required projected states are stored and reused during
inference.

## Open edges

- Measure cached versus uncached decoding work and memory growth.
- Explain cache shapes across layers, heads, grouped-query attention, batch, and
  sequence dimensions.
- Connect grouped-query attention to sharing fewer key/value heads.
- Distinguish self-attention KV caching from encoder-side cross-attention caches.
- Inspect a concrete inference API's cache-enable and cache-return contract.
- Measure logical cache release separately from allocator-reserved GPU memory.

## Reuse opportunities

- A Chapter 4 consequence box immediately after Q/K/V roles.
- A forward reference to the modern architecture chapter's KV-cache accounting.
- `X-ATTN-KV-001`: a reader-facing article connecting attention roles, causal
  immutability, cache lifecycle, and concrete inference frameworks.
- `CAND-ANIM-009`: show each transient query reading a persistent, growing pair
  of key/value tracks during autoregressive generation.
