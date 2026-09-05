# Chapter 4 — Attention and the Causal Information Boundary

Chapter 3 began with a contextual hidden state and asked how it becomes a
next-token prediction. We now examine how a position obtains information from
other positions in the first place.

Consider the illustrative sequence:

> The animal crossed the river.

The representation at “crossed” can use the earlier words to help predict what
comes next. During causal language modeling, it cannot inspect “river” to make
that prediction. The representation at “river,” however, may use the entire
prefix through its own position. These are two different prediction problems at
two different locations in the same sequence.

Attention provides a trainable way for each position to retrieve information
from allowed sources. Its design connects four questions: what should a
position look for, what should each source offer, how should they be combined,
and which sources are legal? Answering them precisely also explains why
inference systems can reuse keys and values while generating a continuation.

## 4.1 Learning outcomes and chapter route

After studying this chapter, you should be able to:

- explain queries, keys, and values as distinct learned projections;
- trace every shape in single-head attention, including the source/query axes;
- distinguish learned compatibility from semantic similarity;
- derive the square-root scaling factor under explicit statistical assumptions;
- implement causal attention and diagnose incorrect masking;
- distinguish attention weights, mixed values, contextual states, and logits;
- derive backward credit through routing and content paths;
- explain the limits of interpreting attention maps;
- derive causal prefix invariance and connect it to per-layer KV caching;
- distinguish cache correctness, memory accounting, and measured speed;
- reproduce the small experiments and state their evidence boundaries.

The route is retrieval (§4.2), compatibility (§4.3), scaling (§4.4), masking
(§4.5), value mixtures (§4.6), implementation (§4.7), gradients (§4.8),
interpretation (§4.9), caching (§§4.10–4.11), and controlled evidence (§4.12).
The chapter closes with exercises and the transition to a complete decoder.

Prerequisites are matrix multiplication, the chain rule, softmax, and the
next-token objective from Chapters 2–3. We use row vectors and zero-based
positions. The derivations concern one sequence and one attention head, with
no dropout unless explicitly stated. Full multi-head blocks, residual streams,
normalization, positional encodings, and feed-forward layers receive their
complete treatment in Chapter 5.

## 4.2 One position has three roles

Let X contain the incoming representations at T positions, each with D features:

$$
X\in\mathbb{R}^{T\times D}.
$$

These are continuous representations, not token IDs. At the first attention
layer they may originate from embeddings and position-dependent processing.
At deeper layers they already incorporate earlier layer computations.

Each position has three learned projections:

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.
$$

The query q_i represents the receiving position's matching criteria. A source's
key k_j represents what that receiver can match against. Its value v_j is the
message returned when that source receives weight.

The retrieval analogy helps explain the roles, but a query need not encode a
literal question and a key is not a dictionary address. They are learned
coordinates. The downstream language-model loss adjusts their projections to
make useful retrieval more likely.

Why separate these roles? What makes a source worth retrieving need not be the
same information that should be retrieved. A receiver may match a source using
one feature while obtaining other features from its value. Distinct query and
key projections also allow directed relationships: position i's interest in j
need not equal j's interest in i.

| Object | Shape | Meaning |
|---|---|---|
| X | [T,D] | Incoming feature rows |
| W_Q, W_K | [D,d_k] | Matching projections |
| W_V | [D,d_v] | Message projection |
| Q, K | [T,d_k] | One query/key per position |
| V | [T,d_v] | One value per position |
| QKᵀ | [T,T] | Receiver-by-source scores |
| A | [T,T] | Receiver-by-source weights |
| O=AV | [T,d_v] | Retrieved output features |

Q and K require compatible feature widths for the dot product. The value width
d_v may differ. Adding a batch axis gives [B,T,d_k] and [B,T,d_v]; multiple
heads add another axis in Chapter 5.

The symbol V here denotes the value matrix. It is not the vocabulary size from
Chapter 3; when both are needed, write N_vocab for vocabulary size.

## 4.3 Dot products are learned compatibility scores

The raw score between receiver i and source j is

$$
s_{ij}=q_i k_j^\top
      =x_iW_QW_K^\top x_j^\top.
$$

This is a learned bilinear relation over incoming representations. The
multiplication contains no linguistic knowledge by itself; the trained
projections determine what counts as compatibility.

A large score can support a dependency between very different words. For
example, a verb might retrieve information about its subject. Conversely,
semantically similar words need not be useful sources for the current query.

The geometric form is

$$
q_i k_j^\top=\|q_i\|\,\|k_j\|\cos\theta.
$$

Both direction and magnitude contribute. Dot-product attention is therefore not
automatically cosine similarity. Nor is its score symmetric: W_QW_Kᵀ need
not be symmetric.

Collecting all queries and keys gives S=QKᵀ. Row i contains a single receiver's
scores over source positions; column j shows how different receivers score one
source. Confusing these axes can produce a matrix of the expected shape that
implements the wrong information flow.

## 4.4 Why divide by the square root of the head width?

Suppose the query and key coordinates are independent, have mean zero and
variance one, and their coordinate products are independent across dimensions.
For one product,

$$
\mathbb{E}[q_mk_m]=0,\qquad
\operatorname{Var}(q_mk_m)
=\mathbb{E}[q_m^2]\mathbb{E}[k_m^2]=1.
$$

The score sums d_k such products:

$$
s=\sum_{m=1}^{d_k}q_mk_m,\qquad
\operatorname{Var}(s)=d_k,\qquad
\operatorname{Std}(s)=\sqrt{d_k}.
$$

Positive and negative contributions partly cancel; the standard deviation grows
as the square root, not linearly with width. Thus

$$
\operatorname{Var}\left(\frac{s}{\sqrt{d_k}}\right)=1.
$$

Without this correction, wider random heads tend to create wider score gaps.
Softmax exponentiates those gaps, producing concentrated distributions even
without stronger learned evidence. Scaling by sqrt(d_k) controls this initial
spread. This is the scaled dot-product mechanism introduced in
[*Attention Is All You Need*](https://arxiv.org/abs/1706.03762).

For a softmax row a, the Jacobian is

$$
J=\operatorname{diag}(a)-aa^\top,\qquad
J_{ij}=a_i(\mathbf{1}[i=j]-a_j).
$$

As a approaches a one-hot vector, these entries approach zero. Its trace,
1−Σ_i a_i², offers a simple measure of local softmax sensitivity. It is not
itself a downstream loss gradient.

This does not contradict Chapter 3's strong p−q gradient for a confidently wrong
vocabulary prediction. That result combines softmax with a particular
cross-entropy objective, whose derivative cancels part of the softmax derivative.
Attention normally receives its learning signal through a weighted value sum;
the same cancellation cannot be assumed.

The IID assumptions explain the scaling rule, not every trained attention
distribution. Learned magnitudes, correlations, normalization, and positional
transformations affect actual score statistics. Dividing by sqrt(d_k) does not
guarantee unit variance or prevent every saturated row.

## 4.5 Causality belongs inside normalization

At position i, allowed source positions satisfy j≤i. The diagonal is included:
the current input token is known, and its output predicts the following token.

Define the additive mask

$$
M_{ij}=
\begin{cases}
0,&j\le i,\\
-\infty,&j>i.
\end{cases}
$$

Scaled, masked scores and weights are

$$
R=\frac{QK^\top}{\sqrt{d_k}}+M,\qquad
A=\operatorname{softmax}_{\mathrm{row}}(R).
$$

For four positions the allowed pattern is

$$
\begin{bmatrix}
1&0&0&0\\
1&1&0&0\\
1&1&1&0\\
1&1&1&1
\end{bmatrix}.
$$

The mask defines permissible information flow; A determines how the model
distributes attention within that boundary.

### Two masking errors

Take one row with scores [0,0,10], where the third source is forbidden.

Correct pre-softmax masking produces

$$
\operatorname{softmax}([0,0,-\infty])=[0.5,0.5,0].
$$

If softmax is applied first, the forbidden score participates in the denominator.
Zeroing its probability afterward produces approximately

$$
[0.0000454,\;0.0000454,\;0].
$$

Although its direct value contribution is removed, the future has changed the
allowed weights. The row no longer sums to one.

Replacing the forbidden score with zero before softmax is also incorrect:

$$
\operatorname{softmax}([0,0,0])=[1/3,1/3,1/3].
$$

Zero is an ordinary score with exponential one. In an additive mask, adding zero
to an allowed score means leave it unchanged; it does not mean remove a source.

There is a useful qualification. Post-softmax zeroing followed by renormalizing
the remaining weights is equivalent in exact arithmetic:

$$
\frac{\exp(s_j)/Z}{\sum_{m\in\mathcal A}\exp(s_m)/Z}
=\frac{\exp(s_j)}{\sum_{m\in\mathcal A}\exp(s_m)},
\quad j\in\mathcal A.
$$

But a large forbidden score can cause all allowed probabilities to underflow
before renormalization, leaving a zero denominator. Masking before a stable
softmax avoids that failure.

### The stronger check: perturb the future

Assume inputs through t are identical, parameters and position handling are
fixed, and stochastic operations are disabled. Correct causal attention obeys

$$
X^{(A)}_{0:t}=X^{(B)}_{0:t}
\Longrightarrow
O^{(A)}_{0:t}=O^{(B)}_{0:t}.
$$

Changing “river” to “road” cannot change the earlier output at “crossed.” The
fourth output may change because its own input changed. The earlier state helps
predict that fourth input; it cannot depend on the answer it is supposed to
predict.

This property assumes earlier layers have also respected causality. A mask
cannot remove future information already hidden in X by a faulty upstream
operation.

### Causal masks, padding masks, and loss masks

A causal mask restricts which time positions a query may read. A padding mask
removes invalid source positions. Their allowed sets are intersected. A loss
mask selects which predictions contribute to the objective; it does not by
itself remove any source from attention.

Every evaluated softmax row must have an allowed source. A row of all negative
infinities is undefined in a naive implementation. Padding-only query rows need
an explicit policy: skip them, or supply a valid numerical computation and
discard their outputs. Do not treat NaNs as masked information.

## 4.6 Attention constructs a value mixture

The output for receiver i is

$$
o_i=\sum_{j\le i}a_{ij}v_j,\qquad O=AV.
$$

With nonnegative weights summing to one, a single head's output lies in the
convex hull of its allowed value vectors. For weights [0.2,0.7,0.1], it is

$$
o_i=0.2v_1+0.7v_2+0.1v_3.
$$

It is a new continuous feature vector, not a token ID. The same source weights
are used across all value coordinates within the head. A sharp distribution
can approximate copying one value, but the operation remains a weighted sum.

The convex-mixture statement applies before attention dropout or subsequent
transformations. Output projection, residual addition, and later nonlinear
layers are not constrained to remain within that same convex hull.

To reconnect with Chapter 3:

$$
O \longrightarrow \text{later decoder computation}
\longrightarrow h_i
\longrightarrow z_i=h_iW_{\mathrm{out}}+b
\longrightarrow p_i=\operatorname{softmax}(z_i).
$$

A's columns index source positions. The final vocabulary logits' columns index
candidate output tokens. These are different competitions with different axes.

## 4.7 A transparent implementation

The implementation below handles one unpadded sequence and one head. Finite
inputs and at least one allowed source per row are assumed.

~~~python
import math
import torch

q, k, v = x @ wq, x @ wk, x @ wv
scores = (q @ k.T) / math.sqrt(q.shape[-1])
allowed = torch.ones_like(scores, dtype=torch.bool).tril()
masked = scores.masked_fill(~allowed, -torch.inf)
weights = torch.softmax(masked, dim=-1)
output = weights @ v
~~~

Use replacement or an additive mask constructed directly. Multiplying a binary
mask by negative infinity creates 0×(−∞), which is NaN.

The companion
[forward implementation](../../src/dongxi_llms/causal_attention_lab.py)
returns intermediate tensors so the notebook can expose each transformation.
A correct implementation should satisfy:

1. nonnegative weights and row sums one before dropout;
2. zero weights at forbidden positions;
3. agreement between AV and an explicit allowed-source sum;
4. unchanged prefix outputs when only future inputs change;
5. agreement with an independently implemented reference at a declared tolerance.

The first notebook compares with PyTorch SDPA using dropout_p=0. For that API,
boolean true in an attention mask means allowed; other APIs can use different
conventions. Its causal handling also depends on query/key alignment, which
matters for cached decoding. Consult the
[SDPA reference](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
rather than transferring a boolean mask convention by name alone.

For dense full-sequence attention, the two main products QKᵀ and AV cost on
the order of T²d_k and T²d_v. Explicitly storing A costs T² elements per head.
The Q/K/V projections add costs proportional to TD times the projected widths.
Fused implementations may avoid materializing the whole score matrix; the small
implementation intentionally retains it for inspection.

## 4.8 How one loss teaches routing and content

Let a downstream next-token loss L send derivative G_O=∂L/∂O into attention.
Because O=AV,

$$
dO=(dA)V+A(dV).
$$

The chain rule separates the two paths:

$$
G_A=G_OV^\top,\qquad G_V=A^\top G_O.
$$

G_V changes the messages that sources transmit. G_A says how the downstream
loss would respond to changing the routing weights, given those messages.

For one softmax row a with incoming derivative g,

$$
\frac{\partial L}{\partial r_j}
=\sum_m g_m a_m(\mathbf{1}[m=j]-a_j)
=a_j\left(g_j-\sum_m a_mg_m\right).
$$

Stacking rows gives

$$
G_R=A\odot\left(G_A-
\operatorname{rowsum}(A\odot G_A)\right),
$$

where the [T,1] row sum broadcasts across sources. Forbidden entries have zero
weight and zero score gradient. M is fixed, so it has no learned update.

Through scaled dot products,

$$
G_Q=\frac{G_RK}{\sqrt{d_k}},\qquad
G_K=\frac{G_R^\top Q}{\sqrt{d_k}}.
$$

Through the learned projections,

$$
G_{W_Q}=X^\top G_Q,\quad
G_{W_K}=X^\top G_K,\quad
G_{W_V}=X^\top G_V,
$$

$$
G_X=G_QW_Q^\top+G_KW_K^\top+G_VW_V^\top.
$$

All derivatives match the shape of their corresponding tensor. No separate
attention-map target is needed. The next-token objective can change where the
receiver reads, what each source advertises, and what it transmits.

For a simple SGD illustration, each matrix updates by
W←W−ηG_W. A derivative's sign and the optimizer's update direction are opposite.
In a shared network, an update affects many examples and scores; these local
derivatives do not imply that every useful edge increases after every step.

### Why prompt states can receive gradients

If only an answer position has direct loss, earlier prompt rows can still
affect that loss through their keys and values. They therefore receive
gradients. A forbidden future row cannot affect that isolated earlier loss,
so its input gradient is zero. Shared parameter updates are not a channel for
future information in the current forward pass.

### Detach as a controlled intervention

Replacing A with A.detach() preserves forward values but disconnects the
routing gradient. In the isolated head, W_Q and W_K lose their paths while
W_V still receives credit. Detaching V instead removes W_V's path while Q/K
can learn which fixed messages to retrieve.

These statements assume independently parameterized projections and no extra
loss paths. The second notebook verifies them with one fixed output head,
one cross-entropy target, and finite differences of the projection weights.

## 4.9 What an attention map does and does not explain

A shows routing coefficients inside a computation. To determine even the head's
output, one also needs V. Different weight patterns may produce the same result.

Let v_1=[2,0], v_2=[0,2], and v_3=[1,1]. Then both

$$
[0.4,0.4,0.2]V=[1,1],\qquad
[0.1,0.1,0.8]V=[1,1].
$$

More generally, if δV=0 and both a and a+δ are valid distributions,

$$
(a+\delta)V=aV.
$$

A complete model adds other heads, residual paths, projections, and later
layers. A high attention weight alone therefore does not establish a source's
causal importance to the final prediction or a human-readable reasoning step.

Attention maps remain useful for diagnosing masking errors and proposing
hypotheses about retrieval. Stronger claims require interventions and
downstream measurements, with their own assumptions. Gradient magnitude alone
also does not settle causal importance.

## 4.10 Causal invariance explains the KV cache

Generation extends a prefix one position at a time. At new position t,

$$
q_t=x_tW_Q,\quad k_t=x_tW_K,\quad v_t=x_tW_V,
$$

$$
K_{\le t}=[K_{<t};k_t],\qquad V_{\le t}=[V_{<t};v_t],
$$

$$
o_t=\operatorname{softmax}\left(
\frac{q_tK_{\le t}^{\top}}{\sqrt{d_k}}\right)V_{\le t}.
$$

The new query reads stored source keys and values. The old queries have already
completed their retrievals and are not needed by this expression.

Why are the stored projections still valid? Earlier states in a causal layer
cannot change when a future token is appended. Starting from unchanged input
representations, this property carries through successive causal layers and
positionwise transformations. Each layer can therefore retain its own earlier
keys and values. This is the mathematical basis of the runtime optimization
described in the [Hugging Face cache explanation](https://huggingface.co/docs/transformers/cache_explanation).

### Same token, different context

A cache is tied to position, prefix, layer, model parameters, and compatible
execution settings. It is not a vocabulary lookup table.

There is an important first-layer qualification. If two occurrences have
identical incoming embeddings and identical position treatment, their initial
linear Q/K/V projections can be identical despite different preceding text.
After contextual attention, deeper-layer incoming states can differ, so deeper
keys and values can differ. Context dependence must be traced through the
actual graph rather than assumed at every projection.

### Prefill and decoding

Prefill computes an entire prompt with causal attention and retains each
layer's K/V. Decoding then projects only new positions, appends their K/V, and
computes their outputs using the prefix cache.

For one new final-position query, all supplied cache keys are past or self.
No additional triangular mask is needed if no padding or other restrictions
exist. An unshifted [1,N] lower triangle would keep only source zero and be wrong.
For multi-token chunks, the mask must include the cached-prefix offset.

KV retention is an optional inference optimization. A full-sequence training
pass normally computes positions in parallel and retains activations for
backward, which is a different purpose. The cache does not add learned knowledge.

## 4.11 Cache cost, lifecycle, and inference systems

For equal-width keys and values, a simple logical-storage model is

$$
\text{KV bytes}=2BLTH_{\mathrm{KV}}d_hb,
$$

where B is batch size, L is layer count, T is retained positions,
H_KV is KV heads per layer, d_h is head width, and b is bytes per scalar.
Unequal key/value widths require summing their sizes separately.

This formula counts payload, not allocation overhead, padding, fragmentation,
temporary buffers, or other model memory. GQA reduces the number of KV heads
relative to query heads; detailed design comparisons belong in Chapter 5.

Caching avoids projecting and processing the unchanged prefix repeatedly. It
does not make a new query's retrieval constant-time: its dot products and value
mixture still grow with the number of retained keys. Small replay experiments
can establish equality and count saved operations, but real latency depends on
hardware, batching, kernels, and memory traffic.

A completed or cancelled request no longer needs its request-local cache.
A runtime can release it while an allocator retains freed memory for reuse.
Some serving systems retain compatible exact-prefix state across requests;
vLLM documents such
[automatic prefix caching](https://docs.vllm.ai/en/latest/design/prefix_caching/).
This is reuse based on a compatible prefix, not merely a matching token string.

Hugging Face model code, vLLM, TensorRT-LLM, and llama.cpp are useful examples
of the systems layer to inspect later. Their cache policies, storage formats,
offloading choices, and defaults are implementation contracts. Verify the
version being used before treating a particular setting as universal.

## 4.12 Three companion notebooks and their evidence

These are executable sections of the chapter. Each includes prediction prompts,
optional learner code, adjacent runnable reference solutions, and explanations.
They can also be read as worked lessons before returning to hands-on study.

| Session | Main question | Evidence |
|---|---|---|
| [1. Forward attention](../../notebooks/day-04/01_causal_attention_forward.ipynb) | Can a forbidden future affect normalization? | Shape checks, SDPA agreement, prefix intervention, broken masks |
| [2. Gradients and failures](../../notebooks/day-04/02_attention_gradients_and_failures.ipynb) | How does loss train routing and content? | Autograd, finite differences, detach interventions, scaling simulation |
| [3. KV-cache equivalence](../../notebooks/day-04/03_kv_cache_equivalence.ipynb) | What can be reused without changing outputs? | Full-prefix versus cached replay, stale-prefix failure, logical memory accounting |

The [forward report](../../experiments/reports/2026-09-05-causal-attention-forward.md)
records exact agreement with SDPA on the fixed teaching fixture, unchanged
earlier outputs under the correct mask, and changed earlier outputs under both
broken masking variants.

The [gradient and cache report](../../experiments/reports/2026-09-05-attention-gradients-cache.md)
records the following CPU float64 observations:

- maximum manual/autograd derivative error: 2.22×10⁻¹⁶;
- maximum central-difference projection-gradient error: 1.83×10⁻¹⁰;
- zero forbidden-score and future-input gradients for the isolated target;
- nonzero gradients into earlier input rows with no direct target;
- agreement of cached and full-prefix logits within 10⁻¹²;
- a stale-prefix cache causing a maximum logit difference of about 1.80691.

In the IID scaling simulation, head widths 8, 64, and 512 produced:

| d_k | Raw score std. | Scaled score std. | Raw entropy (nats) | Scaled entropy (nats) |
|---:|---:|---:|---:|---:|
| 8 | 2.84406 | 1.00553 | 1.24229 | 2.35528 |
| 64 | 7.98888 | 0.99861 | 0.42197 | 2.36165 |
| 512 | 22.28492 | 0.98486 | 0.11479 | 2.35443 |

These are paired finite-sample results under the declared random-coordinate
model, not measurements of a pretrained model.

The cache fixture uses two single-head residual attention layers with width
four and an arbitrary five-class head. It omits a full decoder's normalization,
MLP, and positional encoding. Replaying six fixed input rows with a two-row
prefill requires 12 cached versus 40 uncached layer-position projections for
each of K and V. Final logical cache storage is 768 bytes. This is neither a
measured speedup nor a model-quality result.

## 4.13 Exercises

1. **Shapes and meaning.** For T=5, D=8, d_k=4, and d_v=3, give every shape
   from X through O. Which axis indexes vocabulary candidates?
2. **Directed retrieval.** Explain why q_i k_jᵀ need not equal q_j k_iᵀ,
   even though both projections come from X.
3. **Scale assumptions.** Derive Var(q·k). What changes if coordinate variances
   are σ_q² and σ_k²? Why does scaling not guarantee nonsaturated trained scores?
4. **Mask-order failure.** Explain the three outcomes for [0,0,10] when the last
   source is forbidden. When can post-softmax renormalization repair masking?
5. **Future intervention.** Change only “river” to “road.” Explain the outputs
   at “crossed” and at the changed position. State the assumptions.
6. **Backward credit.** Derive G_V and G_A from O=AV, then trace the path to X.
   Can an earlier prompt position receive gradients with zero direct loss?
7. **Detach and saturation.** Explain the two detach interventions. Reconcile
   attention saturation with the p−q gradient for vocabulary cross-entropy.
8. **Interpretation.** Construct two different attention distributions with the
   same value mixture. What can an attention map establish by itself?
9. **Cache correctness.** Explain why old K/V remain valid when a prefix grows,
   why old Q is unnecessary for ordinary decoding, and why the same token can
   have different deeper-layer keys.
10. **Cache memory and masking.** Derive the 768-byte fixture size. Explain why
    an unshifted [1,N] causal triangle is incorrect for its final-position query.
11. **Experimental claims.** What do the derivative checks, cache replay, and
    operation counts establish? What would require a separate experiment?
12. **Implementation defense.** Starting from the first notebook's five-step
    attention implementation, name a check that catches each of: wrong softmax
    axis, future leakage, zero-score replacement, and an all-masked row.

[Worked solutions](../solutions/04-attention-and-the-causal-information-boundary.md)
provide explanations and links to executable checks.

## 4.14 From one head to a decoder

We can now follow attention in both directions: incoming states produce
matching requests, source keys, and messages; a causal distribution mixes those
messages; downstream loss sends credit back through content and routing.
Causal prefix invariance then permits a runtime to retain the sources that
future queries will need.

A single head is only one component of a language model. Chapter 5 assembles
multiple heads, output projection, residual streams, normalization, position
information, and feed-forward transformations into a decoder. It then studies
modern variants and accounts for parameters, compute, and memory. Recurrent
depth and looped Transformers remain a later architecture extension after this
baseline is established.
