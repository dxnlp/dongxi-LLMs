# Causal Shifting and Loss Masks

- Day: 03
- Date opened: 2026-09-01
- Status: introduced
- Book destination: Chapter 3 sections on causal label alignment and masked loss
- Related evidence: planned manual tensor trace and PyTorch verification
- Related production tasks: candidate `CAND-ANIM-001`; `ANIM-CE-001`

## Questions that drive the discussion

- Which hidden-state position predicts which target token?
- Why can a causal language model consume `input_ids` and labels with the same
  apparent shape while still training on the next token?
- What happens if logits and labels are compared without a one-position shift?
- Where do BOS, EOS, padding, and ignored labels enter the alignment?
- Which positions contribute to the final mean loss?

## Mechanism introduced

For a token sequence:

\[
x_0,x_1,\ldots,x_{T-1},
\]

the contextual state at position $t$ can use only $x_{\le t}$. Its target is
the next token:

\[
y_t=x_{t+1}, \qquad 0\le t<T-1.
\]

In explicit tensor form:

```python
shift_logits = logits[:, :-1, :]   # [B, T-1, V]
shift_labels = input_ids[:, 1:]    # [B, T-1]
```

The shift is an alignment of predictions with targets; it is not permission for
the hidden state to inspect the future. The causal attention boundary must still
ensure that `logits[:, t, :]` depends only on tokens through position $t$.

Comparing `logits[:, t, :]` directly with `input_ids[:, t]` trains the model to
recover a token it has already received at the same position. That objective can
look numerically easy while failing to train next-token prediction.

EOS is normally a meaningful shifted target after the final content token. BOS,
when used, can provide a position from which to predict the first ordinary token.
Padding or deliberately unsupervised positions must be excluded after alignment,
and the loss denominator must count only valid targets.

Some causal language-model APIs accept `labels=input_ids` and perform the shift
inside the model loss. The exact implementation must be inspected rather than
assuming every framework has the same contract.

## Logits, labels, and their dimensions

Let:

- $B$ be the number of sequences in a batch;
- $T$ be the number of token positions in each sequence;
- $D$ be the hidden or embedding dimension;
- $V$ be the model's output-vocabulary dimension (which can include rows not
  exposed as ordinary tokenizer entries).

The main tensor path is:

| Object | Shape | Meaning at one position |
|---|---|---|
| `input_ids` | `[B,T]` | one integer vocabulary ID |
| embeddings | `[B,T,D]` | one $D$-dimensional token vector |
| contextual hidden states | `[B,T,D]` | one $D$-dimensional context-dependent vector |
| logits | `[B,T,V]` | one raw score for every vocabulary candidate |
| labels | `[B,T]` before alignment | one integer ID naming the observed target |

The transformer does not apply softmax directly to a hidden state. Its output
head first maps the $D$-dimensional representation into $V$ token scores:

\[
z_{b,t}=W_{\mathrm{out}}h_{b,t}+b,
\qquad
W_{\mathrm{out}}\in\mathbb{R}^{V\times D},
\qquad
z_{b,t}\in\mathbb{R}^{V}.
\]

For candidate token $i$, the individual score is

\[
z_{b,t,i}=W_{\mathrm{out},i}\cdot h_{b,t}+b_i.
\]

Softmax is then applied to $z_{b,t}$, producing one conditional next-token
distribution over the model vocabulary. The output head is essential because
the $D$ hidden coordinates are internal learned features, not vocabulary-token
coordinates. With weight tying, `W_out` reuses the input embedding matrix; the
mapping still occurs.

### One input position produces scores for all possible outputs

For a single position, one token ID selects one input embedding row. After the
transformer has contextualized that position, its hidden state has shape `[D]`.
The output head compares that one state with all $V$ output rows at once, giving
one logit vector of shape `[V]`:

```text
one contextual state h_t
        |
        +-- score for token 0
        +-- score for token 1
        +-- score for token 2
        |   ...
        +-- score for token V-1
```

For a toy vocabulary `[cat, dog, slept, EOS]`, the state at one position might
emit `[2.0, 1.0, -1.0, 0.3]`. These are four competing scores, not four emitted
tokens. Softmax turns the complete row into four probabilities; decoding later
selects or samples one candidate.

A real input normally contains $T$ token IDs rather than one. The model produces
one such $V$-wide score row for every input position:

```text
input IDs:  [B, T]
logits:     [B, T, V]
```

During generation, only the last position's `[V]` row is normally used to choose
the next token. During teacher-forced training, shifted rows across the sequence
can all be supervised in parallel.

Concrete model interfaces make the scale visible:

| Model snapshot | Tokenizer-exposed IDs | Model output rows / logits per position | Rows beyond tokenizer entries |
|---|---:|---:|---:|
| pinned Qwen3-0.6B | 151,669 | 151,936 | 267 |
| `openai/gpt-oss-20b` | 200,019 | 201,088 | 1,069 |
| `openai/gpt-oss-120b` | 200,019 | 201,088 | 1,069 |

The Qwen values were already measured in the pinned local interface report. The
gpt-oss values were checked on 2026-09-02 from each model's configuration and the
shared tokenizer artifact: the tokenizer has 199,998 base BPE entries plus 21
added tokens, while both model configurations declare `vocab_size=201088`.
Therefore a one-position forward pass has logit shape `[1,1,151936]` for this
Qwen model and `[1,1,201088]` for either gpt-oss model. The extra model rows are an
observed interface fact; these files alone do not establish the designers'
rationale for them.

### The computational cost of a large output vocabulary

For dense output projection,

\[
Z_{[B,T,V]}=H_{[B,T,D]}W_{\mathrm{out},[V,D]}^\top,
\]

so its arithmetic scales as $\Theta(BTDV)$ and the output matrix contains $VD$
weights, plus an optional bias. Softmax adds work proportional to $BTV$, but the
projection performs a length-$D$ dot product for every vocabulary candidate and
is usually the larger arithmetic term.

For the pinned Qwen3-0.6B interface, $D=1024$ and $V=151{,}936$. Its tied
embedding/output matrix therefore contains 155,582,464 weights, and one position
requires roughly 155.6 million multiply-accumulate operations for the vocabulary
projection alone. For gpt-oss, $D=2880$ and $V=201{,}088$, so its configured
output matrix contains 579,133,440 weights; its configuration reports untied
input and output embeddings. These operation counts describe the dense
mathematical projection and omit bias, memory traffic, kernel implementation,
and hardware utilization, so they are not measured latency.

GPUs make this feasible by evaluating the matrix multiplication in parallel.
Training applies it across many positions at once; cached autoregressive
generation normally applies it only to the newly produced final position on each
step. Full `[B,T,V]` logits can also consume substantial memory, motivating fused
or chunked linear-cross-entropy implementations that avoid retaining unnecessary
intermediate probability tensors.

A larger vocabulary is not pure overhead. Better compression can reduce sequence
length $T$, decreasing the number of transformer positions and sometimes the
quadratic attention work, while increasing the $V$-dependent embedding and output
cost. Whether that trade is favorable depends on the tokenizer's actual
compression on the target data, model width, sequence geometry, and hardware.

#### Why fewer vocabulary entries tend to create more positions

In byte-level BPE, the initial inventory can represent every input using byte
tokens. Each learned merge spends one additional vocabulary entry to replace a
frequent adjacent pair with a single reusable token. A tokenizer with a smaller
merge budget leaves more strings decomposed into bytes or short subwords; a larger
budget can store longer recurring fragments.

For example, the same conceptual text might be segmented as:

```text
smaller learned vocabulary:  data | base | govern | ance     T = 4
larger learned vocabulary:   database | governance           T = 2
```

For Chinese text, a byte-fallback tokenizer with few learned Chinese merges might
represent each character through several byte tokens, while a tokenizer with
relevant learned entries might use `数据库 | 治理`. The improvement comes from
which patterns received vocabulary capacity, not from the number $V$ alone. A
smaller language-specific vocabulary can therefore compress its target language
better than a larger vocabulary whose capacity was allocated elsewhere.

Every resulting token occupies a transformer position. At every layer, each
position receives attention and feed-forward projections, residual updates, and
normalization. A simplified full-sequence cost separates three important terms:

\[
\text{cost}
\approx
c_1LTD^2
+
c_2LT^2D
+
c_3TDV,
\]

where $L$ is layer count. The first term represents per-position projections and
feed-forward work, the second represents full self-attention interactions, and
the third represents the dense vocabulary projection. Constants and fused-kernel
behavior are omitted; this equation explains scaling rather than predicting
wall-clock time.

If tokenization doubles $T$, per-position transformer work roughly doubles and
the number of full-attention position pairs grows approximately fourfold. Under
causal attention the exact count is triangular, $T(T+1)/2$, rather than $T^2$,
but the quadratic scaling remains. Flash-style attention can avoid storing the
complete score matrix, yet it does not make all full-attention arithmetic linear.

During cached autoregressive generation, the new token attends to all cached past
positions. Longer tokenization increases KV-cache storage, the attention span of
each later step, and the number of sequential generation steps needed to express
the same text. These effects can matter even when only the newest position passes
through the output head.

The opposite pressure comes from $V$. Consider a deliberately simplified
comparison for one string:

| Design | $T$ | $V$ | Causal attention pairs $T(T+1)/2$ | Vocabulary comparisons $TV$ |
|---|---:|---:|---:|---:|
| smaller vocabulary | 12 | 10,000 | 78 | 120,000 |
| larger vocabulary | 4 | 100,000 | 10 | 400,000 |

The larger vocabulary sharply reduces positions and attention pairs but performs
more total output comparisons for this example. Real transformer-body and output
costs also depend on $D$, $L$, batching, kernels, and whether logits are required
at every position. The table demonstrates a trade-off, not an optimal vocabulary.

The statistical trade-off is equally important. Smaller pieces occur more often
and share evidence across many words, but require the transformer to compose more
steps. Longer vocabulary entries shorten sequences but may be rare, leaving their
embeddings and classifier rows with less training evidence. A fixed token context
window also covers different amounts of human text under different tokenizers.
Vocabulary design therefore balances compression, parameter and output cost,
training frequency, multilingual allocation, context coverage, and hardware—not
only tokenizer file size.

### A token position is not a semantic atom

The learner proposed a useful tokenizer–model distinction using Chinese:
humans recognize relationships among `数`, `据`, `库`, `数据`, and `数据库`, while
a tokenizer may assign each available form a separate ID. The refined statement
is that a token is an atomic unit of representation, computation, and prediction
for this model interface, not necessarily an atomic unit of meaning or
understanding.

A BPE tokenizer may contain entries resembling:

```text
ID a -> 数
ID b -> 据
ID c -> 库
ID d -> 数据
ID e -> 数据库
```

The IDs are arbitrary addresses. Nothing about their numerical values requires
`d` to be related to `a` and `b`, and the model has no built-in constraint such
as

\[
E[e]=E[a]+E[b]+E[c]
\quad\text{or}\quad
E[d]=E[a]+E[b].
\]

BPE training can still record a *procedural* relationship: a longer token may
have been created by repeatedly merging adjacent shorter symbols. At runtime,
however, if the trained encoding selects `数据库` as one token, the language
model ordinarily receives its final ID, not the merge tree and not the three
character IDs simultaneously. The tokenizer's shared spelling or merge history
therefore does not itself provide semantic composition to the transformer.

Next-token training can connect these symbols later through shared usage:

- their tokens occur in similar surrounding contexts;
- shorter forms occur within related sequences and documents;
- predictions made from one form overlap with predictions made from another;
- shared transformer parameters propagate statistical structure;
- tied input/output weights, when present, add another gradient path.

Those learned relationships need not appear as simple cosine similarity among
the static input embedding rows. They can be distributed across attention,
feed-forward layers, and context-dependent hidden states. Vector similarity is
one diagnostic probe, not a complete definition of linguistic knowledge.

The human comparison must also remain qualified. Human readers learn characters,
morphemes, words, and recurring multi-character constructions at several levels;
they do not always construct meaning strictly bottom-up from isolated characters.
The central contrast is therefore not "humans compose while tokenizers do the
opposite." It is that human units are organized through meaning and experience,
whereas tokenizer units are selected primarily by an encoding/compression
procedure; semantic relationships are learned mainly by the model afterward.

### Token IDs are categorical addresses, not numerical quantities

The learner then sharpened the question: because the transformer receives IDs
such as `1, 2, 3, 4`, does it directly learn relationships among numbers and only
indirectly learn language? The implementation-level intuition is useful, but
"numbers" is misleading. An ID is used for indexing, not arithmetic. ID `4` is
not inherently closer to ID `3` than to ID `9000`, and the model is not normally
given the scalar value `4` as a continuous feature.

The lookup operation is:

\[
e_t=E[i_t],
\]

so the ID selects an entire learned row. A consistent vocabulary permutation
makes the arbitrariness precise. Let $\pi$ rename every token ID, and define:

\[
E'[\pi(i)]=E[i],
\qquad
W'_{\mathrm{out}}[\pi(i)]=W_{\mathrm{out}}[i].
\]

If the encoded dataset and decoded outputs are renamed by the same $\pi$, the
model implements the same text-level behavior. This invariance shows that the
integer values are labels; the learned content resides in parameter rows,
sequence structure, and transformations of those rows.

The most precise hierarchy is:

1. the tokenizer maps text fragments to categorical symbols represented by IDs;
2. the embedding table maps those symbols to learned vectors;
3. the transformer learns functions over ordered sequences of those vectors;
4. the output head scores categorical next-token alternatives;
5. the tokenizer decoder maps the selected symbols back to text.

A text-only language model is directly optimized to model token-sequence
probabilities. Because a largely reversible tokenizer preserves the linguistic
sequence, this induces a model of text and justifies saying that it learns
linguistic structure. Syntax, semantic associations, discourse patterns, and
some world regularities can be statistically recoverable because they constrain
human-written token sequences. Still, its access to the world is mediated by its
training data; next-token prediction alone does not establish human-like
understanding or direct grounding.

The roles of hidden states and logits must also stay separate. A contextual
hidden state of width $D$ is the learned representation used to compute the next
prediction; it can encode many distributed features of the prefix. The $V$
logits are a readout—one compatibility score per output symbol—not $V$
independent containers of knowledge. Knowledge is distributed across embeddings,
attention and feed-forward parameters, hidden-state computations, and the output
mapping.

### From a probability vector back to one token ID

The vocabulary axis is indexed by token ID. After softmax,

\[
p_t=[p_t(0),p_t(1),\ldots,p_t(V-1)],
\]

so array position $i$ is the probability of token ID $i$. No separate semantic
search is needed to recover an ID. A decoding rule chooses an index from this
vector.

Greedy decoding selects:

\[
i_{t+1}=\operatorname*{arg\,max}_i p_t(i)
=\operatorname*{arg\,max}_i z_t(i).
\]

The second equality holds because softmax preserves ordering, so an implementation
does not need to materialize probabilities merely to take the maximum. Sampling
instead draws $i_{t+1}$ from the categorical distribution, often after modifying
the logits with temperature, top-$k$, or top-$p$ filtering. Therefore generation
does not always select the largest probability.

For a batch, the last-position path is:

```python
last_logits = logits[:, -1, :]                         # [B, V]
probabilities = softmax(last_logits, dim=-1)            # [B, V]
next_id = probabilities.argmax(dim=-1, keepdim=True)    # [B, 1], greedy only
input_ids = torch.cat([input_ids, next_id], dim=1)      # [B, T+1]
```

The new ID is appended to the prefix, the model runs the next autoregressive
step, and the loop stops at EOS or another stopping condition. Finally, the
tokenizer decoder maps the generated ID sequence back to text. During
teacher-forced training this selection loop is normally absent: the known target
label is used to calculate loss, allowing all shifted positions to train in
parallel.

At a single batch item and position, the model emits a logit vector
$z\in\mathbb{R}^V$, while the stored label $y$ is a scalar integer in
$\{0,\ldots,V-1\}$. Softmax turns the $V$ logits into $V$ probabilities, and
cross-entropy uses $y$ to select the observed token's probability. Conceptually,
$y$ corresponds to a one-hot target distribution $q\in\mathbb{R}^V$, but
implementations normally store only the integer ID rather than materializing
the large one-hot vector.

For next-token alignment, the tensors used by the loss become:

```python
shift_logits = logits[:, :-1, :]   # [B, T-1, V]
shift_labels = labels[:, 1:]       # [B, T-1]
```

Thus the two tensors agree on batch and supervised-position axes. The logits
retain one extra vocabulary axis because each prediction is a competition among
all $V$ candidates; each label needs only one integer to identify the observed
winner. Labels are required for training-time evaluation of the prediction but
are absent during ordinary generation, when the model emits logits and a decoding
rule selects the next token.

## Teacher forcing and generation drift

Correct shifting is necessary, but it does not make training and generation
identical. Under teacher forcing, every prediction is conditioned on the real
prefix from the dataset. During free generation, the model must condition on its
own sampled or selected tokens. One imperfect choice changes the next prefix;
that altered prefix may be rarer in training, making another error more likely.
Small local errors can therefore compound over autoregressive steps.

This is often called exposure bias or train--generation distribution mismatch:

\[
\text{training prefixes}\sim p_{\text{data}},
\qquad
\text{generated prefixes}\sim p_{\text{model}}.
\]

It is distinct from incorrect label alignment. An unshifted objective teaches
the wrong task—recovering the current visible token. Exposure bias can remain
even when labels are shifted correctly and the causal mask is flawless. It also
does not imply that teacher forcing is defective: next-token maximum likelihood
is still the standard local probabilistic objective, while free-running quality
depends on how errors and decoding decisions change later contexts.

## Loss masks: which aligned predictions count

Correct target alignment is necessary but does not imply that every tensor
position should contribute to optimization. Let $m_{b,t}\in\{0,1\}$ indicate
whether one aligned next-token target is valid. The mean loss is

\[
L=
\frac{
\sum_{b,t}m_{b,t}\left[-\log p_{b,t}(y_{b,t})\right]
}{
\sum_{b,t}m_{b,t}
}.
\]

The denominator is the number of valid target tokens, not necessarily $B(T-1)$.
This matters when sequences have different lengths or when prompt, padding, or
otherwise unsupervised regions are present.

For example, after shifting two padded sequences may have:

```text
targets A:    The    cat    slept  EOS
loss mask A:   1      1       1     1

targets B:    Hello  EOS    PAD    PAD
loss mask B:   1      1       0      0
```

There are six supervised targets, so the mean divides by six rather than eight.
EOS is normally meaningful because predicting the end of a sequence is part of
language modeling. Padding is storage structure rather than language evidence and
normally contributes no loss. Frameworks often encode an ignored label with a
sentinel such as `-100` rather than store a separate loss-mask tensor.

Three masks answer different questions:

| Mechanism | Question it answers |
|---|---|
| causal attention mask | May this hidden state read a future token? |
| padding/document attention mask | May this hidden state read this padded or unrelated position? |
| loss mask / ignored label | Should this prediction be graded and update parameters? |

A position may be readable as context while still being excluded from the loss.
For example, supervised fine-tuning can let answer tokens attend to a user prompt
while assigning loss only to the answer. Conversely, excluding a position from
loss does not by itself prevent other positions from attending to it.

This explicit masked-mean mathematics is already within approved task
`ANIM-CE-001`; it strengthens that animation's aggregation act and does not create
a duplicate proposal. Production remains on the Mac Studio.

## Evidence state

- `introduced`: position $t$ predicts token $t+1$.
- `introduced`: explicit shifted shapes `[B,T-1,V]` and `[B,T-1]`.
- `developing`: the learner correctly traced token IDs through the transformer's
  final contextual state into logits, then initially placed softmax directly on
  that hidden state. The refined path is hidden state `[D]` → output head →
  logits `[V]` → softmax → probabilities `[V]`.
- `developing`: the learner is refining the cardinality distinction: one input
  position yields one contextual state but a complete vector of scores over all
  $V$ possible output tokens; a sequence yields one such vector per position.
- `introduced`: the learner inferred that a large vocabulary projection demands
  substantial computation. The refined systems view is a trade-off between
  $V$-dependent output cost and possible $T$ reductions from better token
  compression, with GPUs evaluating the dense projection in parallel.
- `developing`: the learner correctly separated tokenizer-level atomic IDs from
  model-learned linguistic relationships. The refinement replaces "unit of
  understanding" with "unit of representation/computation/prediction," retains
  BPE merge history as procedural rather than semantic structure, and avoids
  reducing model knowledge to static embedding similarity.
- `developing`: the learner recognized that the model operates on atomic IDs and
  asked whether it therefore learns numbers rather than language. The refined
  answer treats IDs as permutation-invariant categorical addresses, distinguishes
  hidden representations from logit readouts, and describes language modeling as
  direct token-sequence learning that induces text-level linguistic structure.
- `introduced`: the learner traced logits through softmax to a probability
  vector and asked how one ID emerges. The vocabulary coordinate already equals
  the token ID; greedy decoding uses `argmax`, sampling draws a categorical
  index, and the chosen ID is appended for the next autoregressive step.
- `developing`: the learner initially connected an unshifted loss with seeing
  future tokens. The discussion separated two independent failure modes:
  unshifted target alignment asks position $t$ to recover the already-visible
  token $x_t$, while a broken causal attention mask lets position $t$ inspect
  future tokens $x_{>t}$.
- `developing`: when asked why teacher-forced performance can exceed free-running
  performance, the learner correctly proposed incorrect label shifting as one
  possible failure mode. The remaining refinement distinguishes that objective
  bug from exposure bias: even a correctly trained model sees gold prefixes in
  teacher forcing but must consume its own potentially drifting prefixes during
  generation.
- `introduced`: loss masks select which correctly aligned targets contribute,
  and the mean-loss denominator counts valid targets rather than padded tensor
  capacity.
- `not yet demonstrated`: learner explanation-back of the corrected distinction,
  exact BOS/EOS trace, separation of attention and loss masks, and
  manual/PyTorch agreement.

## Animation opportunity check

This important tensor alignment automatically triggers animation review. Existing
`CAND-ANIM-001` already preserves token identity while logits, next-token targets,
and per-position losses align. It also covers the output-head bridge from hidden
dimension $D$ to vocabulary dimension $V$, so no duplicate candidate is created. It remains
`discuss` and all production remains on the Mac Studio after approval and
verification.

## Open edges

- Trace one concrete sequence from input IDs to every supervised target.
- Show the failure produced by unshifted labels.
- Contrast gold-prefix teacher forcing with compounding free-generation errors.
- Verify framework shifting and ignore-index behavior in PyTorch.
- Distinguish sequence boundaries from padding and packed-document boundaries.
