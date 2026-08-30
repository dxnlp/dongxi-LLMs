# Embeddings, Context, and Gradient Accumulation

- Day: 02
- Date opened: 2026-08-30
- Status: demonstrated; executable verification open
- Book destination: Chapter 2 embedding sections
- Related evidence: planned Day 2 embedding lab
- Related production tasks: none currently queued

## Questions that drove the discussion

- What shape changes when token IDs pass through an embedding table?
- Is embedding lookup equivalent to multiplying by a one-hot vector?
- What happens when the same token appears twice?
- Which embedding rows receive gradients?
- Why can repeated tokens end with different hidden states?
- Is the embedding layer trained separately, or through the complete language
  model objective?
- How does tying the input embedding and output projection change which rows
  receive gradients?
- What exactly does `h` represent in the output-logit equations?
- What is a logit, and how does it differ from a probability?

## Learner's initial model

The learner first described embedding output as a flat or one-dimensional value,
then refined the model through explicit tensor-shape exercises.

## Refined mental model

For vocabulary size `V` and embedding width `d`, the trainable table is:

```text
E.shape = [V, d]
```

An ID tensor of shape `[B, T]` performs row lookup and returns `[B, T, d]`.
Lookup is mathematically equivalent to a one-hot vector selecting a row, but is
implemented as indexing rather than materializing a huge sparse vector.

Repeated IDs return identical initial vectors. Transformer position handling and
contextual attention then produce generally different hidden states. For Qwen3,
position is applied through RoPE inside attention rather than a learned absolute
position vector simply added at lookup.

The embedding layer is normally trained end to end, not with a separate
"embedding objective." Token lookup supplies vectors to the transformer; the
transformer produces contextual states; the output projection produces logits;
the next-token loss sends gradients backward through the complete computation
graph into the embedding rows.

For a decoder hidden tensor `H ∈ R^(B×T×d)`, lowercase `h = H[b,t,:] ∈ R^d`
denotes the contextual hidden state at one batch item and one sequence position.
It is the vector used to predict the next token at that position. For input
`[cat, sat]`, the final-position state can be written conceptually as
`h_sat = Transformer(E[cat], E[sat])`; unlike the initial row `E[sat]`, it
contains information about the available causal context.

When input and output weights are untied, the lookup path sends direct embedding
gradients only to rows used by the input IDs. When weights are tied, the same
matrix also acts as the output classifier. The softmax loss then supplies an
output-side gradient to every vocabulary row in the exact classifier calculation,
while input rows additionally receive the lookup-path gradient.

## Concrete examples and derivations

```text
Before lookup: [2, 5]
After lookup with d = 8: [2, 5, 8]
```

For a single sequence:

```text
IDs: [2, 5, 2]
Returned vectors: E[2], E[5], E[2]
```

If `X_i = E[token_i]`, then repeated uses accumulate into the shared row:

```text
dL/dE[2] = dL/dX[0] + dL/dX[2]
dL/dE[5] = dL/dX[1]
```

Unused rows receive zero direct embedding gradient from this example.

For one output state `h`, tied output logits have the form:

```text
logit_i = h · E[i]
```

A logit is one raw, unnormalized compatibility score for a candidate next token.
It may be positive, zero, or negative, and logits do not need to sum to one. For
vocabulary size `V`, one state `h ∈ R^d` produces `V` logits. Softmax converts
their relative differences into probabilities:

```text
p_i = exp(logit_i) / sum_j exp(logit_j)
```

Adding the same constant to every logit leaves the probabilities unchanged, so
relative logit differences—not their absolute offset—control the distribution.
With untied weights, the same role is played by a separate output matrix rather
than `E^T`.

The learner requested a slower explanation of the logit-to-probability step. For
the three-token example `[cat, dog, eos]`, use logits `[1, 2, 0]`:

```text
raw logits:             [1,    2,    0]
positive exp weights:   [2.72, 7.39, 1.00]
weight total:           11.11
softmax probabilities:  [2.72/11.11, 7.39/11.11, 1.00/11.11]
                       ≈ [24.5%,      66.5%,      9.0%]
```

Exponentiation makes every weight positive and turns a logit difference into a
ratio: a one-point advantage gives `exp(1) ≈ 2.72` times the unnormalized weight.
Division by the shared total makes the values sum to one. Because every candidate
shares the denominator, increasing one logit generally lowers the probabilities
of the others even when their own logits remain unchanged. Adding the same
constant to all logits multiplies every weight by the same factor and therefore
leaves all normalized probabilities unchanged.

With target token `y`, the output-side contribution is:

```text
dL/dE[i] |_output = (p_i - 1[i = y]) h
```

Thus every output row generally receives a contribution, while the target row's
coefficient is negative when `p_y < 1`. An input row receives the sum of this
output contribution and any gradients arriving through positions where it was
looked up.

The visual shorthand `E ← E - η∇E` represents an SGD update. Production LLM
training commonly uses AdamW, which transforms the same accumulated gradient
through moment estimates and weight decay before updating the parameter.

## Demonstrated understanding

- Correctly calculated `[2, 5] → [2, 5, 8]` for `B=2`, `T=5`, `d=8`.
- Correctly answered that `[2, 5, 2]` returns three vectors while rows 2 and 5
  receive direct gradients.
- Correctly explained that repeated token embeddings are identical before the
  transformer and generally different afterward because context matters.
- Correctly identified rows 1 (`cat`) and 2 (`sat`) as receiving lookup-path
  gradients, and correctly identified those same rows as receiving both paths
  when weights are tied.
- Initially predicted that the tied output path updates rows 1, 2, and target row
  3. The refined result is that ordinary dense softmax supplies an output-path
  contribution to every vocabulary row: `(p_i - 1[i=y])h`. This correction still
  requires an explanation-back and executable verification.
- Correctly inferred that when `p(dog)=0.2`, the target-row gradient
  `(0.2-1)h=-0.8h` makes an SGD update move `E[dog]` toward `h`, increasing their
  dot product when `h` is held fixed for the local calculation.
- Correctly explained the equal-logit case: equal logits produce equal
  exponential weights; normalization by their shared total therefore produces
  equal probabilities. This demonstrates the basic logit-to-softmax mechanism.
- The unequal-logit case and the role of relative logit differences still need
  a learner explanation-back before the complete softmax bridge is marked
  demonstrated.

## Evidence and limitations

These results have been demonstrated analytically in the interactive lesson but
have not yet been verified in the Day 2 executable lab. "Direct embedding
gradient" does not mean that only embedding parameters train; all used upstream
model parameters can receive gradients through the complete computation graph.

## Open edges

- Verify lookup output and repeated-row gradient sums in PyTorch.
- Inspect Qwen3's actual embedding matrix shape and padded vocabulary rows.
- Verify whether input embeddings and the output projection are tied.
- Bridge the final hidden state to logits in Day 3.
- Explain back which embedding rows receive gradients in tied versus untied
  configurations.

## Reuse opportunities

- Chapter 2 tensor-shape walkthrough.
- Small exercise deriving repeated-row gradient accumulation.
- X article task `X-EMB-001`: explain how next-token loss trains embedding rows,
  including repeated IDs and the tied-versus-untied gradient distinction.
- Animation task `ANIM-EMB-001`: IDs → lookup → transformer → tied output loss →
  two backward paths → optimizer update. The planned Mac/Manim version should be
  one continuous forward-and-backward motion, with the shared identity of `E`
  preserved throughout rather than presented as disconnected slides.
