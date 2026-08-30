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

## Demonstrated understanding

- Correctly calculated `[2, 5] → [2, 5, 8]` for `B=2`, `T=5`, `d=8`.
- Correctly answered that `[2, 5, 2]` returns three vectors while rows 2 and 5
  receive direct gradients.
- Correctly explained that repeated token embeddings are identical before the
  transformer and generally different afterward because context matters.

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

## Reuse opportunities

- Chapter 2 tensor-shape walkthrough.
- Small exercise deriving repeated-row gradient accumulation.
- Possible later animation from IDs to shared rows to contextual hidden states.
