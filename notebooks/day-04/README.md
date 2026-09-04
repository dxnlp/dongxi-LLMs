# Day 4 Interactive Attention

These sessions turn the Chapter 4 attention argument into inspectable code. They
begin with one head and tiny tensors so every intermediate object remains
visible. Multi-head integration and a complete decoder block remain in Chapter 5.

## Session sequence

| Session | Deep question | Primary mechanism | Deliberate perturbation | Evidence state |
|---|---|---|---|---|
| `01_causal_attention_forward.ipynb` | Where must the causal boundary be imposed so the future cannot influence even the competition among past tokens? | $QK^\top/\sqrt{d_k}$, pre-softmax mask, row-wise softmax, $AV$, and prefix invariance | Compare correct masking with zeroing weights after softmax and replacing forbidden scores with zero | planned next |
| `02_attention_gradients_and_failures.ipynb` | How can one next-token loss teach both where to retrieve and what content to transmit? | gradients through values, softmax routing, Q/K scores, and projection matrices | Remove scaling as $d_k$ grows; detach routing or value branches; inspect forbidden-edge gradients | planned |
| `03_kv_cache_equivalence.ipynb` | Why may a runtime retain past K/V without changing the model's mathematical result? | causal prefix immutability, prefill, incremental decode, per-layer cache growth, and equality checks | Reuse K/V from a different prefix or modify a cached prefix and expose invalid reuse | planned |

## Session 1 learning path

1. Begin with two sequences that share a prefix but have different futures.
2. Predict which outputs must remain identical under causal attention.
3. Construct tiny $X$, $W_Q$, $W_K$, and $W_V$ tensors and expose $Q$, $K$,
   $V$, raw scores, scaled scores, the mask, attention weights, and output.
4. Check four invariants: allowed-row sum one, future weight zero, output as an
   allowed value mixture, and unchanged-prefix output equality.
5. Break the implementation by applying softmax before masking. Show that a
   forbidden future score still changes the denominator and therefore changes
   the earlier output.
6. Break it again by replacing forbidden logits with zero. Show that zero is an
   ordinary competitive score, not “no probability.”
7. Compare the transparent implementation with PyTorch scaled-dot-product
   attention under a declared numerical tolerance.
8. End with observations, interpretations, and an evidence boundary.

## Session 2 learning path

1. Attach a small downstream loss rather than inventing target attention maps.
2. Retain gradients for $Q$, $K$, $V$, the scaled scores, $A$, and the projection
   matrices.
3. Trace the value/content branch and query-key/routing branch separately.
4. Verify masked future edges have zero routing probability and zero routing
   gradient for the isolated query.
5. Perturb scaling across increasing $d_k$ and inspect score spread, attention
   entropy, and gradient concentration.
6. Detach $A$ or $V$ to isolate what each branch can and cannot learn.
7. State why a visible attention weight is not a complete causal explanation.

## Session 3 learning path

1. Compute a short causal sequence in one full forward pass.
2. Repeat it as prefill followed by one-token decoding while retaining past K/V.
3. Compare every new-token output and logit with the uncached computation.
4. Inspect cache shapes across layers, positions, KV heads, and head width.
5. Demonstrate that cache entries are context-specific states, not vocabulary
   records, by attempting invalid cross-prefix reuse.
6. Separate mathematical equivalence from runtime memory allocation and release.
7. Record what the toy equivalence test proves and what it does not establish
   about serving performance.

## Notebook interaction contract

Each prediction or learner implementation cell is followed immediately by a
collapsed or clearly separated runnable reference solution and mechanism
explanation. Work through the sessions interactively; do not run all cells as a
passive demonstration. Reusable attention logic and verified comparisons move to
`src/dongxi_llms/`, tests, and an experiment report before Day 4 is complete.
