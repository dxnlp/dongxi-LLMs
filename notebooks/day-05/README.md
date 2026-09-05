# Day 5 — Planned Decoder Sessions

These sessions contribute to Chapter 5 alongside the later Day 6 design work.
All are planned; no runnable notebooks are claimed yet.

| Session | Central question | Planned checks |
|---|---|---|
| 01_multi_head_attention.ipynb | What do separate retrieval distributions preserve before head outputs are combined? | Split/merge shapes, independent head computation, output projection, reference agreement, causal invariance |
| 02_residual_norm_and_mlp.ipynb | How do sublayers transform a state without replacing the entire representation at every step? | Residual identity case, featurewise LayerNorm, pre/post-norm gradient paths, linear/nonlinear MLP intervention |
| 03_tiny_decoder_one_batch.ipynb | Can the assembled model learn the declared next-token task end to end? | Embeddings/positions, stacked blocks, vocabulary head, correct shift, consistent one-batch fit, memorization limits |

Every session includes conceptual predictions, inspectable tensors, optional
implementation exercises, adjacent runnable reference answers, explanations,
controlled changes, and evidence boundaries. Reusable computation belongs in
`src/dongxi_llms/`. Create the actual notebooks as the mechanisms are developed.
