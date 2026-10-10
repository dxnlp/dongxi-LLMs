# Notation and Experimental Conventions

All logarithms are natural unless a base is explicitly stated. Negative
log-likelihood and KL are therefore measured in nats. Token IDs are categorical
indices; their numerical distances do not express linguistic similarity.

| Symbol | Meaning | Typical shape |
|---|---|---|
| $B,T,V,D$ | Batch, sequence length, vocabulary, residual width | Scalar dimensions |
| $X$ | Residual states at all input positions | $[B,T,D]$ |
| $E$ | Input embedding table | $[V,D]$ |
| $h_t,z_t,p_t$ | Contextual state, logits, probabilities | $[D]$, $[V]$, $[V]$ |
| $q_t$ | Target distribution, often one-hot | $[V]$ |
| $H,d_k$ | Attention heads and head width | Scalar dimensions |
| $Q,K,V_{mathrm{attn}}$ | Queries, keys and attention values | $[B,H,T,d_k]$ |
| $m_t$ | Valid-target or completion-loss mask | $[B,T]$ |
| $\pi_\theta,\pi_{\mathrm{ref}},\pi_{\mathrm{old}}$ | Current, reference and rollout policies | Conditional distributions |
| $r,R,A$ | Reward score, return and advantage | Scalar or rollout/token arrays |
| $G$ | Number of completions per prompt in grouped optimization | Scalar |

The attention value tensor uses a subscript here to avoid confusing it with
vocabulary size. A chapter can use another local symbol after defining it.
An output table may be tied to $E$; unique parameter storage is then counted
once even though the parameter has two computational paths.

Python tensors commonly use zero-based positions. At logit position $t$, a
causal next-token model predicts token $t+1$. A training API may accept labels
at their original input positions and shift internally; a manual implementation
typically slices logits and labels explicitly. Every implementation in the book
states which alignment it uses.

The mask $m_t$ selects losses. It does not determine which source positions a
hidden state can attend to. Attention masks control that visibility, while
autograd detachment controls gradient propagation. These three boundaries
must be named separately when describing an intervention.

The same word “step” may refer to a generated token, microbatch, optimizer update
or environment action. Reports use the explicit term. Token budgets name whether
they count valid targets, processed positions, unique source targets or generated
tokens. Throughput names both its numerator and timing boundary.

Measured results are linked to reports. Calculated costs are labeled calculations.
Synthetic fixtures are labeled teaching examples. Proposed Spark protocols remain
proposed until their actual measurements exist. Repeatedly inspected prompts
are development probes; final capability claims need an independent contract.
