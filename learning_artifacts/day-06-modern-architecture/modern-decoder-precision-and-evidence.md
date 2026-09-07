# Modern decoder: conceptual boundaries and evidence

## Material-preparation request

After completing the Day 5 chapter foundation, the learner asked for Day 6
notebooks, then directly requested coherent Day 6 chapter material. Reused the
three already-built notebooks rather than creating duplicates; extended the
same Chapter 5 rather than opening a separate chronological chapter. No learner
predictions or mastery of these new mechanisms have yet been demonstrated.

## Distinctions the prose must preserve

- RMSNorm retains a common feature component relative to magnitude, not every
  original value. Epsilon breaks exact positive-scale invariance. Its output
  mean need not be zero and RMS-normalized length is not unit L2 length.
- SwiGLU's SiLU gate is not a probability. A zero gate can suppress the forward
  update while still receiving nonzero gradients through SiLU'(0)=1/2. Bias-free
  zero-output claims and approximately matched parameter budgets are explicit.
- RoPE rotates projected Q/K, not the value branch. Its relative-position
  identity is for fixed unrotated vectors, not a whole-model invariance or a
  long-context quality guarantee. Cached keys retain their original rotations.
- Causal visibility and consistent position offsets are independent requirements;
  a cache can be causal yet numerically wrong.
- GQA shares K/V representations, not query distributions. Compact cache
  storage, temporary repeated tensors, dense arithmetic, and measured latency
  are different quantities.
- This lab's Q/K norm uses RMS statistics per d-coordinate head, but a length-d
  Q scale and length-d K scale are shared across heads per layer: 2d parameters.
  It precedes RoPE and retains sqrt(d) scaling. Do not substitute the original
  L2-plus-learned-score-scale QKNorm definition or its reported results.
- Qwen's pinned config demonstrates D != Hq*d; it does not make the toy decoder
  checkpoint-compatible. The 16K-vocabulary larger candidates are calculations,
  not allocated models or a settled tokenizer/pretraining design.
- Recurrent weight sharing increases applications without multiplying shared
  weight storage. States and K/V generally differ across uses. Explicit
  recursion-specific KV-sharing architectures are a separate choice.

## Sources and publication boundaries

Rechecked on 2026-09-07: primary abstracts for RMSNorm (1910.07467), GLU variants
(2002.05202), RoFormer (2104.09864), GQA (2305.13245), and original QKNorm
(2010.04245); pinned Qwen3-0.6B config at c1899de289a04d12100db370d81485cdf75e47ca;
Geiping et al. recurrent depth (2502.05171v2); Bae et al. Mixture-of-Recursions
(2507.10524v3). Direct citations are placed at the corresponding chapter claims.
Paper summaries are introductions, not independent reproductions of results.
No closed-vendor architecture rumors are used in the factual argument.

X-LOOP-001 and CAND-ANIM-011 retain their existing production dependencies.
The new conceptual introduction is available, but the optional trained,
budget-controlled comparison and architecture defense remain Day 7 work.
Candidate capture does not approve animation production, public drafting, or
publication. Keep rendering on Mac Studio.

## Verification record

The modern chapter code, rounded normalization/SiLU examples, zero-gate
derivative, RoPE identity, parameter/cache/FLOP tables, and accounting-only larger
designs passed CPU checks. All 47 tests passed; 95 local links and seven heading
anchors resolved before final ledger updates. Notebook hashes still match the
three-session readiness run. Exact source identities and evidence boundaries:
`experiments/reports/2026-09-07-day6-chapter-verification.md`. None of these
checks establishes a trained quality advantage or learner mastery.
