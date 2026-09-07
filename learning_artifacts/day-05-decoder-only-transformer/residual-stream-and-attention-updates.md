# Residual Stream and Attention Updates

- Introduced: 2026-09-06, following multi-head attention during the review bridge
- Learning state: explanation introduced; no independent learner restatement
- Book placement: Chapter 5, after head concatenation/output projection;
  integrate into the Days 5–7 chapter synthesis

## Mechanism

Temporarily omitting normalization and dropout, write the attention sublayer
update as X_next = X + MHA(X). The attention output includes W_O and must have
the same [T,D] shape as X. Addition is coordinatewise, not concatenation.
Each position keeps a direct path from its incoming state while the learned
branch can contribute context-dependent changes. X is the current layer's
state, not the original embedding reintroduced afresh at every layer.

If the branch output is zero, the sublayer becomes the identity. In contrast,
the non-residual map X_next=MHA(X) must reconstruct any information that needs
to survive in its output. The residual form provides an identity path but does
not guarantee lossless preservation: learned updates can cancel or modify X.

For column-vector notation, if y=x+f(x), then the incoming gradient g_y gives
g_x=g_y+J_f(x)^T g_y. There is a direct gradient contribution and one through
the branch; cancellation remains possible, so residual connections are not a
universal guarantee against vanishing gradients.

Pre-norm refinement to introduce next: X_next = X + MHA(LN(X)). The residual
path carries X while the branch reads normalized states. Do not equate the
normalized branch input with replacing the entire residual stream.

## Animation

### Live refinement — 2026-09-07

The learner asked whether the residual keeps the original X. Clarified that
the skip carries the incoming state unchanged into the addition, but the next
layer receives only the sum, not a recoverable backup. X=[2,3] plus update
[1,-1] gives [3,2]. Later X means the accumulated state, not the embedding
reinserted from the start. Connect this directly to pre-norm: normalize what
the branch reads, not the skip itself. These are offered explanations; no
independent mastery claim follows from acknowledgement.

Canonical treatment: Chapter 5 sections 5.4–5.5 and worked answers 5–7. The
existing notebooks and 2026-09-06 report verify zero-branch and cancellation
controls; the production candidate remains approval-gated on Mac Studio.

`CAND-ANIM-013` shows X traveling along a persistent path, a branch computing
an update of the same width, and coordinatewise addition. A zero-update control
leaves X unchanged; backward splits into a direct and a learned path. Preserve
the distinction from head concatenation and from vocabulary output. Numerical
controls are now available in notebooks 03–04; production requires approval on Mac.
