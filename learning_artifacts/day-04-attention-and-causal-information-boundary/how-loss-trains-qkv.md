# How Loss Trains Attention Routing and Content

- Day: 04
- Date opened: 2026-09-03
- Status: introduced
- Book destination: Chapter 4, attention gradients and credit assignment
- Related evidence: planned Day 4 autograd notebook and finite-difference checks
- Related production tasks: `CAND-ANIM-008`; approved `ANIM-ATTN-001`

## Questions that drove the discussion

- When the final prediction is wrong, does loss update only the values being
  transported, or also the queries and keys that determine routing?
- How does one gradient split across the two paths?

## Learner's initial model

The question was introduced after the learner established that attention creates
a value mixture and that attention weights alone are not complete explanations.
No answer was claimed before instruction.

## Refined mental model

For one head, write the forward computation as

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V,
$$

$$
R=\frac{QK^\top}{\sqrt{d_k}}+M,\quad
A=\operatorname{softmax}_{\text{row}}(R),\quad
O=AV.
$$

A downstream loss gradient $G_O=\partial L/\partial O$ reaches the attention
output and splits into two conceptually distinct routes.

The value or message-content route is

$$
G_V=A^\top G_O,
\qquad
G_{W_V}=X^\top G_V.
$$

It changes what source positions transmit. The routing route begins with

$$
G_A=G_OV^\top.
$$

For one softmax row, if $g_j=\partial L/\partial a_j$, then

$$
\frac{\partial L}{\partial r_j}
=a_j\left(g_j-\sum_m a_mg_m\right).
$$

Writing the resulting matrix as $G_R$, the query/key gradients are

$$
G_Q=\frac{G_RK}{\sqrt{d_k}},
\qquad
G_K=\frac{G_R^\top Q}{\sqrt{d_k}},
$$

and therefore

$$
G_{W_Q}=X^\top G_Q,
\qquad
G_{W_K}=X^\top G_K.
$$

The loss can consequently improve attention in two complementary ways: modify
the messages available through $V$, and modify which sources are favored through
$Q$ and $K$. The incoming states $X$ also receive the sum of gradient
contributions through all three projection paths.

## Concrete examples and derivations

```text
                         loss L
                            ↑
                       attention O
                       ↙          ↘
          routing weights A       values V
                  ↑                  ↑
          softmax and scores          W_V
              ↙        ↘              ↑
             Q          K              X
             ↑          ↑
            W_Q        W_K
```

If retrieving a source value would have reduced the loss, gradient can increase
its relative score by changing the receiver's query, the source's key, or both.
It can also change the source's value so that future retrieval carries a more
useful message. These are learned jointly under the downstream objective; no
separate supervision labels a “correct attention map.”

## Demonstrated understanding

No explanation-back has yet been recorded. The two-path gradient mechanism is
introduced and awaits conceptual restatement plus executable verification.

## Evidence and limitations

The matrix derivatives are analytical and assume compatible row-vector
conventions. A production implementation may fuse projections and attention
kernels while remaining mathematically equivalent. Exact zero gradient through
forbidden mask locations depends on implementing masked softmax correctly.

Gradient magnitude is not identical to causal importance, and a parameter update
does not have a uniquely human-interpretable meaning. The downstream residual
stream, multiple heads, and later layers create additional paths not represented
in this isolated-head derivation.

## Review bridge — How values teach routing (2026-09-06)

For one receiver, let s_j be its scaled score for allowed source j,
a=softmax(s), o=sum_j a_j v_j, and g=partial L/partial o. Holding the value
vectors fixed for this local derivative gives

$$
\frac{\partial L}{\partial s_j}
=a_j\,g\mathbin{\cdot}(v_j-o).
$$

This follows by substituting partial L/partial a_j = g dot v_j into the
row-softmax derivative. The comparison is against the current mixture, because
raising one score redistributes weight away from the other sources. A negative
score derivative locally favors increasing the score under gradient descent;
the actual shared-parameter update need not move every score independently.

The identity was checked against autograd using the existing CPU float64
teaching fixture, receiver 2, and the fixed cross-entropy head from notebook 2,
at tolerance 1e-12. This is an algebraic refinement of the already verified
matrix derivative, not a new trained-model claim.

The learner predicted that a small routing change could improve prediction even
with identical values. The correction showed that any normalized mixture of
identical vectors is unchanged, so the local score gradient is zero. The learner
requested continuation; independent restatement is still pending. Capture this
comparison in the existing `ANIM-ATTN-001` rather than a new candidate.

The next explanation follows the value branch: for a single receiver with
output gradient g, partial L/partial v_j = a_j g. Changing content can change
the output even when redistributing fixed identical content cannot. Multiple
receivers contribute a sum to each source value, yielding G_V=A^T G_O. These
are gradients of intermediate values, not independent per-token stored
parameters: backward proceeds through the shared W_V and earlier input states.
Zero routing gradient does not imply zero content gradient or guarantee that
an optimizer step will separate previously identical values. This extension
uses the chapter's already verified matrix identity.

## Remaining learning and verification edges

### Requested focus — projection-matrix gradients (2026-09-06)

The learner explicitly chose a direct explanation of how gradients reach the
Q/K/V projection matrices. Keep this within Chapter 4's single head; preceding
multi-head/residual discussion moved ahead of the requested review.

Start with downstream G_O, split at O=AV into G_V=A^T G_O and G_A=G_O V^T,
then pass the routing derivative through row-softmax and the fixed causal mask
to obtain G_S. With S=QK^T/sqrt(d_k), use G_Q=G_S K/sqrt(d_k) and
G_K=G_S^T Q/sqrt(d_k). Finally each shared linear projection obeys
G_W=X^T G_projected. Explain that this product sums per-position outer-product
contributions; it is not an independently stored parameter per token. The
optimizer applies these gradients after backward and changes shared matrices
for subsequent forward computation. These identities are already covered by
the verified Chapter 4 fixture and `ANIM-ATTN-001`. No new learner mastery is
inferred from requesting or receiving this explanation.

### Update principles — 2026-09-06

The learner asked whether Q/K/V weight changes follow principles. Distinguish
trainable projection matrices W_Q/W_K/W_V from activation tensors Q/K/V and
the attention distribution A. The optimizer updates the matrices; forward
computation recomputes Q/K/V and A from current states and parameters.

All three projection gradients are obtained by differentiating the same
downstream objective through the same forward pass. They generally differ in
direction and magnitude. A simple SGD sketch is W_P <- W_P - eta*dL/dW_P for
P in {Q,K,V}; AdamW uses adaptive, stateful updates and weight decay instead of
this exact sketch. There is no hand-authored semantic rule telling W_Q which
linguistic feature to encode. Shared parameters accumulate credit over valid
positions and examples. Masked edges contribute no routing gradient, although
the shared matrices still learn from allowed edges.

Any statement that a useful source's score should rise concerns a local
derivative holding other intermediates fixed. Real shared-parameter steps can
change multiple scores and contents jointly; they do not guarantee that every
individual example improves. This clarification is introduced, not yet assessed.

- Verify every matrix derivative with PyTorch autograd and finite differences.
- Inspect how a forbidden future edge receives zero routing gradient.
- Compare value-path and routing-path gradient norms without treating magnitude
  alone as an explanation.
- Deliberately detach $A$ or $V$ to isolate the two learning routes.

## Reuse opportunities

- Chapter 4 derivation and worked tensor-shape exercise.
- Interactive notebook controls that detach routing or value paths.
- `ANIM-ATTN-001`: use a continuous backward pass that visibly splits at $O=AV$
  and reaches $W_V$ versus $W_Q,W_K$; production belongs on the Mac Studio after
  Day 4 verification.
