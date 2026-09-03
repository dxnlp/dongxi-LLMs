# How Loss Trains Attention Routing and Content

- Day: 04
- Date opened: 2026-09-03
- Status: introduced
- Book destination: Chapter 4, attention gradients and credit assignment
- Related evidence: planned Day 4 autograd notebook and finite-difference checks
- Related production tasks: `CAND-ANIM-008`

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

## Open edges

- Verify every matrix derivative with PyTorch autograd and finite differences.
- Inspect how a forbidden future edge receives zero routing gradient.
- Compare value-path and routing-path gradient norms without treating magnitude
  alone as an explanation.
- Deliberately detach $A$ or $V$ to isolate the two learning routes.

## Reuse opportunities

- Chapter 4 derivation and worked tensor-shape exercise.
- Interactive notebook controls that detach routing or value paths.
- Extend `CAND-ANIM-008` with a backward pass that visibly splits at $O=AV$ and
  reaches $W_V$ versus $W_Q,W_K$.
