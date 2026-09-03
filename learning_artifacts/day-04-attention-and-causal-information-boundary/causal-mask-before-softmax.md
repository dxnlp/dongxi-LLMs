# Why the Causal Mask Must Remove Scores Before Softmax

- Day: 04
- Date opened: 2026-09-03
- Status: introduced
- Book destination: Chapter 4, causal masking and row-wise normalization
- Related evidence: planned Day 4 attention notebook and broken-mask experiment
- Related production tasks: `CAND-ANIM-008`

## Questions that drove the discussion

- Why must future positions be masked before softmax?
- Why is setting forbidden probabilities to zero after softmax insufficient?

## Learner's initial model

The learner correctly identified the causal objective: future information must
not be visible to the model. The remaining refinement was to explain why a
future score can still influence the output through the softmax denominator even
if its final value contribution is later set to zero.

## Refined mental model

For one query row, softmax makes every candidate score participate in a shared
competition:

$$
a_j=\frac{e^{s_j}}{\sum_r e^{s_r}}.
$$

If a forbidden future position remains present during softmax, its score enters
the denominator and changes every allowed probability. Zeroing only its resulting
probability afterward removes its direct value contribution but does not undo
the effect it already had on the normalization. The allowed weights also cease
to sum to one.

The causal mask therefore changes the score domain before normalization:

$$
\tilde{s}_{ij}=s_{ij}+M_{ij},\qquad
M_{ij}=\begin{cases}
0,&j\le i,\\
-\infty,&j>i.
\end{cases}
$$

Then $e^{-\infty}=0$, so forbidden positions contribute neither numerator nor
denominator:

$$
a_{ij}
=\frac{e^{\tilde{s}_{ij}}}{\sum_r e^{\tilde{s}_{ir}}}.
$$

The row becomes a normalized distribution over allowed sources only. Explicitly
zeroing forbidden probabilities and renormalizing the allowed probabilities
would recover the same mathematical distribution, but that is simply a less
direct reimplementation of masking before normalization.

## Concrete examples and derivations

Suppose a query may read the first two positions but not the third, and its raw
scores are

$$
[0,0,10].
$$

Softmax before masking is approximately

$$
[0.000045,0.000045,0.99991].
$$

Zeroing the forbidden third probability afterward gives approximately

$$
[0.000045,0.000045,0],
$$

whose sum is only about $0.00009$. The future score has nearly erased the allowed
message despite its own value being removed. Masking the third score to
$-\infty$ before softmax instead gives

$$
\operatorname{softmax}([0,0,-\infty])=[0.5,0.5,0].
$$

No future-dependent quantity participates in the allowed competition.

## Demonstrated understanding

The learner correctly explained the causal requirement that the model cannot see
the future. The specific normalization mechanism and denominator-leak argument
were then introduced and await explanation-back or executable observation.

## Evidence and limitations

The identities are analytical. Practical kernels often use a sufficiently large
negative representable value rather than constructing literal negative infinity;
correctness requires the resulting forbidden probability and gradient behavior
to satisfy the declared numerical tolerance. Mask orientation and broadcasting
remain common implementation failure modes.

## Open edges

- Distinguish causal masks from padding masks and Day 3 loss masks.
- Verify lower-triangular orientation for explicit tensor shapes.
- Demonstrate denominator leakage in a deliberately broken implementation.
- Inspect gradients at forbidden score locations.

## Reuse opportunities

- Chapter 4 derivation and a broken-mask worked example.
- Day 4 notebook comparison of pre-softmax masking, naive post-softmax zeroing,
  and post-softmax zeroing plus renormalization.
- `CAND-ANIM-008`: show the forbidden future column attempting to consume the
  denominator, then remove it before probability normalization.
