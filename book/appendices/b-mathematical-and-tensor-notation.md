# Appendix B — Mathematical and Tensor Notation

This appendix is a reference for checking a derivation or tensor operation.
The main chapters supply motivation, examples and controlled experiments.

## B.1 Vectors, matrices and batches

A row-vector convention keeps the decoder equations close to tensor code.
States $X\in\mathbb{R}^{B\times T\times D}$ multiplied by a projection
$W\in\mathbb{R}^{D\times V}$ produce logits of shape $[B,T,V]$.
PyTorch's `nn.Linear(D,V)` stores its weight as $[V,D]$ and applies its transpose.
Confusing the stored layout with the mathematical layout is a common source
of shape mistakes.

A reshape only changes how entries are indexed; a permutation changes axis
order. Splitting $[B,T,Hd]$ into $[B,T,H,d]$ and transposing gives
$[B,H,T,d]$. The last two axes of the key tensor are transposed for the query-key
matrix product. A contiguous copy may be required before another reshape.

Broadcasting supplies omitted leading axes, but does not certify their meaning.
A causal $[T,T]$ mask can broadcast across batch and head dimensions. A completion
$[B,T]$ mask cannot be applied to vocabulary logits without an explicit expanded
axis or a loss reduction. Inspect the semantic axis as well as the size.

## B.2 Log probabilities and objective reduction

For vocabulary logits $z$ and target ID $y$,

$$
\log p_y=z_y-\log\sum_j e^{z_j},\qquad L=-\log p_y.
$$

Subtracting $\max_j z_j$ before exponentiation leaves probabilities unchanged
and reduces overflow risk. Use `log_softmax` or a stable `logsumexp` rather than
computing probabilities and taking their logarithm afterward.

For fixed nonnegative weights $m_t$, the valid-target mean is

$$
L=\frac{\sum_t m_t\ell_t}{\sum_t m_t}.
$$

The denominator must be positive. When accumulating several microbatches,
sum their losses and divide by the total valid count. Averaging unequal batch
means changes the objective. A sequence log-probability is instead a sum of
token log-probabilities; replacing it with a mean changes preference or policy
objectives unless explicitly designed and justified.

## B.3 Gradients and optimizer updates

A gradient measures local loss sensitivity. For $L=\sum_j x_j$, each partial
derivative is1 because changing one coordinate by a small amount changes the
sum by that same amount. For $L=x^2$, the derivative is $2x$; it depends on the
current parameter value.

The chain rule adds contributions from every computational path to a shared
parameter. For a tied embedding/output table, input lookup and output projection
both contribute. A parameter absent from the current lookup can still receive
output-classifier gradient. A zero local loss does not eliminate gradients from
later dependent predictions.

For softmax probabilities and a normalized target distribution,

$$
\frac{\partial L}{\partial z_i}=p_i-q_i.
$$

This is the derivative with respect to the logits. Parameter updates also depend
on the Jacobian from parameters to logits and the optimizer's state. AdamW's
first and second moments change its update from plain gradient descent.

## B.4 Stop-gradient and sampled histories

Policy training needs explicit graph boundaries. Rewards, reference-policy
scores, old-policy probabilities and sampled actions are normally treated as
fixed when differentiating the current-policy surrogate. Detaching them is
part of the estimator, not a numerical convenience. If a baseline depends on
the sampled action, the usual unbiased-baseline argument does not apply.

Discrete sampling itself is not differentiated through in the score-function
estimator. The derivative enters through the log-probability of the sampled
trajectory. Tiny enumerable policies allow comparison against the exact
expected-reward gradient; language models normally need sampling.

## B.5 Probabilities, KL and support

For distributions $p,q$ on a common finite support,

$$
D_{\mathrm{KL}}(p\Vert q)=\sum_i p_i\log\frac{p_i}{q_i}.
$$

The direction matters. If $p_i>0$ and $q_i=0$, the forward KL is infinite.
Sampling estimates of a KL require a named sampling distribution and support
condition. An exact categorical sum, a single sampled log ratio, and a
nonnegative transformed ratio estimator have different sample variance and
different gradient interpretations. The policy chapters identify each one.

## B.6 Numerical checks

Check tensor shape before broadcasting, use float64 for tiny finite-difference
or exact-enumeration comparisons, and state absolute/relative tolerances.
Tests should compare independent formulations: manual derivatives against
autograd, weighted accumulation against a combined batch, cached against
uncached decoding, or an empirical estimator against exact enumeration.

A failed finite-difference check can come from an incorrect derivation, step
size, dtype or nondifferentiable boundary. Investigate that discrepancy rather
than widening tolerance until it disappears. A passing check supports the tested
mechanism and input domain, not arbitrary numerical scale or backend behavior.
