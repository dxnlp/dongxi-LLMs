# Appendix B — Mathematical and Tensor Notation

This appendix is a reference for checking a derivation or tensor operation.
The main chapters supply motivation, examples and controlled experiments.

## B.1 Vectors, matrices and batches

A row-vector convention keeps the decoder equations close to tensor code.
States $X\in\mathbb{R}^{B\times n\times D}$ multiply the transpose of the stored
head $W_{\mathrm{out}}\in\mathbb{R}^{V\times D}$ to produce logits of shape
$[B,n,V]$: $Z=XW_{\mathrm{out}}^\top$.
PyTorch's `nn.Linear(D,V)` uses exactly that stored $[V,D]$ layout.
An equivalent mathematical projection of shape $[D,V]$ is
$W_{\mathrm{out}}^\top$, rather than a differently stored parameter.
Confusing the stored layout with the mathematical layout is a common source
of shape mistakes.

A reshape only changes how entries are indexed; a permutation changes axis
order. Splitting $[B,n,Hd_h]$ into $[B,n,H,d_h]$ and transposing gives
$[B,H,n,d_h]$. The last two axes of the key tensor are transposed for the query-key
matrix product. A contiguous copy may be required before another reshape.

Broadcasting supplies omitted leading axes, but does not certify their meaning.
A causal $[n,n]$ mask can broadcast across batch and head dimensions. A completion
$[B,n]$ mask cannot be applied to vocabulary logits without an explicit expanded
axis or a loss reduction. Inspect the semantic axis as well as the size.
Legacy code uses `T` for $n$ and `d`/`d_k` for $d_h$; attention values use
$V_{\mathrm{val}}$, while $V$ remains vocabulary size. The complete shared
[symbol table](../front-matter/notation.md) also separates temperature,
duration, reward sources and objective coefficients.

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
derivative is 1 because changing one coordinate by a small amount changes the
sum by that same amount. For $L=x^2$, the derivative is $2x$; it depends on the
current parameter value.

The chain rule adds contributions from every computational path to a shared
parameter. For a tied embedding/output table, input lookup and output projection
both contribute. A parameter absent from the current lookup can still receive
output-classifier gradient. A zero local loss does not eliminate gradients from
later dependent predictions.

For unscaled cross-entropy $L=-\sum_jq_j\log p_j$, with
$p=\mathrm{softmax}(z)$ and a fixed normalized target distribution $q$,

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

Two checks answer different questions. **Statistical independence:** can the
baseline depend on the pre-action state without depending on the action whose
score it weights? **Graph independence:** does the actor loss treat that baseline
as fixed when differentiating? Detaching an action-dependent reward as its own
baseline still produces zero advantage. A pre-action learned value can be
statistically valid while accidentally adding an unwanted gradient through its
parameters if it is left attached. Chapter 12 works through these distinctions.

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

KL is an average log-probability discrepancy weighted by its first argument.
For $p=(0.8,0.2)$ and $q=(0.5,0.5)$,

$$
D_{\mathrm{KL}}(p\Vert q)=0.8\log1.6+0.2\log0.4\approx0.1927,
\qquad
D_{\mathrm{KL}}(q\Vert p)=0.5\log0.625+0.5\log2.5\approx0.2231.
$$

The distributions are unchanged; the averaging weights change. Individual log
ratios can be negative even though their KL average is nonnegative. Chapter 3
derives the entropy/cross-entropy relationship, and Chapter 11 identifies which
distribution supplies the weights in a policy/reference penalty.

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

## B.7 Architecture operations at a glance

For one feature vector $x\in\mathbb{R}^D$, let
$\mu=D^{-1}\sum_i x_i$ and
$v=D^{-1}\sum_i(x_i-\mu)^2$. With learned feature scale $\gamma_i$,
offset $\beta_i$ and positive stabilizer $\epsilon$,

$$
\mathrm{LayerNorm}(x)_i=\gamma_i\frac{x_i-\mu}{\sqrt{v+\epsilon}}+\beta_i,
\qquad
\mathrm{RMSNorm}(x)_i=\gamma_i\frac{x_i}{\sqrt{D^{-1}\sum_jx_j^2+\epsilon}}.
$$

Both normalize over the feature axis at one position; neither uses batch-wide
statistics. RMSNorm does not subtract the feature mean; scaling can still
change it. The residual update
$x'=x+f(x)$ preserves an identity path alongside the learned branch. Placement
and the branch Jacobian determine how that path combines with normalization;
the formulas alone do not establish optimization stability. Chapter 5 supplies
numeric vectors, branch maps and the baseline/modern architecture comparison.

RoPE rotates coordinate pairs rather than adding a learned position-table row.
For a column pair $u\in\mathbb{R}^2$ at position $m$ and angular frequency
$\omega$, write

$$
R_{m\omega}u=
\begin{bmatrix}\cos(m\omega)&-\sin(m\omega)\\
\sin(m\omega)&\cos(m\omega)\end{bmatrix}u.
$$

Different pairs use different frequencies. Applying these rotations to queries
and keys makes their dot product depend on relative displacement, because
$R_{m\omega}^\top R_{n\omega}=R_{(n-m)\omega}$. This column-pair expression is
equivalent to the row-vector implementation after transposing the rotation.
Keep position offsets consistent when reusing cached keys. Chapter 5 develops
the derivation and the deliberately broken cache example.

## B.8 Optimizer state changes the update

Adam stores an exponentially weighted signed gradient $m_t$ and squared
gradient $v_t$, each with the parameter's shape. For zero-initialized moments,

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,
$$

$$
\widehat m_t=\frac{m_t}{1-\beta_1^t},\qquad
\widehat v_t=\frac{v_t}{1-\beta_2^t},\qquad
\theta_t=(1-\eta_t\lambda_{\mathrm{wd}})\theta_{t-1}
-\eta_t\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon_{\mathrm{Adam}}}.
$$

The square root rescales each coordinate by its recent gradient magnitude;
$v_t$ is not a mean-subtracted variance estimate. Bias correction compensates
for initial zero moments. AdamW applies the displayed shrinkage separately from
the adaptive gradient term. Betas, epsilon, decay groups and the learning-rate
schedule are part of the recipe. Chapter 6 traces their effects numerically;
the optimizer state explains why the same current gradient can produce different
updates after different histories.
