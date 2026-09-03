# Why Scaled Dot-Product Attention Divides by the Square Root of Width

- Day: 04
- Date opened: 2026-09-03
- Status: introduced
- Book destination: Chapter 4, scaled dot-product attention
- Related evidence: planned variance simulation and Day 4 attention notebook
- Related production tasks: `CAND-ANIM-008`

## Questions that drove the discussion

- How does the spread of a query-key dot product change as $d_k$ grows?
- Why does that changing scale cause trouble for softmax and its gradients?
- Why is the divisor $\sqrt{d_k}$ rather than $d_k$?

## Learner's initial model

The learner requested the derivation directly rather than proposing a numerical
prediction.

## Refined mental model

Under the standard teaching assumption that query and key coordinates are
independent, mean zero, and variance one, one coordinate product has

$$
\mathbb{E}[q_mk_m]=0,\qquad \operatorname{Var}(q_mk_m)=1.
$$

The unscaled score is a sum across $d_k$ coordinates:

$$
s=q\cdot k=\sum_{m=1}^{d_k}q_mk_m.
$$

Independent variances add, so

$$
\operatorname{Var}(s)=d_k,
\qquad
\operatorname{Std}(s)=\sqrt{d_k}.
$$

The score's typical scale therefore grows like $\sqrt{d_k}$, not $d_k$: positive
and negative terms partly cancel, while their variance accumulates. Dividing by
$\sqrt{d_k}$ restores approximately unit variance:

$$
\operatorname{Var}\!\left(\frac{s}{\sqrt{d_k}}\right)
=\frac{d_k}{d_k}=1.
$$

Without scaling, increasing head width makes softmax logits increasingly spread
out for statistical reasons unrelated to stronger learned evidence. Exponentials
amplify these gaps, producing nearly one-hot attention rows. In that saturated
regime, the softmax Jacobian

$$
\frac{\partial a_i}{\partial s_j}
=a_i(\mathbf{1}[i=j]-a_j)
$$

contains many values near zero, weakening useful gradient signals. Scaling keeps
the initial score distribution in a more trainable range and makes behavior less
dependent on $d_k$.

## Concrete examples and derivations

- At $d_k=64$, the assumed unscaled score standard deviation is $8$; division by
  $\sqrt{64}=8$ returns it to approximately $1$.
- At $d_k=1024$, the assumed unscaled standard deviation is $32$; division by
  $\sqrt{1024}=32$ again returns it to approximately $1$.

This scaling resembles a fixed softmax temperature because it rescales all score
gaps in a row. It is not cosine normalization: queries and keys are not divided
by their individual norms, so learned vector magnitudes can still affect scores.

## Demonstrated understanding

No explanation-back has yet been recorded. The derivation is introduced and
awaits a conceptual restatement or executable observation.

## Evidence and limitations

The variance derivation depends on simplifying independence and unit-variance
assumptions that are most useful near initialization. Real trained query/key
coordinates need not remain independent or unit variance. Layer normalization,
initialization, QK normalization, attention variants, precision, and learned
correlations modify the empirical distribution. The $1/\sqrt{d_k}$ rule is a
scale-control argument, not a theorem that every trained score has variance one.

## Open edges

- Simulate score variance and softmax entropy across several $d_k$ values with
  and without scaling.
- Inspect gradient behavior as unscaled attention becomes saturated.
- Connect score scaling to causal masking and row-wise softmax.
- Later compare standard scaling with QK normalization in modern architectures.

## Reuse opportunities

- Central Chapter 4 derivation and worked exercise.
- Day 4 notebook experiment that varies head width while holding the coordinate
  distribution fixed.
- `CAND-ANIM-008`: expand the score histogram as $d_k$ grows, then stabilize it
  through division by $\sqrt{d_k}$ before applying softmax.
