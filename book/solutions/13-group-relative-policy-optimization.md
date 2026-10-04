# Chapter 13 — Worked solutions

Read the [chapter](../chapters/13-group-relative-policy-optimization.md) and
make a prediction before opening an answer. Runnable versions are adjacent to
the exercises in the [notebook pathway](../labs/13-group-relative-policy-optimization.md).

## 1. A successful group can teach nothing relatively

If all rewards are one, each centered reward is zero. The standard deviation is
zero, so our explicit constant-group rule returns zero advantages. The group
contains evidence of success, but no information about which sampled response
should become more likely relative to the others. Across later prompts, the
shared parameters may still change. KL regularization or optimizer momentum can
also move them; “zero relative signal” is narrower than “no weight movement.”

## 2. Population versus sample standard deviation

For rewards 0, 1, the mean is 0.5. Population standard deviation is 0.5 and sample
standard deviation is $\sqrt{0.5}$. Normalized advantages are approximately
−1, +1 with the population convention and−0.707, +0.707 with the sample convention.
The sample convention reduces this group's scale by $\sqrt{(G-1)/G}$.
It must be declared because the change affects optimization, especially for small groups.

## 3. The denominator changes prompt weighting

Centering subtracts a local comparison; dividing by a reward-dependent standard
deviation scales that comparison differently across groups. Binary groups with
success fractions near 0 or 1 have a smaller spread than balanced groups.
Moreover, the statistics include each response's own reward. The exact
independent-sample baseline identity from REINFORCE does not make all these
normalized estimators identical or unbiased. The notebook changes reward spread
while retaining signs and exposes the changed advantage magnitudes.

## 4. Zero value, nonzero slope

At initial ratio 1, centered advantages can sum to zero, producing a zero average
surrogate value. The derivative contains each answer's log-probability derivative:

$$
\nabla J=\frac1G\sum_i\hat A_i\nabla\log\pi_\theta(y_i\mid x)
$$

for the simple one-token, unclipped case. Those gradients point in different
directions. The scalar sum of advantages being zero does not make this vector
sum zero. The actual decoder experiment records a first-update gradient norm.

## 5. Negative-advantage clipping

With $A=-1$, ratio 0.5 and clip interval[0.8, 1.2], the unclipped product is−0.5
and clipped product−0.8. The minimum is−0.8, stopping further reward for reducing
the ratio below 0.8. At ratio 1.5, the minimum is−1.5 rather than−1.2, so the
unfavorable increase remains penalized. Clipping treats direction relative to
advantage, not merely numerical distance from one.

## 6. A value identity does not specify a full derivative

The expectation of $k_3$ under the current distribution equals forward KL.
When differentiating that expectation, both the integrand and its distribution
weights depend on parameters. Autograd through a fixed sample includes only the
integrand's path unless a score-function or equivalent correction supplies the
other path. Sampling under an old distribution adds another mismatch. The
notebook computes exact categorical KL and verifies its full logit gradient,
making the implemented regularizer unambiguous at visited states.

## 7. Length reductions

For lengths 2 and 8, response means assign each response half the total nominal
weight, so per-token weights are 0.25 and 0.0625. Global token means assign every
token weight 0.1, yielding total response weights 0.2 and 0.8. Actual gradients
also depend on token log probabilities and advantage signs. The coefficients
alone do not prove how generation length will change after training.

## 8. Prompt gradients survive a response-only loss

Response states attend to earlier prompt states. Differentiating response logits
therefore reaches prompt representations and their embedding/projection
parameters through attention and the residual stack. Masking a prompt's direct
loss removes a local objective term; it does not detach that state's computation.
A deliberate `.detach()` would change the graph and must be justified separately.

## 9. Strict checking can reject valid work

Our one-integer rule rejects a correct mathematical explanation containing extra
words. That is an intentional task-interface constraint, not proof the answer's
reasoning is wrong. Check allowed equivalences and adversarial cases separately.
Tests reject multiple answers, embedded correct substrings and truncation while
accepting the declared whitespace/sign variants. Record the raw response and
checker version to diagnose parser failures without rewriting history.

## 10. Implementation readiness and model evidence

CPU tests show that the selected objective, masks, gradient boundaries and tiny
decoder path execute. They do not measure Qwen3's GPU memory, throughput,
held-out math accuracy or safety. The optional HF runner requires a supplied
local model and pinned revision. Its model-scale experiment remains unexecuted
until a separate report records that run. A four-prompt CPU held-out score is
descriptive, and every failed row remains part of the report.
