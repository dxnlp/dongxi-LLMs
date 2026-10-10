# Chapter 12 — Worked Solutions

Companion to [Language Generation as a Policy](../chapters/12-language-generation-as-a-policy.md).
The [lab guide](../labs/12-language-generation-as-a-policy.md) links the original
finite-policy sessions and probability-accounting/learned-critic extensions.

## 1. Score-function identity

$\nabla J=\sum_yR(y)\nabla\pi(y)=\mathbb{E}[R\nabla\log\pi]$ when reward
is fixed and differentiation and summation are valid. Implement the estimator
with negative detached reward times sampled sequence log-probability. Detachment
prevents spurious reward-gradient paths; the gradient of the sampled token
log-probability remains. If reward explicitly depends on policy parameters,
derive the extra term before deciding how to detach it.

## 2. Relative reward

The softmax derivative gives
$\partial J/\partial z_i=\sum_a p_a r_a(\mathbf1[a=i]-p_i)=p_i(r_i-J)$.
A reward one answer has a negative gradient when expected reward exceeds one.
Gradient ascent then moves mass toward better alternatives. The score need not
be negative in an absolute sense to be relatively undesirable.

## 3. A valid and invalid baseline

For action-independent $b$, the baseline contribution is
$b\sum_a\nabla p_a=b\nabla1=0$. For $b(a)=r_a$, every sample's advantage
is zero, so the estimator vanishes even if the true gradient is nonzero.
Detaching that baseline does not fix the statistical bias: dependence on the
sampled action is the issue.

## 4. Minimum-variance scalar

The mean gradient is unchanged by a constant baseline, so minimize
$\mathbb{E}[(R-b)^2\|s\|^2]$, where $s=\nabla\log\pi$. Setting its
derivative to zero yields
$b^*=\mathbb{E}[R\|s\|^2]/\mathbb{E}[\|s\|^2]$. Expected reward equals
this only under suitable score-norm weighting. The exact lab can enumerate both
values and their covariance traces; the result does not automatically prescribe
an optimal neural state-value baseline.

## 5. Group self-inclusion

$R_i-\bar R=(G-1)(R_i-b_{-i})/G$. Under iid conditional samples, the RLOO
mean has the true gradient expectation, while inclusive mean centering scales
it by $(G-1)/G$. Rescaling restores the same estimator. Coupled samples or
division by a reward-dependent standard deviation require another analysis.
RLOO needs at least two genuinely separate completions per prompt.

## 6. Two frozen policies with different clocks

The old policy generated the current rollout batch and defines importance
ratios. It updates when new rollouts are sampled. The reference defines the
long-term KL anchor and may remain fixed for the whole experiment. Using the
reference in place of the old policy's denominator does not correct sampling
from the actual rollout distribution.

## 7. Sequence and token ratios

The exact trajectory ratio is a product of token ratios, so its logarithm is
the sum of token log-ratio differences. PPO uses local ratios at states from
the old rollout distribution. It does not exactly correct all changes in state
visitation after several updates. Full ratios can have severe variance; local
surrogates trade exact correction for a more manageable update.

## 8. Clipping by advantage sign

For positive $A$, the surrogate is $A\rho$ until the upper threshold and then
flat; decreases below the lower threshold remain penalized. For negative $A$,
the beneficial decrease becomes flat below the lower threshold, while harmful
increases remain penalized. The notebook verifies slopes for ratios .6 and
1.4 and both signs. Clipping a sampled objective does not impose a global KL
bound on a shared-parameter policy.

## 9. KL control variate and missing support

For $a\sim p$ with common positive support,
$\mathbb{E}_p[q(a)/p(a)-1]=\sum_aq(a)-1=0$. Therefore adding this term to
$\log(p/q)$ preserves forward KL expectation and produces nonnegative samples.
If reference mass lies where $p=0$, the sum over sampled support is less than
one, breaking that identity. If $q=0$ where $p>0$, the forward KL is infinite.
The square-log estimator has no such exact expectation identity in general.

## 10. Fairness depends on the question

Match sampled completion count to compare sample efficiency, match gradient
passes or wall time to compare compute efficiency, and report both when they
differ. PPO with three epochs sees the same rollout budget but takes three times
as many gradient passes in this microscope. Use multiple seeds and one fixed
success definition. A finite task with separate prompt logits tests estimator
mechanics, not unseen language generalization.

## 11. Relationship among objectives

SFT maximizes externally demonstrated sequence likelihood. DPO fits recorded
pair judgments using reference-relative likelihood differences. REINFORCE
estimates expected reward gradients from current-policy samples. RLOO changes
its baseline using other samples for the same prompt. PPO uses old-policy data,
advantages, and a clipped surrogate for repeated updates. Common
log-probability computations do not make their training distributions,
objectives, or guarantees identical.

## 12. Reward rises while success falls

Check the reward definition against the independent success rule; inspect
verifier parsing, formatting shortcuts, response length, reward scaling, KL,
entropy, masks, and sample/old-policy identities. A learning-rate change can
hide instability but cannot repair a verifier rewarding the wrong behavior.
Retain paired examples and an untouched checkpoint control. Only after these
checks should a new controlled optimization comparison be proposed.

## 13. Lambda endpoints with a cap

At lambda zero, $\hat A_t=\delta_t$ and $\hat G_t=r_t+\gamma c_tV(s_{t+1})$.
At lambda one, intermediate values cancel, leaving observed discounted rewards
plus the final value after a nonterminal cap. Two zero-reward actions, gamma0.9
and bootstrap1.2 give returns0.972/1.08. Fully observed Monte Carlo targets
require true termination; a cap does not supply future experience.

## 14. EOS is an event and padding is not

Generated EOS zeros continuation even if a nonzero following value was supplied.
A cap leaves continuation alive in the declared task. Post-EOS padding gets no
residual, advantage or loss. Do not count a cap-four forced syntax endpoint as
EOS within cap-three delivery. Breaking the mask changes the return without
changing the observed token prefix.

## 15. Separate learning objectives

Actor gradients reach current likelihoods, not detached advantages or critic
parameters. Critic gradients reach its representation/scalar head, not detached
targets. Reward weights are frozen after verified reload; sampling IDs is not
differentiated. Shared features would receive both losses and let critic
updates move policy representations indirectly. Separate backbones isolate it.

## 16. Pair fitting does not constrain every generated completion

Training pairs used plain/fancy styles. Bare colors and repeated markers visit
text outside those comparisons. Tiny pair loss establishes fit on records,
not reliable extrapolation. Balanced oracle rows chose constant blue and
achieved only0.5 quality. Oracle values do not repair a reward or shortcut.

## 17. Compare quantities with their clocks

All arms match sampled paths and actor steps. Learned arms also train a critic;
oracle/noisy arms enumerate futures. Neither makes compute equal. Distinguish
terminal rewards, bootstraps and diagnostic prefix scores. Inspect every seed,
scheduled checkpoint and truncated/withheld row. Independent quality is not
reward or tuning criterion; finite controls do not rank pretrained PPO stacks.

The [new notebook](../../notebooks/day-20/04_learned_critics_and_frozen_text_rewards.ipynb)
supplies adjacent runnable answers, saved reward identities and measured plots.
