# 12. Language Generation as a Policy

> For the things we have to learn before we can do them, we learn by doing them,
> e.g. men become builders by building and lyre-players by playing the lyre.
>
> — Aristotle, *Nicomachean Ethics*, Book II, translated by W. D. Ross

Every method so far has learned from responses that were written before
training began. SFT copied demonstrations. The reward model and DPO learned
from recorded pairs, and DPO never scored an answer the policy itself produced
during training. Chapter 11 showed where that leaves a gap: DPO could open a
large margin on its recorded pairs while its own generated answers still
missed the required words. A fixed dataset can't tell the model about the
mistakes it would make on its own.

Aristotle's point is that some skills are learned only by practicing them.
This chapter makes that turn. Instead of studying fixed, pre-collected
responses, the model writes its own answers, a judge or checker scores them,
and the score tells it which of its own habits to strengthen. This is the
**reinforcement learning (RL)** view of generation. It finishes the
preference-alignment story of Chapters 10–11 and sets up Chapter 13, where the
score comes from an executable checker: **reinforcement learning with
verifiable rewards (RLVR)**.

The model answers `2 + 3 =` with `4`, `The answer is 5`, or `5`. A judge can
distinguish these complete responses without supplying the desired token at
every position. How does that final judgment reach the earlier choices that
produced the answer? And if we generate the same responses once and train on
them several times, when does their original probability cease to describe
the policy being updated?

Until now, training usually scored responses already present in a dataset.
Here the policy supplies its own candidates. We will first make their reward
and probability completely inspectable, then introduce the uncertainty hidden
by that small example. A story-continuity judgment or a program test changes
the source of reward; the reward-to-gradient mechanism remains the question.
We take it in six steps:

1. Name the pieces: prefixes are states, tokens are actions, and the model is a
   **policy** (§12.1).
2. Derive the basic learning rule, REINFORCE, and compute it exactly on three
   possible answers (§12.2–12.3).
3. Make the noisy rule less noisy with baselines and leave-one-out comparisons
   (§12.4–12.5).
4. Reuse old samples safely: importance ratios, PPO clipping and KL estimates
   (§12.6–12.8).
5. Compare the estimators under declared sampling and update budgets (§12.9).
6. Add a learned critic and generalized advantage estimation, then watch a
   fitted text reward and its critic fail in a measured run (§12.10–12.14).

Days 19–21 derive and test that mechanism before adding a large rollout system.
The prerequisites are autoregressive likelihood, the softmax gradient, and
gradient descent. We will inspect an exact finite policy, prove when baselines
preserve its gradient, examine leave-one-out rewards and proximal policy
optimization (PPO) clipping, and
compare estimators under explicit sampling and update budgets.

## 12.1 Tokens are actions; prefixes are states

To learn from its own attempts, the model needs a vocabulary for "attempt",
"situation" and "outcome". Reinforcement learning supplies one. A game of chess
is a good picture. The board is the situation, a move changes it, and only at
the end do we learn whether we won. Writing an answer works the same way. The
text so far is the board, each new token is a move, and a judge scores the
finished answer.

In RL terms, a **state** is the situation the decision-maker sees, an
**action** is one choice it makes, and a **policy** is its rule for choosing
actions, here a probability for each next token. For a fixed prompt $`x`$, define state $`s_t=(x,y_{<t})`$, action $`a_t=y_t`$, and
policy $`\pi_\theta(a_t\mid s_t)`$. Appending a token determines the next state.
A **trajectory** is the complete sequence $`y`$, including termination under the
chosen convention. Its probability factorizes:

```math
\pi_\theta(y\mid x)=\prod_t\pi_\theta(y_t\mid x,y_{\lt t}),\qquad
\log\pi_\theta(y\mid x)=\sum_t\log\pi_\theta(y_t\mid x,y_{\lt t}).
```

The environment here can be unusually simple: text concatenation is deterministic,
and a terminal verifier evaluates the answer. That does not make the sampling
problem disappear. Different prefixes lead to different states, and a mistake
can change all subsequent predictions. A terminal reward gives credit to many
actions at once; it does not identify which intermediate step was responsible.

Let $`R(x,y)`$ be a fixed **reward**, the number the judge gives a finished
response, and define
$`J(\theta)=\mathbb{E}_{y\sim\pi_\theta}[R(x,y)]`$. For now reward has no
direct parameter dependence. Chapter 11's positive $`\beta`$ weights the
reference penalty in $`J_\beta=\mathbb E[R]-\beta\,\mathrm{KL}(\pi_\theta\Vert\pi_{\mathrm{ref}})`$;
we retain that objective and KL direction when a penalty is added. If a reward includes policy-dependent KL terms,
we must account for that dependence explicitly rather than reuse this assumption
without checking it.

## 12.2 Derive REINFORCE from the probability distribution

We want to raise expected reward, so we want its gradient. The difficulty is
that the model doesn't output a reward. It outputs a random sample, and you
can't differentiate through a coin flip. The way around this is a short identity
that leads to a very plain rule: after sampling an answer, make that answer
more likely in proportion to the reward it earned. That rule is
**REINFORCE**.

For a finite response space, differentiate the expectation:

```math
\nabla_\theta J=\sum_y R(y)\nabla_\theta\pi_\theta(y)
=\sum_y\pi_\theta(y)R(y)\nabla_\theta\log\pi_\theta(y)
=\mathbb{E}[R(y)\nabla_\theta\log\pi_\theta(y)].
```

The identity $`\nabla\pi=\pi\nabla\log\pi`$ replaces a derivative through
discrete sampling with a derivative of the sampled sequence's log-probability.
This score-function family originates in
[Williams' REINFORCE paper](https://doi.org/10.1007/BF00992696).
For autoregressive text, §12.1's factorization turns the sequence score into a
sum of token scores:

```math
\nabla_\theta\log\pi_\theta(y\mid x)
=\sum_t\nabla_\theta\log\pi_\theta(y_t\mid x,y_{\lt t}).
```

Substituting this into the expectation, for terminal reward,

```math
\nabla_\theta J=\mathbb{E}\left[R(y)\sum_t
\nabla_\theta\log\pi_\theta(y_t\mid x,y_{\lt t})\right].
```

A sampled ascent estimate can be implemented by minimizing
$`L=-\mathrm{stopgrad}(R)\sum_t\log\pi_\theta(y_t\mid s_t)`$. The minus
sign matters: the optimizer minimizes a loss while the objective maximizes
reward. Sampled token IDs and verifier rewards are treated as data. This
estimator is noisy but its expectation is the desired gradient under the stated
sampling assumptions. It is not backpropagation through an argmax decision.

Do not confuse teacher forcing during likelihood scoring with using a demonstration
target. Rollouts are generated by the policy; then their sampled tokens are
fed back to compute likelihoods and gradients. A demonstration-based SFT update
instead scores externally selected targets. Both use log-probabilities, but the
distribution of sequences and their weights differ.

## 12.3 A categorical microscope with an exact answer

REINFORCE is an average over samples, and a single sample can mislead. To see
what the rule is aiming at, we need a case small enough to compute the average
exactly. Three possible answers to `2 + 3 =` are enough.

Temporarily collapse each complete response into one action. This is a
single-decision, or bandit, version of the problem; no claim about a model's
language understanding follows from it. Use the following **illustrative**
reward rule: correct arithmetic earns one point, and obeying the bare-integer
format earns a second point only when the answer is correct.

| Action | Complete response | Probability $`p_i`$ | Fixed reward $`R_i`$ | Contribution $`p_iR_i`$ |
|---|---|---:|---:|---:|
| 0 | `4` | 0.5 | 0 | 0 |
| 1 | `The answer is 5` | 0.3 | 1 | 0.3 |
| 2 | `5` | 0.2 | 2 | 0.4 |

The mean reward is $`J=0.7`$. The numbers specify a distribution; they are not
observed frequencies from a trained checkpoint. Logits $`z_i=\log p_i`$ produce
exactly these probabilities under softmax.

**Reader prediction.** The verbose answer earns positive reward. Should we
always raise its probability? What if the bare answer's reward changes from
two to four while all probabilities stay fixed?

Use three complete answers with logits $`z_i`$, probabilities $`p_i`$, and fixed
rewards $`R_i`$. Then $`J=\sum_i p_i R_i`$. Since
$`\partial\log p_a/\partial z_i=\mathbf{1}[a=i]-p_i`$, write
$`\partial p_a/\partial z_i=p_a\,\partial\log p_a/\partial z_i`$ and sum over
answers:

```math
\frac{\partial J}{\partial z_i}=\sum_a R_a\,p_a(\mathbf{1}[a=i]-p_i)
=p_iR_i-p_i\sum_a p_aR_a=p_iR_i-p_iJ.
```

Factoring out $`p_i`$ gives

```math
\frac{\partial J}{\partial z_i}=p_i(R_i-J).
```

For our distribution this is

```math
\nabla_zJ=[0.5(0-0.7),\;0.3(1-0.7),\;0.2(2-0.7)]
=[-0.35,0.09,0.26].
```

Gradient ascent with learning rate 0.1 adds $`[-0.035,0.009,0.026]`$ to the
logits. The coordinates sum to zero, reflecting softmax's insensitivity to
a common logit shift. They are logit changes, not probability changes; we
must renormalize to obtain the next distribution. A minimized actor loss
has the opposite gradient, and gradient descent makes the same update.

Increasing the last reward to four gives $`J=1.1`$ and gradient
$`[-0.55,-0.03,0.58]`$. Now the reward-one response is below average: its logit
receives negative pressure. After the stated learning-rate-0.1 update and
softmax normalization, however, its probability still rises from 0.3 to
approximately 0.303876, compared with 0.305528 under the original rewards.
The first answer's logit falls more sharply, so the verbose answer gains
probability despite its own falling logit. Thus positive reward alone does
not determine either logit pressure or the normalized probability change.
Adding five to **every** reward instead leaves
the original gradient unchanged. The differences from mean reward, rather
than an absolute zero point, determine this exact update.

The correction depends on relative reward, not just whether reward is positive.
An answer receiving reward one can lose probability if the policy's expected
reward is two. At equal rewards every logit gradient is zero: no sampled outcome
distinguishes the actions. Adding a constant to every reward leaves the exact
gradient unchanged because $`R_i-J`$ is unchanged.

Enumerate the three possible sampled gradients
$`g_a=R_a(e_a-p)`$, weight them by $`p_a`$, and compare their mean with both the
formula and automatic differentiation of $`J`$. This three-way check catches sign,
normalization, and detachment errors. The notebook also plots individual sample
vectors: a single update can point away from the expected direction even when
the estimator is unbiased.

For example, sampling action 1 alone gives
$`R_1(e_1-p)=[-0.5,0.7,-0.2]`$. This sample suppresses action 2 even though the
exact expected gradient favors it. Sampling action 2 gives $`[-1,-0.6,1.6]`$;
sampling action 0 gives zero. Weighting these three vectors by 0.3, 0.2 and
0.5 recovers $`[-0.35,0.09,0.26]`$. The disagreement is sampling noise, not
an autograd error.

The same operations are available in
[`exact_gradient` and `estimator_moments`](../../src/dongxi_llms/policy_gradient_lab.py).
The [Day 19 companion](../../notebooks/day-19/01_exact_reinforce_gradient.ipynb)
uses another declared three-action fixture and compares enumeration with
automatic differentiation. Change its reward vector before changing the
learning rate: does the expected direction change as you predicted?

## 12.4 A baseline changes variance, not the expected gradient

Section 12.3 ended with a problem: single samples point in noisy directions.
Part of the noise comes from raw reward having no sense of "expected". A
reward of one might be good news or bad news, depending on what this prompt
usually earns. A good teacher reasons the same way. A B from a student who
usually gets Cs deserves praise. The same B from a student who usually gets As
is a warning sign. Comparing with what was expected is the idea of a
**baseline**.

Subtract a value $`b(x)`$ independent of the sampled action. The return-minus-
baseline difference is an **advantage** estimate: here it is $`R-b`$, a comparison
with what this prompt was expected to achieve. The baseline part contributes
nothing on average, because

```math
b\sum_y\pi(y)\nabla\log\pi(y)=b\sum_y\nabla\pi(y)
=b\,\nabla\sum_y\pi(y)=b\,\nabla1=0.
```

Then

```math
\mathbb{E}[(R-b)\nabla\log\pi]
=\nabla J-b\sum_y\nabla\pi(y)=\nabla J.
```

The probability mass sums to one, so its derivative is zero. For token-level
estimators a baseline may depend on the pre-action state $`s_t`$, but not on the
current sampled action conditional on that state. A learned value model should
be detached in the policy loss; train it with its own prediction objective.
Action independence is a statistical condition, while detachment is an autograd
condition. One does not replace the other.

Not every baseline reduces variance. In the categorical example, the covariance
trace is $`\mathbb{E}[\|g-\mathbb{E}g\|^2]`$. Minimizing its baseline-dependent
part gives

```math
b^*=\frac{\mathbb{E}[R\|\nabla\log\pi\|^2]}{
\mathbb{E}[\|\nabla\log\pi\|^2]}.
```

In detail, write $`s=\nabla\log\pi`$. The mean gradient is independent of
$`b`$, so only $`\mathbb E[(R-b)^2\|s\|^2]`$ matters to this minimization.
Its derivative is $`2b\mathbb E[\|s\|^2]-2\mathbb E[R\|s\|^2]`$.
Setting it to zero gives the displayed optimum when the denominator is positive.

Expected reward is a useful baseline, but it need not equal this variance-optimal
scalar. A wildly inaccurate yet action-independent baseline preserves expectation
and can increase variance dramatically. The lab enumerates exact means and
variances for several choices; it avoids mistaking one noisy batch for a theorem
about variance reduction.

### Keep the gradient, change the noise

Return to $`p=[0.5,0.3,0.2]`$ and $`R=[0,1,2]`$. With baseline $`b=J=0.7`$, the
three possible sampled ascent vectors become

| Sampled action | $`R_a-b`$ | $`(R_a-b)(e_a-p)`$ |
|---|---:|---|
| 0 | −0.7 | $`[-0.35,0.21,0.14]`$ |
| 1 | 0.3 | $`[-0.15,0.21,-0.06]`$ |
| 2 | 1.3 | $`[-0.65,-0.39,1.04]`$ |

Even a zero-reward response now supplies a useful comparison: it was worse
than this prompt's expected outcome. The probability-weighted mean remains
$`[-0.35,0.09,0.26]`$. Exact enumeration gives these covariance traces:

| Detached scalar baseline | Mean ascent gradient | Covariance trace |
|---:|---|---:|
| 0 | $`[-0.35,0.09,0.26]`$ | 0.8198 |
| 0.7 | $`[-0.35,0.09,0.26]`$ | 0.2472 |
| 5 | $`[-0.35,0.09,0.26]`$ | 10.0598 |

The score-vector squared norms here are $`[0.38,0.78,0.98]`$, so the scalar
optimum is $`b^*=0.626/0.62\approx1.009677`$, rather than 0.7. Expected reward
is a natural prediction target; the minimum-variance scalar additionally
weights outcomes by their gradient magnitudes.

![Exact sampled-gradient covariance trace as the detached baseline changes](../../notebooks/figures/chapter-12/day-20-01_baselines_and_rloo-01.png)

The saved [baseline notebook](../../notebooks/day-20/01_baselines_and_rloo.ipynb)
plots the same calculation for its original logits and rewards, not the
numbers in our table. Read the horizontal coordinate as a baseline choice
and the vertical coordinate as estimator noise, not task performance. Moving
far from its minimum makes the noise worse while leaving the expected
gradient unchanged. No optimizer tuning or model training is represented.

An action-dependent “baseline” such as $`b=R`$ makes every advantage zero and
erases learning. More subtly, the mean reward of a batch includes the sample
whose gradient it weights. That dependence changes expectation unless corrected
or otherwise justified.


<!-- chapter-gif: policy-baseline -->
**Predict:** Would using each action's own reward as its baseline preserve the gradient?

![An action-independent detached baseline changes the three sampled-gradient vectors and their covariance while leaving their exact weighted mean unchanged.](../../visuals/chapters/rendered/policy-baseline.gif)

*An action-independent detached baseline changes the three sampled-gradient vectors and their covariance while leaving their exact weighted mean unchanged.*

[Try the notebook](../../notebooks/day-20/01_baselines_and_rloo.ipynb) · [View a still image](../../visuals/chapters/rendered/policy-baseline.png)
<!-- /chapter-gif: policy-baseline -->

## 12.5 RLOO and the self-inclusion factor

A baseline needs an estimate of what the prompt usually earns, and we rarely
know it. A cheap idea is to sample several answers to the same prompt and
compare each one with the others. Think of judging a runner by the average time
of the *other* runners in the same heat. If you include the runner's own time
in that average, you partly compare the runner with themselves. That small
self-comparison has a measurable cost.

The estimator below is **REINFORCE leave-one-out (RLOO)**. For one prompt sample $`G\ge2`$ independent completions. An advantage estimate
compares a realized return with a pre-action baseline: return minus baseline.
For these complete responses, return is the fixed terminal reward, and the
other responses provide the baseline. Define

```math
b_{-i}=\frac{1}{G-1}\sum_{j\ne i}R_j,
\qquad \hat A_i=R_i-b_{-i}.
```

For response $`i`$, write the estimator as $`\hat A_i=R_i-b_{-i}`$.
The same scalar may weight every token in that response through the sequence
log-probability sum. A state-dependent token estimator $`\hat A_{i,t}`$, including
GAE below, is generally different; broadcasting a response advantage does not
turn it into a learned token-level return estimate.

Other samples are independent of completion $`i`$ conditional on the prompt, so
the leave-one-out baseline is action-independent for that sample. Averaging
$`\hat A_i\nabla\log\pi(y_i)`$ preserves the policy gradient. This is the estimator
studied in [the RLOO paper](https://aclanthology.org/2024.acl-long.662/).
It saves a separate value model but requires multiple sampled completions and
can still have high variance.

Compare against subtracting the inclusive group mean $`\bar R`$. The group sum
is $`R_i+(G-1)b_{-i}`$, so

```math
R_i-\bar R=R_i-\frac{R_i+(G-1)b_{-i}}{G}
=\frac{(G-1)R_i-(G-1)b_{-i}}{G}.
```

Factoring out $`G-1`$ gives

```math
R_i-\bar R=\frac{G-1}{G}(R_i-b_{-i}).
```

For iid samples, the mean-gradient expectation is therefore scaled by
$`(G-1)/G`$. Multiplying by $`G/(G-1)`$ recovers the leave-one-out estimator. This
is a precise finite-sample effect. Standardizing by a random sample standard
deviation introduces another dependence and changes the estimator again; do
not call it unbiased merely because the centered group sums to zero.

To see where the factor comes from, let $`s_i=\nabla\log\pi(y_i)`$.
In $`\mathbb E[\bar R s_i]`$, the own-reward term contributes
$`\nabla J/G`$. Each other-reward term contributes zero because it is independent
of $`s_i`$ and $`\mathbb E[s_i]=0`$:

```math
\mathbb E[\bar R s_i]=\frac1G\mathbb E[R_is_i]
+\frac1G\sum_{j\ne i}\mathbb E[R_j]\,\mathbb E[s_i]
=\frac1G\nabla J+0.
```

Thus
$`\mathbb E[(R_i-\bar R)s_i]=(1-1/G)\nabla J`$.
For $`G=4`$ our running example's inclusive-centered mean gradient is
$`[-0.2625,0.0675,0.195]`$; RLOO restores $`[-0.35,0.09,0.26]`$ in expectation.
This statement concerns all iid groups, not one observed group's usefulness.

Independence needs actual independent sampling conditional on the prompt.
Duplicating one completion, imposing a coupled diversity rule, or accidentally
resetting the random generator for every sample violates the proof. Rewards
from different prompts should not be pooled as though they shared the same
conditional value. Chapter 13 will use grouped relative advantages, while
keeping these distinctions visible.

## 12.6 Old policies, references, and importance ratios

Sampling is expensive, so we'd like to use each batch of answers for more than
one update. But after the first update, the answers came from a policy that no
longer exists. Pollsters face the same problem. If a survey reached too many
older voters, they don't throw it away. They give each older respondent less
weight and each younger one more, so the sample stands in for the population
they care about. **Importance weighting** does this for policies.

An **old policy** $`\pi_{\mathrm{old}}`$ generated a rollout batch. A reference policy
$`\pi_{\mathrm{ref}}`$ anchors long-term behavioral movement. These may start at
the same checkpoint but play different roles. The old policy usually updates
when a new batch is collected; the reference often remains fixed.

If data come from an old distribution, the exact full-trajectory importance
ratio is

```math
\rho(y)=\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{old}}(y\mid x)}
=\exp\left(\sum_t[\log\pi_\theta(y_t\mid s_t)-\log\pi_{\mathrm{old}}(y_t\mid s_t)]\right).
```

With support coverage, weighting an old-policy expectation by this ratio
recovers a current-policy expectation:

```math
\mathbb E_{y\sim\pi_{\mathrm{old}}}[\rho(y)f(y)]
=\sum_y\pi_{\mathrm{old}}(y\mid x)\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{old}}(y\mid x)}f(y)
=\mathbb E_{y\sim\pi_\theta}[f(y)].
```

The cancellation needs $`\pi_{\mathrm{old}}(y\mid x)>0`$ wherever
$`\pi_\theta(y\mid x)>0`$; that is what support coverage means here. Long sequence ratios can have enormous
variance because many small token changes multiply. Token-level ratios
$`\rho_t`$ used in PPO provide a local surrogate at states visited by the old
policy. They are not the same as exact full-sequence importance correction when
the state distribution changes.

Stored old log-probabilities and advantages must stay detached during multiple
optimization epochs. Replacing them with freshly recomputed current values
changes the objective. Aggressive reuse makes the old state distribution less
representative, even if a local clipping rule seems quiet.

### What changes when we reuse the answers?

Collection was **on-policy** when our three answers came from the distribution
being differentiated. After an update, the saved answers come from an earlier
policy: their reuse is **off-policy** relative to the new one. Keep the old
probabilities $`[0.5,0.3,0.2]`$ and let the new probabilities be $`[0.4,0.3,0.3]`$.

| Action | Old probability | Current probability | Current/old ratio | Old probability × ratio × reward |
|---|---:|---:|---:|---:|
| 0 | 0.5 | 0.4 | 0.8 | 0 |
| 1 | 0.3 | 0.3 | 1.0 | 0.3 |
| 2 | 0.2 | 0.3 | 1.5 | 0.6 |

The weighted old-policy expectation is 0.9, exactly the current expected
reward. Without the ratios it remains 0.7. In this one-state example every
current action has old-policy support, so the correction is exact. It cannot
create evidence about an action the collector never sampled.

For a text trajectory, four token ratios of 1.1 already multiply to 1.4641.
The [PPO/KL companion](../../notebooks/day-20/02_ppo_clipping_and_kl.ipynb)
checks this product. More positions can magnify variation in trajectory
weights. PPO's next step limits a local sampled incentive, rather than
pretending that the growing product is always a useful estimator.

The [Day 20 probability-accounting notebook](../../notebooks/day-20/03_behavior_probabilities_and_support.ipynb)
turns the coverage assumption into an exact counterexample. It keeps raw model,
temperature/filter-transformed collector and declared update target separate.
Using raw likelihood as the denominator after transformed collection changes both
the expected reward and its gradient. Conditional fixed-support repair is a
different objective, not a universal fix for a full-support target. Chapter 14
§14.3 develops the measured example and its boundary masks.

## 12.7 What PPO clipping actually clips

Section 12.6 ended with ratios that can grow large when we reuse samples. A
large ratio means the policy has already moved far from the one that collected
the data, so the old samples are a poor guide. **Proximal policy optimization
(PPO)** responds with a simple rule. Credit for moving in a good direction
stops growing once the policy has moved far enough. Moving in a bad direction
still counts against it. The clipping below makes that rule exact.

For one response, write $`\hat A_t=\hat A_{i,t}`$ and suppress its index $`i`$.
The clipped per-token surrogate is

```math
L_{\text{clip}}(\theta)=\mathbb{E}\left[
\min\left(\rho_t \hat A_t,
\mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t\right)\right].
```

Maximize this expression, or minimize its negative. For positive advantage,
increasing the ratio beyond $`1+\epsilon`$ gains no more surrogate benefit. For
negative advantage, decreasing it below $`1-\epsilon`$ gains no more benefit.
Movement in a harmful direction is still penalized. The clipping is asymmetric
with respect to advantage sign; it does not simply clamp every gradient to a
fixed interval. The objective and multiple-epoch rollout reuse are introduced
in [the PPO paper](https://arxiv.org/abs/1707.06347).

**Reader prediction.** At $`\epsilon=0.2`$, is every ratio outside
$`[0.8,1.2]`$ ignored? Evaluate both advantage signs before inspecting the table.

| Advantage | Ratio | Unclipped product | Clipped surrogate | Slope with respect to ratio |
|---:|---:|---:|---:|---:|
| +1 | 1.3 | 1.3 | 1.2 | 0 |
| +1 | 0.7 | 0.7 | 0.7 | +1 |
| −1 | 1.3 | −1.3 | −1.3 | −1 |
| −1 | 0.7 | −0.7 | −0.8 | 0 |

Beneficial movement becomes flat beyond the appropriate boundary; harmful
movement remains visible. A blanket clamp would miss the second and third
rows. The value 0.2 is a declared lab choice, not a constant derived from
the reward problem or a universal stable-update threshold.

![PPO surrogate versus probability ratio for positive and negative advantages](../../notebooks/figures/chapter-12/day-20-02_ppo_clipping_and_kl-01.png)

The [plot's runnable source](../../notebooks/day-20/02_ppo_clipping_and_kl.ipynb)
uses advantages +1 and −1. Compare each solid curve with its unclipped dashed
line; the flat region switches sides with the sign. These are objective
schematics evaluated numerically, not measured learning curves.

Our old-policy baseline gives advantages $`[-0.7,0.3,1.3]`$. With the changed
distribution above, the unmodified expected ratio-weighted advantage is
$`0.9-0.7=0.2`$. Clipping reduces it to
$`0.5(-0.56)+0.3(0.3)+0.2(1.56)=0.122`$: the third action's ratio 1.5 receives
no further benefit above 1.2. This is deliberately a modified objective.
At the initial ratios one, the mean advantage and surrogate value are zero,
yet their derivative is the nonzero policy gradient. A loss value alone
cannot tell us whether learning signal exists.

PPO commonly trains a value function to estimate returns and uses advantages
derived from it. Value error, reward shaping, terminal handling, and padding
can all affect the policy update. Our categorical comparison uses the exact
pre-rollout expected reward as a detached baseline, making critic error zero
so the clipping mechanism can be isolated. That is an oracle analysis control,
not a claim about how a neural critic performs.

Clipping is not a hard bound on policy KL. Shared parameters can alter unsampled
actions, repeated updates can move outside the interval, and the average hides
rare changes. Monitor approximate and, where possible, exact KL, entropy, clip
fractions, length, and independent task outcomes. A low clipped loss alone
cannot establish stable learning.


<!-- chapter-gif: ppo-clipping -->
**Predict:** Does this objective impose a hard bound on policy probabilities or KL?

![For positive advantage, the surrogate flattens above the upper ratio boundary; for negative advantage it flattens below the lower boundary. Harmful directions retain a slope.](../../visuals/chapters/rendered/ppo-clipping.gif)

*For positive advantage, the surrogate flattens above the upper ratio boundary; for negative advantage it flattens below the lower boundary. Harmful directions retain a slope.*

[Try the notebook](../../notebooks/day-20/02_ppo_clipping_and_kl.ipynb) · [View a still image](../../visuals/chapters/rendered/ppo-clipping.png)
<!-- /chapter-gif: ppo-clipping -->

## 12.8 KL estimates depend on direction and support

Chapter 11's objective included a KL penalty, and §12.7 said we should monitor
KL because clipping doesn't bound it. For a language model we can't sum over
every response to get KL exactly. We estimate it from the samples we drew.
There are several ways to do this, and they behave differently.

Let $`p`$ be the distribution generating actions and $`q`$ a positive reference on
the same finite support. For $`a\sim p`$, define $`d(a)=\log p(a)-\log q(a)`$.
Then

```math
k_1=d,\qquad k_2=\frac12d^2,\qquad
k_3=\exp(-d)-1+d.
```

$`\mathbb{E}_p[k_1]=\mathrm{KL}(p\Vert q)`$. Individual samples of $`k_1`$ can
be negative although the expectation is nonnegative. $`k_2`$ is nonnegative but
generally biased for KL; its usefulness comes from a local quadratic
approximation near equal distributions. For common full support,
$`\mathbb{E}_p[\exp(-d)]=\sum_a q(a)=1`$. Taking the expectation of each term
of $`k_3`$,

```math
\mathbb{E}_p[k_3]=\sum_a p(a)\frac{q(a)}{p(a)}-1+\mathbb{E}_p[d]
=1-1+\mathrm{KL}(p\Vert q)=\mathrm{KL}(p\Vert q),
```

so $`k_3`$ has the same expectation
as $`k_1`$ and nonnegative individual values because $`u-1-\log u\ge0`$.

This control-variate identity does not guarantee lower variance everywhere.
Rare actions with very large $`q/p`$ can give $`k_3`$ large tails. The notebook
enumerates exact expectations and variances, then perturbs the distributions.
If $`p`$ is zero on some reference mass, the expectation of $`q/p`$ over sampled
support is less than one and the simple identity fails. If $`q`$ is zero where
$`p`$ is positive, forward KL is infinite. Sampling temperature or top-$`k`$ changes
the distribution: log-probabilities must describe the sampler required by the
estimator, not a convenient unmodified model distribution.

Estimator values used for monitoring and a differentiable KL regularizer also
have different gradient contracts. Differentiating a sampled estimate while
treating the sampled distribution as constant may omit score-function terms.
For the finite microscope, compute the exact sum for objective gradients and
use sampled estimators to study measurement variance.

## 12.9 A comparison with declared budgets

We now have four ways to turn sampled rewards into updates: plain REINFORCE, a
baseline, RLOO and PPO. Which works best? That question has no answer until we
say what "equal effort" means. PPO reuses each batch, so it can match the others
in samples drawn while doing more optimizer work.

The CPU task has six arithmetic prompts $`(a,b)`$ and three answer actions;
the correct action is $`(a+b)\bmod3`$. Independent logits per prompt intentionally
remove representation learning. We compare REINFORCE, an exact-value baseline,
RLOO, and PPO using the same prompt set, four samples per prompt, rollout update
count, seed schedule, and exact success metric.

**Reader prediction:** should the four methods finish at the same success
probability when sampled completions match but PPO reuses each rollout three
times? State which comparison would test sampling efficiency and which would
test gradient-work efficiency before revealing the results. PPO uses three gradient epochs
per rollout, so it consumes more optimizer work despite equal sampled tokens.
The report exposes both clocks rather than calling the run compute-matched.

The result in one line: all four methods learn this small task, and PPO ends
highest while using three times the gradient passes.
The [retained CPU report](../../experiments/reports/2026-10-04-preference-policy-cpu.md)
records 160 rollout updates, six contexts and four sampled actions per context:
3,840 sampled completions per arm, with learning rate 1.5 and fixed seeds.

| Arm | Final exact success probability, seeds 1921 / 1922 / 1923 | Gradient passes |
|---|---|---:|
| REINFORCE | 0.980054 / 0.980184 / 0.980117 | 160 |
| Exact-value baseline | 0.980135 / 0.981667 / 0.981941 | 160 |
| RLOO | 0.980318 / 0.981902 / 0.982335 | 160 |
| PPO, three epochs | 0.991820 / 0.991509 / 0.991530 | 480 |

![Measured success trajectories against sampled completions and gradient passes](../../notebooks/figures/chapter-12/day-21-01_policy_algorithm_comparison-01.png)

Read the left panel at matched sampling cost and the right at optimizer work.
Shading is the range of three seeds, not a confidence interval. PPO's higher
endpoint uses more gradient work and an exact pre-rollout value; it supplies
neither a wall-time ranking nor evidence of neural critic quality.

### The finite update in transparent code

The loop below follows [the canonical policy module](../../src/dongxi_llms/policy_gradient_lab.py).
Each row of `logits` is one independent context; columns are the three possible
answers. `correct` contains the verifier's answer action. The surrounding module
initializes the seeded generator and applies this block for the declared updates.

```python
old_logp = logits.detach().log_softmax(-1)
old_p = old_logp.exp()
actions = torch.multinomial(old_p, group, replacement=True, generator=generator)
rewards = (actions == correct[:, None]).to(logits.dtype)
if mode == "rloo":
    advantage = rloo_advantages(rewards)
elif mode in ("baseline", "ppo"):
    advantage = rewards - old_p.gather(1, correct[:, None])
else:
    advantage = rewards
for _ in range(epochs if mode == "ppo" else 1):
    selected = logits.log_softmax(-1).gather(1, actions)
    if mode == "ppo":
        ratio = (selected - old_logp.gather(1, actions)).exp()
        loss = -ppo_surrogate(ratio, advantage).mean()
    else:
        loss = -(selected * advantage).mean()
    (gradient,) = torch.autograd.grad(loss, logits)
    with torch.no_grad():
        logits -= 1.5 * gradient
```

`rloo_advantages` subtracts the mean of the other samples from the same context;
`ppo_surrogate` returns the minimum of `ratio * advantage` and the clipped-ratio
product. The old probabilities and these rewards/advantages have no actor
gradient. The negative sign converts ascent to a minimized loss. Deliberately
recomputing the old probabilities inside each epoch would change this experiment.

This experiment checks estimator behavior and implementation. It has no unseen
arithmetic distribution, no language vocabulary, and no neural generalization
claim. Multiple seeds show sampling variability. A model-scale extension needs
new evidence for rollout memory, throughput, verifier integrity, and retained
capabilities. These operational questions lead into Chapters 13 and 14.

## 12.10 A critic predicts the return before an action

So far the baseline was either known exactly or built from other samples of
the same prompt. Real tasks offer neither luxury at every token. A sports
broadcast shows the idea we need. Before each play, a win-probability display
estimates how likely each team is to win from the current position. After the
play, the change in that estimate tells you how much the play helped. A
**critic** is a model that makes this kind of estimate for a partly written
answer.

The exact-value comparison above deliberately removed representation error.
Now put that difficulty back. A neural critic reads the same pre-action prefix
$`s_t`$ as the actor and predicts $`V_\psi(s_t)`$: the expected discounted future
reward under the current policy. It does not judge the next token alone and
must not see that action when constructing its baseline. Even a prefix with
the correct color can continue into repetition or missing termination.

The critic supplies an estimate before we discover which continuation occurs.
Subtracting that prediction from the realized return gives an **advantage**: how much better
or worse that continuation was than expected at this state. An exact tabular
value, a small value head sharing actor features, or a separate network can
play this role. A critic need not be another policy-sized model; architecture,
shared gradient paths and fitting cost are separate choices. The local
experiment uses separate small backbones to expose those paths.

The critic parameters $`\psi`$ are separate from the learned reward parameters
$`\phi`$. Let $`R_t`$ be the collected reward earned after action $`a_t`$. A frozen
learned score $`r_\phi(x,y)`$ can supply a terminal $`R_t`$, while a verifier can
supply it directly; the return calculation treats either as fixed observed data.
The companion's local per-step reward variables map to $`R_t`$ here.

The **return** is the reward still to come from a state onward. A **discount**
$`\gamma`$, at most one, can make later rewards count a little less; with
$`\gamma=1`$ every future reward counts fully. When
the episode is cut off before it ends, we **bootstrap**: we fill in the
unobserved remainder with the critic's own estimate.
For discount $`\gamma`$ and $`n`$ observed
actions, the bootstrapped return is

```math
G_t=\sum_{k=t}^{n-1}\gamma^{k-t}R_k
+\gamma^{n-t}c_{n-1}V_\psi(s_n).
```

Here $`c_t=0`$ after true termination and $`1`$ otherwise. Sampled EOS ends the
episode: no continuation value follows it. A collector cap stops observation;
unless the task defines that boundary as terminal, $`c_{n-1}=1`$. Bootstrap
estimates unobserved continuation, not a secretly generated EOS or an observed
future reward. Padding contributes no action, reward, target or loss.

With two zero-reward actions, $`\gamma=0.9`$ and bootstrap $`1.2`$, returns are
$`0.972`$ and $`1.08`$. Relabeling the cap as EOS makes both zero. Conversely,
appending a value of nine after actual EOS must not change the completed
return. These checks catch a bug that a finite loss curve can hide.

## 12.11 TD residuals and generalized advantage estimation

The win-probability display gives a natural measure of one play's value: the
estimate after the play, minus the estimate before it. That one-step surprise
is cheap but leans entirely on the critic. Waiting for the final score leans
on no critic but is noisy. Generalized advantage estimation blends the two.

A **temporal-difference (TD) residual** compares the present prediction with one
observed transition and its continuation estimate:

```math
\delta_t=R_t+\gamma c_tV_\psi(s_{t+1})-V_\psi(s_t).
```

**Generalized advantage estimation (GAE)** combines those residuals, giving
nearer ones more weight through a second factor $`\lambda`$. Let $`m_t`$ indicate
a valid action and $`m_n=0`$ immediately beyond observation:

```math
\hat A_t=\delta_t+\gamma\lambda c_t m_{t+1}\hat A_{t+1},
\qquad \hat G_t=\hat A_t+V_\psi(s_t).
```

The last residual includes a nonterminal bootstrap, but no residual is invented
beyond the collected prefix. At $`\lambda=0`$, the target is one-step TD. At
$`\lambda=1`$, intermediate values telescope, leaving $`G_t`$. Only completed
trajectories remove the final estimate and give fully observed Monte Carlo
returns. The [original GAE paper](https://arxiv.org/abs/1506.02438v6) motivates
a bias–variance trade-off, not a guarantee for this text experiment.

An inaccurate critic can distort bootstrapped targets, particularly at states
never observed beyond the cap. Smaller $`\lambda`$ trusts short value predictions
more. Larger $`\lambda`$ uses more observed rewards but cannot recover experience
never collected. “Use GAE” cannot replace checking what terminated and what
was censored.

### Give credit across three positions

Consider a newly calculated, completed three-action trajectory. It earns
$`[0,0,1]`$ and ends with EOS. Use $`\gamma=0.9`$, $`\lambda=0.95`$ and detached
pre-action value predictions $`[0.4,0.5,0.8]`$; terminal continuation value is zero.

| Position $`t`$ | Reward $`R_t`$ | Predicted value $`V_t`$ | TD residual $`\delta_t`$ | GAE $`\hat A_t`$ | Value target $`\hat A_t+V_t`$ |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0.4 | $`0+0.9(0.5)-0.4=0.05`$ | 0.384305 | 0.784305 |
| 1 | 0 | 0.5 | $`0+0.9(0.8)-0.5=0.22`$ | 0.391000 | 0.891000 |
| 2 | 1 | 0.8 | $`1-0.8=0.20`$ | 0.200000 | 1.000000 |

Work backward: $`\hat A_1=0.22+0.855(0.20)=0.391`$ and
$`\hat A_0=0.05+0.855(0.391)=0.384305`$. Thus the final reward influences
earlier actions even though neither earned an immediate reward. The value
predictions decide how surprising each transition is.

**Controlled change.** Set $`\lambda=0`$: the targets are $`[0.45,0.72,1]`$,
using only the next value estimate. Set $`\lambda=1`$: the completed-trajectory
targets become $`[0.81,0.9,1]`$, the observed discounted returns. Our intermediate
choice trades dependence on the imperfect critic against dependence on more
sampled rewards. The discount 0.9 makes that arithmetic visible; a text task
can choose $`\gamma=1`$ to value its eventual terminal reward without discount.
These values are not suggested defaults for every task.

Detach advantages in the actor loss and return targets in the critic loss.
Otherwise the actor can change its supposedly fixed comparison through the
critic computation, or the critic can chase a target that moves through its
own gradient path. This computational separation does not itself establish
that the predictions are accurate. The
[learned-critic companion](../../notebooks/day-20/04_learned_critics_and_frozen_text_rewards.ipynb)
provides runnable return/GAE and EOS/cap controls for that distinction.

## 12.12 Three separate gradient contracts

We now have three models in play: the actor that writes, the critic that
predicts, and the frozen reward that scores. Each should learn only from its own
objective. If gradients leak between them, the actor can game its own
yardstick. This section writes the two losses so that the boundaries are
explicit.

For $`N`$ sampled responses, the actor minimizes a negative clipped token sum per
response; the critic minimizes valid-state mean-square error:

```math
L_\pi=-\frac1N\sum_{i,t}m_{it}
\min\left(\rho_{it}\,\mathrm{stopgrad}(\hat A_{i,t}),
\mathrm{clip}(\rho_{it},1-\epsilon,1+\epsilon)
\,\mathrm{stopgrad}(\hat A_{i,t})\right),
```

```math
L_V=\frac{\sum_{i,t}m_{it}
\left(V_\psi(s_{it})-\mathrm{stopgrad}(\hat G_{i,t})\right)^2}
{\sum_{i,t}m_{it}}.
```

Actor gradients reach current sampled-action likelihoods; old likelihoods and
advantages are data. Critic gradients reach value predictions; targets are
data. The reward is saved, reloaded and frozen: neither loss updates it or
differentiates through sampled IDs. Separate actor/critic causal backbones
make the boundaries directly testable. Shared features would deliberately
receive both objectives and require another analysis.

An exact conditional KL penalty at collected states anchors the initial actor;
the positive coefficient $`\beta`$ weights that penalty separately from $`L_\pi`$.
Collection and current/old/reference scoring use identical finite support.
One fresh actor step per rollout starts at ratios one: clipping has not reached
its flat region. This isolates advantage/critic behavior, not PPO epoch reuse
or exact correction of changing state distributions.

## 12.13 Actual fitted rewards and generated text

Every reward so far was a rule we wrote down. Now we replace it with the kind
of reward Chapter 10 warned about: a model fitted to preference pairs. This
section sets up the test. The next one reports what happened.

Chapter 10 fitted a reward from text pairs. Can a policy exploit what those
pairs did not constrain? Our original task asks for a color. The actor reads
actual prompt word tokens and emits color/style tokens and EOS. The reward
reads printable ASCII prompt/completion characters. Neither model receives an
injected quality feature or reference answer.

Balanced preferences match style; confounded preferences associate correctness
with fancy style. Budgets/source groups match, but information does not.
Actual fitted models are exported and reloaded before policy updates. Training
response strings alone determine score centering/scaling; tanh then bounds the
proxy. Bradley–Terry scores are not success probabilities, and pair margins
do not identify their absolute offset.

Independent quality checks requested color, at most one style word and EOS
within the delivered cap. It never enters rewards, critic targets or checkpoint
selection. Four training, two calibration, two final and two control sources
have distinct actual actor/reward encodings. Fourteen finite terminal paths
define the task: early EOS is sampled; a fourth-position syntax boundary
allows only EOS. Main delivery caps at three, so continuation values belong
to this declared environment, not open language.

## 12.14 The measured experiment retains critic and reward failures

Section 12.13 built a task with a fitted reward and a learned critic. Does the
policy learn the color task? Mostly it doesn't, and the failures are the
lesson: a reward that fits its training pairs almost perfectly can still fail
to teach the policy what the pairs were meant to teach.

Three fixed seeds compare oracle, learned and noisy critics under balanced
reward; confounded reward uses the same learned-critic recipe. Every row
receives forty batches of four prompts and eight paths per prompt. Learned
rows also take one critic step per batch. Oracle/noisy rows enumerate futures
without training a critic. Equal sampled-path budgets do not imply equal
optimizer work, enumeration cost or wall time. No best seed or final-test
checkpoint is selected.

Reward fitting drove pair losses below $`7\times10^{-6}`$, yet color matching
remained unreliable. Balanced oracle rows emitted constant blue and obtained
quality $`0.5`$; bare colors were outside styled reward-training pairs. Two
learned-critic seeds collapsed to capped style prefixes with quality zero;
their high value errors remain in the report. A noisy critic also retains
severe failures. The observations are compatible with reward extrapolation and
bootstrap error, but do not uniquely establish either cause or a universal
ranking of critic methods.

Separate observed rewards at emitted EOS, critic continuation estimates after
a cap, and diagnostic reward scores of delivered prefixes. A positive prefix
score is not earned terminal reward. A cap-four diagnostic supplies the
declared last EOS but cannot repair wrong color or repeated style. All training
paths, frozen controls and withheld responses are in the
[report](../../experiments/reports/2026-10-04-critic-policy.md).
The [new notebook](../../notebooks/day-20/04_learned_critics_and_frozen_text_rewards.ipynb)
traces the difference from return tensors to actual sampled text.

### Measure the gradient component that answers the question

A policy loss can contain task reward, KL pressure and entropy terms. A nonzero total gradient does not identify which term provides it. If all sampled answers in every group receive the same reward, the task's centered advantages vanish even while a reference penalty moves the parameters. Assignment E asks for the task-only gradient norm as well as the total objective's diagnostics.

The [compatible decoder reference](../../notebooks/extensions/cs336/07_posttraining_integration.ipynb) samples answers and the ending action, then reevaluates their likelihood under the declared conditional grammar. Its support and masks therefore remain part of the probability contract. Compare its frozen parent with accepted-response SFT and the reward-based sibling using the same held-out source pairs. Their different losses need not imply different task success.

Reuse this chapter's likelihood-ratio, baseline and support derivations to explain the observed update. The small reference has genuine sampled successes, failures and mixed groups, so it clears a mechanism gate that the earlier all-zero native run did not. Clearing that gate establishes an opportunity for task-directed learning; the independent constant baseline still tests whether the policy acquired the underlying relation.

## Chapter summary

- Earlier chapters learned from fixed, pre-collected responses. Here the model
  learns from its own samples: prefixes are **states**, tokens are **actions**,
  and the model is a **policy** whose complete outputs a judge scores (§12.1).
- **REINFORCE** turns expected reward into a usable gradient: weight the sampled
  response's log-probability gradient by its detached reward. It is unbiased but
  noisy, and it is not backpropagation through sampling (§12.2).
- On three answers the exact logit gradient is $`p_i(R_i-J)`$. What matters is
  reward relative to the policy's average, not whether reward is positive. One
  sample can point the wrong way (§12.3).
- An action-independent **baseline** leaves the expected gradient unchanged and
  changes only its variance, which can go down or up. RLOO builds the baseline
  from the other samples; an inclusive group mean shrinks the expected gradient
  by $`(G-1)/G`$ (§12.4–12.5).
- Reusing samples needs **importance ratios** against the old policy, which is
  different from the fixed reference. PPO clipping removes extra credit for
  beneficial movement but keeps harmful movement penalized, and it is not a hard
  KL bound. Sampled KL estimators differ in sign, bias and tails, and depend on
  support (§12.6–12.8).
- In the CPU comparison, matched sampling did not mean matched optimizer work.
  Report both clocks before ranking methods (§12.9).
- A **critic** predicts the return before an action. TD residuals and GAE turn
  its predictions into advantages, and EOS, caps and padding need different
  masks. Actor, critic and reward each keep separate gradient paths (§12.10–12.12).
- With a fitted text reward, the measured runs kept severe failures despite
  near-zero pair loss. A trained reward is still not a trustworthy definition of
  success, which motivates Chapter 13's verifiable rewards (§12.13–12.14).

## 12.15 Companion route and exercises

Use [Day 19's exact gradient](../../notebooks/day-19/01_exact_reinforce_gradient.ipynb),
[Day 20's baseline microscope](../../notebooks/day-20/01_baselines_and_rloo.ipynb),
[Day 20's PPO/KL laboratory](../../notebooks/day-20/02_ppo_clipping_and_kl.ipynb),
and [Day 21's algorithm comparison](../../notebooks/day-21/01_policy_algorithm_comparison.ipynb).
The [lab guide](../labs/12-language-generation-as-a-policy.md),
[worked solutions](../solutions/12-language-generation-as-a-policy.md), and
[report](../../experiments/reports/2026-10-04-preference-policy-cpu.md) supply
reproduction and evidence boundaries.

1. Derive the score-function gradient and explain why reward is detached.
2. Derive $`p_i(R_i-J)`$ and interpret a positive reward with a negative gradient.
3. Prove the action-independent baseline identity and give a biased counterexample.
4. Derive the variance-optimal scalar baseline; why need it differ from expected reward?
5. Derive the self-inclusion factor for a group mean and explain RLOO's independence condition.
6. Distinguish a rollout's old policy from the fixed reference policy.
7. Explain why multiplying token ratios differs from PPO's local token surrogate.
8. Draw the clipped objective for both advantage signs and explain which directions remain penalized.
9. Prove the $`k_3`$ expectation identity and identify the missing-support failure.
10. Design a fair algorithm comparison when one method reuses samples for several epochs.
11. Connect SFT, DPO, REINFORCE, RLOO, and PPO without claiming identical supervision or objectives.
12. If training reward increases and exact task success decreases, what would you inspect before changing the learning rate?
13. Derive lambda-zero and lambda-one targets including a nonterminal final bootstrap.
14. Why do collector caps and emitted EOS need different return masks?
15. Draw actor, critic and frozen reward gradient paths; what changes with shared features?
16. How can a fitted pair reward mis-score bare or repeated-style policy responses?
17. Compare critics without confusing rollout budget, critic work, prefix scores and quality.

We began with a reward on three complete answers, then separated the expected
gradient from the noisy sampled one. Baselines changed its noise; reuse required
probability accounting; clipping changed its incentive; critics supplied
state-level comparisons whose targets depend on termination. None of those
operations made the reward a trustworthy definition of success.

Chapter 13 now removes the learned critic and compares answers within a prompt
group. The next question is sharp: if every sampled answer is equally wrong,
where could a relative improvement signal come from?
