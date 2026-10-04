# 13. Group-Relative Policy Optimization

Four answers to the same question receive rewards 0, 0, 1, 1. There is no annotated
reasoning trace and no learned critic. Can these comparisons still teach a language
model? Yes: the successful sampled answers can become more probable relative to
the unsuccessful ones. The difficulty is specifying what “relative” means, which
tokens receive that signal, and how far one update may move the policy.

This chapter connects the trajectory mathematics of Chapter 12 to a complete
autoregressive update. You will calculate group advantages, differentiate a
clipped objective, test a verifier, and follow detached rollout tokens back to the
decoder parameters. Days 22–23 provide the notebook route. The CPU implementation
is executable now; the model-scale Qwen experiment is a separate bounded pathway.

## 13.1 A baseline from alternative answers

Let $x$ be a prompt and $y_i=(y_{i1},\ldots,y_{iT_i})$ a response. A behavior
policy $\pi_{\mathrm{old}}$ generates $G$ responses independently conditional on
the same prompt. A verifier or reward model returns $r_i=R(x,y_i)$.
The group supplies a local comparison. Getting reward 1 is an improvement within
a group of mostly failures, but offers no comparison within a group where every
answer succeeds. GRPO replaces a learned value baseline with group statistics;
the original proposal is in [DeepSeekMath](https://arxiv.org/html/2402.03300v3#S4).

We declare the course convention explicitly:

$$
\bar r=\frac1G\sum_{j=1}^{G}r_j,\qquad
s=\sqrt{\frac1G\sum_{j=1}^{G}(r_j-\bar r)^2},\qquad
\hat A_i=\frac{r_i-\bar r}{s+\delta},\quad \delta=10^{-8}.
$$

The implementation uses population standard deviation: Torch's `correction=0`.
Sample standard deviation divides the squared deviations by $G-1$. For a
nonconstant group it is larger by $\sqrt{G/(G-1)}$, so the corresponding normalized
advantages are smaller. At $G=2$ that difference is substantial. An implementation
that changes this choice has changed update scale, even when its code looks
almost identical.

For rewards 0, 0, 1, 1, the mean is 0.5 and population standard deviation 0.5. Ignoring
the tiny epsilon, the advantages are−1, −1, +1, +1. This sign expresses a comparison
inside one prompt's group. It does not say that an answer is globally good or
that a negative-advantage answer is factually false. All-correct groups give zero
relative advantage. All-wrong groups do too. A critic could provide information
across states that this particular relative estimator does not provide.

## 13.2 What normalization changes

Subtracting a baseline and dividing by a standard deviation are different
operations. Centering compares each response to the group's mean. Scaling changes
how strongly groups with different reward spreads contribute. For binary rewards
with empirical success fraction $\hat p$, the population standard deviation is
$\sqrt{\hat p(1-\hat p)}$. A successful answer in a mostly failing group can then
receive a large positive advantage. Across prompts, this is a weighting choice.

The statistics contain the sampled response's own reward. For unnormalized
centering and independent group samples, the expected score-function estimator
has a factor $(G-1)/G$ compared with the estimator using the true expected reward
as baseline. A leave-one-out baseline removes that simple self-inclusion factor.
After standard-deviation normalization, the denominator also depends on all
sampled rewards, so this is no longer merely the same estimator with a fixed
learning-rate rescaling. Chapter 12's baseline identities should not be silently
extended to every normalized group estimator.

If a group is constant, its centered rewards are exactly zero. The implementation
returns exact zero advantages rather than allowing $0/0$. Epsilon protects
near-zero denominators; it cannot invent exploration. Track the fraction of
zero-variance groups. A high fraction may reflect successful saturation, failure
saturation, insufficient sampling, an overly forgiving verifier or an overly
strict one. Its interpretation requires the rewards and representative outputs.

With independent binary success probability $p$, the probability that all $G$
answers are failures is $(1-p)^G$. A larger group may expose an occasional success,
but requires additional decoding. If outputs are effectively duplicates, the
independence approximation exaggerates the benefit. Group size belongs in a
cost-quality comparison, with actual generated token counts recorded.

## 13.3 The objective we will implement

At response position $t$, define state $s_{it}=(x,y_{i,<t})$ and token ratio

$$
\rho_{it}(\theta)=
\exp\left(\log\pi_\theta(y_{it}\mid s_{it})-
\log\pi_{\mathrm{old}}(y_{it}\mid s_{it})\right).
$$

The behavior log probability is recorded at collection time and held fixed.
Our educational objective averages token contributions inside each response,
then averages responses:

$$
J(\theta)=\frac1G\sum_{i=1}^{G}\frac1{T_i}\sum_{t=1}^{T_i}
\left[
\min\left(\rho_{it}\hat A_i,
\mathrm{clip}(\rho_{it},1-\epsilon,1+\epsilon)\hat A_i\right)
-\beta D_{it}(\theta)
\right],\qquad L=-J.
$$

Here $\epsilon=0.2$ in the lab. For several prompts we average this expression
over the prompt batch. The advantages, sampled token IDs, behavior log
probabilities and reference parameters are detached. The current policy logits
and chosen KL term remain differentiable. The optimizer minimizes $L$.

For $\hat A_i>0$, raising a token's ratio improves the surrogate until the upper
clip boundary; beyond it that branch supplies no further positive pressure.
For $\hat A_i<0$, reducing its ratio improves the surrogate until the lower
boundary. An unfavorable movement remains penalized. Clipping is a modification
to a sampled surrogate, not a guarantee that every vocabulary probability stays
inside a range or that the next policy has small KL.

With one fresh rollout batch and one optimizer update, current and old policy
weights initially agree. The ratios equal 1, but their derivatives are nonzero.
Updating weights changes the ratios. Several optimization epochs over the same
rollouts make clipping more active and increase reliance on behavior-policy
correction. Our readable baseline uses one update per freshly collected batch.

## 13.4 KL values and KL gradients

The reference policy $\pi_{\mathrm{ref}}$ is a fixed anchor. It differs from the
behavior snapshot, which is refreshed every update. At an enumerable vocabulary
state, the course computes exact forward categorical KL:

$$
D_{it}=\sum_v p_v\log\frac{p_v}{q_v},\qquad
p_v=\pi_\theta(v\mid s_{it}),\quad q_v=\pi_{\mathrm{ref}}(v\mid s_{it}).
$$

For current logits $z_v$, its derivative is

$$
\frac{\partial D}{\partial z_v}
=p_v\left(\log\frac{p_v}{q_v}-D\right).
$$

This follows by differentiating $p$ through softmax and summing the Jacobian
terms. The expression includes how changing logits changes the distribution
that weights the log ratios. Reference logits are constants. Exact vocabulary
KL avoids sampling noise over candidate tokens at the visited state; it still
averages over states reached by the behavior rollouts rather than every possible
current-policy trajectory.

A common sampled nonnegative estimator is

$$
k_3(v)=\frac{q_v}{p_v}-\log\frac{q_v}{p_v}-1.
$$

When $v$ is sampled from $p$, its expectation is forward KL because
$\sum_v p_v(q_v/p_v)=1$. This value identity does not mean that differentiating
the integrand while treating samples as fixed produces the gradient of the
full expectation. Sampling from an old policy introduces another distribution
change. A framework may deliberately choose a particular surrogate or
importance correction. State that choice; inspect code and tests rather than
equating every reported “KL” value with one universal gradient rule.

## 13.5 Token weighting is part of the algorithm

The response-mean objective gives each response the same total nominal weight.
A response of length 2 allocates that weight across two tokens; one of length 20
allocates it across 20. A global token-mean objective instead gives the longer
response ten times as much total weight, all else equal. Neither reduction is
just a display setting. Both interact with advantage signs, EOS, truncation and
the distribution of lengths.

Use a valid-response mask $m_{it}$ that includes sampled EOS and excludes padding.
Prompt tokens participate in the forward context but receive no direct response
policy-loss term. Their embeddings and earlier states can still receive gradients
through the response computations, as Chapter 3's loss-mask discussion explains.
For response means, divide by each response's valid count; reject empty
responses. Do not average padded zeros as though they were generated actions.

Normalization variants address different biases. [Dr. GRPO's
analysis](https://arxiv.org/html/2503.20783v1) studies reward-spread and length
weighting and proposes changes. [TRL's documentation](https://huggingface.co/docs/trl/main/en/grpo_trainer)
exposes several loss and reward-scaling options. Their existence is a reason to
declare this chapter's formula precisely. It is not evidence that a newly named
variant automatically improves our model, hardware or target task.

## 13.6 The verifier is an executable task definition

“Verifiable” means that a specified program can decide some property. It does
not make the property a complete measure of reasoning. A correct final integer
can accompany an invalid explanation; a valid derivation can be rejected by a
bad parser. Separate answer extraction, canonicalization, task comparison and
reward construction. Retain the original response alongside the extracted answer.

The CPU task has original addition prompts with operands 0..3. The answer must
be one correct numeric token followed by EOS. The text analogue accepts one
ASCII signed integer with surrounding whitespace. It rejects `35 0`, `35.0`,
an embedded answer, Unicode digits, a missing answer and executable-looking text.
It never uses `eval`. This intentionally narrow contract makes failure visible;
it is not a general mathematical-equivalence engine.

Tests must include true positives, ordinary wrong answers, formatting edges,
multiple competing answers, truncation and adversarial strings. A model will be
trained against exactly the behavior of this program. Changing the verifier
changes the task and must change its recorded identity. Freeze training,
development and held-out prompt IDs before collecting rollouts. Procedural data
can leak through identical examples, shared templates or trivial answer coverage;
document each level rather than claiming an exact-ID check proves full isolation.

## 13.7 A complete decoder update

The [source implementation](../../src/dongxi_llms/grpo_lab.py) initializes a
one-layer 16-wide `TinyDecoder` with 12 vocabulary outcomes. It warm-starts on 12
original demonstrations, holds out four operand pairs, freezes a reference and
then executes grouped autoregressive training. This is the same decoder family
constructed in Chapter 5, reduced to make every operation inspectable on CPU.

```python
from dongxi_llms.grpo_lab import decoder_rlvr
result = decoder_rlvr(group_size=4, updates=12, seed=2223)
result['history'][0], result['final']['heldout']
```

Generation occurs under `no_grad`. A response is sampled token by token from a
frozen behavior copy. EOS ends its valid-action mask; later batch padding carries
no policy signal. A training forward pass recomputes current-policy logits using
the same prompt and sampled prefix. The logit at the prompt's last position
predicts the response's first token. Subsequent logits are shifted by one in
exactly the way established in Chapter 3. The reference forward pass supplies
detached distributions at those same states.

Every update records reward, zero-variance fraction, valid tokens, length, gradient
norm, entropy, exact KL, post-update ratio extrema and clipping fraction. A zero
surrogate value at initial ratio 1 can coexist with a nonzero gradient: centered
advantages may sum to zero while their token-specific derivatives do not.
Likewise, a constant-reward group has zero relative policy signal but can still
produce a KL regularization gradient if the current policy has moved from its
reference. AdamW momentum can also move parameters when the current sampled
policy gradient is zero.

## 13.8 Experiments and their reach

The [bounded specification](../../experiments/specs/2026-10-04-grpo-diagnostics-distillation.md)
compares groups 4 and 8 at the same 12-update schedule. The latter consumes more
rollouts, so the report includes token counts and elapsed time rather than
presenting this as an equal-budget algorithm ranking. All initial and final
held-out rows are retained. Four arithmetic prompts cannot establish broad
reasoning generalization. A failed held-out result is useful evidence about the
limits of a tiny demonstration-trained policy, not a reason to hide the report.

The [measured CPU report](../../experiments/reports/2026-10-04-grpo-diagnostics-distillation.md)
retains an instructive negative result. Both arms fit all 12 warm-start training
prompts, while greedy accuracy on the four held-out pairs is zero before and
after the RL stage. The G4 arm consumes 192 valid response tokens; G8 consumes
385, including differing EOS lengths. The objective and gradients execute, but
this setup supplies no observed held-out gain. That is the result to explain,
rather than translating higher sampled reward into a reasoning claim.

For Qwen3-0.6B, use the [Spark experiment contract](../../experiments/specs/day-23-qwen-rlvr.md).
The optional loader accepts an immutable local checkpoint revision, uses the
same loss mechanism, and records full configuration. A ready runner and CPU
tests establish implementation readiness; they do not establish GPU execution,
throughput, numerical safety or mathematical benchmark improvement. Run the
declared smoke mode before a longer experiment and preserve the platform's host
memory reserve.

## 13.9 Deep questions

1. Why does an all-correct group produce no relative policy signal?
2. How does sample standard deviation change the scale of a two-response group?
3. Why is group normalization more than a harmless reward baseline?
4. Can a zero numerical loss have a nonzero derivative in the first update?
5. Which movements does clipping stop for negative advantages?
6. Why does an unbiased KL value estimator not automatically supply its full gradient?
7. How do response means and token means treat short and long responses differently?
8. Can prompt embeddings receive gradients while their direct policy-loss mask is zero?
9. What failures can a strict verifier create, and how would you test them?
10. What does a successful CPU GRPO run justify saying about Qwen training?

[Worked answers](../solutions/13-group-relative-policy-optimization.md) accompany
the [Day 22–23 notebook route](../labs/13-group-relative-policy-optimization.md).
Chapter 14 examines what happens when this exact pipeline faithfully optimizes
the wrong reward or consumes rollouts whose policy identity has changed.
