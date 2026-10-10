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

For an exact finite comparison of equal KL values and unequal frozen-sample
gradients, use [the probability notebook](../../notebooks/day-20/03_behavior_probabilities_and_support.ipynb)
and Chapter14§14.3. The score-function and explicit likelihood-derivative paths
must match the declared regularizer; a scalar estimate alone does not define
what backpropagation should do. The course model adapter deliberately keeps
temperature1/full support while more complicated sampling is studied separately.

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

### 13.5.1 Compare objectives before comparing training runs

Changing a loss and changing its sampled responses at the same time makes an
algorithm comparison hard to interpret. The
[matched-rollout notebook](../../notebooks/day-22/03_objective_weighting_and_filtering.ipynb)
holds responses and initial shared-decoder parameters fixed. Three reductions
and three reward-scaling rules expose objective differences; filtering is a
separate collection experiment.

Write $L=-\sum_{i,t}w_{it}u_{it}$. With $N$ responses, valid-action count $T_i$
and declared generation cap $C$, the coefficients are

$$
w_{it}^{\mathrm{response}}=\frac{m_{it}}{NT_i},\qquad
w_{it}^{\mathrm{token}}=\frac{m_{it}}{\sum_j T_j},\qquad
w_{it}^{\mathrm{fixed}}=\frac{m_{it}}{NC}.
$$

The fixed denominator uses the declared cap, not padded tensor width. At initial
ratio one and away from clipping, the chosen log-probability derivative is

$$
\frac{\partial L}{\partial\ell_{it}}=-w_{it}\hat A_i,
\qquad \ell_{it}=\log\pi_\theta(y_{it}\mid s_{it}).
$$

Changing these coefficients changes the signal reaching shared parameters.
Global-token and fixed-cap derivatives differ by one scalar on a fixed pool.
Response means change individual responses' relative weights, so the parameter
gradient can change direction too. This does not alone predict final generation
length or accuracy.

For reward components $r_{ic}$ with weights $\alpha_c$, normalizing the weighted
total and normalizing components before aggregation are different:

$$
A_i^{\mathrm{total}}=
\frac{\sum_c\alpha_c r_{ic}-\mathrm{mean}_j\sum_c\alpha_c r_{jc}}
{\mathrm{std}_j(\sum_c\alpha_c r_{jc})+\delta},\qquad
A_i^{\mathrm{component}}=\sum_c\alpha_c
\frac{r_{ic}-\mathrm{mean}_j r_{jc}}
{\mathrm{std}_j(r_{jc})+\delta}.
$$

Statistics are within one prompt's group, using population moments. A constant
component contributes zero. A large raw-scale component can dominate total
normalization; component normalization changes that balance. Subsequent weights
still matter: multiplying every $\alpha_c$ by ten multiplies the component-based
advantage by ten. Normalization cannot decide what behavior ought to be rewarded.

The proxy deliberately combines answer-only correctness, one-numeral completion
format and a small verbosity bonus. Independent quality requires the correct
numeral followed by EOS. A long capped answer can earn answer and verbosity
credit while failing the complete task. Save both measurements rather than
defining the shortcut away.

Asymmetric clipping declares lower and upper widths separately. The derivative
is zero for positive advantage above $1+\epsilon_+$, or negative advantage below
$1-\epsilon_-$. Elsewhere it is $-w_{it}\hat A_i\rho_{it}$. At initial ratio one,
symmetric and asymmetric branches agree. A labeled constructed score fixture
exposes clipping boundaries without pretending those ratios were model outputs.

The [three-seed report](../../experiments/reports/2026-10-04-grpo-objective-controls.md)
checks every analytical selected-score derivative against autograd and retains
all actual sampled paths. Response-mean loss is nearly zero at initial ratio one,
but the shared parameter gradient is nonzero. No optimizer step is performed:
this establishes objective differences, not an algorithm ranking or reasoning gain.

### 13.5.2 Filtering has a collection denominator

Discarding constant-quality groups can increase the fraction of retained groups
with a relative signal. Failed collection is not free. Count attempted prompts,
groups, responses and generated actions, including sampled EOS; retain prompts
that never qualify before the retry cap.

All three measured seeds eventually supply five mixed groups of eight responses.
They need nine, nine and seven attempted groups, consuming 144, 142 and 116 valid
actions. Selected actions total only 86, 86 and 89. These successful seeds do not
prove every prompt will qualify: an all-constant test exhausts the three-attempt
cap and selects nothing. Immediate EOS and cap truncation remain recorded events.

The objective matrix still uses the first-attempt pool. Replacing it with filtered
rollouts would change data as well as loss. The questions are motivated by the
pinned [RLHF loss pathway](https://github.com/natolambert/rlhf-book/blob/eecc49e1e1daf1be670d7242eb090240b5bb0f04/code/policy_gradients/loss.py)
and [reasoning objective extensions](https://github.com/rasbt/reasoning-from-scratch/tree/a788466dc85cfe8617b6c0b809ed4ab9084485de/ch07/03_rlvr_grpo_scripts_advanced).
Our original implementation is not a complete named-method reproduction. It uses
a symbolic conditional grammar, not unrestricted language, and does not execute
a Spark campaign.

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

### 13.7.1 A pending rollout is already an experimental observation

Consider a crash after sampling the next group but before applying its gradient.
Reloading policy weights alone and sampling again does not restore that observed
group. Even if the policy is unchanged, a different random draw can change
answers, lengths, rewards, advantages and the gradient. An exact post-collection
restart must use the retained actions, not replace them with more favorable ones.

There are two distinct durable boundaries. At a completed boundary, policy and
Adam state, source cursor, RNG and numerical history all describe the updates
already applied. At a pending boundary, policy and Adam still describe those same
completed updates, but the source cursor and sampling RNG have advanced through
the next full group. That group is an additional retained observation awaiting
one application. Restore it, rescore its actual prefixes under the current
policy and original reference, then apply it once before collecting another group.

The pool includes prompt and response IDs, stop-inclusive valid masks, detached
old selected log probabilities, raw texts, stopping events, rewards and group
advantages. Its behavior-policy version/digest and before/after sampling RNG bind
it to the boundary. The original KL reference is retained separately: replacing
it with the restored trained policy changes the anchor, just as in Chapter11.
Numerical history and counters must distinguish collected work from applied work;
the pending group has incurred decoding cost without a completed optimizer step.

Reading that boundary has its own trust order. Keep an independently retained,
bounded receipt outside the tensor payload. It names the expected payload and
the saved model-work and I/O prefixes. Bind both same physical journals before
inspection, retain every later charge, then inspect/load/check the pool. The
runner's23 work dimensions charge native collection, application and semantic
checks; the shared reader's9 dimensions charge its own byte/tree/tensor and
publication work. Neither allowance can replace the other or authorize a retry.

The saved model-work prefix can precede the save's own semantic validation.
That is not a refund: the external receipt binds exactly the prefix actually
inside the saved state, and reopening the physical journal retains the later
validation suffix. Similarly the I/O prefix precedes the save reservation.
Moving a later completion into the already-written payload would change the
boundary rather than recover it. [Day25 Exercise8](../../notebooks/day-25/02_ragged_kv_cache_and_exact_recovery.ipynb)
compares these phases with actual native tiny-CPU operations and adjacent answers;
its manual publication microscope is not a full runner save schedule.

A crash during generation is not this full-pool guarantee. Retain known partial
attempts and errors, then restart from the last durable boundary under the declared
protocol. Likewise a partial optimizer operation is not a new checkpoint: discard
its poisoned live state and restore the previous valid boundary. Chapter14's
cache lesson explains reusable computation; this lesson explains why the observed
training data and optimizer state must be recovered together. The
[runner-owned recovery plan](../../docs/PRODUCTION_RECOVERY_PLAN.md) separates
actual-loop CPU acceptance from still-gated pretrained/CUDA/BF16 execution.

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

### 13.8.1 Constructive controls: learning can improve without solving the task

The old G4/G8 result stays intact. It begins after fitting all twelve training
demonstrations; a successful warm-start can leave little unsaturated training
signal. To ask whether sampled rewards can change a shared sequence policy, add
a separately identified task whose initial success is low but whose solution is
known. Do not relabel a new easy task as evidence that the old arithmetic task
generalizes.

The [constructive-control notebook](../../notebooks/day-23/03_reasoning_tasks_and_positive_controls.ipynb)
asks a tiny decoder whether the sum of two nonnegative numerals is positive.
Its response is0 or1 followed by EOS. An independently authored integer-arithmetic
oracle defines correctness. The neural input is four symbolic tokens, not the
English sentence: BOS, instruction, first numeral, second numeral. Shared token
embeddings, one causal attention block and an MLP produce the outputs; there is
no per-problem lookup in this decoder and no SFT warm-start.

We deliberately supply a syntax grammar, not the answer. The first response
position permits $C_1=\{0,1\}$; the next permits
$C_2=\{0,1,\mathrm{EOS}\}$. The model samples EOS and receives its loss: termination
is not inserted free. With independently fixed correct answer $a^\ast$, the
complete-path probability is

$$
P_\theta(\text{complete correct}\mid x)=
p_\theta(a^\ast\mid x,C_1)
p_\theta(\mathrm{EOS}\mid x,a^\ast,C_2).
$$

This exact probability is a diagnostic, not the training objective. Actual
rollouts supply binary complete-path rewards and group advantages to the sampled
GRPO update. Temperature is1. Collection and recomputed old/current/reference
likelihoods all renormalize the same allowed logits. The conditional grammar is
part of the objective; using raw-vocabulary likelihoods in its denominator would
repeat Chapter14's sampling mismatch. The existing Qwen baseline remains
temperature-one/full support.

The [predeclared experiment](../../experiments/specs/2026-10-04-reasoning-controls.md)
uses seeds2301,2302,2303,120 updates each, four prompts with sixteen responses
per prompt and a two-token cap. The [measured report](../../experiments/reports/2026-10-04-reasoning-controls.md)
retains every seed and evaluates the last scheduled checkpoint. Mean complete-path
probability improves in all three runs; that is genuine bounded sampled sequence
learning. It does not mean every learnable example improves. Two seeds still
fail the sole zero-sum training example while fitting the three positive-sum
examples. Learned stopping and a majority-answer shortcut can improve the mean
together. Inspect answer-only probabilities as well as the EOS-dependent score.

The frozen paired policy retains the same initial weights and decoding seeds.
A separate exact-expectation lookup arm fits only observed problem keys; unseen
keys keep their initial probabilities. It is a transparent counterexample, not
a substitute for sequence-model learning. An oracle establishes feasibility,
while constant0/1 baselines expose imbalance: always returning1 plus EOS gets
three of four training items and all positive-sum unseen-source items correct.
High held-out percentages on that slice alone cannot establish the arithmetic
rule has been learned.

Source groups and underlying problems never cross splits. Report new-source,
new-source-plus-template and wholly new-family scores separately. New-template
rows also use new sources, so template effects are not isolated. Odd-sum tasks
are a withheld family; a constant answer can receive half-credit there by chance.
The report preserves those failures rather than treating a tiny average as
evidence of reasoning transfer.

### 13.8.2 From symbolic controls to a real reasoning protocol

The companion [original math panel](../../fixtures/reasoning-controls/README.md)
contains arithmetic, linear equations and two-step quantity problems. Its
references are independently checked with exact arithmetic and the bounded
grader. Oracle, wrong-answer, invalid-answer and truncation probes exercise that
evaluation contract. These authored strings are not generations from the tiny
decoder, which does not parse the English panel. They define broader questions
for a separately approved pretrained experiment.

A model's weights, prompt interface and thinking behavior are distinct choices:

| Intervention | Weight identity | What must be recorded |
|---|---|---|
| Base to instruct | Different actual checkpoints | Parent/data history and both weight hashes |
| Raw to chat on fixed weights | Unchanged | Actual rendered tokens and template identity |
| Supported thinking toggle | Same checkpoint | Flag support, template/path change and token cost |
| Output cap32 to128 | Unchanged | Truncation, answer validity, latency and generated tokens |

A base model is not obtained by disabling an instruct model's thinking flag.
Nor does a flag create reasoning capability if that checkpoint/template does not
support it. The planned budget interventions are not promises that32 tokens
solve a task or128 tokens produce faithful reasoning. Our synthetic two-token
cap has a different token vocabulary and purpose entirely.

Even a correct numeral can fail to complete a trajectory. Cap1 preserves a
possibly correct answer while preventing EOS, so its strict reward is zero.
Removing the grammar changes support and can change invalid/truncation rates;
it is a separate answer-producing system, not an unchanged evaluation recipe.
Keep every row in its denominator and report answer correctness, format validity
and completed-path success separately. Exact final-answer verification does not
establish that a visible rationale is valid or caused the answer. The model-scale
baseline remains separately gated; CPU controls do not execute it by implication.

### 13.8.3 The first native row: grading is an interface

The [actual native evidence report](../../experiments/reports/2026-10-05-native-reasoning-rlvr-evidence.md)
retains the first accepted pretrained baseline: Qwen3-0.6B Instruct, native chat,
thinking off, cap32. Four full-support sampled seeds each answer the original
twenty prompts; a separate greedy diagnostic answers the same twenty. The whole
supervised child exits0, all100 records are retained and coverage is complete.
That establishes execution, not useful reasoning or completed RLVR training.
Base/raw/cap32 and Base/raw/cap128 now separately complete all100 original
records each with exit0. Instruct/thinking-off/cap128 now completes another100
records with exit0. Base/custom-chat at both caps32 and128 instead fails
with40 retained records and60 missing each. Both thinking-on conditions now also
complete100 records and exit0; every one of those responses caps and is graded
unsupported. G4 recovery run01 now fails its observed-descendant cleanup gate
despite three native exit codes0; its CPU comparison was not reached and dependent
pilot was not launched at that failed boundary. Matched post-training outcomes
are now measured separately below.
The separately declared fresh G4 recovery02 and original G8 recovery01 are now accepted, without changing
failed01 or implying completed pilot training or improved reasoning.
G4 pilot01 subsequently completes native sixteen iterations/export but fails
terminal logging acknowledgment. Its export is now separately consistency-closed
without changing that failed verdict. G8 pilot16 is accepted; both per-cap
common-panel comparisons are now complete.

Read the result along several axes rather than as one success flag:

| Actual policy/interface | Decoding | Controlled held-out attempts | Declared graded-correct | Natural stop | Capped |
|---|---|---:|---:|---:|---:|
| Instruct/off/cap32 | Four sampled seeds | 44 | 0/44 | 3/44 | 41/44 |
| Instruct/off/cap32 | Separate greedy | 11 | 0/11 | 1/11 | 10/11 |
| Instruct/off/cap128 | Four sampled seeds | 44 | 13/44 | 35/44 | 9/44 |
| Instruct/off/cap128 | Separate greedy | 11 | 7/11 | 11/11 | 0/11 |
| Instruct/on/cap32 | Four sampled seeds | 44 | 0/44 | 0/44 | 44/44 |
| Instruct/on/cap32 | Separate greedy | 11 | 0/11 | 0/11 | 11/11 |
| Instruct/on/cap128 | Four sampled seeds | 44 | 0/44 | 0/44 | 44/44 |
| Instruct/on/cap128 | Separate greedy | 11 | 0/11 | 0/11 | 11/11 |
| Base/raw/cap32 | Four sampled seeds | 44 | 0/44 | 1/44 | 43/44 |
| Base/raw/cap32 | Separate greedy | 11 | 0/11 | 0/11 | 11/11 |
| Base/raw/cap128 | Four sampled seeds | 44 | 2/44 | 9/44 | 35/44 |
| Base/raw/cap128 | Separate greedy | 11 | 1/11 | 2/11 | 9/11 |

These are repeated draws from eleven problems, not fifty-five independent
questions. Nine other prompts remain development/seen diagnostics, including
math9's RLVR-fixture-overlap problem despite its original held-out-source label.
The nine diagnostics are eight original fixture-train items math1–math8 plus
that math9 problem. Only math2 and math9 have true RLVR-overlap flags. This is a
panel-exclusion rule, not evidence that assistant full400/chosen/DPO policies
trained on the math rows or that upstream checkpoint contamination is proven.
Do not pool sampled and greedy policies, or count availability among four
candidates as the accuracy of an unimplemented best-of-four selector.

Base/raw versus Instruct/native-chat changes checkpoint weights and serialization
together; these descriptive differences cannot isolate instruction training's
effect. The two Base/raw caps hold those identities fixed while changing the
budget, not proving that a longer answer always improves a prompt or is faithful.
The two correct held-out samples at cap128 answer different problems from the
single correct greedy response. Their any-candidate availability is2/11 problems,
not the accuracy of an oracle selector the experiment never deployed. All three
held-out successes naturally stop and use supported boxed extraction. Two other
correct cap128 responses are seen/overlap diagnostics and stay out of the headline.

The Instruct thinking-off32→128 comparison holds checkpoint byte inventory,
native interface, source, original items and decoding settings fixed while
changing the output cap. Its13/44 sampled correct answers include two capped
trajectories with supported boxes: sample1009/math19 and sample1019/math18.
The parser can mark their answers correct while stopping flags still say
`max_tokens`; that is not the strict EOS-dependent complete-path reward from
the symbolic experiment. All seven correct greedy answers naturally stop.
Any-candidate availability among four samples is8/11 controlled problems, not
the7/11 greedy-delivered result or an implemented oracle selector. This bounded
cap intervention improves the observed declared grades, not the model's weights,
and does not establish a faithful rationale or monotonic per-prompt benefit.

Thinking-on supplies a different bounded failure from a decode error. Both caps
pass execution and retain all100 records without adapter errors; all are
format-valid under `any`, yet unsupported and capped. At128, sample1009/math10
generates an open `<think>` discussion of5+6 without completing an answer. No
thinking-on response at either budget closes `</think>` or supplies a boxed answer.
The frozen extraction retains the whole output, not a rescued internal numeral.
That is not proof every visible mathematical statement is wrong, that thinking
is universally harmful, or that a longer unrun horizon would fail.

The supported flag uses the same Instruct checkpoint, tokenizer and native
template but changes every rendered assistant prefix. Thinking-off closes an
empty thinking region in the prompt; thinking-on lets the model generate its
thinking text. Within on32→128, all100 attempt seeds/prompt IDs match and the
short action IDs are exact prefixes of the longer responses. This measured
pairing is not a promise of GPU determinism across new runs or devices. At fixed
cap, off/on changes serialization/output path, not weights; cap32/128 changes
budget. On32 records3200 actions/124960 full-prefix positions; on128 records
12800/1114240, without a natural stop or supported answer. These uncached costs
are not FLOPs or serving-speed results. A supported flag, visible trace, parser
success and faithful reasoning remain different claims.

The negative declared score has an important interface boundary. On
sample1009/math10, the prompt `Compute 5 + 6.` yields scoring text `5 + 6 = 11`
and a natural turn stop. The frozen `boxed_or_final` extraction finds no supported
explicit marker and falls back to the whole equation. The exact-rational parser
marks it `UNSUPPORTED`, not a supported answer proven wrong. Its accepted grade
stays negative. This qualitative example explains what the **declared whole-output
grading** rejected; it is not a secretly rescored correct headline or proof that
all visible mathematics is wrong.

The item format policy is `any`, so all100 observed non-error Instruct/off32
responses are format-valid even though99 are `UNSUPPORTED` and one is `INVALID`.
Base/raw32 likewise has no declared correct grade, while Base/raw128 has a few
supported correct boxes amid92 unsupported and3 invalid responses. Format validity,
parser support, mathematical correctness, natural termination and truncation
are separate predicates. Natural stopping does not cure an unsupported answer;
a capped response can contain a plausible numeral without completing. Requiring
a bare numeral or adopting different extraction could be a useful new declared
intervention. Neither changes this run retrospectively. Preserve the raw text,
grader version, unsupported rows and denominators.

Actual baseline generation loads BF16 weights with eager CUDA full-prefix
forwards. This is not an optimized inference benchmark, a MATH-kernel experiment
or the FP32-plus-autocast training recipe from another stage. Base/instruct
weights, raw/custom/native chat, supported thinking flags and caps32/128 remain
distinct axes. These differently serialized checkpoints and two raw-output caps
cannot isolate or broadly rank those interventions or establish
rationale faithfulness.

The failed Base/custom-chat32 and128 invocations supply a separate lesson. At
sample1019/math6 its24-action trajectory ends in ID151768, which has no tokenizer
mapping. The decode-error record preserves that selected action; it is not a
mathematical `UNSUPPORTED` grade, a natural stop or a cap. The actual whole child
exits1 in each condition:56.54799546097638 seconds for32 and132.66686874401057
for128, with no external deadline stop or cleanup error. Each retains40 records
from two cells, while three later cells have60 missing responses. Cap128's partial
records retain8 natural stops,31 caps and one error, plus4491 actions and377459
full-prefix positions; a larger cap did not repair the unmapped action.
Do not present those missing draws as wrong answers, remove the error row, filter
the vocabulary or resample a decodable continuation and call it the same recipe.
Failed work remains spent as other independent predeclared conditions proceed
under the unchanged recipe. All eight initial conditions were attempted: six
accepted complete invocations retain600 records and two failed partial invocations
retain80, including two errors. Their680/800 coverage leaves120 missing; this
does not mean600 mathematically successful answers or full800-slot
coverage or a complete comparison of missing draws. No new requirement to obtain
successful Base/chat runs is invented; their literal failed outcomes remain.
A failed partial row differs from both an accepted negative row and a pending row.

For later RLVR evidence, also separate zero-variance groups from optimizer
applications. A constant-reward group has zero relative policy advantages; the
native loop's fixed path can still rescore, include exact KL, backpropagate and
invoke AdamW. A completed cursor is not a count of groups with positive learning
signal, and optimizer execution is not evidence of held-out improvement. Read the
actual history, committed state and collected/applied work together. Pending
recovery must consume its already spent group without resampling. Completed,
explicitly bound pilot16 exports and the same original twenty-item, per-cap evaluation can support
the later matched comparison; the runner's four diagnostic items cannot stand
in for it.

G4 recovery run01 makes both distinctions concrete without supplying an accepted
replay. Source2 and both resumes have actual native exit0, but the pending
supervisor fails with `Observed descendant cleanup timed out`; the adapter stops
before reaching the adapter's three-way CPU state comparison. Completed helper cleanup and later
process absence do not erase that separate failed gate. Every child retains its
original600-second/25GiB envelope; journals, snapshots and reports remain spent
evidence, not authorization for its dependent pilot.

Its shared work journal measures four optimizer applications but only three
collections: source2, completed resume1 and pending resume1 applications versus
2/1/0 collections. Pending applies64 already retained response targets without
resampling. Do not sum the three restored numerical cursors of2 as six fresh
updates. Both source groups have zero rewards/advantages, yet measured gradient
norms are5.811452865600586e-7 and0.1796875; pre-update exact KL at update2 is
0.0010778990108519793. The recorded post-update policy identities differ. The
zero task-relative signal is not proof of a skipped AdamW call or a zero numerical
derivative, and that movement is not proof of reward learning or improved
reasoning. These failed-attempt observations do not replace the unrun exact-state
comparison, accepted pilot16 export or common-panel evaluation.

The subsequent [controller continuation](../../experiments/reports/2026-10-05-native-rlvr-controller-runtime-independent.md)
defers numerical imports to keep spawn startup lightweight; it does not change
native learning, supervision, acknowledgment timeouts, work/I/O caps or failed
run01 evidence. Fresh G4 run02 now passes all twelve actual recovery checks,
including exact final policy/reference/optimizer/RNG/loop history and update2
metrics. Its three supervisors complete with native exits0 and empty cleanup
errors, within the original600-second bounds. All three numerical cursors finish
at2 while its shared journal retains four applications and three collections;
pending applies its fixed pool without resampling. This is an accepted two-update
replay, not a positive-reward or held-out-quality result. Failed01's distinct
journals remain spent and failed. Pending02 observes no owned descendant, so this
run does not reproduce the original failed pending-helper circumstance or prove
its historical startup cause. G4 pilot01 must explicitly select accepted02;
the default01 is not silently replaced. G8 retains its original01 selection,
now with its own twelve actual recovery checks/ten-component final equality and
three completed native exits0. Pilot quality requires its separate common panel.

G8's physical journal again measures four applications and three collections,
but314 freshly valid actions and434 applied targets. Its384 dense batch slots/
multinomial draws include stopped-row spending and are not384 valid emitted
targets. Pending applies120 saved targets without recollection. Source groups
have seven EOS/one cap then two EOS/six caps, yet all eight rewards/advantages
are zero in both. The measured norms4.3585896492004395e-7 and0.146484375 and
update2 KL0.0009698036592453718 are not proof of positive reward learning.
Different group sizes do not imply equal sampled token costs or a method ranking.
Accepted exact recovery remains distinct from a pilot's own committed16/export,
its declared export-consumer boundary and separate common20 comparison.

That pilot now supplies a further actual boundary: its native report is completed
at16 and the child exits0, but the supervisor returns `Final logging timed out`
and `final_record_retained=false`. At that adapter stop there was no accepted
pilot/final-export gate or publication evaluation. Empty cleanup errors and terminated helpers do not
replace the missing writer acknowledgment. Native iterations were really spent:
sixteen applications/collections,1508 dense sampled slots versus1004 valid
targets, and33 snapshot saves. All64 training responses naturally stop, yet all
sixteen four-member strict-reward/advantage groups are zero. Numerical KL/AdamW
movement is not positive task-relative learning or proof of better reasoning.
Do not describe this as no training, accepted pilot16, or a quality result.
The later explicit G4 export-consistency consumer preserves that failed verdict
instead of refitting the policy. G8 now also completes and passes its own pilot16
gate. Its128 responses all stop naturally, but all sixteen eight-member groups
again have zero strict rewards and advantages. Sixteen optimizer applications
are not sixteen learning successes: G8 spends4888 dense slots for2335 valid
targets, versus G4's1508/1004. Both retain33 saves. G8's final pre-update exact KL
is0.0010826910147443414; nonzero regularized/numerical movement does not create
a reward signal. The [actual training table](../../experiments/reports/2026-10-05-native-reasoning-rlvr-evidence.md#46-actual-native-pilots-completed-updates-are-not-a-reward-signal)
binds both outcomes. Neither four-item diagnostics nor group size supplies a
method ranking; the actual common panel provides the following narrower result.

### 13.8.4 Read the matched outcome, not the pooled count

The [actual cap32/cap128 comparisons](../../experiments/reports/2026-10-05-native-reasoning-rlvr-evidence.md#47-actual-common20-comparisons-the-denominator-changes-the-conclusion)
join unchanged Instruct, G4 and G8 on every original item/seed contract. Four
fresh evaluations retain400 records;200 existing baseline records are reused,
not generated again. The original BF16/eager, thinking-off, full-support interface
and frozen whole-output grader remain unchanged. The headline excludes the nine
diagnostic items; sampled44 means four repetitions of eleven held-out problems,
not44 independent problems or55 answers pooled with greedy.

| Cap | Policy | Sampled correct /44 | Sampled natural / cap | Greedy correct /11 | Greedy natural / cap |
|---|---|---:|---:|---:|---:|
| 32 | Unchanged Instruct | 0/44 | 3 / 41 | 0/11 | 1 / 10 |
| 32 | RLVR G4 | 0/44 | 3 / 41 | 0/11 | 1 / 10 |
| 32 | RLVR G8 | 0/44 | 3 / 41 | 0/11 | 1 / 10 |
| 128 | Unchanged Instruct | 13/44 | 35 / 9 | 7/11 | 11 / 0 |
| 128 | RLVR G4 | 11/44 | 37 / 7 | 7/11 | 11 / 0 |
| 128 | RLVR G8 | 13/44 | 38 / 6 | 7/11 | 11 / 0 |

All-original100 cap128 correct counts rise34→35→36, but held-out sampled counts
are13→11→13 and greedy stays7/11. Diagnostic successes do not establish held-out
learning. G4's small sampled decrease is not a general regression, and G8's
larger group supplies no correct-count gain here. Every training group had zero
relative advantages: optimizer calls and numerical/KL movement are not proof of
reward-driven improvement. This fixed negative result does not invalidate GRPO
or the separate symbolic positive controls.

Correct-but-capped held-out samples number2/2/1 at cap128. Fewer caps and more
natural stops are distinct from correctness; `any` format validity is distinct
from successful extraction. Unsupported and invalid responses stay in the
denominator. No parser rescue, oracle best-of-four selection or rationale-
faithfulness claim is justified by these measurements. Some evaluation time
overlaps CPU notebook verification; recorded whole-child times are not isolated
serving-throughput benchmarks.

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
11. Why can genuine sampled learning coexist with a failed known-solvable training item?
12. What would an always-one baseline reveal about a100% unseen-source score here?
13. Which changes distinguish base/instruct weights, prompt format, thinking support and output budget?
14. Why can response means rotate a parameter gradient, while token means and a
    fixed-cap denominator only rescale it on a fixed pool?
15. Does component normalization make subsequent component weights irrelevant?
16. Which derivative changes when only the upper clip boundary increases?
17. If two filtered batches have forty selected responses each, what else must
    be measured before calling their collection budgets equal?
18. After a crash with a fully collected group awaiting its update, why is
    restoring weights and collecting a fresh group not the same experiment?
19. A native response naturally stops after `5 + 6 = 11`, yet receives a negative
    declared grade and passes the format predicate. Which retained fields explain
    this combination, and what can you say without changing the experiment?
20. What evidence would distinguish a completed RLVR iteration, an optimizer
    application, a nonzero relative policy signal and an improved delivered answer?
21. Both thinking-on caps complete execution but cap every response with an
    unsupported grade. Which checkpoint, template, prompt, seed and cost joins
    make that result interpretable without a general thinking-mode ranking?
22. Three G4 recovery children exit0, all work/I/O tickets close and all helper
    cleanup receipts complete, yet the adapter fails. Which independent gate is
    missing, and what can the retained zero-advantage, nonzero-gradient history
    establish without claiming accepted replay or reward learning?
23. Which new run02 receipts establish accepted exact recovery without erasing
    failed01? Why are its twelve passed checks, unchanged source/input closure
    and four physical applications still insufficient to claim improved reasoning?
24. Why does accepted G8 recovery retain384 sampled slots but314 newly valid
    actions and434 applied targets? Which masks, rollout/restoration histories
    and stopping fields explain those different counts without calling them FLOPs
    or an equal-compute quality comparison?
25. A pilot completes sixteen native iterations/export and exits0, then fails
    final logging acknowledgment. Which facts remain measured, which acceptance
    and publication claims are still unavailable, and why is natural stopping
    across64 zero-reward training responses not a learning-success proxy?

[Worked answers](../solutions/13-group-relative-policy-optimization.md) accompany
the [Day 22–23 notebook route](../labs/13-group-relative-policy-optimization.md).
Chapter 14 examines what happens when this exact pipeline faithfully optimizes
the wrong reward or consumes rollouts whose policy identity has changed.
