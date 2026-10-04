# 10. Preferences and Reward Models

Our story model learned to make locally plausible predictions while still
repeating itself and losing track of characters. Supervised fine-tuning can
demonstrate a better continuation, but demonstrations alone leave another useful
source of information unused: a reader can often recognize a better story
without being able to write an ideal one. This chapter asks how that comparison
becomes a trainable signal, and what the resulting scalar actually measures.

The prerequisites are next-token likelihood, gradient descent, and the evaluation
contract from Chapter 7. By the end, the reader should be able to derive a
pairwise preference likelihood, fit a transparent reward model, audit its
probabilities, and explain why a high reward is a fallible measurement. Days 15
and 16 share this argument: first define judgments; then model and challenge them.

## 10.1 A judgment needs a question

Imagine two answers to “Where is Lily's ball?” after a story explicitly puts it
in a red box. One answer says “In the red box.” Another gives a polished paragraph
about Lily happily playing outside without answering the question. Which is
better? A rubric emphasizing factual continuity favors the first; a rubric
emphasizing engaging prose might distract an annotator toward the second.

A preference record therefore includes a prompt $x$, two completions $y_a,y_b$,
the judgment, and the conditions under which it was made. Store the rubric,
source identifiers, presentation order, annotator or judge identity, abstentions,
and the reason for a tie. Preserve the original answers. “Chosen” and “rejected”
are convenient names for a selected pair, but they do not mean objectively true
and false. A rejected response can contain useful content, and both can be wrong.

Separate substantive criteria such as correctness, instruction compliance, and
coherence from nuisance criteria such as verbosity, headings, confidence, and
position on the screen. Some users genuinely prefer a particular style; that is
valid if the objective explicitly includes it. The danger is accidental style
selection being interpreted as improved reasoning. This continues Chapter 8's
interface argument: the training target includes every choice made while
constructing the record, not only the intended lesson.

## 10.2 From two scores to one preference probability

Let $r_\phi(x,y)$ be a real-valued score from a parameterized model. Fix one
prompt and suppress $x$ briefly. Give each completion a positive strength
$s_y=\exp(r_y)$. If a comparison allocates probability in proportion to strength,
then

$$
P(a\succ b)=\frac{\exp(r_a)}{\exp(r_a)+\exp(r_b)}
=\frac{1}{1+\exp(-(r_a-r_b))}=\sigma(\Delta),\qquad \Delta=r_a-r_b.
$$

This is the Bradley–Terry model. The score difference is a binary classification
logit. A margin of zero means probability one half. A positive margin favors
$a$; a large magnitude expresses confidence under the model. The use of
pairwise likelihoods for language-model reward fitting follows the formulation
in [the DPO paper's preliminaries](https://arxiv.org/html/2305.18290v3#S3).

Now record $q=P_{\text{labels}}(a\succ b)$. A hard selection has $q=1$ or $0$;
aggregated repeated votes can give a fraction. The negative log-likelihood is

$$
L(\Delta,q)=-q\log\sigma(\Delta)-(1-q)\log(1-\sigma(\Delta))
=\mathrm{softplus}(\Delta)-q\Delta.
$$

Use a stable binary-cross-entropy-with-logits implementation. Computing the
sigmoid, then taking logarithms, can turn extreme but finite scores into
infinite losses. For an ordered winning pair, $q=1$ and
$L=\mathrm{softplus}(-\Delta)$. A pair has one likelihood, not two independent
absolute quality labels.

## 10.3 The gradient explains the update

Differentiate the stable expression:

$$
\frac{\partial L}{\partial\Delta}=\sigma(\Delta)-q,
\qquad
\frac{\partial L}{\partial r_a}=\sigma(\Delta)-q,
\qquad
\frac{\partial L}{\partial r_b}=q-\sigma(\Delta).
$$

Suppose annotators favor $a$, but the model currently favors $b$. The first
derivative is negative, so gradient descent raises $r_a$; the second is positive,
so it lowers $r_b$. If the model already predicts the preference confidently,
the correction becomes small. This is the same probability-minus-target
mechanism from Chapter 3, applied to a pair rather than a vocabulary.

Actual network updates also pass through score Jacobians. Two answers share
parameters, so increasing one scalar and decreasing the other are local
directions, not guarantees about every score after an optimizer step. An
embedding or transformer parameter can affect both sides. The transparent lab
first treats scores as independent leaves, then replaces them with
$r_\phi=w^\top f(x,y)$. It makes this distinction visible rather than hiding it
inside a large training library.

## 10.4 Disagreement carries information

Suppose ten independent judgments favor $a$ seven times and $b$ three times.
Under a repeated identical comparison and the binary model, the empirical
target is $q=.7$. Minimizing expected likelihood gives

$$
\sigma(\Delta^*)=q,\qquad
\Delta^*=\log\frac{q}{1-q}.
$$

The optimal margin is finite. The remaining loss is the entropy of the vote
distribution, not evidence that optimization failed. Repeating a deterministic
winner label for a genuinely ambiguous pair instead asks for unlimited margin
in an unconstrained separable model. A regularizer, finite training, or limited
model capacity may stop the scores growing, but these are different mechanisms.

The fraction does not explain *why* people disagree. They may interpret the
rubric differently, have different expertise, make errors, or represent
legitimate conflicting preferences. Annotator-stratified analysis can reveal
that one global probability conceals stable subgroups. A Bradley–Terry scalar
also imposes structure: its implied pair odds satisfy additive log-odds across
alternatives. Strong cyclic judgments cannot generally be represented exactly
by one score per answer. This is a model limitation, not a reason to delete
awkward labels until the dataset looks consistent.

Ties require a declared convention. A soft $q=.5$ says the model should predict
an even binary choice; it does not model a separate “equally good” event. If
ties are frequent and meaningful, use a model with an explicit tie outcome or
retain them for analysis. Abstention because the judge cannot assess a response
is a different event and should not automatically become a half-win label.

## 10.5 Reward has a gauge

For any function $c(x)$, replacing every score for one prompt by
$r'(x,y)=r(x,y)+c(x)$ leaves all within-prompt differences unchanged. Preferences
cannot identify the absolute origin of the reward. An intercept shared by the
two answers has zero gradient under pure pairwise loss. Adding fifty to every
score therefore says nothing about improvement.

Nor are raw scores from different reward checkpoints directly comparable.
One checkpoint can stretch margins while retaining identical rankings. Fixed
label-noise assumptions determine the meaning of that stretch: unlike an
additive offset, multiplying all scores changes preference probabilities.
Centering rewards fixes a convenient origin; temperature calibration fixes a
probability mapping. Neither establishes that the scores represent human welfare
or task accuracy on new prompts.

This freedom will matter twice. DPO cancels prompt-specific reward offsets.
Policy-gradient baselines can subtract an action-independent value without
changing the expected gradient. The algebra resembles the reward gauge, but
the purpose differs: identifiability for a preference model versus variance
reduction for an estimator.

## 10.6 Fitting a reward model

The small lab uses three features: substantive quality, response length, and
polished format. These are explicit synthetic coordinates, not measurements
inferred from real stories. Labels depend only on the first feature. A linear
score allows us to read every coefficient and ask which shortcut training found.

A neural reward model instead takes the prompt and completion through a
transformer and maps a selected hidden state to a scalar, for example
$r_\phi=w^\top h_{\text{end}}+b$. Selecting the final nonpadding position
requires an attention-mask-aware index. Padding at a fixed maximum index is not
the answer's end. Whether EOS is included, whether the chat template adds a
turn-ending token, and whether the reward head sees the prompt must be frozen
in the data contract.

Both answer branches contribute gradients to shared parameters. A language-model
initialization can supply representations, but the scalar head must still learn
from comparisons. Outcome rewards evaluate a completed answer; process rewards
evaluate intermediate steps under a step-level annotation contract. They are
not interchangeable simply because both return numbers. A process verifier
can reward valid intermediate reasoning yet miss an invalid final claim, or
vice versa. The course's bounded experiment fits an outcome preference model;
it makes no process-supervision claim.

Split by prompt or source group before forming answer pairs. Splitting individual
pairs can put the same prompt, near-duplicate answer, or generation family in
both training and validation. Randomly swapping presentation order is valuable,
but a swap is not a new independent example. Score both orders during audits
to detect an order-dependent judge or preprocessing bug.

## 10.7 Ranking accuracy and calibration ask different questions

Pair accuracy checks whether the sign of $\Delta$ agrees with a winner. It
does not punish unjustified certainty. Probability metrics do. On independent
binary labels $l_i$, the Brier score is

$$
\frac1N\sum_i(p_i-l_i)^2.
$$

In our simulator the true soft probability $q_i$ is known, so the *expected*
binary Brier score is
$N^{-1}\sum_i[q_i(1-p_i)^2+(1-q_i)p_i^2]$. It includes irreducible label
variance. The simple squared discrepancy $(p_i-q_i)^2$ measures another
quantity and should not be mislabeled as the observed binary Brier score.

A reliability diagram bins predicted probabilities and compares mean prediction
against mean label frequency. Show counts in each bin. Empty or tiny bins do
not provide precise estimates. Expected calibration error depends on binning
and can conceal bad subgroup behavior. Keep held-out NLL, Brier score, ranking
accuracy, and slices together. Fit any temperature on a calibration split and
evaluate it on a separate final split; fitting and judging on the same labels
overstates evidence.

Temperature scaling replaces $\sigma(\Delta)$ with $\sigma(\Delta/\tau)$.
For positive $\tau$ it preserves pair rankings while changing confidence. It
cannot repair a systematically reversed ranking or a model that uses length
instead of correctness. Calibration is a conditional statement about a specified
population and label process, not a permanent property of the model.

## 10.8 A controlled shortcut experiment

The predeclared CPU experiment changes the training distribution while keeping
the linear score, seed, optimizer, number of examples, and held-out distribution
fixed. In the confounded arm, length and polished format nearly equal quality.
In the balanced arm, they vary independently. Soft preference labels follow
$\sigma(2\,\Delta\text{quality})$ in both. The held-out distribution and an
adversarial pair are independent of the training fixture.

Predict before running: both arms can fit training comparisons; the confounded
arm has insufficient evidence to isolate quality from correlated nuisance
features. On a pair whose first answer is substantively worse but much longer
and more polished, it may assign the wrong preference. The balanced arm can
identify the intended coordinate more clearly. The report preserves all
coefficients, learning curves, held-out NLL, probability calibration, and the
adversarial prediction. These are measured properties of the synthetic model,
not empirical claims about Qwen or human judges.

The mechanism generalizes as a diagnostic question: when reward improves,
which feature changed? Audit matched-length pairs, matched-format pairs,
correct-but-plain answers, incorrect-but-polished answers, and refusal versus
valid response cases. Matching can itself alter the population, so report the
construction rule. Do not infer from one adversarial failure that all training
comparisons are useless, or from a high held-out score that a policy may safely
optimize the reward without encountering new failures.

## 10.9 When optimization changes the population

A held-out comparison dataset usually contains responses from a fixed generation
process. A policy maximizing the reward changes that process. It can discover
regions where the reward extrapolates badly: repetitive praise, excessive length,
format tricks, confident nonsense, or phrases correlated with high training
scores. This is the link from measurement to optimization. The reward model
may remain accurate on ordinary held-out comparisons while failing on responses
actively selected to exploit it.

Maintain independent outcome measurements and periodically audit samples from
the evolving policy. Freeze evaluation prompts and decoding settings before
comparing checkpoints. Store negative cases. A larger score is an observation;
improved usefulness requires another argument. Chapter 11 learns directly from
comparisons without a separately trained scalar head, but it still inherits the
preference data's biases and coverage limits.

## 10.10 Companion route and exercises

Run [Day 15's margin microscope](../../notebooks/day-15/01_bradley_terry_margin.ipynb),
[Day 15's disagreement/calibration lab](../../notebooks/day-15/02_disagreement_and_calibration.ipynb),
and [Day 16's shortcut audit](../../notebooks/day-16/01_reward_model_bias_audit.ipynb).
The [lab guide](../labs/10-preferences-and-reward-models.md) records the machine
route and reproduction command; [worked solutions](../solutions/10-preferences-and-reward-models.md)
follow each exercise. Use the [specification](../../experiments/specs/2026-10-04-preference-policy-cpu.md)
and [measured report](../../experiments/reports/2026-10-04-preference-policy-cpu.md)
to distinguish predictions from results.

1. Derive the score gradients for a soft preference target and explain their signs.
2. What does adding a prompt-specific constant to all rewards change? What does multiplying rewards by two change?
3. Why can a well-trained preference model have nonzero minimum loss on repeated judgments?
4. Construct cyclic preferences and explain the restriction imposed by one score per answer.
5. Design a split that prevents a prompt with many candidate pairs from leaking into validation.
6. Explain how two models can have identical pair accuracy and different Brier scores.
7. Why should calibration temperature be fitted on a different split from final evaluation?
8. Diagnose a reward model that likes longer incorrect answers. What would distinguish intended style from a shortcut?
9. Which hidden state should a padded transformer reward head score? Explain an off-by-one failure.
10. Why can optimization reveal reward failures absent from a fixed held-out pair set?

The next chapter retains the pairwise probability model, but replaces the scalar
reward with a policy's change in log-probability relative to a frozen reference.
Understanding the score difference, its gauge, and its evidence limits makes
that transformation much easier to interpret.
