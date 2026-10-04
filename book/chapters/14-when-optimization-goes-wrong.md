# 14. When Optimization Goes Wrong

A model finds a way to make its reward rise while giving worse answers. The
optimizer has succeeded at the program it was given. The experiment has failed
at the capability it was meant to improve. This distinction connects the
repetitive TinyStories samples of Chapter 6, reward-model selection in Chapter 10,
and the verifier-driven policy updates of Chapter 13.

Days 24–25 study failures as interventions rather than slogans. We deliberately
build an exploitable reward, repair it, compare length reductions, reject stale
rollouts and estimate a rollout pipeline's cost. By the end, a “training is
unstable” report should become a specific account of measurements, hypotheses,
controls and evidence.

## 14.1 A real reward-hacking microscope

Imagine a task whose correct answer is 35. Our policy can choose `35`, `35 0` or
`0`. Strict correctness accepts only one integer equal 35. A deliberately broken
proxy accepts text that contains 35 and adds a bonus for apparent extra work:

$$
R_{\mathrm{proxy}}=(1,2,0),\qquad
R_{\mathrm{truth}}=(1,0,0).
$$

With logits $z$ and $p=\mathrm{softmax}(z)$, optimize the exact expected proxy:

$$
J(z)=\sum_a p_a R_a,\qquad
\frac{\partial J}{\partial z_a}=p_a(R_a-J).
$$

The derivative follows from the same softmax Jacobian used for cross-entropy.
Actions above expected reward gain probability under ascent; actions below it
lose probability. At uniform initialization, the highest-reward malformed answer
is favored. No noisy reward estimator, model-size limitation or distributed bug
is needed to create the problem.

The [CPU experiment](../../src/dongxi_llms/optimization_diagnostics_lab.py)
performs 60 actual SGD updates on these logits. It records expected proxy,
strict accuracy, entropy and every action probability. A paired repair repeats
the same update budget from the same initial logits using the strict verifier.
This isolates the effect of reward definition. It does not show that a hacked
language-model checkpoint can always be repaired with one replacement rule.

In the [measured run](../../experiments/reports/2026-10-04-grpo-diagnostics-distillation.md),
the broken policy's expected proxy rises from 1.000000 to 1.966561 while strict
accuracy falls from 0.333333 to 0.016816. The paired strict-reward restart reaches
0.964428 accuracy. These numbers directly support the finite-policy mechanism:
the optimizer can succeed at the proxy while failing at the task. They do not
measure generalization or repair of any pretrained language model.

This mechanism is an example of optimizing an imperfect measurement. Research
on learned reward overoptimization documents related failure under model-scale
selection and training; see [Gao et al.](https://arxiv.org/abs/2210.10760).
Our three-action experiment supplies its own evidence, without inheriting those
papers' quantitative conclusions.

## 14.2 Separate the task from its checker

A verifier should have a test suite independent of the model's favored outputs.
Test ordinary correct answers, equivalent formatting where allowed, malicious
multiple answers, missing answers, overlong strings, truncated generations and
unexpected encodings. Decide whether leading zeros, signs, decimals or units
are accepted. An arbitrary parser convention can become a hidden training target.

Keep raw response, extracted answer, verifier version and reward components.
If reward mixes correctness, format and length, log each component separately.
Otherwise an increase in total reward can hide declining correctness. Treat
timeouts, exceptions and invalid outputs explicitly. Silently dropping difficult
examples changes the optimization distribution and can make the remaining
reward appear better.

A repair must be evaluated outside the reward used to select it. Repeatedly
writing exceptions for development failures can overfit a checker just as
training can overfit examples. Add adversarial cases to a declared development
suite, preserve a separate test suite and record the checker's revision. A
simple arithmetic verifier can check final correctness; it cannot certify that
every intermediate step in a rationale is valid or that the model used that
rationale internally.

## 14.3 Entropy collapse: certainty can mean several things

At a state, categorical entropy is

$$
H(p)=-\sum_v p_v\log p_v.
$$

Low entropy indicates a concentrated next-token distribution. It may reflect
mastery of an easy deterministic answer, repetition of one successful template,
overconfident failure, or the loss of alternatives needed for exploration. Average
entropy over valid response positions and report which states produced it.
Prompt-token entropy, padded rows and EOS-dominated tails answer different
questions. The entropy of a marginal mixture can also differ from average
conditional entropy over prompts.

An entropy bonus $\alpha H(p)$ encourages broader distributions locally. It
does not make low-probability actions useful. On a strict syntax task it can
increase invalid outputs; on a nearly saturated task it can trade correctness
for exploration. Compare valid fraction, correctness and diversity along with
entropy. A diverse set of wrong answers is not task progress.

Policy sampling temperature is another intervention. Raising it changes
collection and inference behavior. If training ratios assume samples came from
the untempered policy while generation used a temperature or top-p restriction,
the behavior likelihood has been misrecorded. A rollout record needs the actual
collection distribution, including truncation and renormalization rules. Our
basic GRPO runner samples at temperature 1 with no top-k or top-p restriction.

## 14.4 KL is an anchor and a diagnostic

Forward KL measures departure from a specified reference at specified states.
It is not a measure of truth or a universal acceptable-change threshold. Two
references define different anchors. The reference may be a weak base model,
an SFT assistant or the start of a particular RL phase. Checkpoint identity must
travel with the KL curve.

A sudden rise can reflect a large learning rate, repeated epochs on one rollout
batch, missing behavior correction, incorrect log-probability alignment, a reward
scale change or expected movement toward a new task. The first diagnostic is
whether the log-probabilities and masks are correct. Recompute a small batch
in full precision; verify ratios equal 1 before an update with fresh rollouts;
test whether exact KL is zero for identical policies. Then make one bounded
intervention with a predicted result.

Increasing $\beta$ strengthens the penalty in Chapter 13's objective. If
$\beta$ is too large, it can suppress useful adaptation. If too small, it may
allow harmful drift. A finite per-token KL does not bound total sequence KL
independently of length, and average KL can hide a few extreme prompts. Record
quantiles, length-stratified summaries and representative high-KL cases when
the data budget permits. Do not equate a sampled estimator's occasional negative
value with proof that the true KL is negative.

## 14.5 Length bias begins in a denominator

Two responses of lengths 2 and 8 receive the same sequence advantage. Under
response means, their total nominal loss weights are 0.5 and 0.5. Under a global
token mean, they are 0.2 and 0.8. Within response means, each token of the shorter
response receives four times the coefficient of each token in the longer one.
This creates different gradient pressures even before reward contains any
explicit length bonus.

The realized effect depends on the token gradients, advantage signs, sampled
states and EOS. “Longer is better” or “shorter is better” cannot be inferred
from a normalization coefficient alone. [Dr. GRPO](https://arxiv.org/html/2503.20783v1)
analyzes consequences of particular normalization choices; our notebooks expose
the coefficients and let the reader test simple counterexamples.

Always record generated valid tokens, natural EOS, token-limit termination and
reward by length bin. A correct answer truncated before its required closing
marker can become an apparent failure. Conversely, a substring-based checker
may reward a long response that lists many possible answers. Treat response
length caps as part of the experimental contract; changing them changes both
computation and the answer distribution.

## 14.6 The rollout system is part of the experiment

A synchronous update has a clear sequence: publish policy version $k$, generate
responses under $k$, verify them, optimize, then publish version $k+1$. The rollout
records retain behavior log probabilities and version $k$. The reference version
is separate and fixed. This makes freshness inspectable without guessing from
timestamps.

Asynchronous systems can generate while a learner updates. That improves hardware
utilization but can increase lag and complicate importance weighting. Accepting
a lagged rollout intentionally differs from accidentally treating it as fresh.
Record the lag distribution, policy weight identity, tokenizer/chat template,
sampling rules, prompt ID, group ID and verifier identity. Every answer in one
group should correspond to the intended prompt and collection contract.

A rollout generated under $k$ does not become a rollout under $k+2$ after weights
are synchronized. Its probabilities must still use $k$ as behavior. Overwriting
the stored denominator defeats the correction. A tokenizer mismatch is more
severe: the token IDs can represent different actions, making the probability
ratio uninterpretable.

The notebook's `RolloutIdentity` rejects stale versions under a strict
zero-lag teaching contract, changed tokenizer/verifier hashes and empty
responses. Passing this check establishes metadata consistency. It does not
prove that the recorded weights really match the hash, that the sampling engine
implemented the requested distribution or that accepted lag is statistically
harmless. Those require integration checks with the actual engine.

## 14.7 Generation can dominate the budget

For $B$ prompts, group size $G$, average valid response length $T$ and useful
generation rate $q$, a first projected synchronous budget is

$$
N= BGT,\qquad
\tau_{\mathrm{update}}\approx N/q+
\tau_{\mathrm{verify}}+\tau_{\mathrm{learn}}+\tau_{\mathrm{sync}}.
$$

The estimate ignores queuing and changes in throughput with batch geometry. It
is labeled projected. The lab's default 128 tokens/s is an illustrative control,
not a Spark benchmark. Vary $G$ and $T$, then inspect how the same 6-second learner
step becomes a small fraction of total elapsed time. A faster backward pass
may barely change throughput if generation dominates.

Batching can improve generation throughput while increasing active KV cache,
padding, latency or shared memory pressure. Qwen weights, frozen reference,
optimizer moments, gradients, backward activations, rollout tokens and an
inference engine's cache are distinct memory consumers. The Spark's shared host
and GPU memory makes host `MemAvailable` a relevant protection signal in
addition to Torch allocator measurements. Preserve the validated 25 GiB reserve.
An allocation projection is not a measured safe batch size.

vLLM exposes a production inference route; [its documentation](https://docs.vllm.ai/)
describes engine interfaces and serving behavior. Do not import a changing
API into the conceptual definition of GRPO. The course baseline uses a readable
synchronous Torch path first. A later engine adapter must reproduce token IDs,
EOS/masks and behavior log probabilities before its speed comparison is useful.
An engine throughput number without policy correctness is not an RL result.

## 14.8 Monitoring should distinguish incidents from causes

The [diagnostic module](../../src/dongxi_llms/optimization_diagnostics_lab.py)
returns reasons to investigate: nonfinite loss, exhausted host reserve, policy
lag, low entropy, high declared KL or disagreement between reward and evaluation.
Thresholds are pedagogical configuration values, not universally valid limits.
Do not let a green dashboard substitute for a checked process exit, complete
checkpoint or frozen evaluation.

Write an incident table with time, policy version, prompt group, measurements,
first failing invariant, hypothesis, intervention and outcome. Preserve the
failure checkpoint and a bounded response sample. Recovery must restore model,
optimizer, scheduler, RNG, sampler position and reference identity; otherwise
it is a new trajectory from related weights. Replaying the same sample schedule
can distinguish a recovery bug from expected stochastic variation.

The monitoring questions become concrete: Did useful target throughput fall
because responses grew? Did zero-variance groups rise because correctness
saturated? Did proxy reward improve while an independent evaluator declined?
Did the verifier change mid-run? Did ratios explode before clipping? Each asks
for a different measurement. Chapter 15 uses these records to decide what can
be defended at release time.

## 14.9 Deep questions

1. How can rising reward and falling accuracy both indicate successful optimization?
2. Why is a repaired checker insufficient evidence that the hacked model is repaired?
3. When can low entropy be a desired result, and when is it dangerous?
4. Why must collection temperature appear in behavior-policy bookkeeping?
5. What information is missing from one average KL number?
6. How can a loss denominator change the pressure on response length?
7. Why does weight synchronization not make an old rollout fresh?
8. Which speedup matters when generation consumes most update time?
9. Which checkpoint state is required to resume the same experiment trajectory?
10. What is an actionable monitoring alert, as opposed to a causal conclusion?

[Worked answers](../solutions/14-when-optimization-goes-wrong.md) and the
[Day 24–25 notebook route](../labs/14-when-optimization-goes-wrong.md) include
the reward exploit, reduction audit, version contract and bounded system budget.
