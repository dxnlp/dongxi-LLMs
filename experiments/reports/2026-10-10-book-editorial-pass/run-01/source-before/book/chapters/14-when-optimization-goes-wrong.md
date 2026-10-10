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

### Learned measurements can fail before the optimizer exploits them

The three-action proxy deliberately encodes a mistake. A text reward model can
instead learn an inadequate measurement from plausible examples. Before blaming
policy optimization, ask whether that measurement can distinguish the examples
and whether its distinctions generalize.

Chapter 10's word-tokenized reward experiment found complete held-out pairs
that became identical model inputs. No weight update could rank two identical
inputs differently. Its [character-tokenizer intervention](../../experiments/reports/2026-10-04-text-reward-character.md)
removes those collisions under a frozen representation contract. Yet all three
seeds still rank only half the ordinary held-out pairs correctly. Removing an
impossibility is necessary for that comparison; it is not evidence that the
model has learned the intended preference. Calibration reduces overconfidence
on its own declared slice without repairing those rankings.

Chapter 12 adds a separate difficulty: a learned critic predicts the future
return of the current policy. At a nonterminal collection cap its bootstrap
can influence the advantage even when no terminal reward has been delivered.
The [learned-critic notebook](../../notebooks/day-20/04_learned_critics_and_frozen_text_rewards.ipynb)
retains actual capped style-token repetitions and weak independent task scores.
A reward-model score displayed for an unfinished prefix is a diagnostic, not
the terminal reward paid by that experiment. Confusing the two would hide the
failure boundary.

Compare oracle, learned and deliberately noisy values, inspect completed and
capped paths separately, and retain the frozen reward's training distribution.
These controls expose plausible failure mechanisms; they do not uniquely prove
that every repetitive response was caused by critic error. A finite policy can
also collapse to a constant answer, and a reward can favor text outside its
training distribution. Diagnose the first failed invariant rather than naming
all three phenomena “reward hacking.”

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

### Three distributions, not three names for one vector

A model says an action has probability0.30. A sampler lowers temperature,
removes unlikely actions and renormalizes. The action ID has not changed, but
its probability has. An update that divides by0.30 anyway is not using a
record of what generated that action. This can alter an otherwise finite,
apparently healthy gradient.

At one fixed prefix, distinguish raw model probabilities $p$, actual collection
probabilities $b$, and separately declared target probabilities $t_\theta$.
The old policy describes collection; the reference policy defines an anchor.
For collection temperature $\tau_b$ and retained support $S$, our sampler is

$$
b(a)=\frac{\boldsymbol{1}_{a\in S}e^{z_a^{\mathrm{old}}/\tau_b}}
{\sum_{j\in S}e^{z_j^{\mathrm{old}}/\tau_b}}.
$$

Its explicit order is temperature, top-k, renormalization, top-p, then final
renormalization. Top-p retains the action crossing its cumulative threshold;
ties use smaller action ID. Different orders can produce different distributions.
Save the transformation and selected action log probabilities during collection.
A later raw model forward pass need not reproduce them.

A conditional target on the **fixed** retained support can declare its own
temperature $\tau_t$:

$$
t_\theta(a)=\frac{\boldsymbol{1}_{a\in S}e^{z_a^\theta/\tau_t}}
{\sum_{j\in S}e^{z_j^\theta/\tau_t}},\qquad
\rho(a)=\frac{t_\theta(a)}{b(a)}.
$$

Matching logits and support do not imply ratio1: temperature and renormalization
must match too. Fixing support defines a local conditional objective, not a
derivative through a discontinuous top-k/top-p choice. Recomputing its support
after an update is a different procedure with different boundary behavior.

For fixed rewards, exact enumeration verifies

$$
J(\theta)=\sum_a t_\theta(a)R(a)
=\sum_{a:b(a)>0}b(a)\rho(a)R(a),\qquad
t_\theta(a)>0\Rightarrow b(a)>0.
$$

On fixed support the logit gradient is $t_i(R_i-J)/\tau_t$ and is zero outside
it. Collection probabilities and rewards detach. Replacing the denominator
with raw model probabilities leaves a multiplier $b/p$ in the sum, changing
both value and gradient. PPO clipping does not make an incorrect denominator
true. This single-state action identity also does not solve a complete
autoregressive off-policy problem: different prefix-state distributions and
trajectory weights require their own objective and correction.

### Missing support is missing evidence

The [original CPU microscope](../../src/dongxi_llms/sampling_likelihood_lab.py)
starts with raw probabilities0.55,0.30,0.15 and rewards0,1,4. Temperature0.5,
top-k2 and top-p0.9 produce behavior approximately0.7707,0.2293,0. A declared
temperature-one target on its retained support is approximately0.6471,0.3529,0.
Correct denominators recover expected reward0.352941; raw model denominators
produce0.269764. The [measured report](../../experiments/reports/2026-10-04-sampling-support.md)
retains both gradient vectors, rather than inferring correctness from a
plausible scalar loss. These are finite fixture measurements, not LLM results.

Now ask this collector to optimize the original full-support target. Action2
has target mass0.15 and contributes0.60 to expected reward, but cannot be
sampled. The importance routine rejects this missing support. Zeroing its
ratio would silently omit that contribution. A retained mask cannot repair
the original objective; it can help define a different, conditional objective.
Full-support collection is an alternative when the original raw target must
be retained.

No universal “keep the sampling mask” rule follows. Temperature, renormalization,
target support, changing selection boundaries and prefix-state distributions
all matter. The course Qwen adapter deliberately keeps its simpler
temperature-one/full-support contract. The microscope is motivated by the
[pinned reasoning companion's sampling-mask mechanism](https://github.com/rasbt/reasoning-from-scratch/blob/a788466dc85cfe8617b6c0b809ed4ab9084485de/ch07/03_rlvr_grpo_scripts_advanced/7_7_improvements/deepseek_v32_style.py);
its implementation, prose and fixtures are independent.

### A correct KL value can provide a different KL gradient

At one fixed prefix with positive full-support current $p$ and frozen raw
reference $q$, two sample statistics are

$$
k_1(a)=\log\frac{p(a)}{q(a)},\qquad
k_3(a)=\frac{q(a)}{p(a)}-1+\log\frac{p(a)}{q(a)}.
$$

Both have expectation $D_{\mathrm{KL}}(p\Vert q)$ under $a\sim p$.
For $k_3$ the extra terms cancel because $\sum_a q(a)=\sum_a p(a)=1$.
Shared full support matters: reference mass outside truncated support can
invalidate that cancellation.

Already collected actions do not resample when parameters move. Freeze their
distribution at $b=p_0$ and differentiate each sample term alone. At the fresh
point $p=p_0$, expected current-logit gradients are

$$
\mathbb{E}_b[\nabla_z k_1]=0,\qquad
\mathbb{E}_b[\nabla_z k_3]=p-q.
$$

Neither generally equals the full categorical forward-KL derivative:

$$
\frac{\partial D_{\mathrm{KL}}(p\Vert q)}{\partial z_i}
=p_i\left(\log\frac{p_i}{q_i}-D_{\mathrm{KL}}(p\Vert q)\right).
$$

Their forward values agree; their autodiff contracts differ. Exact categorical
KL differentiates both probabilities and log probabilities. Alternatively,
in this finite full-support example, differentiating the **entire**
$\mathbb{E}_b[(p_\theta/b)k_\theta]$ expression restores the action-distribution
derivative. Detaching its weight removes that term again. This identity is not
a recipe for every trajectory-KL or policy-gradient implementation. Declare
whether an expression is a diagnostic or a chosen differentiable surrogate,
its sampling distribution and its detached terms.

### Stops and gradient boundaries belong in the contract

A sampled EOS is a response action and receives policy loss. Prompt tokens,
padding and positions after the first generated stop do not. If EOS is also
the padding ID, use prompt boundaries and valid lengths, not token values
alone. A token cap does not assert that the model chose to terminate. In a
continuing-task critic it can permit bootstrapping from the remaining state;
an explicitly finite-horizon task can instead declare its cap terminal.
State that task convention before constructing targets.

The utility tests these distinctions and rejects an unexplained early end.
Recorded old likelihoods, advantages and reference parameters detach; current
target probabilities remain differentiable. Select valid positions before
likelihood arithmetic: zero times a padded NaN is still NaN. A token mask must
still be shifted/sliced once to align the logits predicting those tokens.

Use [Day20's probability notebook](../../notebooks/day-20/03_behavior_probabilities_and_support.ipynb)
as the bridge from Chapter12 policy ratios to this failure analysis. Its six
prediction/reference exercises show probabilities, wrong-denominator gradients,
missing support, KL-gradient differences, masks and detachment. Execution
verifies finite mathematics and record integrity, not Mac setup, model-scale
stability, full PPO or a pretrained capability gain.

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

The [matched-rollout objective notebook](../../notebooks/day-22/03_objective_weighting_and_filtering.ipynb)
compares these pressures on the same generated responses and unchanged decoder
weights. Its analytical selected-score derivatives match autograd. Token-mean
and fixed-cap reductions change scale while sharing a numerator; response means
can also rotate the parameter-gradient direction. Centering a component total
differs from centering and scaling each component before combining them.
None of these gradient observations establishes a better trained policy.

Clipping is a separate intervention. Fresh matching policies have ratio one,
so asymmetric upper clipping cannot improve that fresh surrogate by itself.
A labeled constructed-ratio control shows which pressures change once ratios
move. A separate bounded retry ledger retains rejected groups and their cost.
The [measured CPU comparison](../../experiments/reports/2026-10-04-grpo-objective-controls.md)
attempts 200 responses but selects only 120. Reporting selected tokens alone
would obscure the cost of changing the training population. Filtering is not
silently inserted into the fixed pool used to compare objectives.

### Retention is another measurement, not another name for preference loss

A decreasing DPO loss says the recorded winner is improving relative to its
loser and reference. It does not establish that the winner is a valid answer,
that unrelated skills survive, or that longer winners teach the intended style.
Chapter 11's [retention experiment](../../notebooks/day-18/02_dpo_retention_and_preference_controls.ipynb)
keeps a copy task, an unrelated parity task and held-out sources separate. It
compares preference-only training, chosen-response NLL, rehearsal NLL and their
combination under clean, noisy and length-controlled pairs.

The warm policy did not perfectly solve parity. Rehearsal can therefore continue
learning that task as well as retain what was already learned. Calling every
gain “prevented forgetting” would assume a stronger starting capability than
the evidence supplies. Fixed canonical-answer-and-EOS evaluation also differs
from the preference labels when winners contain extra style tokens. Keep both
measurements: changing evaluation to bless the trained style would conceal
the disagreement rather than diagnose it.

Auxiliary losses consume extra training computation. Report their weights,
tokens, forwards and source overlap; an equal update count is not automatically
an equal compute budget. Negative held-out and noisy-label results remain part
of this comparison. Neither rehearsal nor chosen-response likelihood is a
universal repair for a weak representation or an incorrect preference target.

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

### Fewer forwarded positions is not automatically faster serving

The [ragged-cache reference](../../experiments/reports/2026-10-04-batched-cache-recovery.md)
uses the actual modern tiny decoder, compact GQA keys/values and differently
sized prompts. Left-padding is storage, not token validity. Logical RoPE
positions follow each row's valid prefix, while an emitted EOS is a real action
even when its ID also serves as padding. Finished rows leave the active cache;
a cap ends collection without inventing EOS.

Under one fixed greedy panel the four paths emit the same12 useful actions.
Sequential uncached processing forwards66 positions; batched uncached forwards
90, including24 padding positions. Sequential caching forwards21; batched
caching27, including6 padding positions. Selected log probabilities match to
the declared tolerance. Those counts isolate saved prefix work, not FLOPs or
latency. Cached batching is slightly slower on this tiny CPU workload; Python
bookkeeping and small kernels can outweigh the work reduction. Do not carry
that timing, or an assumed batching speedup, into a Spark serving claim.

For $L$ layers, active batch $B$, compact KV heads $H_{\mathrm{kv}}$, physical
cache length $S$, head width $d$ and element bytes $b$, tensor payload is

$$
M_{\mathrm{KV}}=2LBH_{\mathrm{kv}}Sdb.
$$

The factor2 counts keys and values. This excludes allocator overhead,
snapshots, query-head expansion, activations and model/optimizer storage. Our
explicit tensor counts test this formula; they are not a device-memory profile.

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

### Recovery must replay the pending update, not collect a new one

The cache reference also performs real tiny DPO and verifier-reward updates.
It interrupts after completed updates and separately after collection but
before optimization. Restoring weights, Adam, frozen reference, data cursor,
RNG, version and pending rollout reproduces the retained trajectory in the same
CPU environment. A restored pending rollout is consumed before any new sample
is collected. Omitting applicable optimizer/cursor/RNG state supplies controlled
divergences. This tests these isolated interfaces, not recovery in the optional
pretrained Spark runners.

Source review found a more basic failure: the initial state-digest encoding
omitted dictionary boundaries, so two differently nested states could serialize
identically before hashing. A SHA256 label cannot repair ambiguous input
serialization. Version2 uses typed length-framed containers and tensor
dtype/shape/bytes, rejects obsolete contracts before deserialization, and keeps
all historical reports unchanged. The migration compares actual saved
policy/reference/Adam/data tensors and numerical actions, not rewritten old
hash strings. The [independent recovery verification](../../experiments/reports/2026-10-04-batched-cache-recovery-root-verification.json)
records that bridge. Local byte checks are still not external authentication
or a sandbox for hostile checkpoint inputs.

Byte identity must also preserve the actual tensor representation. Direct
numerical conversion to NumPy fails for BF16 in the inspected environment.
Flattening a contiguous tensor and viewing its payload as bytes fixes the dense
serialization boundary without converting its numerical dtype. The
[follow-up check](../../experiments/reports/2026-10-05-recovery-tensor-bytes.md)
preserves all13 archived snapshot identities and their CPU continuations; this
is not evidence that a BF16 pretrained training trajectory can be recovered.

### Safe shutdown is a separate outcome

A leader process can exit0 while a worker still runs. Consequently, record the
leader's actual exit and the shutdown condition separately. The
[owned-worker controls](../../experiments/reports/2026-10-05-owned-workers-and-disk-guards.md)
demonstrate that failure: a successful leader leaves a TERM-ignoring worker,
which the supervisor identifies and stops. Another worker starts an independent
session, so a process-group signal alone cannot reach it. Retained PID/create-time
handles permit cleanup of that discovered worker without adopting unrelated
processes from a conflict scan. Unknown status is not proof of death.

Disk controls have similarly distinct meanings. A hard per-file size limit does
not limit the sum of many files. Our fixed writers separately demonstrate a
SIGXFSZ exit at one file's limit and a sampled aggregate-byte stop. The latter can
overshoot between samples; free filesystem bytes, logical artifact bytes and a
hard filesystem quota are not interchangeable. Failure of the observer or its
journal must not prevent cleanup. Independent fault controls exercise denied
inspection, a denied TERM attempt and continued KILL/reap/closure.

These are bounded cooperative CPU controls, not a production launcher or cgroup
sandbox. Sampled discovery can miss rapid reparenting; a process-group identity
check cannot eliminate every check/signal race. The
[runner-owned recovery plan](../../docs/PRODUCTION_RECOVERY_PLAN.md) keeps approved
pretrained/CUDA/BF16 snapshot replay and Spark containment gates open. Passing a deliberately
failing fixture means the control caught that failure, not that a model trained
successfully.

### The deadline must not wait for the observer

A training loop's runtime check only runs when control returns from a model
operation. If that operation hangs, the next check never happens. Moving the
check to a parent process helps, but introduces another failure: the parent can
itself hang while sampling memory, reading a partial probe response or flushing
its incident log. A timeout written in configuration is not an enforced deadline.

Keep the deadline loop independent of those operations. Bound observation and
logging separately, retain their failures, then stop and reap only the owned
invocation. A successful leader exit must not silently certify its remaining
workers. Process shutdown and durable evidence are distinct outcomes: killing
the job cannot guarantee that a failed filesystem stored the final receipt.

The [native-profile verification contract](../../experiments/specs/2026-10-05-native-profile-watchdog.md)
turns these ideas into deliberately failing, inert CPU controls for the same
controller intended to wrap one pinned SFT profile. It does not authorize that
profile. Sampled host memory, conservative conflict inspection and bounded owned
cleanup still do not prove continuous memory safety, GPU idleness, a hard disk
quota or containment of hostile unobserved descendants.

The [first native DPO recovery attempt](../../experiments/reports/2026-10-05-native-dpo-recovery-deadline.md)
shows why those distinctions matter in a real model run. Its two independent
two-update children exit0, but the fresh completed1→2 child reaches its
600-second external deadline. Externally sampled available memory stays above
the25GiB reserve; that does not change the observed stop reason into memory
exhaustion or a nonfinite-gradient diagnosis. The receipt also retains an
owned-leader reap timeout and an unknown native exit. The adapter's exit1
cannot fill that missing field. A later absent PID is a later observation,
not a retroactive clean shutdown.

The second attempt retains an even sharper distinction. With actual CPU8
thread witnesses, its resumed update2, committed saves, validation, generation
and policy export exist. Yet its final `result.json` does not: the child again
reaches the deadline, with an unknown native exit and reap timeout. A durable
numerical boundary is necessary for recovery; it is not a whole-job success
receipt. Keep both attempts' work, failures and intermediate artifacts.

### A later complete replay does not repair the earlier receipts

The predeclared third attempt passes its own gate. It retains the selected
full400 parent, original 8/4/4 location fixtures, DPO beta 0.1, learning rate 5e-7,
seed 1818, accumulation 4, length 512, generation cap 64 and checkpoint cadence 1.
Policy/reference weights remain FP32 with BF16 CUDA autocast. Actual stdout
witnesses report OMP=8, Torch intra-op 8/inter-op 1 and four complete-file hash
workers in the three native children and the separately supervised CPU
comparison.

| Actual run03 invocation | Child seconds, rounded to six decimals | Minimum sampled available bytes | Outcome |
|---|---:|---:|---|
| Independent clean2 |319.773992|95,734,423,552|Completed, native exit 0|
| Source2 retaining completed1 |296.337269|96,150,581,248|Completed, native exit 0|
| Fresh completed1→2 resume |392.908058|86,374,494,208|Completed, native exit 0|
| CPU three-way state comparison |127.596575|101,931,466,752|Completed, comparison exit 0|

Each invocation keeps its own 600-second external guard and 25 GiB sampled
reserve. This is not a 600-second ceiling on the entire serial campaign. The
[report](../../experiments/reports/2026-10-05-native-dpo-recovery-deadline.md#run-03-whole-child-recovery-and-comparison-pass)
distinguishes child time, supervisor time and the separately retained outer
launcher observation. Point-sampled memory is not continuous safety.

The CPU comparison admits the three final payload reads under their original
shared I/O journals, then compares the typed numerical components: policy,
Adam state, frozen reference, Torch/CUDA and sampler RNG, sample history,
counters, schema and completed cursor. All ten component identities agree;
the original reference remains unchanged and the resumed numerical metric tail
matches update2. Operational prefixes need not match. The source and resume
share a work journal retaining three actual optimizer updates—two in the source,
one repeated after restoring completed1—even though each final numerical state
has cursor2. Comparison reads add I/O spending rather than disappearing because
the weights agree.

The implementation change concerns scheduling, not weaker validation.
`hash_workers=1` remains the default serial inventory path. The explicit four-worker path
hashes independent files with at most four outstanding digest tasks/file
descriptors and 1-MiB buffers. Every inventory invocation and complete-file SHA,
no-follow/type/ownership, cap, missing-file, hardlink and race check remains;
errors retain the original order and submitted workers drain on failure. There
is no metadata checksum cache or omitted repeated check. Worker count is an
execution setting, not a changed scientific contract or replenished allowance.

Run03 establishes this local, bounded completed1→2 recovery, including actual
whole-child exits and comparison. It does not prove a causal hashing speedup:
the earlier deadline-capped attempts and this complete run are not a matched
performance experiment. All retained before/after location panels still have
0/4 strict exact answers despite 4/4 declared natural stops; stopping is not
correctness. Nor does this replay complete pilot100, independent preference
quality, Mac recovery, physical quotas or hostile-process containment. Technical
recovery and useful preference behavior remain different claims.

### A bounded update is not a bounded job

Suppose a DPO recipe promises 100 updates, four accumulated pairs per update,
and sequence length at most 512. The usual update geometry gives 204,800 positions.
But the actual loop scores both chosen and rejected sequences under both policy
and reference. After its one-token input/target shift, the corresponding upper
bound is 817,600 logical input positions across the two networks. Neither bound
includes baseline/final evaluation, generation, failed attempts or replay. Neither
is a FLOP count, and activation recomputation can add backward dispatches.

Distinguish three records rather than forcing all costs into a token counter:

| Record | What it protects or measures | What it does not establish |
|---|---|---|
| Whole-operation reservation | Enough declared capacity before a complete update, evaluation or generation attempt | Exact work inside a failed library call |
| Known successful/partial work | Completed calls, input positions, targets and emitted tokens that were actually observed | FLOPs, total GPU time or unseen partial execution |
| Durable cumulative ledger | Earlier attempts remain charged when a new output directory resumes old optimizer state | Authentication against an adversarial writer or cross-host replay |

An operation crossing any configured dimension must be refused **before** its
sampler or forward pass runs. Dropping the longest response to fit the remaining
budget would change the training population. Shortening accumulation would change
the update. A conservative reservation can charge more than ultimately observed;
record that difference instead of pretending its upper bound was measured work.

This also changes the meaning of recovery. A checkpoint can restore a policy
from update 3 while its separate durable work journal already records an attempted
update 4. Retrying 4 consumes additional capacity. Restoring the journal to the
checkpoint's old cursor would erase an attempt that genuinely happened. Keep the
snapshot's trusted prefix and replay the journal's retained later entries. A new
output folder is not a new allowance.

Storage has the same peak-versus-final distinction. While saving a new snapshot,
the previous snapshot, partial new payload, temporary marker and published marker
may coexist. Reserve bytes and directory entries before creating those files;
retain the charge for failed partials. A cooperative writer can enforce its own
envelope, but arbitrary library exports and child writes require a separately
tested physical aggregate quota. An accounting class cannot manufacture that
backend.

### Replaying randomness for a check is not resampling the experiment

Suppose a DPO checkpoint records $u$ completed updates and $a$ accumulated
pairs per update. Its retained sampler history should agree with $ua$ draws
from the declared seeded stream. A validator can replay those draws using a
temporary generator. The private replay consumes verification work, but it
does not advance the live training sampler or invent new training examples.
After validation, restore the saved live sampler state; the next training draw
must remain the next draw of the original trajectory.

This is why a single "random draws" counter is ambiguous. Separate training
draws from verification draws. Likewise, a save may validate its state once,
while a load callback and a subsequent restore function each validate the same
state. If both calls execute, count both panels. Their numerical outcome can
agree while their cumulative work differs. Budget tests should spy on the actual
calls and compare the resulting weights, optimizer moments, reference and sampler,
not merely assert that a descriptive validation field exists.

### A callback cannot reserve work that happened before it

The shared reader first verifies expected bytes, performs a restricted tensor
load and scans a bounded state tree. Only then does it invoke the runner's
semantic validator. Reserving a history or RNG panel at that callback protects
those later checks; it does not retroactively charge the earlier byte hashing,
deserialization or generic finite scans. Explicit file envelopes are useful,
but remain distinct from a work ledger and a hostile-input sandbox.

There is a dependency to resolve before pre-read reservation. A resumed journal
must be bound to a trusted prefix before granting more work. If the only prefix
is inside the checkpoint, finding it already requires reading the checkpoint.
The reader boundary therefore needs an independently retained prefix
receipt, checked against the same physical journal and scientific contract
before the expensive read. The separate shared I/O ledger now provides that
source path; its
[CPU verification record](../../experiments/reports/2026-10-05-snapshot-io-readiness.md)
separates released checks from work still being verified. The
[checkpoint-inspection plan](../../docs/SNAPSHOT_INSPECTION_BUDGET_PLAN.md)
keeps its original dependency and acceptance rules. Neither path resets an old allowance nor proves cross-machine
recovery.

### Identical restored weights do not make the next read free

Consider the [Day25 reader microscope](../../notebooks/day-25/02_ragged_kv_cache_and_exact_recovery.ipynb).
It saves one tiny tensor, inspects its bytes and loads it twice. Both successful
loads return identical weights. A third load reads the same payload and performs
the generic checks, but a deliberately rejecting callback prevents acceptance.
The I/O allowance permits three load attempts. After reopening the journal from
the checkpoint's older pre-save receipt, the fourth load is still refused.
Numerical sameness does not erase an operation or make a failed check free.

The trust and cost order is deliberately asymmetric:

```text
small independently retained receipt
    → bind the same physical journals, retaining their later spending
    → reserve the complete shared operation
    → verify payload bytes → restricted load → bounded generic checks
    → compare payload with receipt → runner semantic checks → apply state
```

The adjacent commit marker is not the independent expectation. Nor can a prefix
hidden inside the checkpoint authorize work already performed to find it.
Identity collection must respect the same order: hashing the entire resume
payload as an ordinary input before admission would bypass this boundary even
if the later `load_snapshot` call were correctly guarded.

A save has another ordering constraint. Its payload records the I/O prefix
immediately before the save reservation. The save's completion belongs in later
journal history and an independently retained receipt. Trying to serialize that
completion inside its own already-written payload creates a circular dependency.
An interrupted publication can leave bytes or a marker; preserve them and the
failure rather than inventing a completed external receipt.

File size alone is not the generic-check envelope. A repeated/expanded tensor
view can have few stored bytes but many logical elements, and repeated aliases
can require several clones. Reserve declared nodes, elements, primitive bytes,
clone bytes and serialization bytes separately. Check tensor elements **and
bytes** before finite scans or cloning, and accumulate visits across all phases
of one operation. These are cooperative logical units, not exact CPU operations,
deserializer allocation containment or a sandbox for hostile checkpoints.

### A recovery schedule belongs beside the training schedule

Let $C$ contain the deduplicated initial, periodic and final commit cursors.
Let $S_c$, $I_c$ and $L_c$ be whole-operation save, inspect and load cost vectors,
and $D$ the declared final diagnostic-load count. The separate
[fixture calculator](../../src/dongxi_llms/snapshot_io_schedule.py) uses

$$
W_{\mathrm{fresh}}=\sum_{c\in C}S_c+D L_U,
$$

$$
W_{\mathrm{resume}}(k)=I_k+L_k+S_k+\sum_{c\in C,\ c>k}S_c+D L_U.
$$

The resumed path inspects and loads its parent, then commits the restored
boundary into the new invocation before continuing. A second semantic validation
during restore is not a second payload load; count each actual operation in its
own ledger. For complete-attempt capacity $A$, an allowance can combine fresh
work with $(A-1)$ times a **componentwise** maximum of declared resumed vectors.
That conservative envelope need not describe any one measured trajectory, and
it does not authorize the attempts. Keep the original model/semantic limits
unchanged rather than silently appending or refilling dimensions.

### A pending pool changes the publication schedule

The completed-only calculator must not be silently reused for native RLVR.
Its lifecycle publishes an initial boundary, every complete collected pool and
every applied update. For $U$ total updates, let $C_u$ denote the completed
boundary after update $u$ and $P_u$ the next collected pool while only $u$
updates have been applied. Fresh execution saves $C_0$ and the alternating
sequence $P_0,C_1,\ldots,P_{U-1},C_U$: $1+2U$ saves.

A restart republishes its restored boundary into the new invocation. Resuming
$C_k$ requires a new collection before application; resuming $P_k$ already has
that observation. The source-derived save counts are

$$
N_{\mathrm{completed}}(k)=1+2(U-k),\qquad
N_{\mathrm{pending}}(k)=2+2(U-k-1),\quad 0\leq k<U.
$$

The pending path saves restored $P_k$, applies it without recollection, saves
$C_{k+1}$ and then resumes the alternating sequence. Each resume also performs
its separately admitted inspect and load. Saving only a trained weights cursor
would omit both the retained observation and its publication cost. A completed
terminal boundary $C_U$ can be republished without more updates; $P_U$ is
invalid and must refuse before journal opening or model allocation.

[Day25 Exercise8](../../notebooks/day-25/02_ragged_kv_cache_and_exact_recovery.ipynb)
is a smaller controlled microscope: manually publish one boundary per phase,
retain a rejected load, reopen both same physical journals and compare the
first native application. Its actual counts must not be relabeled as the full
lifecycle algebra. Both arms recover the same original tiny trajectory; only
the completed arm needs another collection before its first resumed update.
Logical reservations, realized actions and successful operations remain
different quantities. None establishes pretrained quality, physical containment,
cross-machine equivalence or permission to launch an external experiment.

### Checking recovery can itself consume work

There is a subtle third question after “Were the weights restored?” and “Was the
spending preserved?”: what work is required to establish that the recovered
state is valid? A pending RLVR pool contains old selected-token likelihoods. A
fresh forward can verify them against the retained collecting policy. Replaying
its sampling trace can require additional temporary-generator draws. These are
not new training examples or optimizer updates, but they still consume work.
Reserve verification before it runs; restoring an older snapshot must not refund
the verification either. The RLVR CPU source controls explicitly charge its
recovery likelihood/RNG checks. The current
[SFT](../../experiments/reports/2026-10-05-sft-semantic-validation-work.md) and
[DPO](../../experiments/reports/2026-10-05-dpo-recovery-validation-budget.md)
schemas also charge each actual runner-owned history/tensor/RNG semantic panel,
including repeated callbacks and restore checks. Their original scientific caps
remain unchanged; old budget schemas refuse rather than infer an upgrade.
The shared reader's hashing/loading/tree checks and save cloning/serialization
use a separate explicit nine-dimensional I/O contract, not those runner units.
Bootstrap/journal processing, caller state capture, application, inventory hashes
and other outputs remain named exclusions. None supplies a physical
quota, a complete CPU-work bound or pretrained recovery evidence.

Likewise, “generation positions” needs a computation contract. The SFT observer
keeps its original cached greedy decoding; the DPO and basic RLVR observers keep
their original uncached paths. An uncached worst-case envelope can safely reserve
capacity for cached SFT without becoming its measured count. Record the actual
cached input positions separately. Changing the generation algorithm to make a
budget test pass would change what the test compares.

The [actual DPO snapshot integration](../../experiments/reports/2026-10-05-dpo-snapshot-artifacts.md)
provides a concrete storage bridge. Its numerical snapshot carries an old artifact
journal prefix, while the live ledger keeps later snapshots and an eight-byte
failed file charged at its full256-byte reservation. Fresh replay restores
update3 and finishes6 without deleting those files. A separate joint control
ends with ten update allowances charged for a six-update numerical trajectory.
Both differences are expected history, not corrupted counters. Snapshot coverage
does not cover HF exports, metrics or arbitrary child writes.

Before using a compiled stage, connect its declared recipe to **actual encoded
lengths and masks**, all evaluation panels and an explicit retry allowance. A
length ceiling alone does not approve the resulting work vector. A byte-pinned
record of supplied encodings also does not prove that the live tokenizer produced
them: the runner must independently re-encode and compare. Finally, the bytes
parsed as caps must match the input evidence; hashing a changed file later would
document a different input. These are distinct integrity and authorization gates,
not a reason to infer approval from a well-formed JSON file.

Finally, a configuration claiming a private cgroup, quota or observer timeout is
not evidence that the platform enforced it. The
[preflight interface](../../src/dongxi_llms/production_preflight.py) checks two
fresh, consistently owned observations and treats missing backends, incomplete
GPU-owner inspection and failed drain as refusals. Its
[CPU control](../../experiments/specs/2026-10-05-production-preflight.md) uses only
authored observations. Even its positive receipt says non-production and cannot
launch anything. Real platform adapters, continuously bounded observers and
authorized model runs remain separate gates in the
[supervisor plan](../../docs/PRODUCTION_SUPERVISOR_PLAN.md).

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
11. Why can identical raw logits still produce an incorrect importance ratio?
12. How can two KL statistics agree in value but disagree in gradient?
13. Why must a sampled EOS and a collection cap produce different critic targets?
14. If a vocabulary intervention removes all held-out input collisions but ranking remains at chance, what has it established?
15. When can rehearsal improvement be continued learning rather than evidence of prevented forgetting?
16. Why can a cached batch forward fewer positions but take longer on a tiny CPU model?
17. What must happen first when resuming a snapshot taken after rollout collection but before its update?
18. Why is a strong hash insufficient if two different states share its serialization?
19. Why can a leader exit0 while the overall shutdown condition fails?
20. Why does a hard per-file limit fail to provide a total experiment disk quota?
21. Why can 204,800 positions of DPO update geometry exclude most logical forward work?
22. Why must an attempted update after a snapshot remain charged when that snapshot is restored?
23. Why can the final checkpoint fit an artifact cap while its safe publication cannot?
24. Why is validating a mock backend receipt different from proving real platform enforcement?
25. Why can checking a recovered pending pool spend work even without a new optimizer update?
26. Why do actual encoded lengths make a budget more useful without making it authorized?
27. How can cached generation use an uncached reservation without misreporting its work?
28. Why does budgeted snapshot publication not establish a whole-output disk quota?
29. How can replaying sampler draws validate a checkpoint without changing the training trajectory?
30. Why cannot a work prefix stored only inside a checkpoint authorize the work required to read it?
31. Why cannot a checkpoint payload contain its own completed save charge, and what should recovery do with later save failures?
32. Why does a guarded loader fail to bound pre-read work if identity collection already hashed the checkpoint? What should be bound instead?
33. Why do completed and pending RLVR snapshots with the same applied-update cursor require different first actions and save schedules? Can a pre-validation saved work prefix still retain later validation charges?
34. Why can an external runtime check still fail if its resource observer or incident logger blocks? What must remain uncertain after a logging failure?
35. How can bounded parallel file hashing preserve a recovery contract, and why are a completed snapshot, a whole-child exit and a three-way replay comparison still different claims?

[Worked answers](../solutions/14-when-optimization-goes-wrong.md) and the
[Day 24–25 notebook route](../labs/14-when-optimization-goes-wrong.md) include
the reward exploit, reduction audit, version contract and bounded system budget.
