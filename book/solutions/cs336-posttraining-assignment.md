# Worked solution: a compatible post-training assignment

The question is whether a delivered answer improves because the policy learns
the task, learns to stop, or receives broader source coverage. The executable
route is [the reward audit](../../notebooks/extensions/cs336/04_reward_signal_audit.ipynb)
followed by [the compatible chain](../../notebooks/extensions/cs336/07_posttraining_integration.ipynb).
Each checkpoint has its runnable reference and explanation immediately adjacent.
The consumer is [posttraining_assignment.py](../../src/dongxi_llms/posttraining_assignment.py);
it reuses the existing decoder, SFT loss, sampler, group advantages, clipping,
categorical KL, and mathematical grader.

## The interface makes the comparison possible

The original fixture encodes parity with ten tokens. Inputs are
`[BOS, operation, NUMBER_START+a, NUMBER_START+b]`; responses are an answer token
and EOS. English fixture prompts describe the problem but are not parsed by the
decoder. A source is the underlying ordered operand pair and task family.
Training, development, and held-out parity sources are disjoint. Both direct and
alias forms are present on each source, including the identical held-out panel.
The independently withheld equality operation uses a new operation token;
overlapping operands do not make it a trained task.

All four splits have at least two distinct sources per answer class. Constant 0,
constant 1, and the training-majority baseline each have accuracy 0.5. The
baselines emit EOS, so their complete-response accuracy is also 0.5. Omitting this
partial baseline would make a formatting gain look like a reasoning gain.

The SFT labels mask the first four positions. After the loss's single causal
shift, position 3 predicts the answer and position 4 predicts EOS. Coverage uses
eight sources in two templates; repetition uses two sources in two templates,
each repeated four times. Both arms see 32 supervised labels per update and 768
across 24 updates, starting from the same actual random parent. Adding prompt
labels changes both the objective and exposure; it is the deliberately broken
mask in checkpoint 1, not an alternative result selected after evaluation.

SFT uses full-vocabulary cross-entropy. Reward collection uses the declared
position-specific conditional support: answer 0/1 first, then EOS/0/1. EOS is
sampled, never appended by an oracle. Old-policy scores, entropy, and KL use
that same conditional support. Full-vocabulary and cap 1 generation are separately
labeled fixed-checkpoint interventions.

## Actual ancestry and sibling budgets

The frozen protocol uses seeds 33671, 33672, 33673, 24 SFT updates and 16 sibling
updates, with a 180-second total CPU cap. The actual reference completed in
2.173 seconds on the task-local CPU environment. All seeds and raw paths are
retained. Random parent → coverage-SFT is the common sibling lineage; the
repetition-SFT arm is a same-random-parent comparison. The historical
TinyStories, Qwen Base/SFT/preference, and acquired Instruct/RLVR checkpoints
remain separate roots.

Accepted-response SFT trains only on the actual sampled correct complete paths.
It uses accepted-token-mean full-vocabulary CE. Failed paths are not replaced by
oracle answers, and an empty accepted pool skips the optimizer step. Reward
siblings include every eligible response and use the existing response-mean
clipped objective, with either population-std or center-only group advantages.
The frozen SFT parent supplies the categorical KL reference.

The first sibling pool is exactly shared because weights and RNG agree. Later
fresh collections follow the diverging policies. All main sibling arms attempt
2048 responses and 4096 generated response positions per seed, but accepted SFT
uses 1692, 1572, 1746 supervised labels across the three seeds. Those are distinct
clocks. The reports additionally count physical generation prefixes, scoring
inputs, and accepted-SFT forward inputs; equal attempts are not equal compute.

## What the measurements support

The exact held-out conditional complete-path probabilities are:

| Seed | Coverage SFT | Repetition SFT | Accepted-response SFT | GRPO std | GRPO center |
|---|---:|---:|---:|---:|---:|
|33671|0.3792|0.2104|0.4602|0.4719|0.4713|
|33672|0.3562|0.3505|0.4481|0.4627|0.4624|
|33673|0.3863|0.1539|0.4608|0.4719|0.4712|

These are exact probabilities of the two-token correct path under the frozen
conditional policy, not sampled fractions. The notebook also shows actual
fixed-seeded sampled answers, stops, and complete responses. Coverage improves
this delivery probability relative to repetition on the retained panel, with a
small difference in seed 33672. Siblings improve termination while answer-only
transfer remains weak. Every final exact score remains below the always-stopping
constant baseline 0.5. These measurements support a useful signal and delivery
diagnostic, not established parity generalization or English capability.

Every seed supplies training/development successes and failures, mixed groups,
and a separately measured task-reward gradient. The independently withheld
equality negatives, source-paired regressions, and descriptive source standard
errors remain in the reports. Two templates do not double the number of
independent problems. Nothing on the held-out panel chooses the recipe, caps,
difficulty, or checkpoint.

## Finite groups and the cost of difficulty selection

Let $r_i$ be an independent Bernoulli reward with success probability $p$ and
let $p$ be parameterized by a logit $\theta$. Its score is $r_i-p$, and the true
reward gradient is $p(1-p)$. For the inclusive group baseline
$\bar r=G^{-1}\sum_i r_i$, the center-only estimate is

$$
\hat g=\frac{1}{G}\sum_{i=1}^{G}(r_i-\bar r)(r_i-p),\qquad
\mathbb{E}[\hat g]=\frac{G-1}{G}p(1-p).
$$

The response participates in its own baseline. Dividing by the group's sampled
population standard deviation introduces another dependence. The reference
enumerates every joint group and its probability; it never substitutes the
expected denominator or sums independent normalized response variances. At
$G=2$, std-normalized mixed pairs give expectation
$p(1-p)/(1+2\varepsilon)$, with the kernel's $\varepsilon=10^{-8}$. All-constant
groups give zero. The large-group std-normalized limit is
$\sqrt{p(1-p)}$, while center-only approaches $p(1-p)$; those are different
scales, not a promise of universal variance reduction.

Keeping mixed groups has probability $1-p^G-(1-p)^G$. Originally uniform
prompt slices are reweighted by that probability after selection. Discarded
groups still consume collection work, and the conditional expected gradient is
a different estimand. The assignment chooses prompt eligibility once from
frozen-parent **training** probabilities in the predeclared range 0.05–0.95;
development is a signal audit only. The actual reference kept all training
prompts. Its mixed-only selection analysis is explicitly counterfactual;
constant groups are never silently removed from the real update.

The all-zero-reward gradient probe changes the policy away from its frozen
reference. Its task gradient is exactly zero while KL produces a nonzero
gradient. The real updates therefore report task and weighted-KL gradients
separately, their dot product, and the actual combined objective gradient.
Accepted-SFT's GRPO components are labeled counterfactual diagnostics; its
actual update remains CE.

## The historical verifier stays historical

The audit replays all 192 retained G4/G8 responses and reproduces every original
zero reward from the original stop and `verify_integer`. The separately named
`cs336-bounded-answer-audit-v1` accepts one terminated boxed 9 rationale for 4+5;
the other 191 responses have unsupported richer parses. Full text, token IDs,
original stops, archive hashes, and reason codes are retained. This diagnostic
does not rewrite the rewards that produced the old updates.

Authored probes separately test answer correctness, parser support, format,
natural termination, and truncation against independent references. A capped
correct answer earns no complete reward. A richer correct prose answer can fail
a stricter `single_integer` format policy. Unsupported and unsafe expressions
receive no answer credit; nothing is evaluated as Python.

The frozen specification, commands, all seed checkpoint states, complete raw
collections, source identities, executed notebooks and labeled figures are in
[the new report directory](../../experiments/reports/2026-10-11-cs336-upgrade/run-01/posttraining/).
The optional native capability audit has a separate named protocol and execution
record. Same-parent native SFT/RL training remains a future gated experiment;
the CPU mechanism result does not authorize or demonstrate it.
