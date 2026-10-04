# Chapter 14 — Worked solutions

Companion to the [chapter](../chapters/14-when-optimization-goes-wrong.md) and
[Day 24–25 notebook route](../labs/14-when-optimization-goes-wrong.md).

## 1. Correct optimization of a flawed proxy

The malformed action `35 0` has proxy reward 2 while the correct `35` has reward 1.
The exact expected-reward gradient increases probability of actions above the
current average. It therefore favors the malformed answer. Rising proxy and
falling strict accuracy are predicted consequences of the programmed objective.
The paired plot measures both; reporting only reward would hide the task failure.

## 2. Repairing a checker differs from repairing weights

A corrected verifier changes future updates. It does not restore diversity,
general skills or representations already lost during previous optimization.
Our repair experiment restarts from the same initial logits to isolate objective
quality. Repairing the actual hacked checkpoint would require another controlled
run, a recovery budget and external evaluation. Do not substitute one experiment's
result for the other's claim.

## 3. Low entropy has a context

A deterministic arithmetic task may warrant concentration on the correct
answer. On an unsolved prompt, concentration on one wrong answer destroys
exploration. Inspect correctness, valid fraction, output diversity and conditional
state identity alongside entropy. An entropy bonus can broaden outputs, but
cannot make the broadened candidates correct by itself.

## 4. The actual collection distribution is the behavior policy

Temperature scales logits; top-p removes outcomes and renormalizes the rest.
These operations change the sampling probabilities. Importance ratios must use
the probabilities of the distribution that generated the stored actions, not
the unmodified model distribution unless it was actually used. The basic course
runner avoids this ambiguity with temperature 1 and full-support sampling.

## 5. Average KL hides identity and tails

A KL value requires a named reference, state distribution, estimator and
reduction. A mean can hide extreme prompts and changes in response length.
Measure quantiles, fixed-prompt values and validity. First check identical-policy
KL, logit/label alignment and masks; then test the learning-rate or regularizer
hypothesis. A threshold breach is a reason to investigate, not a completed cause.

## 6. Denominator pressure

Response means spread equal response weight over variable token counts;
token means give long responses more total weight. With positive and negative
advantages this can change relative pressures on answer content and stopping.
Record the exact formula, EOS handling and token cap. A plot of nominal weights
isolates this mechanism, while a trained length-distribution comparison is
needed to establish the realized behavioral effect.

## 7. Synchronization does not rewrite history

A response collected under version 4 retains version 4 behavior probabilities
after the learner reaches version 6. Assigning version 6 to its denominator makes
the ratio an incorrect description of collection. Synchronization affects future
generation. Metadata checks reject unintended lag; intentional asynchronous
training needs an explicit lag/correction contract and engine integration tests.

## 8. Optimize the dominating cost

If generation takes 32 seconds, learning 6, verification 0.1 and synchronization 0.5,
halving learning time saves 3 seconds out of 38.6. Doubling useful generation
throughput saves 16 seconds under this simple model. Actual batching and memory
can change these numbers; the notebook labels them projected rather than Spark
measurements. Compare useful tokens, complete update time and safety together.

## 9. Resume the trajectory, not only its weights

Restore optimizer moments and step, scheduler, RNG, data/sampler position,
policy/reference identities and rollout progress. Reusing weights with a fresh
optimizer can be a valid new experiment, but should receive a new identity.
For replay checks, use a small fixed input and restored RNG to compare the next
sample and update under the same environment. Save recovery failures honestly.

## 10. Alerts versus explanations

An actionable alert names a failed configured condition and points to evidence:
nonfinite loss, reserve below 25 GiB, changed verifier, unexpected lag or reward
improvement alongside evaluation decline. A causal conclusion additionally
needs a mechanism and discriminating intervention. “Entropy fell” is a
measurement; “the model learned the correct deterministic action” needs
correctness evidence and alternate explanations checked.
