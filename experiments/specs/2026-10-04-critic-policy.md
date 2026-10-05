# Learned critics and a frozen text reward policy loop

This specification and the original JSON inputs are written before fitting or
policy measurements. The experiment is a bounded CPU mechanism, not a pretrained
PPO stack, natural-language reasoning benchmark or human-feedback study.

## Frozen task and inputs

Use `fixtures/critic-policy/protocol.json` and the two preference fixtures.
Four train sources contain cup/key crossed with red/blue; bag supplies two
calibration sources, box two final sources and mug two withheld controls. All
source groups and actual actor/reward encodings must remain split-disjoint.
The actor sees the actual word-tokenized prompt, not a reference or source ID.
The fixed vocabulary declares all grammar words independently of held-out
labels; the reward model preserves printable ASCII rather than mapping new
words to UNK. Only train pair records fit the reward model.

The actor emits red/blue first, then EOS/plain/fancy, then EOS/plain/fancy, then
EOS only at the declared four-action environment horizon. There are fourteen
finite terminal paths. EOS at position four is the declared grammar's forced
boundary; earlier EOS is genuinely sampled. The main collector caps at three
actions, so a non-EOS prefix is censored, not terminal. Its continuation value
comes from the declared underlying environment, not an invented generated EOS.
A separate delivered cap-four evaluation is descriptive, not a repair selected
after final-test inspection.

Independent quality requires the requested color, no more than one style word
and actual EOS within the delivered cap. It is not used as training reward,
critic target, scaling criterion or checkpoint selector. Invalid/truncated
responses stay in every denominator. Pair fixtures and quality rules are
authored, not independently gathered human judgments.

## Preference models and frozen reward

Use the additive shared character reward implementation: width24, heads2,
float64 CPU, causal scalar-head decoder, printable ASCII, no dropout. Fit each
balanced/confounded arm for120 AdamW updates with seed1611, learning rate0.02
and weight decay0.01. Equal source-group weighting and pair counts do not make
the arms informationally equivalent: the confounded arm repeats the same
quality/style contrast in reverse orientation. Preserve both fitted histories
and first-backward evidence.

Save each actual model with fixture/source/interface identity, reload with its
expected file digest, and forbid all reward gradients. Score every finite path
using that saved model. Freeze the affine score center/scale on train response
strings only, with minimum scale0.1; terminal reward is the resulting tanh
score. This is an explicit bounded proxy, not a calibrated success probability.
The preference objective cannot identify an absolute reward offset by itself.

## Actor and critic budgets

Seeds2001/2002/2003 initialize one-layer causal text actors and separate neural
state-value models, width16, heads2, feed-forward width32, float64 CPU. Identical
actor initialization is paired across arms. Compare balanced/oracle,
balanced/learned, balanced/noisy and confounded/learned. All twelve rows receive
40 fresh rollout batches, four train prompts and eight independently sampled
paths per prompt, cap3, gamma1, lambda0.95. Use one actor update per batch:
AdamW lr0.01, weight decay0, gradient clip1, local clipped surrogate epsilon0.2,
masked token sum then response mean, exact conditional KL beta0.02 against a
frozen initial reference. With fresh single updates the initial ratios are one;
this isolates advantages and critic error, not PPO epoch reuse.

The learned critic receives one separate AdamW lr0.01/weight decay0 MSE update
per batch against detached lambda-return targets, valid-state mean and clip1.
Actor and critic have separate backbones. Oracle values enumerate all possible
future paths under the actual current actor and frozen reward. Noisy values add
predeclared deterministic state-keyed noise of amplitude2 to the oracle;
noise is pre-action and cannot inspect the chosen next action. Oracle/noisy
arms take zero learned critic steps, so equal rollout budgets are not equal
optimizer or wall-clock budgets. Archive both work clocks.

Rewards occur only at emitted EOS. TD/GAE masks distinguish terminal EOS from
collector truncation, ignore post-EOS padding, and bootstrap capped prefixes.
Policy advantages, old likelihoods and critic targets are detached. Critic
MSE and policy losses have separate gradient graphs; reward has neither.

## Evaluation and retained evidence

At scheduled updates0/20/40, retain greedy plus eight fixed-seeded sampled
responses for every source. Never select a best seed/checkpoint after observing
test quality. Record all token IDs, valid masks, rendered response, natural
termination, truncation, raw/scaled proxy scores, independent quality, stop
reason, prefix/response-token costs and actual wall time. Exact finite-policy
expected proxy/quality values are diagnostics, not policy training gradients.
Keep a frozen unchanged policy control and final cap4 intervention.

Keep all train rollout tokens, old log probabilities, rewards, values,
bootstraps, TD residuals, advantages, critic targets, losses and measured work.
Retain negative trajectories and reward/quality disagreement. Fitted reward
weights and actor/critic parameter-byte identities are actual identities; no
pretrained ancestry, throughput, Mac or GPU claim follows from them.

## Acceptance checks

Independently enumerate returns/TD/GAE and lambda0/1 endpoints; verify cap
bootstrap versus deliberately broken cap-as-EOS and padding invariance. Test
likelihood support, sampled shift, critic/actor/reward detach boundaries, frozen
state identities and deterministic seeds. Report proxy and quality across all
arms/sources without promising gains or claiming critic superiority from one
small task. Execute the new notebook in a fresh isolated CPU kernel; export and
inspect measured figures; verify math, focused tests and report source hashes.
No installation, model acquisition, GPU, service, Git mutation or animation
production is included. The learner remains on Day9.

