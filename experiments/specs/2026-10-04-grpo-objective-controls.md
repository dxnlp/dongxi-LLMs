# GRPO weighting and filtering controls

Mode: bounded CPU reference. This protocol is frozen before collecting rollouts
or calculating result matrices. It completes a mechanism comparison, not a
pretrained reasoning campaign or a named-method reproduction.

## Question and fixed design

Which differences come from token reduction, group reward scaling, component
aggregation or clipping, rather than new data? Compare every objective on the
same first-attempt rollouts and initial policy parameters. Separately measure
the cost of rejecting constant-quality groups and attempting new groups.

Use seeds 2401, 2402 and 2403, all retained. Initialize the existing one-layer
`TinyDecoder`: width 16, four attention heads, two KV heads, untied vocabulary
head, ten token IDs, context eight, float64 CPU. No SFT or RL training is performed
in this matrix. Prompts encode sum-positive for operand pairs (0,0), (0,1),
(1,0), (1,1), (0,2); they are symbolic inputs, not English understood by the model.
At every generated position condition on the three syntax actions EOS, 0, 1.
Temperature is one. This grammar does not supply the correct answer and allows
immediate EOS, excessive numbers and cap truncation.

Generate eight responses per prompt with cap three, including the chosen EOS
as a valid action. Preserve every ID, valid mask, conditional behavior log
probability and termination reason. Collection and recomputation use identical
conditional support. The policy is untrained; any success is an actual sample,
not evidence of reasoning acquisition.

## Reward and objective interventions

The independently defined quality target is exactly the correct numeral followed
by EOS. The scalar proxy combines three declared components:

- answer-only correctness of the first numeral, weight 1;
- exactly one numeral followed by EOS, weight 0.4;
- numeral count divided by cap three, weight 0.2.

The verbosity component is deliberately a potential shortcut. Save component
values and independent completed-task quality separately. Compare centered total
reward, total-reward population-std scaling, and component-wise population-std
scaling before weighted aggregation. Constant components contribute zero after
centering. Use epsilon 1e-8. All rewards/advantages and old probabilities detach.

Compare response-mean, global valid-token mean and fixed denominator
`responses * cap`. Include symmetric clip (0.2, 0.2) and asymmetric clip
(0.2, 0.4). At initial ratio one, the clipping variants must agree. An explicitly
constructed log-probability fixture, not a model generation, supplies ratios
0.6, 0.9, 1.1, 1.3, 1.6 for positive and negative advantages to inspect clipping.
Also compare component weights after multiplying all weights by ten; do not
silently claim component normalization makes the weighting policy invariant.

Report each objective's value, selected-log-probability gradient, decoder
parameter-gradient norm/hash, weight denominator and pairwise gradient cosine.
An independent analytic derivative must match autograd on the constructed
fixture and actual collected token scores, away from clipping boundaries.
Padding must not affect the value or gradient, including nonfinite padded scores.
Reject empty responses, non-prefix masks and nonfinite valid inputs.

## Filtering and cost accounting

Keep the first mixed independent-quality group for each prompt, if one appears.
Attempt at most three groups per prompt; retry only prompts still without a mixed
group. Preserve every rejected all-zero/all-one group and every unsatisfied
prompt. Measure attempted/selected prompts, responses and valid generated tokens,
including EOS, caps and failures. This is a collection-cost experiment, not an
equal-budget performance ranking. Do not replace first-attempt objective pools
with selected groups: that would change both data and objective.

## Acceptance and limits

Analytic derivatives, population moments, component rescaling/degeneracy,
padding, asymmetric clipping and bounded retry accounting need independent
tests. Save actual environment, source/spec hashes, seeds, conditional support,
all rollout records and failures. Retain historical GRPO negative reports.
Add a focused visual Day 22 extension with predictions, adjacent solutions,
measured arrays and a fixed reference preview; integrate into Chapters 13/14.

No selection of seeds, coefficients, tasks or caps after inspecting results.
No model download, external API, GPU, installation, server or animation rendering.
Original independent implementation; upstream references are mechanism anchors,
not correctness oracles. No complete DAPO, Dr. GRPO or GDPO reproduction is claimed.
