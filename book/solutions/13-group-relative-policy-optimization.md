# Chapter 13 — Worked solutions

Read the [chapter](../chapters/13-group-relative-policy-optimization.md) and
make a prediction before opening an answer. Runnable versions are adjacent to
the exercises in the [notebook pathway](../labs/13-group-relative-policy-optimization.md).

## 1. A successful group can teach nothing relatively

If all rewards are one, each centered reward is zero. The standard deviation is
zero, so our explicit constant-group rule returns zero advantages. The group
contains evidence of success, but no information about which sampled response
should become more likely relative to the others. Across later prompts, the
shared parameters may still change. KL regularization or optimizer momentum can
also move them; “zero relative signal” is narrower than “no weight movement.”

## 2. Population versus sample standard deviation

For rewards 0, 1, the mean is 0.5. Population standard deviation is 0.5 and sample
standard deviation is $\sqrt{0.5}$. Normalized advantages are approximately
−1, +1 with the population convention and−0.707, +0.707 with the sample convention.
The sample convention reduces this group's scale by $\sqrt{(G-1)/G}$.
It must be declared because the change affects optimization, especially for small groups.

## 3. The denominator changes prompt weighting

Centering subtracts a local comparison; dividing by a reward-dependent standard
deviation scales that comparison differently across groups. Binary groups with
success fractions near 0 or 1 have a smaller spread than balanced groups.
Moreover, the statistics include each response's own reward. The exact
independent-sample baseline identity from REINFORCE does not make all these
normalized estimators identical or unbiased. The notebook changes reward spread
while retaining signs and exposes the changed advantage magnitudes.

## 4. Zero value, nonzero slope

At initial ratio 1, centered advantages can sum to zero, producing a zero average
surrogate value. The derivative contains each answer's log-probability derivative:

$$
\nabla J=\frac1G\sum_i\hat A_i\nabla\log\pi_\theta(y_i\mid x)
$$

for the simple one-token, unclipped case. Those gradients point in different
directions. The scalar sum of advantages being zero does not make this vector
sum zero. The actual decoder experiment records a first-update gradient norm.

## 5. Negative-advantage clipping

With $A=-1$, ratio 0.5 and clip interval[0.8, 1.2], the unclipped product is−0.5
and clipped product−0.8. The minimum is−0.8, stopping further reward for reducing
the ratio below 0.8. At ratio 1.5, the minimum is−1.5 rather than−1.2, so the
unfavorable increase remains penalized. Clipping treats direction relative to
advantage, not merely numerical distance from one.

## 6. A value identity does not specify a full derivative

The expectation of $k_3$ under the current distribution equals forward KL.
When differentiating that expectation, both the integrand and its distribution
weights depend on parameters. Autograd through a fixed sample includes only the
integrand's path unless a score-function or equivalent correction supplies the
other path. Sampling under an old distribution adds another mismatch. The
notebook computes exact categorical KL and verifies its full logit gradient,
making the implemented regularizer unambiguous at visited states.

## 7. Length reductions

For lengths 2 and 8, response means assign each response half the total nominal
weight, so per-token weights are 0.25 and 0.0625. Global token means assign every
token weight 0.1, yielding total response weights 0.2 and 0.8. Actual gradients
also depend on token log probabilities and advantage signs. The coefficients
alone do not prove how generation length will change after training.

## 8. Prompt gradients survive a response-only loss

Response states attend to earlier prompt states. Differentiating response logits
therefore reaches prompt representations and their embedding/projection
parameters through attention and the residual stack. Masking a prompt's direct
loss removes a local objective term; it does not detach that state's computation.
A deliberate `.detach()` would change the graph and must be justified separately.

## 9. Strict checking can reject valid work

Our one-integer rule rejects a correct mathematical explanation containing extra
words. That is an intentional task-interface constraint, not proof the answer's
reasoning is wrong. Check allowed equivalences and adversarial cases separately.
Tests reject multiple answers, embedded correct substrings and truncation while
accepting the declared whitespace/sign variants. Record the raw response and
checker version to diagnose parser failures without rewriting history.

## 10. Implementation readiness and model evidence

CPU tests show that the selected objective, masks, gradient boundaries and tiny
decoder path execute. They do not measure Qwen3's GPU memory, throughput,
held-out math accuracy or safety. The optional HF runner requires a supplied
local model and pinned revision. Its model-scale experiment remains unexecuted
until a separate report records that run. A four-prompt CPU held-out score is
descriptive, and every failed row remains part of the report.

## 11. A constructive control is not a guarantee of every-item learning

Known solvability establishes that a correct path exists, not that stochastic
optimization must discover it for every initialization. The new shared decoder
receives actual sampled response rewards, unlike the separate exact lookup arm.
Its first answer and conditional EOS probabilities both change. All three
predeclared seeds improve mean complete-path probability; two still fail 0+0.
That minority example can disappear into a majority-answer shortcut while
stopping improves. Keep the item-level probabilities, failed seed rows, frozen
policy and oracle. Do not choose another checkpoint after reading the test data.

The adjacent references in the
[constructive-control notebook](../../notebooks/day-23/03_reasoning_tasks_and_positive_controls.ipynb)
show the actual sampled tokens, old likelihoods, EOS gradient and layer-gradient
reach. The diagnostic exact expectation is not differentiated as decoder training
loss. EOS is sampled within the declared grammar and included in the loss.

## 12. A constant baseline can explain a perfect slice without reasoning

Every new-source sum-positive item has at least one positive operand. Always
returning 1 plus EOS therefore scores 100% on that slice,75% on the training set,
and 50% on the odd-sum family. Those percentages match a shortcut's availability,
not proof the decoder learned the intended arithmetic rule. Four prompts per
slice provide little coverage. Examine zero-sum cases, balanced task variants
and generated answers before proposing a new development experiment. Do not
silently retune the current final test or replace its frozen panel.

The lookup arm makes a related limitation explicit: trained keys improve, but
all unseen keys retain probability 0.5. It supplies EOS by construction and has
a different objective/budget, so it cannot be ranked against sampled decoder
training as though those conditions matched.

## 13. Weight changes, inference changes and rationale claims

Base and instruct refer to different weight checkpoints and training histories.
Raw versus chat is a prompt-interface choice on specified weights. A supported
thinking toggle changes a template/output path, not the underlying checkpoint.
Changing the response cap changes a computational constraint; it is not training.
Record all four axes independently, including actual tokenizer/template/weight
identities, stopping and costs.

Cap 1 can contain a correct numeral without EOS. The answer grader may mark it
correct while the complete-path reward is zero. A longer cap is an intervention,
not proof of reasoning, and final-answer correctness does not validate rationale
faithfulness. The original English arithmetic/algebra/two-step panel currently
has independently authored oracle/parser controls, not a tiny-decoder language
score. A [separate actual report](../../experiments/reports/2026-10-05-native-reasoning-rlvr-evidence.md)
now incorporates all eight initial conditions: six accepted complete invocations
and two failed Base/custom-chat partial invocations. It establishes bounded
native off/on and cap observations, not complete 800-response coverage, a broad
model ranking or native RLVR gain. Negative declared grades expose a model-output/grader-interface
limitation. Keep those measured rows separate from the historical symbolic controls.

## 14. Changing scale versus changing direction

Global token mean and fixed-cap mean share the same numerator on a fixed pool.
Their parameter gradients differ by $\sum_in_i/(NC)$. Response means insert a
separate $1/n_i$ inside that numerator. They change relative contributions from
short and long responses, so the summed gradient can rotate when individual
token gradients are not collinear. Cosine one still does not imply equal
magnitudes or equal behavior after repeated training.

The [notebook](../../notebooks/day-22/03_objective_weighting_and_filtering.ipynb)
shows exact coefficients, three-seed measured gradient norms and a gradient-cosine
matrix. It applies no update, isolating an objective fact from a performance claim.

## 15. Normalization and preference weights remain separate

Component normalization removes each component's observed within-group mean and
spread. Its subsequent weight still changes its contribution. Multiplying all
weights by ten multiplies component-aggregated advantages by ten. Normalizing the
total can largely cancel common positive scaling, subject to epsilon; these
operations are not interchangeable. Constant components provide no relative signal.

A normalized verbosity reward remains a verbosity reward. Separate strict task
quality detects capped answers that earn proxy credit without completing with EOS.

## 16. Clip directions follow advantage signs

For positive advantage, enlarging the upper boundary permits more encouragement
to increase the ratio. For negative advantage, the favorable movement is reducing
the ratio, controlled by the lower boundary. Changing only the upper boundary
does not move that lower threshold. Unfavorable movements remain penalized.
The constructed ratio-1.3 fixture has zero positive-advantage derivative at upper
boundary 1.2 but a negative derivative at boundary 1.4. Gradient descent increases
the selected log probability only in the latter case. At ratio one both agree.

## 17. A selected batch hides rejected collection

Retain all attempted groups and valid actions, including unsuccessful groups,
sampled EOS, caps and unqualified prompts. The measured seeds select forty
responses each but attempt seventy-two, seventy-two and fifty-six. Selected
token counts differ too. Uncached full-prefix position counts are not FLOPs;
collection and analysis rescoring add work. Independent quality and a matched
budget are required before an algorithm-performance comparison.

## 18. Preserve observations, not just weights

The retained group is already a realized draw from the behavior policy. Replacing
it can change its actions, stopping lengths, rewards, normalized advantages and
resulting update. A pending restart keeps policy/Adam at the completed update
boundary but restores the post-collection source cursor and RNG together with
that exact pool. Its detached old likelihoods and policy version remain part of
the objective. Recompute current/reference scores on the retained prefixes and
apply once before new collection; preserve the original reference, not a copy of
the updated policy.

Collected and applied work differ at this boundary. Retain both counters and
the complete committed numerical history. A no-collection spy can verify that
pending restoration consumes its group rather than silently sampling again;
weights-only equality cannot prove that property. Partial generation and partial
optimizer work are separate failed attempts, not accepted pending/completed states.
Tiny actual-loop CPU replay does not establish pretrained/CUDA/BF16 recovery.

The external receipt must bind both the exact sampled state and its saved
model-work/I/O prefixes before payload inspection. Those prefixes need not equal
the journals' final cursor: save validation and publication create later work.
Reopen the same physical journals and retain those suffixes. The native save's
historical model-work capture occurs before semantic validation; preserving it
while retaining the later charge is correct. Refreshing the prefix merely to
make counters look aligned would change the saved boundary. Exercise 8 in the
Day 25 recovery notebook pairs this distinction with a native completed/pending
comparison; the model-work 23 and shared-reader 9 dimensions stay separate.

## 19. Follow one response through independent predicates

Begin with the actual receipt and raw record, not a rewritten answer.
`Sample1009`/`math10` records raw `5 + 6 = 11<|im_end|>` and scoring text
`5 + 6 = 11`. The declared turn marker explains natural termination and
`truncated=false`. `format_policy="any"` explains `format_valid=true`; this is
not a promise that the rational parser can understand an equation. The frozen
`boxed_or_final` extraction falls back to `mechanism="whole"`, whose answer
contains the whole equation. Canonicalization is `UNSUPPORTED`, with reason
`outside exact rational grammar`. Consequently accepted `correct` and
`task_success` remain false.

Execution acceptance proves all planned records exist and the whole child exited 0.
It does not promote an unsupported grade to a supported wrong answer, and it does
not prove every visible mathematical statement is false. Conversely, noticing
the valid equation does not license a replacement headline. Preserve the frozen
grade and describe this case qualitatively as a grading-interface limitation.
A new bare-numeral instruction or extraction policy needs separate declaration,
controls and actual evaluation. Final-answer correctness would still not prove
the rationale's faithfulness.

Use the controlled held-out denominator: eleven distinct problems,44 sampled
attempts and 11 separate greedy attempts. Exclude the nine diagnostic problems,
including `math9`'s RLVR training overlap. No best-of-four selector was deployed;
candidate availability and delivered-answer accuracy would be different systems.
More precisely, the nine diagnostic roles comprise the eight original fixture-train
items `math1`–`math8` and `math9`'s known RLVR-fixture overlap; only `math2` and `math9`
carry true RLVR-overlap flags. Those roles do not establish that full400/chosen/DPO
policies consumed this math data or that upstream checkpoint contamination occurred.

The newly accepted Base/raw 32 and Base/raw 128 children also retain all 100 original
records each and exit 0. Base/raw 32's controlled sampled slice has 1 natural stop
and 43 caps; all 11 controlled greedy attempts cap, with no declared correct row.
Base/raw 128's sampled slice has 2/44 declared correct,9 natural stops and 35 caps;
its greedy slice has 1/11 correct,2 stops and 9 caps. Those three held-out correct
answers use supported boxes and naturally terminate. Two other correct cap 128
rows are seen/overlap diagnostics, not held-out success. Any-candidate availability
among its four samples is 2/11 problems; greedy delivers a correct answer on a
different 1/11. No selector delivers the sampled successes.

Base/raw versus Instruct/native-chat changes both weights and serialization, not
only instruction fine-tuning. The two Base/raw caps hold those identities fixed
while changing the allowance; a few supported boxes at 128 do not promise monotonic
per-problem gain or validate visible reasoning. Keep the failed same-weights chat
rows distinct from the accepted negative thinking-on rows; neither is a license
to invent scores or complete missing comparisons.

Instruct/thinking-off 128 is accepted with all 100 records and exit 0, using the
same weights, source, original panel, native interface and decoding settings as 32.
The output cap is the intervention. Its controlled sampled answer grade is 13/44,
with 35 natural stops and 9 caps; greedy is 7/11 with 11 natural stops and no caps.
All 100 satisfy permissive `any` format. Two correct sampled boxes still cap 128:
`math19`/seed 1009 and `math18`/seed 1019. `correct` and `task_success` in this parser
panel are true for those boxes even though natural termination is false. Do not
turn them into strict EOS-complete paths, rewrite the flags or validate the
rationale by implication. Availability among four samples is 8/11 controlled
problems, distinct from 7/11 greedy-delivered answers and any unimplemented selector.
This controlled budget observation neither trains new weights nor guarantees
monotonic per-prompt improvement or faithful reasoning.

Base/custom-chat 32 and 128 are actual failures, not pending outcomes.
`Sample1019`/`math6` ends its 24-action trajectory in unmapped ID151768 and records a
decode error. For cap 32 the child exits 1;40 records and 60 missing responses remain. Its
56.54799546097638 child seconds and 56.59703186398838 supervisor seconds are
different, overlapping scopes. Minimum sampled available bytes are 122019704832,
stop reason is null and cleanup errors are empty; no deadline stopped this child.
The 39 unsupported records, one error and all their cost/IDs remain evidence.
Do not count missing cells as wrong answers, or convert the failed row to success
by silently filtering, replacing or resampling an unrepresentable action. The
cap 128 condition independently fails at the same 24th unmapped action with
40 retained/60 missing,39 unsupported/one error,8 natural stops and 31 caps.
Its native exit is 1 after 132.66686874401057 seconds; supervisor interval is
132.72932148602558, minimum sampled bytes 120977350656, stop reason null and
cleanup errors empty. It retains 4491 actions and 377459 positions without a
filter, retry or source/gate change. This is a second fixed-budget condition,
not a repaired cap 32 run. Thinking-on now has actual negative results at both
caps. G4 recovery run 01 has a cleanup-gate failure and fresh 02 has accepted exact
recovery; G8/01 also has accepted exact recovery. G8 pilot 16 now passes, while
G4's completed 16 export is separately consistency-closed without rewriting its
failed supervision. Common-panel outcomes are now measured separately in the
[matched results](../chapters/13-group-relative-policy-optimization.md#1384-read-the-matched-outcome-not-the-pooled-count).

The 680 retained records across eight invoked conditions include two errors;
800 planned slots leave 120 missing. Six complete accepted invocations account
for 600 records, not 600 correct answers. Do not call the 680 successful completions,
pool them into an accuracy score, invent missing observations or claim all 800
original panel slots are covered. Retaining failed Base/chat evidence also does
not create a new mandatory gate requiring a successful rerun. Continue to apply
the original criteria and dependency contracts, not a retrospectively altered task.

## 20. An iteration is not a quality claim

The completed loop cursor identifies a committed numerical boundary. An optimizer
application is an actual operation in that iteration; nonzero relative policy
signal additionally requires nonconstant group rewards. All-correct and
all-wrong groups both have zero relative advantages. The fixed native application
path still recomputes likelihoods and exact KL, backpropagates, clips and invokes
AdamW when valid. Zero advantages do not by themselves prove no optimizer call,
no momentum effect or no parameter movement. Inspect actual histories and state
identities rather than infer these events from baseline parser grades; the RLVR
strict integer reward and the publication parser are distinct instruments.

Exact recovery adds another obligation: source 2, completed 1→2 and pending 1→2
must agree in numerical state and history while retaining the same physical
journals' later work. Applying the retained pending group without collecting
again preserves a realized observation, not merely a seed. A completed pilot 16
then needs a completed horizon and bound final export through its explicitly
declared consumer. Neither recovery nor
training acceptance proves quality improvement. That requires unchanged
Instruct/G4/G8 evaluations on the common twenty-item contract, separately per
cap and decoding cell, with overlap excluded and failures/costs retained.

## 21. A supported thinking flag is not a completed answer

Both native thinking-on caps are accepted execution outcomes with 100 records,
zero adapter errors and all 100 capped/unsupported grades each. For either cap,
the controlled sampled system is 0/44 and the separate greedy system 0/11; no
natural stop occurs. `format_policy="any"` makes each record format-valid, not
parser-supported. Raw output keeps the thinking text; no response at either
budget closes `</think>` or supplies a box. `Sample1009`/`math10` at 128 opens a
discussion of 5+6 but does not complete its answer. Do not rescue a numeral,
declare all mathematics false or infer the result of a longer unrun horizon.

Off/on shares the actual weights, tokenizer, native template and semantic
interface. The supported flag nevertheless changes every rendered prompt: off
precloses an empty thinking region while on permits generated `<think>` text.
That is a serialization/output-path intervention, not a new checkpoint or an
unchanged prompt. Within thinking-on 32→128, all 100 attempt seeds and prompt IDs
agree and the shorter action sequence is exactly the longer run's prefix. This
within-run-pair evidence does not establish cross-platform bitwise determinism.

On 32's 3200 actions and 124960 uncached prefix positions become 12800 actions and
1114240 positions at 128, with zero supported grades at both. Whole-child times
are 144.33200714498525 and 364.1156065230025 seconds; supervisor intervals are
144.3781207689899 and 364.1712747570127, respectively. Those scopes overlap;
positions are not FLOPs and recorded eager/BF16 configuration is not a kernel
trace or serving benchmark. The contrast with off 128's bounded delivered scores
does not establish universal mode superiority, rationale faithfulness or method
ranking. A new budget or extraction rule requires a new declared experiment.

## Evidence-reading extensions

The chapter keeps the mechanisms and conclusions; this section retains the
original operational questions and their evidence paths. Read these after
[Appendix D's supervision distinction](../appendices/d-reproduction-and-environments.md#supervision-and-durable-evidence).

22. Three G4 recovery children exit 0, all work/I/O tickets close and all helper
    cleanup receipts complete, yet the adapter fails. Which independent gate is
    missing, and what can the retained zero-advantage, nonzero-gradient history
    establish without claiming accepted replay or reward learning?
23. Which new run 02 receipts establish accepted exact recovery without erasing
    failed 01? Why are its twelve passed checks, unchanged source/input closure
    and four physical applications still insufficient to claim improved reasoning?
24. Why does accepted G8 recovery retain 384 sampled slots but 314 newly valid
    actions and 434 applied targets? Which masks, rollout/restoration histories
    and stopping fields explain those different counts without calling them FLOPs
    or an equal-compute quality comparison?
25. A pilot completes sixteen native iterations/export and exits 0, then fails
    final logging acknowledgment. Which facts remain measured, which acceptance
    and publication claims are still unavailable, and why is natural stopping
    across 64 zero-reward training responses not a learning-success proxy?

## 22. Native exit 0 does not waive the cleanup and replay gates

The actual G4 run 01 source 2, completed 1→2 and pending 1→2 native children all
exit 0. Their child durations are 179.35619652300375,131.65557820600225 and
107.62926277797669 seconds. The last supervisor nevertheless records
`Observed descendant cleanup timed out` and status `failed`. Its three helper
cleanup receipts show terminated helpers and both feeder threads stop, but
those fields address different processes from the observed-descendant gate.
The adapter fails before reaching its CPU comparison and writes no acceptance.
Later process absence, readable reports or completed snapshots cannot substitute
for an accepted whole invocation plus exact numerical/state comparison. Each
child's 600-second/25GiB bounds and all failed spending remain unchanged.

Read the shared journal rather than add restored report histories. Four measured
`train_updates` split 2/1/1 among the children; three collections split 2/1/0.
Pending applies 64 saved targets without collection. The journal retains 160
newly collected valid tokens and 224 application targets, with twenty-five closed
work reservations. Fourteen closed I/O reservations retain ten saves, two
inspections and two loads. None of this repairs the cleanup failure.

Both source groups have zero rewards and advantages. Actual gradient norms are
5.811452865600586e-7 and 0.1796875, with update 2 pre-update exact KL
0.0010778990108519793 and different recorded post-update policy identities.
The code applies a clipped objective plus exact-KL regularizer and calls AdamW
with zero weight decay when valid; it does not skip constant groups. A displayed
zero objective value need not imply a numerically zero derivative. Do not call
these operations zero work, infer skipped updates, or attribute their movement
to a nonzero task-relative reward signal. Nor do these failed-attempt measurements
prove exact recovery, verifier success or common-panel reasoning improvement.

The separately reviewed controller correction defers numerical imports before
spawn cleanup and permits an explicit, sealed recovery 02 selector. It preserves
the original native code,0.5-second acknowledgment,600-second children,23-work/
9-I/O ceilings and run 01 failure. Fresh G4 run 02 is now accepted; pilot 01 may
select 02 only explicitly after that own gate.
Neither CPU controls nor current queue activity retroactively passes run 01 or
pre-signs pilots or common-panel quality. G8's own accepted recovery is separate.

## 23. An accepted replay still needs a separate quality experiment

G4 run 02's actual acceptance records twelve literal true checks, three completed
supervisors/native exits 0, null stop reasons and empty cleanup errors. The
adapter now reaches the streamed three-way comparison: all ten final scientific
components and update 2 metrics agree. That covers policy, original reference,
optimizer, RNG, loop contract/history/cursor and numerical work/pending state,
while excluding only producer invocation and physical spending prefix. It is
stronger than successful saves, readable exports or matching scalar metrics.
Its retained closing receipt binds the same source/input/model/prerequisite
bytes before and after execution; only controller/test source hashes differ
from failed 01, not the Qwen implementation, supervisor, baseline or recipe.

The fresh 02 journal again measures four applications, three collections,160
fresh valid tokens and 224 applied targets; pending applies 64 retained targets
without resampling. I/O retains ten saves, two inspections and two loads. Those
shared run 02 histories do not reset or refund failed 01's distinct journals.
All source rewards/advantages remain zero despite recorded gradient/KL movement.
Exact replay does not establish nonzero task signal, verifier quality or improved
answers. Pending 02 observes no owned descendant; accepted execution is not a
captured explanation of the historical cleanup failure.

Native child seconds are 176.87867476296378,143.31851184100378 and
109.66025628399802. The separate after-the-job owner transcription records
outer 0/482.61591269401833 seconds, including overlapping preparation,
supervision, inline CPU comparison and closure. Do not add those intervals or
call the outer observation a native prelaunch archive. G4 pilot 01 must explicitly
bind accepted recovery 02, then its actual sixteen-iteration export and unchanged
common 20 per-cap evaluations must exist before any quality comparison. G8/01 now
has its own accepted recovery and later accepted pilot 16. Common-panel results
are a separate observation, now completed under both original caps.
The later G4 pilot 01 attempt completes native 16/export but fails terminal logging
acknowledgment; it does not establish an accepted pilot or change failed 01.

## 24. A batch slot, valid action and replayed target are different costs

G8/01 has all twelve actual recovery checks true, three completed supervisors/
native exits 0, empty cleanup errors and exact ten-component/update 2 equality.
Its original 600-second/25GiB bounds remain unchanged. All three roles observe no
owned descendant; this does not explain the historical G4 cleanup timeout.

Its shared physical journal retains four optimizer applications versus three
collections. Dense batch sampling spends 384 slots/draws, including stopped-row
slots, while the response mask retains 314 newly valid actions. The numerical
source has 74 valid tokens with seven EOS/one cap, then 120 with two EOS/six caps;
completed resume freshly collects another 120. Pending applies its saved 120
without collection. Applications score 194 source targets plus 120 plus 120,
giving 434, not 384 or three restored cursors summed as six fresh updates.
The 14 I/O operations are ten saves, two inspections and two loads; their later
spending and failed G4/01 costs remain retained under distinct journal identities.

Both source groups have eight zero rewards/advantages despite some natural stops.
Their gradient norms are 4.3585896492004395e-7 and 0.146484375, with update 2 KL
0.0009698036592453718. Constant strict-reward groups supply no task-relative
signal, not an automatic no-op for the numerical exact-KL/AdamW application path.
Recovered state equality proves this fixed run replays; it does not prove
reward-driven quality improvement, equal token budgets across G4/G8 or faithfulness.
Keep the pilot 16 cap 64 and common evaluation caps 32/128 separate from recovery 16.

## 25. Trained but unaccepted is neither no training nor a quality pass

G4 pilot 01 explicitly selects accepted recovery 02, uses the original cap 64/
sixteen-iteration recipe and actually completes native cursor 16/export with
child exit 0 after 1275.7366294650128 seconds. The supervisor interval is
1275.7819221359678 seconds with minimum sampled available bytes 108300517376,
null stop reason, empty cleanup and no observed descendants. Its terminal writer
acknowledgment is nevertheless false and journal error is `Final logging timed
out`. That distinct failed gate is not a native crash, cleanup timeout or external
1800-second deadline. The adapter writes failure, not acceptance/closing/export
inventory. Surviving weights are not an accepted evaluation parent.

The native journal retains 67 closed reservations, sixteen applications and
sixteen collections. It spends 1508 dense slots/draws versus 1004 valid targets;
all 64 responses naturally stop but every reward/advantage group is zero.
Thirty-three saves and 173146264423 serialized bytes remain spent. All sixteen
recorded post-update policy identities differ, and the final gradient/KL are
0.07666015625/0.0006887196796014905. Those observations do not establish positive
task-relative signal or mathematical quality, and closed native tickets do not
satisfy terminal record retention. Initial/final four-item diagnostics are not
the common twenty-item panel. The owner's outer 1/1293.7624233269598-second
observation overlaps this work and is not added as another model invocation.

The later explicitly declared export-consistency consumer checks the actual
completed 16 state/export without changing the failed supervisor into acceptance.
G8 separately passes pilot 16. Its 128 responses naturally stop, but all sixteen
groups also have zero rewards/advantages; sixteen actual applications and
4888 slots/2335 valid targets do not establish positive reward learning or an
equal-budget group-size comparison. Keep these policy observations separate from
the actual common 20 evaluations. Do not erase the original failure or refund
spending. At cap 128 the all 100 correct counts 34→35→36 include diagnostics;
held-out sampled counts instead read 13→11→13/44, with greedy 7/11 unchanged.
At cap 32 all three policies have zero graded-correct held-out responses. The
fixed pilots do not improve this held-out correct-count headline, not a universal
ranking or proof that GRPO fails. Correct-but-capped sample counts 2/2/1 also show
why stopping is a separate outcome. Read the matched per-seed receipts, retain
the frozen parser and distinguish four new 100-record evaluations from reused
baseline records.
