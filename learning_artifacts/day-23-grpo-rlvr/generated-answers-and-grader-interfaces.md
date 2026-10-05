# Generated answers and grader interfaces

Recorded: 2026-10-05. Status: focused course-production evidence; live learner
explanation and correction remain unassessed. The learner's Day9 position is
unchanged. This artifact does not mark Day23 complete.

## The evidence boundary

The [partial native report](../../experiments/reports/2026-10-05-native-reasoning-rlvr-evidence.md)
incorporates the first accepted pretrained baseline: Qwen3-0.6B Instruct with native
thinking-off chat, cap32, four sampled seeds and a separate greedy diagnostic.
All100 planned response records are retained and the actual whole child exits0.
Base/raw/cap32 and Base/raw/cap128 now separately retain another100 original
records each and exit0. Instruct/thinking-off128 separately completes100 records
and exits0. Base/custom-chat at both caps32 and128 has instead failed
with40 retained records and60 missing each. Both thinking-on conditions now
complete100 records each but cap every response with an unsupported grade.
G4 recovery run01 now fails its cleanup gate despite three native exits0; its
CPU comparison was not reached and dependent pilot was not launched at that
boundary. Common20 outcomes now have four actual post-evaluations and two
matched per-cap comparisons.
Fresh G4 recovery02 and original G8 recovery01 are now accepted; neither erases
the original failure nor establishes quality. G8 pilot16 now passes; G4's
completed16 export has a separate consistency closure retaining its failed
supervision, not an acceptance retrofit.
The older symbolic
negative and constructive positive controls remain unchanged.

## A response is not one success bit

Follow the actual sample1009/math10 record. Its prompt is `Compute 5 + 6.`;
the raw output is `5 + 6 = 11<|im_end|>` and scoring text is `5 + 6 = 11`.
The declared terminal-stop removal is not an answer rewrite.

| Instrument | Actual observation | What it does not prove |
|---|---|---|
| Whole-child supervisor | Completed, actual exit0 | Mathematical success |
| Response stopping | Natural turn stop, not capped | Parser support or correctness |
| Format predicate | Valid under `format_policy="any"` | A single parseable rational answer |
| Answer extraction | `boxed_or_final` falls back to `whole` | Permission to select the last numeral |
| Exact-rational canonicalizer | `UNSUPPORTED`: outside exact rational grammar | A supported answer proven mathematically wrong |
| Accepted task fields | `correct=false`, `task_success=false` | Every visible statement is false; no reasoning occurred |

This example makes the **declared whole-output grading** limitation concrete.
It is a qualitative illustration, not a rescored headline. All100 accepted
Instruct/off32 grades remain negative:99 unsupported and1 invalid. Format validity is100/100
because the declared format is permissive. Do not collapse format, support,
correctness, termination and truncation into a single proxy.

Eleven controlled held-out problems yield44 sampled attempts and11 separate
greedy attempts. Their declared correct counts are0/44 and0/11; natural stops
are3/44 and1/11, while caps are41/44 and10/11. Nine other prompts are
development/seen diagnostic roles: eight original fixture-train items math1–math8
plus math9's known RLVR-fixture overlap. Only math2 and math9 have true RLVR-overlap
flags. This panel rule does not prove that actual full400/chosen/DPO policies
trained on the math rows or that upstream checkpoint contamination occurred.
Repeated draws are not independent source problems. No deployed best-of-four
selector is measured; availability of any graded-correct sample would be a
separate diagnostic from delivered-answer accuracy.

Base/raw32 separately has0/44 controlled sampled grade,1 natural stop and43 caps;
all11 controlled greedy responses cap with zero declared correct. Base/raw128
has2/44 sampled correct,9 stops and35 caps, plus1/11 greedy correct,2 stops and
9 caps. Its three controlled successes use supported boxed extraction and
naturally stop; two other correct responses are overlap/seen diagnostics.
Availability among four samples is2/11 distinct controlled problems, while the
greedy response is correct on a different1/11. No oracle selector was deployed.
Do not merge those denominators or silently count the seen successes as held-out.

Instruct/thinking-off128 holds the original weights, source, native interface,
panel and decoding settings fixed against32; only the cap changes. It has13/44
controlled sampled correct answers with35 natural stops/9 caps, and7/11 greedy
correct with11 stops/no caps. All100 are format-valid under `any`. Two correct
sampled boxes still cap128 (math19/seed1009 and math18/seed1019); parser answer
success is not strict EOS-complete trajectory success. Any-candidate availability
among four samples is8/11 controlled problems, not the7/11 greedy-delivered score
or an implemented selector. More supported answers under this bounded cap
intervention do not establish newly trained weights, per-prompt monotonic gain
or the rationale's faithfulness.

Thinking-on at both budgets retains all100 original records without adapter
errors, but each sampled controlled slice is0/44 with44 caps and each greedy
slice0/11 with11 caps; no natural stop occurs. All are format-valid under `any`
yet unsupported, with whole-output extraction retaining their unfinished thinking
material. No response closes `</think>` or supplies a box. Sample1009/math10 at128
opens a discussion of5+6 without completing a final answer; this is not proof of
wrong mathematics, absent ability or a universally harmful thinking mode.

The flag uses the same Instruct weights/tokenizer/native template but changes
all rendered assistant prefixes: off precloses an empty thinking region, while
on permits generated `<think>` text. Within on32→128, all100 attempt seeds,
prompt IDs and short-action prefixes match. These observed pairings do not imply
cross-platform GPU determinism. On32 consumes3200 actions/124960 prefix positions
and on12812800/1114240, all capped, under the same fixed900-second/25GiB child
envelope. Uncached position counts are not FLOPs or optimized serving throughput;
visible traces and parser success do not establish rationale faithfulness.

## A repair proposal would be a new experiment

Requiring a bare numeral, changing extraction, expanding the parser or increasing
the output cap could each be declared and tested separately. Name the intervention,
freeze its input/contract, retain independent positive/negative/parser controls
and run it before claiming an effect. Do not alter current outputs, discard
unsupported rows or retrofit a more favorable denominator. Correct final answers
would still not certify rationale faithfulness.

The current native generation path is BF16-loaded/eager CUDA with uncached
full-prefix forwards. This is not a MATH-kernel control, FP32-plus-autocast
training or an optimized serving benchmark. Base/instruct checkpoint changes,
raw/chat serialization, supported thinking flags and caps32/128 are different
interventions. Base/raw versus Instruct/native chat changes both weights and
serialization, so their descriptive differences cannot isolate instruction
training. The two raw-output caps hold those identities fixed but do not promise
monotonic per-prompt improvement or establish rationale faithfulness. The
remaining RLVR outcomes have no measured scores, not zero-valued measurements.
Both same-weights custom-chat invocations are failed/partial instead
of an accepted raw/chat contrast.

## A selected action can be an interface failure

The actual Base/custom-chat32 child exits1 after56.54799546097638 seconds;
supervision covers56.59703186398838 seconds with minimum sampled available
bytes122019704832. There is no external deadline stop or cleanup error. Its
sample1019/math6 trajectory retains24 generated IDs ending in151768, which lacks
a tokenizer mapping. This is a decode `ERROR`, not an unsupported mathematical
answer, a cap or natural termination. Forty records, sixty missing responses,
the full error IDs, partial decoded text and spent cost stay retained. Missing
draws are not fabricated wrong answers; a vocabulary filter, replacement ID or
resampled continuation would change this declared full-support experiment.
The independent cap128 condition also exits1 at that24th unmapped action, with
child132.66686874401057 seconds, supervisor132.72932148602558 seconds and minimum
sampled available bytes120977350656; stop reason remains null and cleanup empty.
Its40 records retain39 unsupported/one error,8 natural stops/31 caps,4491
actions and377459 positions. It is not a repaired retry of cap32. Other independent
predeclared conditions proceed with source/gates/contracts unchanged and no filter.
All eight initial conditions have actual receipts: six accepted complete
invocations supply600 records; two failed partial invocations supply80, including
two errors. The680/800 retained coverage leaves120 missing, not600 correct answers.
This is not full800-slot coverage, a pooled quality score
or a complete missing-sixty comparison. Failed rows remain failed without inventing
a new mandatory gate for successful Base/chat reruns. The original goal stays open.

## Keep the later RLVR questions open

A constant-reward group has zero relative advantages, but the fixed native
application path can still run exact-KL scoring, backward, clipping and AdamW.
Inspect completed iteration cursors, actual optimizer applications, nonconstant
rewards, policy movement and delivered held-out quality separately. The baseline
publication parser is not the RLVR strict integer reward instrument. Pending
recovery must retain its already collected actions and journal spending and apply
the group without resampling. No accepted recovery or pilot outcome follows from
this source reading.

The actual failed G4 run01 is retained evidence, not a pending or passed replay.
All three native children exit0, but the pending supervisor reports an observed
descendant cleanup timeout. Completed helper cleanup and later process absence
cannot waive that distinct gate. CPU comparison was not reached and acceptance
is absent; the600-second/25GiB child bounds and all spending remain unchanged.
The shared journal records four actual optimizer applications and three
collections, split2/1/1 and2/1/0 across source/completed/pending. Pending applies64
saved response targets without resampling. Summing restored cursors would
misreport physical work.

Both source groups have zero rewards/advantages, but measured gradient norms
are5.811452865600586e-7 and0.1796875, with pre-update KL0.0010778990108519793
at update2 and different recorded post-update policy identities. The actual
AdamW-plus-exact-KL path has no constant-group skip. Zero task-relative signal,
numerical gradient, optimizer execution, state movement and held-out learning
are separate claims. These failed-attempt observations establish neither accepted
exact recovery nor positive reward learning. Follow
[worked answer22](../../book/solutions/13-group-relative-policy-optimization.md#22-native-exit0-does-not-waive-the-cleanup-and-replay-gates)
before predicting later pilot or common-panel results.

The owner has subsequently launched one declared fresh G4 run02 after a
CPU-reviewed controller-only lazy-import correction. Numerical science,
supervision and the0.5-second cleanup acknowledgment stay fixed; run01 remains
failed/spent. Fresh02 now passes all twelve actual recovery checks, including
three completed supervisors/native exits0, exact ten-component final scientific
state and update2 metrics, unchanged before/after source/input/model closure,
retained same-physical-journal spending and pending no-resample behavior. Its
four measured applications/three collections do not become positive reward
learning or erase failed01's distinct journals. Pending02 observes no owned
descendant, so fresh acceptance does not capture the historical timeout cause.
Its dependent pilot01 must explicitly select accepted02, not silently search or fall back.
This runtime continuation is not a positive-reward result or permission to
repair baseline outputs, change the verifier or forecast common-panel scores.
Read [worked answer23](../../book/solutions/13-group-relative-policy-optimization.md#23-an-accepted-replay-still-needs-a-separate-quality-experiment)
to distinguish actual replay from the separate pilot/common20 observations.

G8's own twelve checks and ten-component/update2 comparisons pass with three
supervised native exits0 and empty cleanup errors. Its four applications/three
collections retain384 dense sampled slots but314 fresh valid actions and434
applied targets. Stopped-row sampling still spends slots; pending consumes120
saved valid targets without resampling. Source groups have seven EOS/one cap
then two EOS/six caps, but all rewards/advantages are zero. Their nonzero measured
gradients and update2 KL do not create a positive-reward result or equal-compute
group-size comparison. See [worked answer24](../../book/solutions/13-group-relative-policy-optimization.md#24-a-batch-slot-valid-action-and-replayed-target-are-different-costs).
G4 pilot01 explicitly selects accepted02, but its subsequent terminal-retention
failure does not supply an accepted final16 export or common20 score.

## Native sixteen iterations can still be an unaccepted pilot

G4 pilot01 now completes native cursor16/export and exits0, but its supervisor
fails with `Final logging timed out` and `final_record_retained=false`. Cleanup
errors and descendant lists are empty: this is a separate terminal writer gate,
not a repeated cleanup failure or1800-second deadline. No acceptance/closing/
bound export existed at that failed queue boundary. The later explicit export
consumer validates the completed state and exported bytes without reclassifying
supervision; it does not rerun training or create a quality score.

Its retained native measurements are real: sixteen applications/collections,
1508 dense sampled slots versus1004 valid targets,67 closed work reservations
and33 saves. All64 responses naturally stop, but all sixteen strict-reward/
advantage groups remain zero. Nonzero numerical gradients/KL and differing
recorded policy identities do not create positive task-relative learning or
common20 quality. Do not call the attempt no training, discard its costs or
turn surviving weights into an accepted publication parent. Read
[worked answer25](../../book/solutions/13-group-relative-policy-optimization.md#25-trained-but-unaccepted-is-neither-no-training-nor-a-quality-pass).

G8's separately accepted pilot16 generates128 naturally stopped responses with
zero strict rewards and zero advantages in all sixteen groups. The actual
application path still executes sixteen optimizer updates. Its4888 sampled
slots versus2335 valid targets, compared with G4's1508/1004, show why group size
alone is not equal compute. Numerical KL/parameter movement does not establish
positive task-relative learning. The [actual training table](../../experiments/reports/2026-10-05-native-reasoning-rlvr-evidence.md#46-actual-native-pilots-completed-updates-are-not-a-reward-signal)
retains those distinctions. The [actual common-panel results](../../experiments/reports/2026-10-05-native-reasoning-rlvr-evidence.md#47-actual-common20-comparisons-the-denominator-changes-the-conclusion)
now join400 new RLVR records with200 reused baseline records. At cap32 all three
policies have zero graded-correct held-out responses. At cap128 their sampled
correct counts are13→11→13/44 and greedy is7/11 for all three, despite rising
all100 counts34→35→36. Diagnostic gains are not held-out learning. Correct-but-
capped samples2/2/1 keep correctness separate from stopping; no general method
ranking or rationale-faithfulness claim follows. Neither fixed native pilot
increases this held-out correct-count headline, and zero training advantages
do not become a positive reward signal merely because optimizer work completed.

## Next live-learning action

Use [Lab13's evidence-reading exercise](../../book/labs/13-group-relative-policy-optimization.md#read-actual-evidence-without-repairing-the-result)
and [worked answer19](../../book/solutions/13-group-relative-policy-optimization.md#19-follow-one-response-through-independent-predicates).
Ask the learner to explain the apparent contradiction before revealing the
retained extraction fields. Record their own prediction, correction and unresolved
edge during live study; agent inspection alone is not mastery evidence.

Suggested X article: “What did your reasoning benchmark actually grade?”
Related suggested article angle: “Zero rewards do not mean zero updates.” Anchor
it to the accepted G4/G8 recovery histories, valid-mask/draw distinctions and
retained pending pool, not an unmeasured quality gain or an attributed historical
cleanup cause. This remains a suggestion, not a commissioned article.
Suggested CAND-ANIM-022 extension: keep the same actual response visible while
separate raw-text, stopping and grading lanes expose the unsupported whole-equation
fallback. Both remain suggestions for separately authorized Mac production.
No article draft, animation, rendering, upload or public claim is produced here.
