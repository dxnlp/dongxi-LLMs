# Defend the experiment, not just the training loss

Status: scoped native comparisons, scientific integration and final local
verification are complete. Whole-goal acceptance is recorded in the
[original-criteria signoff](2026-10-05-final-original-criteria-signoff.md), not
inferred from training exits. Missing optional/later outcomes below remain
missing, not zero scores or predictions. The historical
[short-run defense](2026-10-05-current-model-defense.md) remains a separate
source-bound observation. The learner remains on Day 9.

## Three genealogies, three questions

| Branch | Actual starting point | Controlled question | Evidence boundary |
| --- | --- | --- | --- |
| Story pretraining | Two fresh random initializations, seed 909 | What changes by the predeclared first 400 updates when only peak/floor learning rate is halved? | Original 14,000-update schedule remains unfinished; later checkpoints are missing. |
| Instruction and preference | Pinned Qwen3-0.6B-Base, then the original fresh full400 export | Does the declared full/LoRA recipe learn the instruction interface, and what does preference training change relative to that fixed parent? | Three shared instruction templates and the separate location preference fixtures are not general assistant competence. |
| Reasoning and RLVR | Separately pinned Qwen3-0.6B Instruct for RLVR; actual Base and Instruct checkpoints for interface controls | How do actual checkpoint, raw/chat interface, thinking support, cap and fixed group-size intervention change delivered answers? | Upstream training history is not reconstructed; reasoning traces are not certificates of causal faithfulness. |

No inference connects the randomly initialized story model to Qwen's SFT/DPO
weights. Likewise, turning off thinking does not turn Instruct into Base.
Recovery and profiling artifacts are disposable gates, not pilot parents.

## A trained story model can improve NLL and still write poor stories

The [fresh first400 comparison](2026-10-05-native-story-first400-comparison.md)
retains equal training exposure: 1,389,548 valid targets and 6,553,600 processed
positions in each arm. Control and half-rate development NLL are respectively
3.388772 and 3.652813 on the frozen 64-window slice. These are not whole-corpus
losses or a proof that every training-source evaluation window was seen.

Four available checkpoints produce 192 real continuations; 288 later planned
cells remain missing. Two separately submitted AI reviewers each rate every
candidate under the declared rubric. Separate instances do not establish
independent underlying models, human consensus, or perfectly successful
blinding. Their source-opening uncertainty is descriptive, not uncertainty over
training seeds or the human population.

Control400 reaches EOS in 45 of 48 continuations, yet its mean ending-quality
rating is only 0.020833 on the 0–2 scale. There are 56 candidates with at least
one reviewer disagreement, comprising 64 dimension-level disagreements.
Stopping, story resolution, grammar and repetition therefore remain separate
observations. Repetition alone does not diagnose overfitting.

The [model-free archive](native-story-publication-20261005-01-archive/acceptance.json)
binds the actual measurements and metadata; the
[separate rating report](native-story-publication-20261005-01/ratings-evaluation-01/report.json)
retains supplied reviews. Neither is a portable weight archive. Deterministic
MATH-only training/recovery and automatic-SDPA publication are distinct recorded
execution paths, not a claimed cross-backend equality.

## Learning an instruction interface is a narrow, real capability

The [full/LoRA400 study](2026-10-05-native-sft400-comparison.md)
uses fresh copies of the same pinned Base, equal 9,321 training labels and
63,378 processed training positions, seed 1212 and learning rate 0.00002.
Full tuning updates 596,049,920 parameters; rank8 Q/V LoRA updates 1,146,880.
The same exposure and learning rate do not make their parameter spaces or
best possible recipes equivalent.

On the original 120 held-out-value instruction items in 40 source groups, full
tuning gives 120 exact whole answers and 120 natural message endings. Base
and the separately verified FP32-merged LoRA give zero of each and exhaust all
64-token caps. The permissive `format_policy=any` score passes every response
in every arm; it is not an accuracy score. Constant source-group differences
give degenerate descriptive bootstrap intervals, not certainty about new tasks.

The FP32 merge/reload pass does not erase the historical BF16 merge failure.
Publication generation loads BF16 weights and uses eager CUDA; merge arithmetic,
stored export precision and inference precision are separate claims. The
[assistant-study card](../../docs/model_cards/native-assistant-interface-study.md)
records intended educational use and excluded deployment claims.

## Recovery passes do not establish preference quality

Chosen-only recovery and the later
[DPO03 replay](2026-10-05-native-dpo-replay03-independent-review.md)
pass fresh-process numerical equality and actual whole-child completion. Both
two-update location panels still give 0/4 exact answers despite 4/4 natural
stops. DPO01/02 remain deadline failures with unknown native exits and retained
cleanup errors, even though durable intermediate state exists.

The accepted DPO03 children preserve the same scientific recipe and original
600-second guards. Actual witnesses attest CPU8 and bounded complete-file
hash worker count 4; the default remains serial and every integrity check is
retained. This is not a causal speedup benchmark or a physical quota guarantee.
Independent state comparison is CPU work outside the model-training journal.

The fixed fresh chosen100/DPO100 pilots and their
[separately accepted common evaluation](2026-10-05-native-preference-comparison.md)
now supply actual evidence. Both match100 updates,400 replacement draws and
2,047 chosen labels from the same original full400 parent. DPO also processes
1,600 rejected targets and800 frozen-reference training forwards. Their
objective reduction, forward layout and information/compute are not equal.

| Common separate generation panels | Original full400 | Chosen-only100 | DPO100 |
| --- | ---: | ---: | ---: |
| Strict location answers |0/4|4/4|1/4|
| Original instruction answers |120/120|120/120|120/120|
| Annotated reasoning, bounded-parser correct |5/20|6/20|6/20|

Every common response stops naturally; no cap is reached. Both descendants
retain the same originally correct IDs and add only `math-10`, with no
correctness loss in these panels. Nine seen/development reasoning items remain
separate from eleven controlled held-out items. Their0/11→1/11 held-out change
does not establish general reasoning gain, faithful traces or mathematical-step
validity. Instruction retention spans three shared templates, not arbitrary
assistant behavior.

DPO's mean unscaled four-pair relative margin is71.370116 nats versus13.488587
for chosen-only, yet its exact location score is lower. Both improve absolute
chosen likelihood here; DPO also strongly lowers rejected likelihood. Neither
a favorable ratio nor natural stopping specifies the exact greedy continuation.
Missing articles and missing nouns remain different qualitative nonmatches
under the unchanged whole-answer contract. The actual figures join complete
raw records and metadata without another model run or score generation.

All52 common checks pass and all three native evaluation children actually
exit0 under their original900-second/25GiB guards. Their native child times
are258.947192/254.523008/246.119601 seconds; the outer1086.313452 interval
overlaps them and includes preparation/closure. Likelihood uses FP32 policy
and reference weights with BF16 autocast/unforced SDPA; generation uses
BF16-loaded eager CUDA and common saved-template greedy64. Own-run diagnostics
and original native-Instruct32/128 reasoning remain distinct, not silently
relabeled into this precision or ancestry.

## Reasoning: interface and output budget are interventions

The four predeclared Base-raw, Base-chat, Instruct-thinking-off and
Instruct-thinking-on rows retain both 32/128-token caps, four sampled seeds and
the separate greedy diagnostic. The nine fixture-development/known-overlap
diagnostics stay separate from eleven controlled held-out items. Only two
items have the actual `rlvr_train_problem_overlap` flag; this annotation does
not claim that the assistant/preference policies were trained on mathematics
or establish upstream contamination. Nothing is silently removed from raw
records. All eight original initial conditions have now been attempted: six
complete100-response conditions and two failed40-response conditions retain680
of800 planned responses. The120 missing responses preclude a complete
four-interface comparison. G4run02/G8run01 recovery is accepted and both native
fixed16 training children and all four common post-training evaluations have
completed. Final evidence/source verification and original-goal signoff remain
separate from these scientific outcomes.

| Complete initial conditions, held-out only | Sampled correct, four attempts/item | Greedy correct, one attempt/item |
| --- | ---: | ---: |
| Base/raw32 |0/44|0/11|
| Base/raw128 |2/44|1/11|
| Instruct/thinking-off32 |0/44|0/11|
| Instruct/thinking-off128 |13/44|7/11|
| Instruct/thinking-on32 |0/44|0/11|
| Instruct/thinking-on128 |0/44|0/11|

Instruct/off128 preserves the same weights, interface, attempt seeds and saved
token prefixes as off32, but allows enough output for more gradeable answers.
Its held-out sampled records include35 natural stops and9 caps; two capped
samples still pass the frozen bounded mathematical grader. Its eleven greedy
answers all stop naturally. Correctness therefore does not silently become
strict EOS-complete reward, and8/11 any-correct sampled availability is not
7/11 delivered greedy accuracy or a tested selector. Both thinking-on caps
produce100 capped `UNSUPPORTED` records each, despite successful execution.
This is negative evidence under the chosen budgets/parser, not proof that
thinking is universally worse or that every trace is mathematically wrong.
The [source-bound reasoning report](2026-10-05-native-reasoning-rlvr-evidence.md)
and [independent review](2026-10-05-native-reasoning-baselines-independent-review.md)
retain all raw/stopping/cost joins. These initial generations use BF16-loaded
eager CUDA/full-prefix uncached decoding; no instrumented kernel-trace claim
follows.

Base/chat32run01 is an actual protocol failure, not an unrun row or0/100
accuracy. It retains40 records (39UNSUPPORTED,1ERROR) and60 missing planned
responses. One sampled continuation selects ID151768 without a tokenizer
mapping after24 actions, retaining its full IDs, selected likelihoods and
decode-error cost. Whole-child exit is1 after56.547995 seconds, not an unknown
exit; cleanup is clean and no deadline or reserve failure occurs. No vocabulary
mask, resampling, discarded error or retroactive acceptance is introduced.
The remaining independent original rows continue unchanged, while RLVR still
requires its own accepted Instruct-off32/128 and recovery gates. An incomplete
Base/chat row cannot establish a complete four-interface paired comparison.
The independently planned cap128 row also fails, retaining40 records and60
missing, eight natural stops,31caps and one decode error. Its actual exit1
occurs after132.666869 seconds with no deadline or cleanup error. Both errors
share the same problem/seed/24-action trajectory, ending at151768; they are
distinct failed invocations, not two independent rare-event draws. The ID is
inside151936 configured output rows but absent from151669 mapped decoder IDs.
Input byte coverage and output row addressability are different interfaces.
The [Day2 bridge](../../learning_artifacts/day-02-text-tokens-and-embeddings/output-vocabulary-and-decoder-coverage.md)
retains the exact cached metadata and the single-prefix conditional likelihood.

A correct extracted numeral, valid format, natural stop, cap exhaustion,
sampled success and delivered greedy answer answer different questions. Neither
an any-correct candidate pool nor a long visible trace guarantees the answer
a user actually receives. Final integration must retain these denominators
and paired source-group comparisons rather than pool unrelated percentages.

G4 recoveryrun01 is another distinct negative: all three native children exit0,
but the pending-resume supervisor fails its observed-descendant cleanup
acknowledgment. Exact final state comparison is not reached and recovery
acceptance is absent. Later absence of the exact owned PIDs does not repair that
original terminal receipt. Its shared journals retain four actual optimizer
applications across source/completed/pending processes and three collections,
even though task rewards and advantages are zero. Measured nonzero gradients
and reference KL are not a task-reward improvement or a fabricated skipped
update. The [diagnosis](2026-10-05-native-rlvr-recovery-cleanup.md) and
[independently reviewed controller correction](2026-10-05-native-rlvr-controller-runtime-independent.md)
preserve the failure and source bytes. A separately declared fresh retry02
defers model-heavy controller imports while preserving the original0.5-second
cleanup acknowledgment bound, numerical implementation and scientific recipe.
The freshG4run02 now passes all twelve recovery checks, three supervised
exits0 and exact final numerical component/update2-metric comparison. Its
completed-resume role has an observed descendant and clean acknowledgment;
the pending role has none. This is not a controlled historical import-latency
benchmark. Shared journals still count repeated physical applications rather
than rewinding to a numerical prefix. G8run01 also passes all twelve recovery
checks, exact final numerical state/update2 metrics and three supervised exits0.
Its182.051605/137.331498/101.909522-second children retain clean cleanup and
sampled memory above the original25GiB floor. The enclosing504.835865-second
adapter interval overlaps those child times. At that recovery milestone both
fixed16 pilots and their common evaluations were pending. No timeout waiver, automatic
retry or refunded spending is introduced; exact replay is not a positive
task-quality result.

Subsequent G4pilot01 completes all16 numerical updates and exports its final
policy; the actual native process exits0. Its returned supervisor nevertheless
fails the final-logging acknowledgment, so original pilot acceptance remains
absent. The [separate export observation](native-rlvr-g4-pilot-20261005-run-01/export-observation.json)
verifies that completed16 artifact without changing the failed receipt or
refitting. All64 training rewards/advantages are zero; nonzero KL gradients
and16 physical optimizer applications do not establish reward-driven learning.
The explicit run02 common evaluation is a different consumer of this verified
artifact, not a retroactive pass or a replacement training attempt.

G8pilot01 now passes all three pilot checks and completed supervision/native
exit0. It likewise has16 physical applications/collections and zero task rewards
or advantages across128 naturally stopped training responses. G4/G8 process
1,004/2,335 valid targets, versus1,508/4,888 dense slots/sampling draws that
also include masked post-stop work. Equal update counts are not equal rollout
exposure. Its four greedy strict diagnostics remain0/4 before and after, although
the actual strings change. These are not the common20 held-out panels. Both
predeclared final16 exports now have consistency observations.

The separately executed common run02 [cap32](native-rlvr-common20-cap32-20261005-run-02.json)
and [cap128](native-rlvr-common20-cap128-20261005-run-02.json) comparisons now
retain400 new G4/G8 responses plus200 reused accepted unchanged-Instruct
responses. All four new model evaluation children exit0 with completed
supervision and their four checks true. That does not change failed G4 training
supervision into an accepted pilot.

| Controlled held-out slice | Unchanged Instruct | G4 final16 | G8 final16 |
| --- | ---: | ---: | ---: |
| Cap32, sampled |0/44|0/44|0/44|
| Cap32, greedy |0/11|0/11|0/11|
| Cap128, sampled |13/44|11/44|13/44|
| Cap128, greedy |7/11|7/11|7/11|

Cap128 all-original correctness, pooling sampled and greedy diagnostics over20
items, rises34/100→35/100→36/100 while the held-out metrics do not improve.
These denominators answer different questions. Nine development/known-overlap
diagnostics are not nine proven RLVR training overlaps; only two items have
that actual flag. A two-sample regression also does not establish population
degradation. Correct-but-capped sampled held-outs number2/2/1, so correctness
does not become strict EOS-complete reward. The predetermined final16 policies
were not selected using this evaluation. Neither largerG nor nonzero KL updates
demonstrates reward-driven arithmetic learning here.

The four actual evaluation child times are202.253392,333.345343,201.023033,
336.047204 seconds in G4cap32/G4cap128/G8cap32/G8cap128 order. They are whole
child intervals, not pure decode/FLOP or isolated throughput measurements;
the CPU notebook panel overlaps parts of the G8 evaluation. New resource
sampling is prospectively1Hz with the original900-second/25GiB limits, not a
continuous-floor proof. CPU comparison adapter times26.768527/34.050493 seconds
are separate and overlap their own internal work, not extra model generations.

## Read every cost at its own boundary

| Completed fresh training arm | Valid training labels | Training policy/model positions | Frozen-reference training positions | Whole native child seconds |
| --- | ---: | ---: | ---: | ---: |
| Story control400 |1,389,548|6,553,600|not applicable|1,187.172993|
| Story half-rate400 |1,389,548|6,553,600|not applicable|1,273.519688|
| Assistant full400 |9,321|63,378|not applicable|519.034220|
| Assistant Q/V-LoRA400 |9,321|63,378|not applicable|439.013098|
| Chosen-only100 |2,047|13,343|0|426.071011|
| DPO100 |2,047 chosen +1,600 rejected|25,439|25,439|986.490389|

These are actual counts for different questions and parameter spaces, not a
cross-task efficiency leaderboard. Training-position columns exclude separate
generation/evaluation; child duration includes that child's loading, training,
validation, saving and terminal work. Durations are rounded here, with full
precision in the linked branch reports and actual supervisor receipts. Failed
recovery attempts and separately invoked publication evaluations remain
additional physical work; they are not hidden in a recovered numerical cursor.

Numerical updates, valid training labels, physical repeated presentations,
padded positions, policy/reference calls, generation tokens, snapshot visits,
external child time and sampled memory are not interchangeable units. A
recovered trajectory can be exact while its shared physical journal includes
the tail twice. Failed attempts remain spent. Snapshot clone, hash and
serialization counters can traverse the same bytes; their sum is not unique
disk traffic. CUDA allocated/reserved memory is not total unified-memory use.

Native child time and enclosing supervisor/wrapper time overlap. Evaluations
inside a training child are not extra wall time to add again. Independent
comparison, final artifact hashing and review have separate CPU boundaries.
Sampled MemAvailable above 25 GiB does not establish a continuous floor.

Historical hardware warnings also expire. The [current applicability review](../../docs/REFERENCE_REPOSITORY_AUDIT.md)
records upstream resolution of the old unified-memory hang and the updated
local stack. That old warning is not a current reason to build custom monitoring
infrastructure. Ordinary finite-resource bounds, the actual model's measured
cost and a terminal-logging failure are separate issues. Assess a proposed gap
for current relevance before treating more infrastructure as course progress.

## Final joins completed

The [closed campaign archive](native-campaign-evidence-20261005-run-01/closing-bindings.json)
retains all45 original stage declarations, accepted exports, explicitly
observed-but-failed-supervision G4 export, raw responses, actual genealogy,
failures, source/lock/interface identities and distinct invocation costs.
Its74 terminal model-directed children comprise66 completed and8 failed
observations; this does not establish74 GPU-forward executions or success at
all declared maxima. Snapshot/gzip closure and2,244 archived outcome/input/output
bindings plus six source bindings are independently reviewed.

The measured results are integrated into chapters, adjacent solutions, labs
and model cards. [Final current-source verification](2026-10-05-final-course-verification.md)
passes1,533 tests and all76 fresh notebook references,535 executed reference
cells/210 images, preserving four unfinished learner exercises. All238
executable/workflow identities remain frozen. The
[signed original-criteria review](2026-10-05-final-original-criteria-signoff.md)
checks all18 packages/86 unchanged criteria and dependencies.

The archive deliberately retains the assembly-time15/18 ledger. The later
[status projection](2026-10-05-final-goal-status-projection.json) records final
17→03→18 package closure without rewriting historical receipts or missing
optional/later cells.

Actual Linux execution, unverified Mac portability and unexecuted hosted CI
must remain separate. The original criteria require that separation, not
invented Mac/hosted success. Media production and publication remain separately
approved, Mac-assigned work; this report does not authorize either.
