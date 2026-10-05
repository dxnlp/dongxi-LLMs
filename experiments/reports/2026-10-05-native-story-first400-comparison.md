# A matched first tranche is not a completed training schedule

Both arms completed the predeclared first 400 updates of the original
14,000-update TinyStories schedule on Spark. Four publication children then
generated 192 continuations, now scored by two separately blinded AI readers.
The control has lower fixed-panel NLL and more natural EOS stops at update 400;
the rubric does not support reliable coherent storytelling by either arm. This
is a trained learning-rate intervention, not a sampling comparison or a
replacement for the historical September 14,000-update run.

Both fresh arms use seed 909, the same pinned prepared data and tokenizer bytes,
document windows, masks, context 1024 and effective batch 16. The control peak/floor
rates are 0.0003/0.00003; the intervention halves both. Warmup stays 200 updates
and decay retains its 14,000-update horizon. The first 400 stop was declared before
publication quality inspection; it does not compress decay into 400 updates.
The 50M valid-target and 14,400-second limits remain ceilings, not promised spend.

## Actual training boundaries

| Observation | Control | Half learning rate |
|---|---:|---:|
| Completed updates |400|400|
| Successful valid training targets |1,389,548|1,389,548|
| Processed training positions |6,553,600|6,553,600|
| External supervised child seconds |1,187.173|1,273.520|
| Runner seconds |1,184.413|1,270.712|
| Minimum externally sampled MemAvailable bytes |109,796,323,328|110,089,650,176|
| Actual trainer exit |0|0|
| Requested 400 boundary reached |true|true|
| Full 14,000 schedule completed |false|false|

The [control acceptance](native-story-control-pilot-20261005-01/acceptance.json)
and [half-rate acceptance](native-story-half-lr-pilot-20261005-01/acceptance.json)
bind actual producer source, data, interpreter/lock and each successful owned
child. Their completion records and physical work journals retain successful work,
reservations and observers separately. Training positions are measured input
elements, not all model work; validation, development generation, activation
probes and saves are additional operations. External sampling does not prove a
continuous memory floor. Trainer and supervisor timers have different boundaries.

![Actual online batch NLL versus matched valid-label exposure and learning rate versus completed update](native-story-first400-figures/learning-curves.png)

The [figure receipt](native-story-first400-figures/inputs.json) binds both complete
400-row metric streams before and after rendering. Every update has the same
valid-target and processed-position count across arms. The control's online batch
loss falls faster in this single matched trajectory. That curve alone is not a
story-quality difference or a matched frozen-checkpoint train/development gap;
the separate measurements below address those questions.
The nearly flat rates after warmup show why retaining the 14,000-step horizon is
different from ending a 400-step decay schedule. Twelve authored consumer tests
cover admission, identity/metric mismatch, exclusive output and mutation failure;
their inert plots are separate from this actual figure.

Actual deterministic profile and completed-state replay gates preceded both
pilots. The initial automatic-SDPA recovery mismatch remains failed even though
its children exited zero. The separately declared deterministic entry witnesses
strict deterministic algorithms, disabled TF32 and SDPA MATH only. The current
source-bound profile and both recovery receipts use revision 03. Exact replay in
that tested configuration is not proof that every attention backend is equivalent
or that every future host will reproduce it.

## Actual publication coverage and fixed-target prediction

The original plan retains twelve openings, four decoding recipes and five
predetermined checkpoints per arm: 0/400/4000/8000/14000. Only actual 0/400 weights
are available from this first tranche. The
[coverage record](native-story-publication-20261005-01/coverage.json) and
[raw continuations](native-story-publication-20261005-01/records.jsonl) contain
192 actual records from four successful native children, with zero generation
failures. The six later checkpoint declarations still account for 288 missing
cells of the original 480-cell panel. They are not zeros or forecasts. The
generation coverage receipt's original awaiting-ratings status is retained;
the later ratings receipt supplies the subsequent review evidence.

Each available checkpoint uses the same fixed 64 training-source windows
(seed 409, 13,285 valid targets) and 64 development windows (seed 909, 13,132
valid targets). Each selection forwards 65,536 physical positions. These are
target-weighted slice NLLs, not whole-split losses or the September run's
512-window development measure. Membership in the training source does not
establish that every selected window was presented during these 400 updates.

| Checkpoint | Training-source NLL | Development NLL | Natural EOS / 48 | Token cap / 48 |
|---|---:|---:|---:|---:|
| Control, initialization |10.896202|10.907969|0|48|
| Half-rate, initialization |10.896202|10.907969|0|48|
| Control, update 400 |3.264446|3.388772|45|3|
| Half-rate, update 400 |3.519832|3.652813|25|23|

The [portable measurement archive](native-story-publication-20261005-01-archive/archive.json)
retains all four raw NLL/child measurement documents, original returned
supervision and their source/input byte bindings. Its
[archival acceptance](native-story-publication-20261005-01-archive/acceptance.json)
passes before/after checks;15 pure-standard-library controls and independent
review pass. The original ignored native outputs stay intact. This archives
measurement evidence for another checkout, not checkpoint weights or a new
model/corpus-body verification. The archive's semantic digest is
`5c710289f621a2915c1136b7ad3c341de4dc5ac7d2e3ba7d5873adac639201ac`;
its distinct file-byte digest is retained in acceptance.

At update 400, control development NLL is lower by 0.264040 on this fixed slice.
That supports better observed-target prediction here, not a universal winner,
overfitting diagnosis or expected performance at update 14,000. The small
training-source/development gaps cannot identify memorization or causal
generalization on their own.

Publication loads FP32 checkpoints and uses CUDA BF16 forward autocast,
FP64 likelihood arithmetic and automatic causal SDPA dispatch, with activation
checkpointing off. It does not use the deterministic MATH-only training and
recovery entry. All four publication children use this same declared setting;
we did not measure the selected kernel per forward or establish bitwise
equivalence across these backends.

## What the blinded rubric actually says

Two fresh AI subagents independently read the shuffled anonymous packet without
checkpoint/arm labels, the private codebook or the other reader's judgments.
Their runtime model identifiers were not independently observed: two separate
instances are not two humans or proof of independent underlying models. The
[supplied-ratings report](native-story-publication-20261005-01/ratings-evaluation-01/report.json)
and [consumer receipt](native-story-publication-20261005-01/ratings-evaluation-01/receipt.json)
retain both actual submissions, notes and the predeclared two-rater means.
The [independent numerical review](2026-10-05-native-story-rating-independent.md)
recomputes the completed paired outputs and verifies actual record/rating joins.
The consumer exited zero; it processed supplied ratings rather than generating
stories, supplying human ratings or authenticating the declared independence.

There are 384 scored reader-candidate rows, with no abstentions. The readers
disagree on at least one dimension for 56 of the 192 candidates: 64
dimension-level disagreements, including 41 on repetition. The mean policy
leaves zero pending adjudications, but averaging is not an independent third
review or evidence of consensus. Every row below averages both readers across
48 candidates per checkpoint; the five dimensions remain separate, on their
original 0–2 scales, with higher scores better.

| Rubric dimension | Initialization, either arm | Control, update 400 | Half-rate, update 400 |
|---|---:|---:|---:|
| Grammar |0|0.072917|0|
| Entity/object consistency |0|0.260417|0.145833|
| Causal continuity |0|0.197917|0.093750|
| Repetition: freedom from obstructive loops |1.500000|0.843750|0.979167|
| Meaningful ending |0|0.020833|0|

This is a useful counterexample to two tempting shortcuts. Control reaches EOS
in 45/48 continuations but has an ending mean of only 0.020833: terminating the
token stream does not resolve the story. Initialization scores 1.5 for freedom
from repetition but zero on all other dimensions: high-entropy nonsense can
avoid repeated phrases without becoming language. Half-rate's higher repetition
mean likewise does not overturn its lower scores on the other dimensions.
No composite score or percentage of coherent stories was defined by this rubric.

The matched update-400 comparison keeps all four decoding recipes together when
resampling each of the twelve source openings. With 800 bootstrap draws and
seed 1010, the actual half-rate-minus-control results are:

| Dimension | Paired mean difference | Source-opening percentile interval |
|---|---:|---:|
| Grammar |−0.072917|[−0.145833, −0.010417]|
| Entity/object consistency |−0.114583|[−0.239583, 0.010417]|
| Causal continuity |−0.104167|[−0.177083, −0.010417]|
| Repetition |0.135417|[−0.072917, 0.322917]|
| Meaningful ending |−0.020833|[−0.052083, 0]|

The equal-recipe mean is an explicit mixture of one greedy and three sampled
recipes at temperature 0.8, seeds 909/1909/2909, without top-k or top-p filtering,
and a 256-new-token cap. Recipe-specific results remain in the raw report.
There are twelve resampling groups, not 48 independent stories or 384 independent
training runs. These small-panel descriptive intervals do not estimate
between-training-seed uncertainty, human agreement, a new opening population or
the unrun later checkpoints. They do not certify publication rights or replace
the separate [corpus-overlap audit](2026-10-05-story-panel-audit.md).

## Stopping changes actual generation work

The publication children measured different amounts of work because EOS ends
generation early. This is not evidence that one arm has a faster kernel.

| Checkpoint | Generated tokens | Forwarded generation positions | External child seconds |
|---|---:|---:|---:|
| Control, initialization |12,288|1,745,920|393.566|
| Half-rate, initialization |12,288|1,745,920|393.724|
| Control, update 400 |8,085|842,538|266.801|
| Half-rate, update 400 |10,632|1,370,480|345.424|

The four children together generated 43,293 tokens and forwarded 5,704,858
generation positions. Fixed NLL additionally used 105,668 valid-target
presentations and 524,288 physical positions. Those totals are separate from
each arm's 1,389,548 training targets and 6,553,600 training positions. All four
publication children exited zero under the 900-second child deadline and
25-GiB host reserve; their lowest sampled MemAvailable was 110,756,626,432 bytes.
Sampling is not continuous monitoring. Each child's CUDA allocated/reserved
peaks were 8,537,298,432 / 12,339,642,368 bytes, distinct from host availability
and not quantities to add across sequential children. The
[checkpoint registry](native-story-publication-20261005-01/checkpoints.json)
and their returned supervision records retain the individual boundaries.

## Launcher refusal does not imply a failed trainer

After control completion, the sequential queue initially refused progression
because its launcher expected a nonexistent `checks` field in this producer's
acceptance format. The producer receipt instead records status, source bindings,
distinct children, their actual exits and the requested boundary. The launcher
check was corrected against that existing schema; producer code, scientific
criteria and resource limits were unchanged. No native child was rerun or new
GPU work charged for the launcher refusal. This administrative error must not be
described as a failed 400-update experiment.

Larger continuation, additional seeds, actual Mac portability, hosted CI,
rendered animations and public model/course release remain separate work. This
report does not advance the learner beyond Day 9.
