# Native reasoning interfaces and RLVR: actual bounded results

Date: 2026-10-05. Status: **initial eight conditions attempted; G4/G8 recovery accepted; both native pilots completed16; four post-evaluations and two matched per-cap comparisons complete**.
All initial baseline conditions now have actual receipts: six accepted complete
invocations and two failed, partial Base/custom-chat invocations. The retained
coverage is680 records of800 planned, including two decode-error records;120
responses remain missing. G4 recovery run01 has an actual cleanup-gate failure;
the separately declared fresh G4 run02 and original G8 run01 are now accepted.
G4 pilot01 completes sixteen native iterations/export but fails its final logging
acknowledgment gate. G8 pilot16 is now accepted, and G4's completed export has a separate
consistency closure without reclassifying its failed supervision. Common-panel
post-evaluations and both matched per-cap comparisons now have actual receipts.
Failed rows are not replaced
by the owner's continuation through other independent predeclared conditions.
Queued or prepared stages are not measured outcomes.

## 1. What ran, and what was accepted

The fixed [preparation](native-reasoning-instruct-thinking-off-cap32-20261005-run-01/preparation.json),
[actual acceptance](native-reasoning-instruct-thinking-off-cap32-20261005-run-01/acceptance.json)
and [returned supervisor receipt](native-reasoning-instruct-thinking-off-cap32-20261005-run-01/returned-supervision.json)
retain one complete interface/cap invocation. Acceptance is `passed`; the actual
whole child exited0, all five cells completed their original twenty responses,
and no missing response or unreadable JSONL line was accepted. These are execution
and coverage checks, not a positive mathematical-quality gate.

The unchanged checkpoint is Qwen/Qwen3-0.6B at revision
`c1899de289a04d12100db370d81485cdf75e47ca`. The five cell summaries record
`local-hf-sha256:9798aefa7180a5af3d5128aaf07992cdfcc539fe5e36e908a5a830752178f0d8`.
This report reads retained identities; it does not reload weights or rehash their
bodies. Native Instruct chat has thinking disabled. Generation loads BF16 weights
and uses the eager CUDA, uncached full-prefix implementation. It is neither an
optimized KV-cache timing benchmark nor FP32 training with BF16 autocast; no MATH
kernel intervention is implied.
Those precision/kernel labels identify the fixed generation configuration and
implementation; the receipts are not an independently instrumented kernel trace.

The four full-support, temperature-one sampling seeds are1009,1019,1029,1039.
Each generates twenty original prompts. A separate greedy diagnostic uses seed1009
and the same twenty prompts. The external900-second deadline and25GiB sampled
available-memory reserve cover the whole five-cell child, not900 seconds per
prompt. Context is512 and each response cap is32; the maximum emitted action
allowance is3200. EOS151643, turn stop151645 and pad151643 remain declared.
Raw decoding retains the terminal marker; scoring text removes only the declared
terminal stop.

| Timing/memory boundary | Retained value | Interpretation |
|---|---:|---|
| Child seconds | 129.81141823600046 | Whole native baseline child |
| Supervisor seconds | 129.86850639100885 | Its surrounding supervisor interval |
| Minimum sampled available bytes | 121394933760 | Observed reserve witness, not peak model allocation |
| Completed response-record seconds | 66.32834011112573 | Sum of recorded response work, not the whole child |

Supervisor status is `completed`, actual exit is0, stop reason is null and cleanup
errors are empty. The child and supervisor intervals overlap; they are not added.
The completed raw records retain2692 generated actions,2692 forward calls and
117798 attempted/model full-prefix positions. Position counts are not FLOPs.
Source, input, interpreter/lock, contracts and local model inventories were bound
by the producer before and after execution; retained closing bindings accompany
the acceptance. These do not make an ignored-output directory a portable model
archive.

### 1.1 Actual Base raw continuation at caps32 and128

Base/raw's cap32 [preparation](native-reasoning-base-raw-cap32-20261005-run-01/preparation.json),
[acceptance](native-reasoning-base-raw-cap32-20261005-run-01/acceptance.json) and
[returned supervisor receipt](native-reasoning-base-raw-cap32-20261005-run-01/returned-supervision.json)
retain a second complete five-cell invocation. Its cap128
[preparation](native-reasoning-base-raw-cap128-20261005-run-01/preparation.json),
[acceptance](native-reasoning-base-raw-cap128-20261005-run-01/acceptance.json) and
[returned receipt](native-reasoning-base-raw-cap128-20261005-run-01/returned-supervision.json)
retain a third. Both pass all four literal execution/coverage checks with actual
exit0 and100 original responses each; incomplete/unreadable/missing collections
are empty. Retained closing source/input/model inventories equal preparation,
and every raw response field agrees with its accepted graded row. No weight-body
hashes or new grades were computed for this integration.

The fixed Base checkpoint is Qwen/Qwen3-0.6B-Base revision
`da87bfb608c14b7cf20ba1ce41287e8de496c0cd`, recorded in both rows as
`local-hf-sha256:a14dcabce74aa3c5a7751288f4d83e85e0f0277648a6586b6cae541c3a2e1df2`.
Original prompts are raw text, with no chat template or added special tokens;
thinking is not applicable. Both use the same fixed BF16/eager implementation,
four sampled seeds, separate greedy cell, context512 and900-second/25GiB
whole-child envelope. Cap128 allows12800 emitted actions, not a new execution
deadline. Base/raw's semantic interface SHA is
`e6d9c3b00216043e0bf6ae129d11d2a988050b85c8c0036fed59ed16a19354cb`.
It is not Instruct with thinking disabled.

| Base/raw boundary | Cap32 | Cap128 |
|---|---:|---:|
| Whole child seconds | 136.51633585902164 | 334.02569088502787 |
| Supervisor seconds | 136.58701521303738 | 334.07604531402467 |
| Minimum sampled available bytes | 121993846784 | 121081409536 |
| Completed response-record seconds | 76.99439720402006 | 270.4933472422417 |
| Generated actions / forward calls | 3163 / 3163 | 12041 / 12041 |
| Attempted/model full-prefix positions | 97810 / 97810 | 930423 / 930423 |

Both supervisor statuses are completed, stop reasons null and cleanup errors
empty. These overlapping time boundaries are not added, positions are not FLOPs,
and uncached implementation timings are not optimized throughput measurements.
Base/raw versus Instruct/native-chat changes both weights and serialization;
its descriptive differences cannot isolate instruction training's causal effect
or establish a broad model/method ranking. The two Base/raw caps hold checkpoint
and serialization fixed while changing the cap, but do not promise a monotonic
gain on each prompt, a faithful rationale or general reasoning improvement.

### 1.2 Failed Base/custom-chat/cap32: preserve the unrepresentable action

The Base/chat/cap32 [preparation](native-reasoning-base-chat-cap32-20261005-run-01/preparation.json),
[failed verdict](native-reasoning-base-chat-cap32-20261005-run-01/acceptance.json),
[failure record](native-reasoning-base-chat-cap32-20261005-run-01/failure.json) and
[returned supervisor receipt](native-reasoning-base-chat-cap32-20261005-run-01/returned-supervision.json)
do **not** establish an accepted baseline. The actual child exits1. Its exit0,
five-cell completion and original-response coverage checks are false; unreadable
JSONL remains empty. Forty records are retained and sixty planned responses are
missing, not negative answers that were generated.

Sample1009 has twenty records and a completed summary. Sample1019 has twenty
records and `completed_with_errors`; sample1029, sample1039 and greedy have no
summary or response records. At sample1019/math6, the retained24-action
trajectory ends in selected ID151768. The decode error states that this ID has
no tokenizer mapping and retains the full IDs. This is a recorded `ERROR` with
`stop_reason="error"`, not `UNSUPPORTED` mathematical parsing, a natural stop,
a cap, or a repaired/filtered draw. The partial decoded text and actual likelihood,
sampling and cost evidence remain unchanged. No vocabulary mask, replacement
token, resampling or output repair was introduced by this prose integration.

| Failed-child boundary | Retained value |
|---|---:|
| Actual native exit | 1 |
| Whole child seconds | 56.54799546097638 |
| Supervisor seconds | 56.59703186398838 |
| Minimum sampled available bytes | 122019704832 |
| Retained / missing planned records | 40 / 60 |
| Retained generated actions / calls | 1244 / 1244 |
| Retained attempted/model positions | 48378 / 48378 |
| Retained response-record seconds | 30.84977986908052 |

Supervisor status is failed, stop reason null and cleanup errors empty: no external
deadline or reserve stop caused this exit. Source/input/model closing bindings
remain unchanged, and all forty raw records match the failed verdict's corresponding
graded rows. Within those partial records are2 natural stops,37 caps,39
`UNSUPPORTED` grades and1 `ERROR`. Format-valid is39/40 because the error row
fails that predicate; it must not be laundered into a valid answer by `any`.
The partial held-out slice has22 records, all capped and unsupported, but it is
not a completed44-sample plus11-greedy comparison. The missing cells have no
measured score or cost. The error action's24 generated actions/forward calls
and828 positions, plus all other failed-invocation spending, remain retained.
The owner initially paused for diagnosis, then continued other independent
predeclared conditions without changing source/gates or repairing this failed
row. This report does not authorize a new job, sampling change or claim that the
interface failure has been repaired.

### 1.3 Failed Base/custom-chat/cap128: a second partial condition, not a retry

The separate cap128 [preparation](native-reasoning-base-chat-cap128-20261005-run-01/preparation.json),
[failed verdict](native-reasoning-base-chat-cap128-20261005-run-01/acceptance.json),
[failure record](native-reasoning-base-chat-cap128-20261005-run-01/failure.json) and
[returned supervisor receipt](native-reasoning-base-chat-cap128-20261005-run-01/returned-supervision.json)
also retain native exit1, forty records and sixty missing. This was the other
fixed Base/chat budget condition, not a repaired retry of cap32. Sample1009
completes twenty records; sample1019 records twenty with `completed_with_errors`;
the remaining two sample seeds and greedy are not entered. The same
sample1019/math6 decode error retains24 generated IDs with selected ID151768
as the24th action. Source, contracts and gates remain bound; no vocabulary
filter, new draw, token substitution or rescore was added.

| Base/custom-chat/cap128 failed boundary | Retained value |
|---|---:|
| Actual native exit | 1 |
| Whole child seconds | 132.66686874401057 |
| Supervisor seconds | 132.72932148602558 |
| Minimum sampled available bytes | 120977350656 |
| Retained / missing planned records | 40 / 60 |
| Natural stops / caps / decode errors | 8 / 31 / 1 |
| Retained generated actions / calls | 4491 / 4491 |
| Retained attempted/model positions | 377459 / 377459 |
| Retained response-record seconds | 105.78654449281748 |

Supervisor stop reason is null and cleanup errors are empty; this is not a
deadline or reserve stop. Retained closing bindings and all forty raw records
match the failed verdict. Its grades are39 `UNSUPPORTED` and1 `ERROR`, and
format-valid is39/40. The error trajectory retains24 actions/calls and828
positions. The partial held-out slice contains22 records, with3 natural stops
and19 caps, all unsupported. Neither a full44-sample slice nor greedy coverage
exists, so these records do not become a complete paired interface comparison.

Across all eight invoked conditions there are680 retained response records,
including the two error records, out of800 planned response slots;120 are missing
from the failed Base/chat invocations. Six accepted invocations supply600 records;
two failed invocations supply80 partial records. Those are execution/coverage
counts, not600 mathematically successful answers, a pooled accuracy denominator,
or complete800-response coverage. Both Base/chat rows remain **failed**. No new mandatory requirement to
make those rows successful is added here, and no missing-sixty paired comparison
or full-panel completeness is claimed. The original criteria, dependency gates,
full-support sampling, quotas and failed-work retention remain unchanged.

### 1.4 Actual Instruct/thinking-off/cap128: more budget, unchanged weights

The cap128 [preparation](native-reasoning-instruct-thinking-off-cap128-20261005-run-01/preparation.json),
[acceptance](native-reasoning-instruct-thinking-off-cap128-20261005-run-01/acceptance.json)
and [returned receipt](native-reasoning-instruct-thinking-off-cap128-20261005-run-01/returned-supervision.json)
pass all four literal execution/coverage checks. All100 original records are
retained, with no missing, unreadable or incomplete record; the whole child exits0.
Retained closing bindings and every raw response field match the accepted rows.
The original logical panel, source bindings, Instruct checkpoint byte inventory,
native thinking-off template/interface, decoding seeds and stopping settings
equal cap32. Only the output allowance changes32→128; this is not a newly trained
checkpoint, a thinking-toggle result or an alteration of the parser.
All100 paired attempt seeds and rendered prompt IDs agree; each retained cap32
action sequence is exactly a prefix of its cap128 counterpart. This is an
observed within-pair record join, not a promise of GPU determinism across new runs.

| Instruct/off/cap128 boundary | Retained value |
|---|---:|
| Whole child seconds | 247.0372236270341 |
| Supervisor seconds | 247.08396206301404 |
| Minimum sampled available bytes | 120799416320 |
| Completed response-record seconds | 181.13693934329785 |
| Generated actions / forward calls | 7727 / 7727 |
| Attempted/model full-prefix positions | 627977 / 627977 |

Status is completed, stop reason null and cleanup errors empty. The900-second
whole-child deadline and25GiB reserve stay unchanged. Parent-launcher observations
are separate from these native-child and supervisor intervals. Neither positions
nor the fixed BF16/eager code path constitute FLOPs, a kernel trace or optimized
serving throughput.

The coverage aggregate records34 correct,90 natural stops,10 caps,65 unsupported
and1 invalid grade; it mixes four sampled repetitions with greedy and is not a
deployment headline. On the eleven controlled held-out problems, the actual
sampled split is13/44 correct with35 natural stops and9 caps; greedy is7/11
correct with11 natural stops and no caps. All100 pass the permissive `any` format
predicate. This is a bounded answer-grading observation on the original contract,
not proof of general reasoning or rationale faithfulness.

Two of the thirteen correct held-out samples still hit cap128:
sample1009/math19 has supported boxed `1.5`, and sample1019/math18 has supported
boxed `2`. Their accepted `correct` and `task_success` are true while
`truncated=true` and `stop_reason="max_tokens"`. Do not erase those flags or
invent natural completion. The publication parser's answer-grade success is not
the strict EOS-dependent complete-path reward of the symbolic control. Eleven
other correct held-out samples naturally stop; all seven correct greedy answers
naturally stop. Any-candidate availability across the four samples is8/11 distinct
controlled problems, not the7/11 greedy-delivered score or an implemented
best-of-four service. The fixed checkpoints and interface make this a cap
intervention, but neither more supported answers nor longer visible traces
establish per-prompt monotonic improvement or the causal correctness of a rationale.
For example, sample1009/math11 caps at32 but reaches a supported boxed `3` and
natural turn stop after108 actions at128. By contrast sample1009/math10 is the
same nine-action, naturally stopped `5 + 6 = 11` and remains unsupported at both
caps. More allowance does not force every output into the parser's grammar.

### 1.5 Actual native thinking-on at caps32 and128: traces are not final answers

Both thinking-on conditions now have accepted actual receipts: cap32
[preparation](native-reasoning-instruct-thinking-on-cap32-20261005-run-01/preparation.json),
[acceptance](native-reasoning-instruct-thinking-on-cap32-20261005-run-01/acceptance.json)
and [returned receipt](native-reasoning-instruct-thinking-on-cap32-20261005-run-01/returned-supervision.json);
cap128 [preparation](native-reasoning-instruct-thinking-on-cap128-20261005-run-01/preparation.json),
[acceptance](native-reasoning-instruct-thinking-on-cap128-20261005-run-01/acceptance.json)
and [returned receipt](native-reasoning-instruct-thinking-on-cap128-20261005-run-01/returned-supervision.json).
Each exits0, passes all four literal execution checks and retains100 original
responses, with no unreadable, missing or incomplete record. All200 raw records
agree with the accepted fields and retained closing bindings; there is no model
execution or regrading by this integration.

| Thinking-on cost boundary | Cap32 | Cap128 |
|---|---:|---:|
| Whole child seconds | 144.33200714498525 | 364.1156065230025 |
| Supervisor seconds | 144.3781207689899 | 364.1712747570127 |
| Minimum sampled available bytes | 121987284992 | 120767070208 |
| Completed response-record seconds | 79.3077671159408 | 297.00581288244575 |
| Generated actions / forward calls | 3200 / 3200 | 12800 / 12800 |
| Attempted/model full-prefix positions | 124960 / 124960 | 1114240 / 1114240 |

The sampled and greedy cost modes also remain separate. At32 the eighty sampled
records retain2560 actions/99968 positions, while greedy's twenty retain640/24992.
At128 sampled retains10240/891392 and greedy2560/222848. These are actual record
costs, not a pooled quality score or independent problem count.

Both supervisors are completed with null stop reason and empty cleanup errors;
the whole-child900-second/25GiB envelope remains unchanged. Every response emits
its full declared cap; none naturally stops or records an adapter error. The
held-out sampled systems separately have0/44 correct,0 natural stops and44 caps
at32 and at128. Each separate greedy system has0/11 correct,0 stops and11 caps.
All100 records per cap are format-valid under `any` and graded `UNSUPPORTED`.
Execution acceptance is not mathematical success; permissive formatting is not
parser support, and zero accepted correct answers do not prove all visible
mathematical statements are wrong.

Thinking on/off uses the same actual Instruct weights, tokenizer, native template
and semantic interface SHA `13252c7df16dc0ef0863a5b5e55d9414b1ceeb9da8eed6ac6ec23a4caf95789a`.
The supported flag changes rendered prompt IDs on all100 paired problems: off
renders an empty, closed thinking region in its assistant prefix, while on leaves
the assistant prefix open for the model's generated `<think>` text. Do not call
that a new model or an unchanged rendered prompt. Within thinking-on32→128,
all100 attempt seeds/prompt IDs agree and the32-action IDs are exact prefixes
of the128-action trajectories. This is observed pairing within these runs, not
GPU determinism across future platforms/runs.

At sample1009/math10, the128-action response begins `<think>` and a discussion
of computing5+6, but caps before a final answer. Neither budget's hundred
responses contains a closing `</think>` or boxed answer; accepted extraction
falls back to `whole`. Raw/scoring text retains the thinking material rather
than silently removing it or rescuing a numeral. These are budget-limited traces
under the frozen final-answer/parser interface, not a general demonstration that
thinking is harmful or incapable of mathematics. Comparing off/on at a fixed cap
is a supported flag/serialization intervention on fixed weights; a longer output
budget is another axis. Their actual costs and stops stay separate. A longer
thinking budget, explicit final-answer instruction or different extraction would
be a new declared experiment, not retrospective repair or a rationale-faithfulness
claim. No best-of-four selector is measured; any-candidate graded-correct
availability is0/11 at each thinking-on cap.

## 2. The denominator is not a hundred independent problems

The [logical panel](native-reasoning-instruct-thinking-off-cap32-20261005-run-01/logical-panel.json)
contains twenty original problems. In each completed invocation, four sampled
repetitions and one greedy diagnostic produce100 response attempts, not100 independent source problems or
five independent models. The sampled and greedy systems stay separate:

The following table is the controlled eleven-problem slice only; failed Base/chat
partial records are not inserted as complete comparable systems:

| Accepted policy/interface | Decoding system | Attempts | Declared graded-correct | Format valid | Natural stop | Cap truncation |
|---|---|---:|---:|---:|---:|---:|
| Base/raw/cap32 | Four samples | 44 | 0/44 | 44/44 | 1/44 | 43/44 |
| Base/raw/cap32 | Greedy | 11 | 0/11 | 11/11 | 0/11 | 11/11 |
| Base/raw/cap128 | Four samples | 44 | 2/44 | 44/44 | 9/44 | 35/44 |
| Base/raw/cap128 | Greedy | 11 | 1/11 | 11/11 | 2/11 | 9/11 |
| Instruct/off/cap32 | Four samples | 44 | 0/44 | 44/44 | 3/44 | 41/44 |
| Instruct/off/cap32 | Greedy | 11 | 0/11 | 11/11 | 1/11 | 10/11 |
| Instruct/off/cap128 | Four samples | 44 | 13/44 | 44/44 | 35/44 | 9/44 |
| Instruct/off/cap128 | Greedy | 11 | 7/11 | 11/11 | 11/11 | 0/11 |
| Instruct/on/cap32 | Four samples | 44 | 0/44 | 44/44 | 0/44 | 44/44 |
| Instruct/on/cap32 | Greedy | 11 | 0/11 | 11/11 | 0/11 | 11/11 |
| Instruct/on/cap128 | Four samples | 44 | 0/44 | 44/44 | 0/44 | 44/44 |
| Instruct/on/cap128 | Greedy | 11 | 0/11 | 11/11 | 0/11 | 11/11 |

The eleven controlled held-out IDs are math10 through math20. Arithmetic and
linear-algebra source/template slices and the multi-step family slice remain
separate problem roles. The original math1–math8 train rows and math9, whose
underlying4+5 problem overlaps the RLVR training fixture, are development/seen
diagnostics. Here the nine diagnostic roles mean eight **original fixture-train**
items math1–math8 plus math9's known RLVR-fixture overlap. Only math2 and math9
have `rlvr_train_problem_overlap=true`; the other seven diagnostic items do not.
These panel annotations do not mean the actual assistant full400, chosen or DPO
policies trained on these math rows, or prove upstream checkpoint contamination.
The string `heldout-source` on math9 does not erase its known fixture overlap.
Headline held-out evidence excludes all nine diagnostic items. A template slice
also changes source problems, so it is not a pure causal template intervention.

The Instruct/off32 acceptance's100-row coverage aggregate has23 natural stops,77 capped rows,
99 `UNSUPPORTED` grades and1 `INVALID` grade. It is retained for coverage, not
promoted to a pooled method-ranking score. All55 held-out attempts together have
4 natural stops and51 caps; the table above preserves their actual sampled versus
greedy denominators. No selector delivered a best-of-four answer. “Any sampled
candidate correct” under the declared grader is an availability diagnostic, not
the accuracy of a deployed best-of-N system. Instruct/off32 and Base/raw32 have
no graded-correct samples. Base/raw32's separate100-row coverage aggregate has
2 natural stops,98 caps,98 `UNSUPPORTED` and2 `INVALID` grades; its55 held-out
attempts have1 natural stop and54 caps, all `UNSUPPORTED`.

Base/raw128's100-row coverage aggregate has14 natural stops,86 caps,5 `CORRECT`,
92 `UNSUPPORTED` and3 `INVALID` grades. Two of those five correct rows are
development/seen diagnostics: sampled math2 and greedy math9. They do not enter
the controlled held-out headline. The held-out correct sampled rows are math18
at seed1019 and math10 at seed1039; the greedy correct row is math13. All three
are naturally stopped and use supported boxed extraction. Any-candidate
availability among four samples is2/11 distinct held-out problems, while the
actually generated greedy answer is correct on1/11, a different problem. No
oracle selector delivered the sampled successes, and no automatic best-of-four
service is measured. All100 rows per completed invocation remain format-valid
under `any`; pending rows have no measured grade or cost, not fabricated zeros.

## 3. What the declared whole-output grading actually rejected

The original item contract uses `extraction="boxed_or_final"` and
`format_policy="any"`. A unique supported box or exact `Final answer:` line can
provide an extracted answer; without one, the frozen extraction falls back to
the whole response. The bounded rational parser does not infer a last numeral
from arbitrary prose or equations. `UNSUPPORTED` is distinct from a supported
answer proven wrong. The permissive format predicate is also distinct from
successful extraction/canonicalization: `any` makes all100 observed non-error
responses in each completed invocation format-valid, including responses the
parser cannot grade.

Inspect [sample1009 raw records](../../outputs/native-reasoning-instruct-thinking-off-cap32-20261005-run-01/sample-1009/responses.jsonl)
at math10, not only the zero in an aggregate. The original prompt is
`Compute 5 + 6.` and reference is `11`. The actual raw text is
`5 + 6 = 11<|im_end|>`; scoring text is `5 + 6 = 11`. It naturally stops at the
declared turn marker, is not capped, and is format-valid under `any`. Extraction
retains the whole equation; canonicalization reports `UNSUPPORTED`, with reason
`outside exact rational grammar`. Its accepted `correct` and `task_success`
fields remain false.

This is a qualitative illustration of a model-output/grader-interface mismatch,
not a silently rescored correct headline. Instruct/off32's result remains **zero
accepted correct responses under the declared whole-output grading**, not evidence that every
visible mathematical statement is false or that the model performs no reasoning.
A stopped answer can be unsupported; a capped response can contain a plausible
numeral without completing; a format-valid response can fail canonicalization.
Keep all three failures visible. Final-answer correctness, even if established,
would not establish the faithfulness or causal role of a rationale.

Changing instructions to require a bare numeral, changing extraction, extending
the parser, or increasing the cap would be a new declared intervention. It may
be worth studying, but would require a new frozen interface, independent grader
controls and actual receipt. Do not rewrite these raw outputs, replace their
grades, exclude unsupported rows from denominators, or retrospectively turn this
run into that different experiment.

## 4. Declared interfaces and RLVR gates

These are the fixed baseline axes, not an implied ranking. All eight initial
conditions were invoked; failed partial rows remain failed rather than being
replaced or promoted to complete coverage:

| Weight/interface row | Cap32 | Cap128 |
|---|---|---|
| Qwen3-0.6B-Base, raw continuation | Actual accepted §1.1 | Actual accepted §1.1 |
| Same Base, original custom chat template | Failed/partial:40 retained,60 missing (§1.2) | Failed/partial:40 retained,60 missing (§1.3) |
| Qwen3-0.6B Instruct, native thinking off | Actual accepted above | Actual accepted §1.4 |
| Same Instruct, native thinking on | Actual accepted §1.5 | Actual accepted §1.5 |

Base is a different checkpoint, not Instruct with thinking disabled. Raw/chat
on fixed Base weights is a serialization intervention; supported thinking off/on
on fixed Instruct is a native template/output-path intervention. Cap32/128 is a
budget intervention. Prompt IDs, semantic interfaces, checkpoint inventories,
stopping, unsupported grades, errors and costs must stay attached to every row.
More visible text or a larger cap is not by itself stronger or more faithful
reasoning.

The frozen [RLVR stage adapter](../../scripts/run_native_rlvr_stages.py) and
[common20 evaluator](../../scripts/run_native_rlvr_evaluation.py) define later
evidence; their existence is not a measured outcome. Recovery must compare actual
source2, completed1→2 and pending1→2 children and exact final numerical history,
policy/reference/optimizer/RNG/source state, update2 metrics and the same physical
work/I/O journals. A saved fully collected rollout is spent evidence. Pending
resume applies its retained actions/rewards/old likelihoods without resampling;
later spending is not refunded merely because the numerical cursor rewinds.

Each accepted G4/G8 pilot must actually complete its declared16-iteration horizon,
bind its final committed state and exact policy export, then pass fresh common20
generation. Pilot-generation cap64 and recovery cap16 are not the common20
evaluation caps32/128. The runner's four held-out diagnostic problems are not the
publication panel. The common evaluation separately joins unchanged Instruct,
G4 and G8 on the same original twenty items, per cap and decoding cell, with known
overlap excluded from the held-out headline. All raw, capped, failed and partial
cost records remain evidence; no quality selection of checkpoints or seeds is
allowed.

Constant-reward groups have zero relative advantages whether all answers are
correct or all are wrong. That does **not** establish that the actual native loop
skips AdamW: its fixed application path also rescores policy/reference, includes
the exact-KL term, backpropagates, clips and invokes the optimizer when valid.
Completed iteration cursor, optimizer applications, nonzero relative policy
signal, policy-state movement and held-out improvement are different facts.
Later reports must read the actual histories rather than infer these counters
from a low baseline grade. RLVR training uses its own strict integer verifier;
the native panel's broader extraction/parser status is not its reward history.
Post-training common-panel generation uses BF16/eager; that does not relabel
the training loader's SDPA kernel path.

### 4.1 Actual G4 recovery run01: native exit0 is necessary, not sufficient

The G4 [preparation](native-rlvr-g4-recovery-20261005-run-01/preparation.json)
fixes source2, completed1→2 and pending1→2 at seed2323, learning rate1e-6,
KL coefficient0.02, group4 and recovery output cap16. It starts from the original
Instruct checkpoint with native thinking off, not the assistant full400 parent.
All three children retain completed native reports and actual exit0, but the
[adapter failure](native-rlvr-g4-recovery-20261005-run-01/failure.json) is real:
the pending child's supervisor records `Observed descendant cleanup timed out`.
There is no recovery acceptance and the adapter's three-way CPU comparison was
not reached.

| Recovery role | Actual native exit | Native child seconds | Supervisor seconds | Minimum sampled available bytes | Supervisor verdict |
|---|---:|---:|---:|---:|---|
| Source2 | 0 | 179.35619652300375 | 179.40738152200356 | 109313617920 | completed |
| Completed1→2 | 0 | 131.65557820600225 | 131.7237069410039 | 103996592128 | completed |
| Pending1→2 | 0 | 107.62926277797669 | 107.69894294301048 | 103994474496 | failed: descendant cleanup |

Each [returned receipt](native-rlvr-g4-recovery-20261005-run-01/returned-supervision-1.json)
([completed](native-rlvr-g4-recovery-20261005-run-01/returned-supervision-2.json),
[pending](native-rlvr-g4-recovery-20261005-run-01/returned-supervision-3.json))
keeps the600-second whole-child deadline and25GiB reserve. Stop reasons are null;
the first two cleanup lists are empty. In the pending receipt, all three helper
cleanup records retain actual exit−15, `cleanup_complete=true` and
`still_alive=false`, and both queue feeder threads are stopped. Those helper
receipts do not negate the separate observed-descendant cleanup timeout.
The owner's [retained outer observation](native-rlvr-g4-recovery-20261005-run-01/outer-launcher-observation.json)
is adapter exit1 after456.23976223001955 seconds;
that parent interval is not a native-child measurement or a supervisor receipt.
Later observed absence of GPU processes does not retroactively repair the failed
cleanup gate. No dependent G4 pilot or exact-replay/quality pass follows.

The failed attempt still contains interpretable, permanently spent measurements.
The [shared work journal](../../outputs/native-rlvr-g4-recovery-20261005-run-01-journals/work.jsonl)
has25 reservations and25 completions, with four actual `train_updates`: two in
source, one in completed resume and one in pending resume. Its three actual
collections split2/1/0 across those invocations. It retains160 newly collected
valid response tokens and224 applied response targets; the pending invocation
applies64 retained targets without a new collection. These are physical shared
journal measurements, not sums of the three reports' restored numerical cursors.
All three cursors finish at2; copied cumulative histories are not additional
fresh work. The [I/O journal](../../outputs/native-rlvr-g4-recovery-20261005-run-01-journals/io.jsonl)
retains fourteen completed reservations: ten saves, two inspections and two loads.
Neither journal has an open ticket at this retained boundary. Closed work/I/O
tickets do not supply the missing cleanup or exact-state comparison gate.

In the retained [source report](../../outputs/native-rlvr-g4-recovery-20261005-run-01-source/report.json),
both groups have four zero rewards and four zero advantages. Update1 nevertheless
records gradient norm5.811452865600586e-7 with displayed loss−0.0 and exact KL0.0;
update2 records norm0.1796875, loss0.000021557978470809758 and pre-update exact
KL0.0010778990108519793. The two recorded post-update policy identities differ.
The actual application path backpropagates the clipped-plus-exact-KL objective,
clips the gradient and calls AdamW with zero weight decay; it has no
constant-reward skipping branch. Thus zero task advantages do not imply no
optimizer operation or numerically zero parameter derivative. The numerical
gradient at a displayed zero loss and the later KL term must not be relabeled
as positive task-reward learning, verifier success or held-out improvement.
These remain measurements inside a failed recovery attempt; we did not load
state bodies or perform its unreached CPU comparison.

### 4.2 Separately declared controller continuation: no silent fallback

The [retained diagnosis](2026-10-05-native-rlvr-recovery-cleanup.md) identifies a
code-supported startup hypothesis: a late spawn helper imports the controller's
main module before its cleanup target, and the old controller imported numerical
modules at module scope. That startup is inside the original0.5-second cleanup
acknowledgment window. No historical helper stack or separated startup timing was
captured, so the diagnosis is not a measured causal explanation of run01.

The [explicit continuation declaration](2026-10-05-native-rlvr-lazy-import-continuation.json)
moves those controller imports into numerical functions and seals an explicit
run01/run02 pilot recovery selector. Its [independent CPU review](2026-10-05-native-rlvr-controller-runtime-independent.md)
retains89 passing tests, unchanged before/after source/declaration identities,
a fresh lightweight-controller check and bounded spawn-cleanup controls. The
native Qwen implementation, supervisor, baseline sources, model/data/interfaces,
sampler/rewards/reference, seed2323, learning rate1e-6, beta0.02, group sizes,
23-work/9-I/O allowances,600-second recovery and1800-second pilot deadlines,
25GiB reserve and0.5-second cleanup bound remain unchanged. This is an execution
startup correction, not permission to omit checks, extend deadlines or alter
learning science; authored CPU fixture passes are not pretrained recovery.

The owner reports starting the separately declared ten-job queue at13:02:07UTC,
beginning fresh G4 recovery run02. The declaration itself is retained as
`declared-not-launched`; that earlier prelaunch document is not an actual child
receipt. At that earlier launch boundary, run02 had no acceptance. It had to
pass its own three supervised native invocations and the adapter's twelve
recovery checks before G4 pilot01 may explicitly select `--recovery-run-id run-02`.
The default still selects01 and refuses failed/missing01; no automatic search or
fallback exists. G8 selects its original recovery01. The old G4 failure/journals
remain failed and spent. Planned pilots, four post-evaluations and two CPU
comparisons in that queue are not measured outcomes or additional campaign rows.

### 4.3 Actual accepted G4 recovery run02: exact replay is not a quality result

The fresh run02 [preparation](native-rlvr-g4-recovery-20261005-run-02/preparation.json),
[acceptance](native-rlvr-g4-recovery-20261005-run-02/acceptance.json) and
[closing bindings](native-rlvr-g4-recovery-20261005-run-02/closing-bindings.json)
now retain `passed`, all twelve literal checks true, three completed supervisors
and three actual native exits0. All stop reasons are null and cleanup errors
empty. The original600-second/25GiB envelope remains unchanged per invocation.

| Accepted recovery role | Native child seconds | Supervisor seconds | Minimum sampled available bytes |
|---|---:|---:|---:|
| Source2 | 176.87867476296378 | 176.92320748697966 | 109353230336 |
| Completed1→2 | 143.31851184100378 | 143.3822326939553 | 104016809984 |
| Pending1→2 | 109.66025628399802 | 109.72791634697933 | 103915323392 |

The [source](native-rlvr-g4-recovery-20261005-run-02/returned-supervision-1.json),
[completed](native-rlvr-g4-recovery-20261005-run-02/returned-supervision-2.json)
and [pending](native-rlvr-g4-recovery-20261005-run-02/returned-supervision-3.json)
receipts retain their actual cleanup details. Pending observes no owned
descendant and has two logger/observer helper cleanup entries, not the original
failed pending invocation's late-helper circumstance. Completed resume does
retain one observed descendant and three helper cleanup entries. We do not infer
a historical helper stack or a causal timing improvement from this fresh run;
the deliberately exercised0.5-second acknowledgment cases are separately
authored CPU controls. The [owner's outer transcript](native-rlvr-g4-recovery-20261005-run-02/outer-launcher-observation.json)
records adapter exit0 after482.61591269401833 seconds. That after-the-job typed
transcription overlaps preparation, native supervision, inline CPU comparison
and closure; it is not a prelaunch archive, an extra model invocation or an
independently isolated comparison duration.

The adapter actually reaches its streamed CPU comparison. All three final
completed2 descriptors agree, including ten components: policy, original frozen
reference, optimizer, loop contract, cursor, numerical collection/application
work, history, pending state and RNG. The retained scientific-state SHA is
`32c7e71b446895d564b0483737edf1adf924eb552313ee1ffa6f2fc522659f52`.
Source update2 metrics exactly equal both resume rows. The comparison excludes
only producer invocation and the physical work prefix, not numerical history or
original reference. It is actual native replay evidence, not simply file-load,
snapshot/export existence or the matching JSON metrics alone. This integration
reads the comparison receipt; it does not rehash tensor bodies.

The run02 [work journal](../../outputs/native-rlvr-g4-recovery-20261005-run-02-journals/work.jsonl)
retains25 reservations/25 completions and four actual optimizer applications
split2/1/1, but three collections split2/1/0. Fresh valid tokens are160 and
applied response targets224; pending consumes64 already collected targets
without resampling. Its [I/O journal](../../outputs/native-rlvr-g4-recovery-20261005-run-02-journals/io.jsonl)
retains14 reservations/completions: ten saves, two inspections and two loads.
The three run02 children share one physical main journal and one physical I/O
journal, retaining later source/resume spending; no open/failed ticket is accepted.
These are distinct from failed run01's retained journal identities and identical
operation-count geometry. A fresh run does not refund failed01 spending or
rewrite its cleanup status. Streamed comparison/archive hashes and export work
are outside the shared23-work/9-I/O ledger scope; those counters are not all
physical CPU work, FLOPs or aggregate memory containment.

Before/after closure retains all fourteen source bindings, five small input
bindings, the cached model inventory and original prerequisite bindings unchanged.
Compared with failed01, recipe, encoded geometry, allowances, model inventory
and corresponding input-byte SHAs agree; only the controller and its tests have
new source hashes. Role-specific input paths differ because this is fresh02.
Its source groups again have all-zero rewards/advantages and the exact gradient/
KL values reported in §4.1. Acceptance proves the fixed two-update recovery
mechanism, including no resampling and unchanged reference, not positive
task-relative learning or improved twenty-item answers. G8's subsequent accepted
recovery appears in §4.4. At that recovery-only boundary the pilot16 exports,
common20 post-evaluations and matched comparisons were still unmeasured; their
later actual outcomes appear in §4.6–§4.7.

The [independent recovery review](2026-10-05-native-rlvr-recovery-independent-review.md)
now joins G4 run02's twelve gates, current/prepared/closing small bindings,
recorded component maps/update2 history, actual native receipts and independently
validated journal chains, inode identities, saved prefixes and final totals.
It did not rehash5.39GB tensor bodies or rerun learning. Its completed-role
descendant-cleanup positive witness remains distinct from the original failed
pending-role circumstance and the separately authored delayed CPU negative.
This accepted metadata/replay review's G4 boundary did not pre-sign G8, pilots or quality.

### 4.4 Actual accepted G8 recovery run01: sampled slots are not valid targets

G8's original run01 [preparation](native-rlvr-g8-recovery-20261005-run-01/preparation.json),
[acceptance](native-rlvr-g8-recovery-20261005-run-01/acceptance.json) and
[closing bindings](native-rlvr-g8-recovery-20261005-run-01/closing-bindings.json)
now retain all twelve original checks true. Source2, completed1→2 and pending1→2
each has a completed supervisor, actual native exit0, null stop reason and empty
cleanup errors. All three observe no owned descendant and retain two
logger/observer helper cleanup records. Each original600-second whole-child
deadline and25GiB reserve remains unchanged.

| G8 recovery role | Native child seconds | Supervisor seconds | Minimum sampled available bytes |
|---|---:|---:|---:|
| Source2 | 182.05160450399853 | 182.09738468396245 | 109504069632 |
| Completed1→2 | 137.33149761601817 | 137.3943130170228 | 104302305280 |
| Pending1→2 | 101.90952163102338 | 101.97952055704081 | 104201089024 |

The [source](native-rlvr-g8-recovery-20261005-run-01/returned-supervision-1.json),
[completed](native-rlvr-g8-recovery-20261005-run-01/returned-supervision-2.json)
and [pending](native-rlvr-g8-recovery-20261005-run-01/returned-supervision-3.json)
receipts keep those distinct measured boundaries. The separately retained,
after-the-job [owner transcript](native-rlvr-g8-recovery-20261005-run-01/outer-launcher-observation.json)
records outer exit0/504.83586531702895 seconds; it overlaps preparation,
supervision, inline CPU comparison and closure rather than adding a fourth model
child or supplying a prelaunch-bound terminal archive.

The actual three-way comparison agrees on the same ten named scientific
components as G4, and source update2 exactly equals both resumed records.
G8's scientific-state SHA is
`6457e0efac59436ee4c0671923ea5284531c6a7967e4e39810f520294155421a`.
That is its own exact replay identity, not equality of G4/G8 final policies.
The original frozen-reference component agrees across both group-size runs.
Fourteen source/five input/model/prerequisite maps match before/after closure;
G4/02 and G8/01 retain the same source bindings, parent inventory, original
prompt geometry and non-group recipe. Group size differs, so their realized
rollouts and costs need not be equal.

The G8 [physical work journal](../../outputs/native-rlvr-g8-recovery-20261005-run-01-journals/work.jsonl)
has25 reservations/25 completions, again four optimizer applications split2/1/1
and three collections split2/1/0. Fresh valid actions are314 and applied response
targets434; pending applies120 saved targets without recollection. However,
the dense full-group sampler spends384 generated batch slots and384 multinomial
draws, including stopped-row slots. These are not384 emitted valid targets:
the response mask retains314 valid collection positions. The
[I/O journal](../../outputs/native-rlvr-g8-recovery-20261005-run-01-journals/io.jsonl)
again retains14 completed reservations, ten saves/two inspections/two loads,
with no open/failed ticket. Later spending remains in each shared physical
journal; neither G4's separate histories nor failed G4/01 are refunded.

Source update1 retains74 valid tokens with seven EOS stops and one cap; update2
retains120 with two EOS stops and six caps. Both eight-member reward/advantage
groups are all zero. Their gradient norms are4.3585896492004395e-7 and0.146484375;
update2 pre-update exact KL is0.0009698036592453718 and loss
0.000019645012798719108. Natural stops, valid-token masks, strict verifier
rewards and optimizer applications remain different facts. The BF16/SDPA
training configuration is not the BF16/eager common-panel generation path or
an independently profiled kernel trace. These two-update observations establish
the accepted recovery/no-resample mechanism, not nonzero task-relative learning,
equal-compute group-size comparison or reasoning superiority.

The owner started G4 pilot16/01 with explicit `--recovery-run-id run-02`.
Its subsequent trained-but-unaccepted failure is retained in §4.5. Accepted
pilots, four post-evaluations and the two original per-cap matched comparisons
were absent at that recovery-only boundary; no common20 quality score is filled from the runner's four
diagnostic items or from accepted recovery.

The [incremental independent recovery review](2026-10-05-native-rlvr-recovery-independent-review.md)
now also covers G8's separate retained receipts, chains/inodes/prefixes, masks,
stored ten-component comparison/update2 metrics and current before/after
fourteen-source/five-input bindings. It independently retains the384-slot,
314-valid,434-application distinction, while not repeating the actual producer's
large tensor-body comparison. Both selected recoveries are reviewed at their
own scope; neither pre-signs a pilot or publication-panel outcome.

### 4.5 Actual G4 pilot01: sixteen native iterations, failed terminal retention

The fixed [pilot preparation](native-rlvr-g4-pilot-20261005-run-01/preparation.json)
explicitly selects accepted G4 recovery02 and both original Instruct/off32/128
baselines. Its unchanged science is sixteen iterations, group4, output cap64,
seed2323, learning rate1e-6, beta0.02, native thinking off,1800 seconds and25GiB.
The retained [native report](../../outputs/native-rlvr-g4-pilot-20261005-run-01/report.json)
is completed with committed cursor16, sixteen update rows, final completed16
snapshot and exported policy present. The actual whole native child exits0.
Those facts do **not** make this an accepted pilot.

The [returned supervisor receipt](native-rlvr-g4-pilot-20261005-run-01/returned-supervision-1.json)
is failed: `journal_error.type="TimeoutError"`, message `Final logging timed out`,
and `final_record_retained=false`. Native failure and stop reason are null;
cleanup errors and observed-descendant lists are empty. Its two helpers are
terminated with cleanup complete. This is not another descendant-cleanup failure,
an1800-second external deadline, a reserve stop or a nonzero native exit. The
[adapter failure](native-rlvr-g4-pilot-20261005-run-01/failure.json) preserves
reports, checkpoints, export and spending. No pilot acceptance/closing-bindings
or accepted final-state-to-export inventory exists; the adapter never reaches
that acceptance path. The policy export cannot be used as an accepted publication
parent simply because the training horizon completed.

| Failed G4 pilot boundary | Actual retained value |
|---|---:|
| Native child exit / supervisor verdict | 0 / failed |
| Native child seconds | 1275.7366294650128 |
| Supervisor seconds | 1275.7819221359678 |
| Minimum sampled available bytes | 108300517376 |
| Final-record writer acknowledgment | false |
| Completed native iterations / optimizer applications | 16 / 16 |

The owner reports outer adapter exit1 after1293.7624233269598 seconds, a separate
after-job console observation overlapping preparation/native supervision and
terminal-retention failure, not an extra model run or a native receipt. A native
report's completed status, terminated helpers and later quiet GPU do not supply
the missing terminal writer acknowledgment. We have not diagnosed its historical
logger stack or promoted any surviving bytes to acceptance.

The [main journal](../../outputs/native-rlvr-g4-pilot-20261005-run-01-journals/work.jsonl)
retains67 reservations/67 completions, sixteen actual optimizer applications and
sixteen collections. Dense slots/draws are1508, while valid collection positions
and applied targets are1004 each;504 spent stopped-row slots are outside the
valid mask. All sixty-four training responses naturally stop at EOS, but every
four-member strict-reward group has zero rewards and zero advantages. All sixteen
recorded post-update policy identities differ; neither optimizer execution nor
state movement establishes nonzero task-relative reward learning. The final
gradient norm is0.07666015625, loss0.000014048184311832301 and pre-update exact
KL0.0006887196796014905. These are retained regularized numerical observations,
not a common-panel score or proof that every visible mathematical statement is
wrong. The strict integer verifier and publication parser remain distinct.

The [I/O journal](../../outputs/native-rlvr-g4-pilot-20261005-run-01-journals/io.jsonl)
has33 completed save reservations, no inspection/load and no open/failed native
ticket; recorded serialized bytes are173146264423. Native logical work/I/O
completion does not erase the later external logging failure. Initial/final
four-item diagnostics account for eight evaluation examples, not a twenty-item
publication run. Costs from failed G4 recovery01, both accepted recoveries and
this distinct failed pilot journal remain spent, with no reset or refund.

At the session91701 stop, G8 pilot01, all four original
post-evaluation conditions and both per-cap matched comparisons were not started.
At that boundary recovery prerequisites were accepted2/2, accepted pilots0/2, with one
trained-but-unaccepted attempt. No post-training quality outcome or completed
whole-goal claim was justified by that retained cursor16/export alone. The later
explicit export consumer and G8 pilot are separate observations below.

### 4.6 Actual native pilots: completed updates are not a reward signal

G8 pilot01 now has a completed16 [native report](../../outputs/native-rlvr-g8-pilot-20261005-run-01/report.json)
and [acceptance](native-rlvr-g8-pilot-20261005-run-01/acceptance.json) with all three
original checks true. Its [closing bindings](native-rlvr-g8-pilot-20261005-run-01/closing-bindings.json)
retain the original source, inputs, cached Instruct parent and prerequisites.
It uses the same seed2323, learning rate1e-6, beta0.02, cap64, sixteen iterations
and native thinking-off interface, changing group size from4 to8. Training is
BF16/SDPA; the actual dispatched kernel is not identified by that backend label.

| Actual native training measurement | G4 pilot01 | G8 pilot01 |
|---|---:|---:|
| Committed iterations / optimizer applications / collections | 16 / 16 / 16 | 16 / 16 / 16 |
| Training responses | 64 | 128 |
| Positive strict rewards / nonconstant-advantage groups | 0 / 0 | 0 / 0 |
| Naturally stopped responses / capped responses | 64 / 0 | 128 / 0 |
| Dense sampled slots / multinomial draws | 1508 / 1508 | 4888 / 4888 |
| Fresh valid actions / applied targets | 1004 / 1004 | 2335 / 2335 |
| Completed snapshot saves | 33 | 33 |
| Recorded serialized bytes | 173146264423 | 173147185063 |
| Update16 pre-update exact KL | 0.0006887196796014905 | 0.0010826910147443414 |
| Final gradient norm | 0.07666015625 | 0.0966796875 |

The G8 [work journal](../../outputs/native-rlvr-g8-pilot-20261005-run-01-journals/work.jsonl)
has67 reservations/completions and its [I/O journal](../../outputs/native-rlvr-g8-pilot-20261005-run-01-journals/io.jsonl)
has33 saves, no inspection/load and no open/failed ticket. All sixteen groups in
each arm have zero rewards and zero advantages. The application path still calls
AdamW; exact-KL regularization and finite-precision numerical movement are not
task-relative reward learning. For example, G8's retained training text
`2 + 3 = 5` can be mathematically intelligible while failing the strict
integer-only reward. Natural stopping does not repair that interface mismatch.
G8 spends2553 stopped-row slots outside the valid mask versus G4's504; doubling
group size is not an equal-token-budget comparison. Both initial/final four-item
diagnostics remain separate from the common twenty-item publication panel.
G8's initial and final diagnostics both score0/4, but their retained responses
differ; equal aggregate scores do not prove identical generations. The
[independent pilot review](2026-10-05-native-rlvr-g8-pilot01-independent-review.md)
checks these metadata/record joins without rehashing model/state bodies.

G4's [export observation](native-rlvr-g4-pilot-20261005-run-01/export-observation.json)
separately validates the committed16 payload, exact exported files and genealogy,
with all three consistency checks true. Its original failed supervisor remains
failed; the receipt is not pilot acceptance or a new training run. The
[independent consumer review](2026-10-05-native-rlvr-observed-export-independent-review.md)
retains that boundary. These two completed policies may enter the explicitly
declared observed-export comparison, whose measured results follow.

### 4.7 Actual common20 comparisons: the denominator changes the conclusion

All four fresh post-evaluations pass their four execution/coverage checks with
native exit0 and100 retained responses each:
[G4/cap32](native-rlvr-evaluation-g4-cap32-20261005-run-02/acceptance.json),
[G4/cap128](native-rlvr-evaluation-g4-cap128-20261005-run-02/acceptance.json),
[G8/cap32](native-rlvr-evaluation-g8-cap32-20261005-run-02/acceptance.json) and
[G8/cap128](native-rlvr-evaluation-g8-cap128-20261005-run-02/acceptance.json).
The actual [cap32 comparison](native-rlvr-common20-cap32-20261005-run-02.json)
and [cap128 comparison](native-rlvr-common20-cap128-20261005-run-02.json) join
unchanged Instruct/G4/G8 on the same original item/seed/decoding contract in each
cell, binding policy exports, raw records, inputs and producer sources. The
400 new RLVR records plus200 reused unchanged-Instruct baseline records supply
600 comparison records, not600 new generations. All earlier failed/partial
baseline records and failed G4 supervision remain retained.

Generation uses the original native Instruct thinking-off template, BF16-loaded
policies, eager attention, full-prefix no-KV execution and full-support
temperature1 sampling at the four original seeds1009/1019/1029/1039, plus a
separate greedy diagnostic. No instruction/parser repair, vocabulary filtering,
quality selection or new training is introduced. The held-out slice excludes
all nine diagnostic items; the two actual RLVR-overlap flags remain math2/math9.
The sampled44 observations repeat eleven problems four times; they are not44
independent source problems and are not pooled with eleven greedy answers.

| Cap | Policy | Sampled correct /44 | Sampled natural / cap | Greedy correct /11 | Greedy natural / cap |
|---|---|---:|---:|---:|---:|
| 32 | Unchanged Instruct | 0/44 | 3 / 41 | 0/11 | 1 / 10 |
| 32 | RLVR G4 | 0/44 | 3 / 41 | 0/11 | 1 / 10 |
| 32 | RLVR G8 | 0/44 | 3 / 41 | 0/11 | 1 / 10 |
| 128 | Unchanged Instruct | 13/44 | 35 / 9 | 7/11 | 11 / 0 |
| 128 | RLVR G4 | 11/44 | 37 / 7 | 7/11 | 11 / 0 |
| 128 | RLVR G8 | 13/44 | 38 / 6 | 7/11 | 11 / 0 |

The all-original100 cap128 correct counts rise34→35→36, while the held-out
sampled counts are13→11→13 and greedy remains7→7→7. Correct diagnostic counts
instead rise14→17→16. The all-panel increase is therefore not a held-out
learning headline. G4's two fewer sampled successes are a small fixed-slice
observation, not a general regression or population ranking. Per-seed matched
results remain in the comparison receipts; their existing descriptive source-
cluster intervals are not a new claim of method superiority.

At cap128, correct-but-capped held-out samples number2/2/1 for unchanged/G4/G8.
More natural stops do not by themselves establish more correct answers. The
permissive `any` format predicate remains separate from extraction and grading;
`UNSUPPORTED`, `INVALID` and caps remain in their denominators, and none of the
four new evaluations has an `ERROR` record. Neither native pilot increases
held-out graded-correct counts here, and both training histories had zero
relative rewards/advantages. This is a measured bounded negative result, not
proof that GRPO cannot work, that visible mathematics is wrong, or that a
generated rationale is faithful. The separate symbolic positive controls stand.

Measured native evaluation child seconds are202.25339220499154/333.345343247056
for G4 cap32/128 and201.02303303498775/336.047204387025 for G8 cap32/128.
Some evaluations overlap GPU-hidden CPU notebook verification; these whole-child
intervals include loading and evidence work and are not isolated throughput
benchmarks. The owner-observed CPU comparison intervals26.768526927975472 and
34.05049322597915 seconds are separate overlapping boundaries, not model time
or FLOPs. Raw token/forward-position costs remain in the receipts.

## 5. Course integration and current boundary

[Chapter13§13.8.3](../../book/chapters/13-group-relative-policy-optimization.md#1383-the-first-native-row-grading-is-an-interface)
and [Lab13](../../book/labs/13-group-relative-policy-optimization.md#read-actual-evidence-without-repairing-the-result)
turn the retained baseline into an evidence-reading exercise. The focused
[Day23 artifact](../../learning_artifacts/day-23-grpo-rlvr/generated-answers-and-grader-interfaces.md)
records the distinction without advancing the learner's Day9 position or claiming
that an agent's production work demonstrates learner mastery. Historical symbolic
negative and positive controls remain unchanged and separately scoped.

Suggested article and animation ideas are source captures only. Mac drafting or
rendering would need separate production authority; no article, animation,
upload, new inference run or grading intervention was produced by this report.
Current defensible conclusion: six native interface/cap baselines fully executed.
All three accepted cap32 conditions have negative declared grades; Base/raw128 has a few supported,
naturally stopped correct answers, alongside pervasive truncation and unsupported
outputs. Instruct/off128 records13/44 sampled and7/11 greedy held-out graded-correct
answers, with correct-but-capped samples retained. The Instruct/off32 example
makes the grading-interface limitation visible. Both thinking-on budgets cap
all100 responses and have negative declared grades, without proving visible
mathematics wrong or supplying a general thinking-mode ranking.
Both Base/custom-chat caps actually failed with40 retained/60 missing records
each and the unmapped selected action. Neither is an accepted row or a zero-filled
complete result. Initial coverage is680/800 including two errors, not full-panel
completion. G4 recovery run01 fails its observed-descendant cleanup gate despite
three native exits0; its CPU comparison was not reached. Fresh G4 run02 actually
passes all twelve recovery checks and its exact final comparison, without
rewriting failed01 or establishing learning quality. G8/01 also passes all twelve
recovery checks, preserving distinct sampled-slot/valid-target counts.
G8 pilot16 is now accepted; G4 pilot01 completes native16/export and has a separate
export consistency closure, without changing failed supervision. Both arms have
sixteen zero-reward/zero-advantage groups, not evidence of positive task-relative
learning. Common20 post-evaluations and matched per-cap comparisons are now
measured: no held-out correct-count gain at either cap, despite increased
all-panel cap128 counts and fewer caps. The initial680/800 coverage still
includes two failures and120 missing records; it is not repaired into800/800.
These bounded outcomes are not a universal method ranking or learner-mastery claim.

A separate read-only metadata review of the initial Instruct/off32 row reproduced
its sampled/greedy and held-out
counts, timings, costs, overlap annotation, closing identity bindings and literal
math10 extraction/grade from the retained receipts and five small raw JSONLs.
That review performed no model execution, weight/state-body hashing or rescoring;
it does not expand the report's current scientific boundary.
The retained [incremental independent review](2026-10-05-native-reasoning-baselines-independent-review.md)
now covers all eight initial invocations and680 actual records, with zero
unchanged-grader mismatches and checked identity/coverage/resource joins. Its
earlier340- and380-record prefixes remain historical boundaries, not advance
acceptance of later rows. This completed baseline review does not pre-sign any
RLVR recovery, pilot or post-training comparison.
