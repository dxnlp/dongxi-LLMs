# Independent reasoning-baseline review: initial evidence boundary

Latest boundary: all eight fixed conditions have now been attempted and reviewed
in the incremental sections below. Six accepted conditions retain100 records
each; two failed Base/chat conditions retain40 each, with120 records missing.
Earlier boundary statements are preserved chronologically, not current claims
that later accepted conditions remain unrun. Active RLVR and goal closure are
still outside this baseline-only review.

This is an **incremental review, not final eight-condition acceptance**. The
initial inspection covers three accepted invocations and one failed invocation;
the other four interface/cap conditions had no acceptance receipt at that
boundary. It neither pre-signs their outcomes nor completes the active RLVR
comparison or the eighteen-package goal.

Only this report was written. Review used source inspection, bounded saved
metadata/JSONL reads, source-file SHA checks and the unchanged model-free grader.
No model/tokenizer load, weight-body hash, GPU operation, producer/test launch,
resampling, acquisition, environment installation, source/spec/tracker change,
or Git mutation was performed. Native experiments below were executed by the
parent's separately supervised campaign, not by this reviewer.

## Initial actual matrix

Every successful condition plans five separate cells: sampled seeds
1009/1019/1029/1039, 20 original problems each, plus an independent greedy1009
diagnostic on the same 20 problems. A complete condition therefore has 80
sampled and 20 greedy records, not 100 independent problems or a best-of-four
selected answer. Its controlled heldout subset has 44 sampled and 11 greedy
records from 11 problems.

| Condition, run01 | Actual native exit | Recorded / missing | Parser-confirmed correct | Natural / cap stops | Controlled heldout correct |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base/raw32: accepted | 0 | 100 / 0 | 0/100 | 2 / 98 | 0/55 |
| Base/raw128: accepted | 0 | 100 / 0 | 5/100 | 14 / 86 | 3/55 |
| Base/custom-chat32: **failed** | 1 | 40 / 60 | 0/40 observed | 2 / 37; one error | 0/22 observed |
| Base/custom-chat128 | null: not yet reviewed | null | null | null | null |
| Instruct/native-thinking-off32: accepted | 0 | 100 / 0 | 0/100 | 23 / 77 | 0/55 |
| Instruct/native-thinking-off128 | null: not yet reviewed | null | null | null | null |
| Same Instruct/native-thinking-on32 | null: not yet reviewed | null | null | null | null |
| Same Instruct/native-thinking-on128 | null: not yet reviewed | null | null | null | null |

Actual anchors are the acceptances for
[Base/raw32](native-reasoning-base-raw-cap32-20261005-run-01/acceptance.json),
[Base/raw128](native-reasoning-base-raw-cap128-20261005-run-01/acceptance.json),
[Instruct/off32](native-reasoning-instruct-thinking-off-cap32-20261005-run-01/acceptance.json),
and the failed
[Base/chat32](native-reasoning-base-chat-cap32-20261005-run-01/acceptance.json).
The first three pass their four literal execution/coverage checks. Failed
Base/chat32 passes only the no-unreadable-JSONL check. There are no unknown
native exits among these four recorded invocations; an unobserved condition is
not assigned exit0 or a zero score.

Base/raw128's sampled cells have correct counts 1/1/0/1 out of 20; its greedy
cell has 2/20. Heldout counts are 0/1/0/1 sampled and 1/11 greedy. The pooled
5/100 and 3/55 counts are explicitly mixed decoding records, not deployment
accuracy or five distinct solved problems. No model-ranking or statistical
confidence claim is inferred from them.

## Original sources, selected weights and serialized inputs

All ten source/test hashes matched the frozen preparations at initial review.
The baseline driver is
`249102e3dc4d6d81cf1c86ae71b65a9023dbb2f10c32698ad1eebf747cfa54e4`.
The source's default route prepares only; execution selects a fixed interface,
32/128 cap and one closed run identifier, with no arbitrary argv or acquisition.

The accepted and failed preparations have valid self-SHAs. Their closing
source/input/local-model bindings exactly equal preparation; logical-panel
hashes, complete annotated items, retained launch commands and supervisor
PID/status/exit joins match. The 17 observed generation cells have consistent
identity self-SHAs, source and input maps, selected model/tokenizer declarations,
lock/interpreter identities, observed tokenizer semantics, contract/config and
command joins. All 340 recorded prompts' serialized text and token IDs equal
their tokenizer-preparation encodings. No new tensor inventory was measured.

Base/raw and Base/custom-chat select the same local
`Qwen/Qwen3-0.6B-Base` revision
`da87bfb608c14b7cf20ba1ce41287e8de496c0cd`. The Base intervention changes raw
text versus the explicit custom chat serialization, not weights. Instruct/off
selects `Qwen/Qwen3-0.6B` revision
`c1899de289a04d12100db370d81485cdf75e47ca`; its planned thinking-on counterpart
must use that same Instruct checkpoint, not a newly trained model. Base and
Instruct are distinct checkpoints; disabling thinking does not turn Instruct
into Base. The native Instruct template explicitly supports `enable_thinking`;
raw input has no thinking flag, and Base/custom-chat uses template-default.

Local revisions are retained declarations plus the producer's historical
streamed byte identities, not newly remotely authenticated upstream provenance.
The observed model-loaded events report CUDA, BF16, `Qwen3ForCausalLM` and
596,049,920 parameters. Source selects eager, uncached full-prefix generation.
That declaration and equal recorded configuration are not a kernel trace,
bitwise GPU determinism proof, MATH-backend claim or serving-speed benchmark.

The original prompts and references are unchanged. Context512 and caps32/128
fit without prompt truncation. Sampling uses temperature1, no top-k filter and
top-p1; selected behavior and raw log probabilities agree at every observed
sampled step. Greedy is a separate cell. Natural stops bind the actual EOS151643
or turn-stop151645; raw IDs/costs retain the stop token, while scored text removes
only that terminal token, not thinking text or unfavorable prose.

## Grader boundaries and the actual overlap annotation

Independent replay of all **340 observed records** found zero stored
status/correctness/format/termination/truncation/cost mismatches. No failed or
unsupported record was rewritten or rescued by selecting its last number.

| Condition | Actual retained grade statuses |
| --- | --- |
| Base/raw32 | 98 UNSUPPORTED, 2 INVALID |
| Base/raw128 | 92 UNSUPPORTED, 3 INVALID, 5 CORRECT |
| Base/custom-chat32, partial | 39 UNSUPPORTED, 1 ERROR |
| Instruct/off32 | 99 UNSUPPORTED, 1 INVALID |

`UNSUPPORTED` means outside the bounded verifier's grammar, not proof that an
answer is mathematically wrong. `INVALID` is a parsing/extraction boundary;
`ERROR` is an adapter/runtime failure. The rational/set/interval parser, explicit
answer extraction, unit identity and ambiguity rules remain unchanged. A
`format_policy='any'` success is not an accuracy or demanding-format score:
all 300 accepted-condition records are format-valid, including unsupported
answers; the failed condition has 39/40 format-valid records, excluding its
error. Natural termination, cap truncation and answer correctness are separate.

The nine `development-or-seen-diagnostic` items are **eight original fixture-train
rows math1–math8 plus math9's known RLVR training-problem overlap**. Only math2
and math9 carry true `rlvr_train_problem_overlap` flags. Math9's original
`heldout-source` label does not erase that known overlap. Math10–math20 form the
11-item controlled heldout slice. Neither annotation proves unseen upstream
pretraining data or template-family transfer.

Recommended cross-chapter wording is: “Eight fixture-development rows and the
additional known RLVR-overlap math9 form nine diagnostic rows; the remaining
eleven are the controlled heldout slice. These annotations do not mean that
assistant full400, chosen100 or DPO100 trained on those math rows.” The
preference-common20 panel reuses the annotation to make future RLVR comparisons
consistent; its assistant/preference policies were trained on different task
data. It must not be described as nine arithmetic rows seen by those policies.

## Base/chat32 failure: output mapping, not a deadline or wrong-math grade

The failed [returned receipt](native-reasoning-base-chat-cap32-20261005-run-01/returned-supervision.json)
records native PID298583, exit1 after 56.547995 seconds, no deadline signal and
no cleanup error. Sample1009 has 20 records and completes normally. Sample1019
has 20 records but returns `completed_with_errors`: math6's 24-action trajectory
ends in ID151768, above the observed tokenizer's maximum ID151668. The cached
model config declares an output vocabulary of 151936, so this is a model output
slot without a tokenizer mapping, not an out-of-model-range tensor index.

The raw error is: `The model selected an ID without a tokenizer mapping; full
IDs are retained`. Its stage is `decode`, stop reason is `error`, truncation is
false, and the full 24 IDs/selected likelihoods remain recorded. Its 24 completed
forwards consumed 828 full-prefix positions and 0.634719 recorded seconds.
This is not rational `UNSUPPORTED`, a natural stop, a token-cap failure or a
600/900-second timeout. The native wrapper then refuses the cell status because
it requires `completed`, not `completed_with_errors`; the three later cells are
never entered. The 40 observed records do not fill or replace the 60 missing.

The underlying generator already records an ordinary response error and its
cost, rather than silently filtering the vocabulary or retrying. A separately
declared coverage consumer could in principle distinguish terminal error records
from invocation-wide failures and continue later cells without changing sampling
settings. **No such source/gate change or new run is approved by this report.**
The parent's chosen continuation preserves the original stop-on-error contract,
retains failed run01 unchanged, and runs other independent predeclared conditions.
There is no sampler repair, filtered-vocabulary intervention, retry, rescore or
replacement of the failed row in this review.

## Actual cost and literal acceptance scope

Every observed response has matching attempted/completed call and position
counts, and positions equal the exact sum of its prompt-plus-growing-prefix
lengths. All cap stops emit exactly the declared cap. Stop IDs and all likelihood
vector lengths match the recorded trajectories. Persisted partial events are
evolving versions of those same attempts, not extra independent generations.
No latest unfinished attempt remains beyond the 40 failed-condition records;
missing future work is unknown, not zero-spent completed work.

| Condition | Whole native child seconds | Recorded-response seconds | Emitted tokens / completed forwards | Full-prefix positions | Minimum sampled available bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base/raw32 | 136.516336 | 76.994397 | 3,163 | 97,810 | 121,993,846,784 |
| Base/raw128 | 334.025691 | 270.493347 | 12,041 | 930,423 | 121,081,409,536 |
| Base/custom-chat32, failed | 56.547995 | 30.849780 | 1,244 | 48,378 | 122,019,704,832 |
| Instruct/off32 | 129.811418 | 66.328340 | 2,692 | 117,798 | 121,394,933,760 |

Each invocation has one external900-second whole-child envelope and a sampled
25GiB reserve, not five independent900-second model allowances. All recorded
resource minima rederive from their saved samples; conflict probes are clear,
helper cleanup completes and queue feeders end. Model loading, byte/interface
verification and administrative work are outside per-response intervals but
inside the native child. These overlapping time scopes are not added together
or mislabeled GPU FLOPs, billing, physical quota or continuous memory proof.

Original [DXI-03](../../docs/COURSE_IMPROVEMENT_PLAN.md#dxi-03-staged-spark-evidence-and-actual-checkpoint-comparison)
criterion3 requires distinct base→SFT→DPO and reasoning/RLVR branches, comparison
on appropriate frozen slices and preservation of every planned row. Criterion5
requires smoke/recovery/interface gates before pilots and actual exits,
identities, failures, costs and negative results. Neither requires every
baseline to succeed or spending all45 stage maxima/full14000 story horizon.
A preserved failed condition is valid negative evidence under those criteria,
but cannot justify complete eight-condition100-response coverage or a paired
comparison involving its missing records. The still-active RLVR comparison must
meet its own accepted Instruct off32/off128, interface, smoke/recovery and pilot
prerequisites and supply actual G4/G8 common-slice results. This report waives
none of those gates and does not mark DXI-03 or the whole goal complete.

## Incremental receipt: Base/custom-chat128 also fails without a repair

After the initial boundary above, the independently launched original
[Base/chat128 run01](native-reasoning-base-chat-cap128-20261005-run-01/acceptance.json)
also failed. This additional review inspected its saved preparation, launch,
closing bindings, returned supervision, two observed cells and 40 raw records.
All ten prepared source hashes still match actual source bytes; preparation and
identity self-hashes, closing source/input/model maps, original prompt encodings,
launch/supervisor command and sampled resource minimum joins match. Independent
model-free grading found **zero mismatches in the additional 40 records**.
Across the five reviewed invocations, that is 380 observed records, including
two errors, out of 500 planned; 120 are missing. These are coverage counts,
not one pooled accuracy estimate or complete coverage of all eight conditions.

| Base/chat128 observed evidence | Actual result |
| --- | --- |
| Native PID / exit | 299364 / 1 |
| Completed-response records / missing | 40 / 60 |
| Correct / grade statuses | 0/40 observed; 39 UNSUPPORTED, 1 ERROR |
| Natural / cap / error stops | 8 / 31 / 1 |
| Controlled heldout | 0/22 observed; 3 natural, 19 cap stops |
| Emitted actions / completed forwards | 4,491 / 4,491 |
| Full-prefix positions | 377,459 |
| Recorded-response wall seconds | 105.786544 |
| Whole native child / supervisor seconds | 132.666869 / 132.729321 |
| Minimum sampled available bytes | 120,977,350,656 |

Sample1009 finishes its 20 records with two EOS stops and 18 caps. Sample1019
finishes 20 records with six EOS stops, 13 caps and one decode error, causing
`completed_with_errors`; sample1029, sample1039 and greedy1009 are never entered.
Every observed cap stop emits exactly 128 actions. Recorded action, selected
likelihood/support vector, attempted/completed forward and growing-prefix
position counts match. No failed record is relabeled a successful termination.

The error is again sample1019/math6 at the **24th selected ID151768**, with
the same prompt IDs and complete 24-ID trajectory as the cap32 failure. Both
saved records have selected raw and behavior log probability
−14.50478178687033 for that action (probability approximately
5.019417364343248e−7), with retained sampling support151936. The cap128 error
records 24 forwards, 828 positions and 0.582172 response seconds. This establishes
the observed mapping failure, not a counterfactual claim that a filtered sampler
would improve reasoning. No vocabulary filter, resample, source/gate change,
automatic retry or replacement was introduced.

The returned native receipt has null signal/stop reason, no deadline or cleanup
errors, complete helper cleanup and ended queue feeders. The source wrapper
still refuses the ordinary response-error cell and retains the remaining 60
records as missing. The separately retained owner-console outer interval,
146.414474 seconds, is a transcribed overlapping preparation/supervision/closure
interval, not another model invocation or an independently bound terminal
archive. The native child's precise exit and duration come from its own receipt.

Thus the current reviewed evidence is **three accepted conditions and two
failed, partial conditions**. The three remaining Instruct conditions and all
active RLVR comparison outcomes are outside this receipt boundary. The original
DXI-03 criterion3/criterion5 interpretation above remains unchanged: preserve
the negative rows and costs, but do not invent a complete 800-response matrix,
zero-fill missing answers, or waive the active RLVR prerequisites.

## Incremental receipt: Instruct/native-thinking-off128 accepted

The next fixed [Instruct/off128 run01](native-reasoning-instruct-thinking-off-cap128-20261005-run-01/acceptance.json)
passes all four execution/coverage checks: actual native exit0, five completed
cells, all100 original response records and no unreadable JSONL. Review of its
saved records independently reproduces every grade and cost predicate, with
**zero mismatches in these additional100 records**. All ten prepared source
hashes match current bytes; preparation/identity self-hashes, launch command,
closing source/input/model bindings, serialized prompt encodings and sampled
minimum join correctly. No weight body was rehashed or model loaded by review.

The selected local checkpoint and native thinking-disabled interface are exactly
the same as off32, including observed semantic interface SHA
`13252c7df16dc0ef0863a5b5e55d9414b1ceeb9da8eed6ac6ec23a4caf95789a`.
Original annotated items and prompt IDs remain identical. All100 corresponding
attempt seeds match; the actual short-cap token sequences equal the prefixes of
their retained cap128 counterparts. This is an observed same-recipe pairing,
not an unmeasured GPU-determinism or kernel-tracing guarantee. Only the declared
response cap changes, not weights or thinking mode.

| Instruct/off128 slice | Records | Declared correct | Natural stops | Caps | Grade statuses |
| --- | ---: | ---: | ---: | ---: | --- |
| Four sampled seeds, all original problems | 80 | 23/80 | 70 | 10 | 56 UNSUPPORTED, 23 CORRECT, 1 INVALID |
| Separate greedy diagnostic, all problems | 20 | 11/20 | 20 | 0 | 9 UNSUPPORTED, 11 CORRECT |
| Four samples, controlled heldout | 44 | 13/44 | 35 | 9 | 31 UNSUPPORTED, 13 CORRECT |
| Separate greedy, controlled heldout | 11 | 7/11 | 11 | 0 | 4 UNSUPPORTED, 7 CORRECT |

The four sample cells' all-problem correct counts are5/7/8/3 of20; their heldout
counts are3/3/5/2 of11. All100 responses are format-valid under the unchanged
`any` predicate. The coverage aggregate has34 correct,90 natural stops and10
caps, but its mixed34/100 and20/55 heldout counters are not the sampled or
greedy headline. This comparison does not pool decoding policies or count
repeated problems as independent population observations.

The longer allowance changes delivered length and whether a supported explicit
answer is available to the fixed grader. For example, sample1009/math11 reaches
a boxed3 with a natural turn stop at108 actions. Sample1009/math19 has a
supported correct boxed1.5 while still capped at128, demonstrating that
correctness and natural termination remain independent. Conversely,
sample1009/math10 retains the same nine-action `5 + 6 = 11` natural response
and whole-output `UNSUPPORTED` grade at both caps. No last-number rescue or
thinking-text removal was introduced. These observations do not establish that
the checkpoint learned between invocations, that every longer answer improves,
or that a visible explanation is faithful.

Native PID299699 exits0 after247.0372236270341 child seconds; the surrounding
supervisor interval is247.08396206301404 seconds. Its minimum sampled available
memory is120,799,416,320 bytes; stop reason is null, cleanup errors are empty and
all helper cleanup/queue shutdown receipts finish. The100 recorded responses
consume7,727 actions/forwards and627,977 full-prefix positions, with
181.13693934329785 response seconds. The sampled cells contribute6,279 actions,
514,679 positions and150.55549336317927 seconds; greedy contributes1,448,
113,298 and30.581445980118588 respectively. Those response intervals lie inside
the whole child; no sum of overlapping child/supervisor times or FLOP claim is
made. Action/log-probability vector lengths, exact growing-prefix counts,
cap/stop IDs, context512 and sampled behavior/raw likelihood agreement match
the saved trajectories. Eager BF16-loaded CUDA is a retained implementation
label, not an observed MATH-backend trace or optimized serving result.

This increment leaves **four accepted and two failed conditions** independently
reviewed:480 observed slots of600 invoked plans, with the same120 missing from
the two unchanged Base/chat failures. The two thinking-enabled conditions and
all active RLVR outcomes remain outside this review boundary. The original
dependency, smoke/recovery/interface and frozen-slice obligations still apply;
this report does not pre-approve those results or complete the goal.

## Final baseline boundary: both thinking-enabled caps accepted, all eight attempted

The original [Instruct/thinking-on32](native-reasoning-instruct-thinking-on-cap32-20261005-run-01/acceptance.json)
and [Instruct/thinking-on128](native-reasoning-instruct-thinking-on-cap128-20261005-run-01/acceptance.json)
invocations each pass all four execution/coverage checks, exit0 and retain100
original records across the unchanged four sampled seeds plus separate greedy
diagnostic. Their missing, unreadable and incomplete response collections are
empty. Independent model-free replay of these additional200 responses finds
zero stored grade/cost predicate mismatches. This completes review of all680
observed records, not all800 planned records.

Each preparation self-hash and all ten prepared current-source byte hashes
match. Closing source/input/model maps, launch and supervisor command joins,
generation identity self-hashes, checkpoint metadata, observed tokenizer/template
interfaces and all serialized prompt encodings match the retained contracts.
Both choose the same Instruct revision and recorded596,049,920-parameter BF16
CUDA model as the thinking-disabled conditions. Their native tokenizer/template
semantic SHA remains `13252c7df16dc0ef0863a5b5e55d9414b1ceeb9da8eed6ac6ec23a4caf95789a`;
the **thinking-enabled serialization setting** is separately bound in each
contract/record. Every corresponding serialized prompt differs from the
thinking-off prompt, although its original problem text does not change.
Thinking-on is not a different checkpoint or an additional learned model.

Between the two thinking-on caps, all100 attempt seeds and short token prefixes
match the retained long-cap counterparts. Every response reaches its cap, has
zero natural stops or adapter errors and is `UNSUPPORTED` under the unchanged
whole-output grading contract. All are format-valid under `any`; that does not
turn them into supported correct answers. Reasoning text is neither stripped
nor rescued by last-number extraction. This is a bounded delivered-output
result, not proof of absent mathematical ability, rationale unfaithfulness or
thinking-on being universally worse. No counterfactual longer horizon is run.

| Thinking-on actual scope | Cap32 | Cap128 |
| --- | ---: | ---: |
| Sampled: correct / original records | 0/80 | 0/80 |
| Greedy: correct / original records | 0/20 | 0/20 |
| Heldout sampled: correct / records; caps | 0/44; 44 | 0/44; 44 |
| Heldout greedy: correct / records; caps | 0/11; 11 | 0/11; 11 |
| Native PID / exit | 300181 / 0 | 300438 / 0 |
| Whole child seconds | 144.33200714498525 | 364.1156065230025 |
| Supervisor seconds | 144.3781207689899 | 364.1712747570127 |
| Minimum sampled available bytes | 121,987,284,992 | 120,767,070,208 |
| Recorded-response seconds | 79.3077671159408 | 297.00581288244575 |
| Generated actions / completed forwards | 3,200 / 3,200 | 12,800 / 12,800 |
| Full-prefix positions | 124,960 | 1,114,240 |

Both returned stop reasons are null and cleanup errors empty; all helper/queue
shutdowns complete. The saved observation samples reproduce each resource
minimum. Exact attempted/completed forward counts, support/selected likelihood
vector lengths, cap IDs, emitted actions and growing-prefix positions match
all trajectories. Sampled behavior/raw log probabilities agree at temperature1
with full support; greedy behavior log probabilities are0. The cap128 sample
cells consume10,240 actions/891,392 positions, versus2,560/222,848 for greedy;
cap32 consumes2,560/99,968 versus640/24,992 respectively. Whole-child,
supervisor and recorded-response intervals overlap rather than add; they are
not model billing or FLOP estimates. Same recorded eager BF16 configuration
does not establish a kernel trace or physical containment guarantee.

### Closed baseline inventory, not an invented complete response matrix

| Original condition | Execution verdict | Recorded / missing | Heldout sampled correct | Heldout greedy correct |
| --- | --- | ---: | ---: | ---: |
| Base/raw32 | Accepted negative result | 100 / 0 | 0/44 | 0/11 |
| Base/raw128 | Accepted bounded result | 100 / 0 | 2/44 | 1/11 |
| Base/custom-chat32 | Failed, partial | 40 / 60 | 0/22 observed; two cells only | Not generated |
| Base/custom-chat128 | Failed, partial | 40 / 60 | 0/22 observed; two cells only | Not generated |
| Instruct/thinking-off32 | Accepted negative result | 100 / 0 | 0/44 | 0/11 |
| Instruct/thinking-off128 | Accepted bounded result | 100 / 0 | 13/44 | 7/11 |
| Same Instruct/thinking-on32 | Accepted negative result | 100 / 0 | 0/44 | 0/11 |
| Same Instruct/thinking-on128 | Accepted negative result | 100 / 0 | 0/44 | 0/11 |

All eight conditions were attempted: **six complete100-record invocations plus
two partial40-record invocations give680 observed/800 planned, with120 missing**.
The two decode-error rows, their child failures and all spending remain in that
680; they are not silently excluded or repaired. No pooled accuracy score,
missing-sixty paired comparison or full four-interface800-response completion
is claimed. The fixed failures do not create a new compulsory successful
Base/chat rerun gate, and no source, settings, support filter or response gate
was changed to obtain a favorable verdict.

The overlap roles remain eight original fixture-train items plus math9's known
RLVR problem overlap; only math2/math9 have true overlap flags. These diagnostics
remain outside heldout headlines and do not imply that assistant full400,
chosen100 or DPO100 trained on the math rows. These baseline receipts also do
not substitute for the separately gated active RLVR smoke/recovery, G4/G8
pilots, unchanged common-slice checkpoint comparison, final course integration
or current-source verification. No DXI-03 or whole-goal completion is signed by
this baseline-only review.
