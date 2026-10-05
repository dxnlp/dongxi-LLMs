# Teacher attempts and matched rejection SFT

The bounded pipeline preserves provenance and matched controls, but better
teacher selection did not guarantee student transfer. All nine trained students
failed the polite-prefix control under greedy generation. Those failures, wrong
demonstrations, retries and rejected raw responses remain in the evidence.

## Frozen inputs and actual teacher

The [specification](../specs/2026-10-04-teacher-data.md) and
[protocol](../../fixtures/teacher-data/protocol.json) precede collection and
student fits. The original fixture has twelve train and four each development,
test and control prompts. Actual prompt IDs and underlying operation/color-tuple
source groups are split-disjoint. Colors and task forms are shared, so held-out
results concern new combinations and the declared polite-prefix control, not
unseen vocabulary or general assistance.

The teacher is an executed programmatic copy/reverse function with eight
declared response/fault slots. It is not an API, pretrained model, or human
source. One local error per train prompt receives one recorded retry. Its trace
and final answer are separate fields; the trace is retained but not supervised
and makes no claim about faithful language-model reasoning.

The reference collected nine records, paused, resumed to 108 unique attempt
records, then resumed again without adding another committed record. There are
twelve error attempts and twelve retry attempts. Every attempt retains its raw
text, symbolic IDs when representable, stop/error, coordinates, actual executor,
frozen source/interface contract and content terms. The
[raw journal](../../fixtures/teacher-data/reference-journal/attempts.jsonl) and
[execution events](../../fixtures/teacher-data/reference-journal/events.jsonl)
remain available, including rejected records.

The single-writer journal synchronizes records and rejects changed contracts,
duplicate committed IDs and complete corrupt lines. Tests exercise interruption
and explicit partial-tail recovery: corrupt bytes and the recovery reason are
saved before removing only an uncommitted suffix. A started but uncommitted
execution may physically repeat, with the repeated start and unknown lost cost
visible. This is not exactly-once external requests or production crash recovery.

## Acceptance and selection are different

The filter rejects format, length, unsupported IDs, missing END, source leakage
and within-prompt duplicates, **not correctness**. Seventy-two attempts are
rejected; 36 unique candidates survive: twelve correct and 24 wrong.

| Rejection reason | Attempts with reason |
|---|---:|
| empty answer | 24 |
| overlength answer | 12 |
| unsupported answer | 12 |
| missing END | 24 |
| teacher error | 12 |
| duplicate candidate | 12 |

An attempt can have multiple reasons, so these counts do not sum to 72.
Retries that duplicate an earlier correct answer add cost, not independent
candidate information. Wrong well-formed responses remain selectable.

The fixed training-task verifier computes copy/reverse correctness from the
training prompt and candidate text, never teacher mode labels or reference
fields. This is a known executable task verifier, not a fitted reward model.
All selectors consume the identical frozen pool. Top chooses by task score,
shorter target length and stable identity; random uses a fixed item-keyed seed
912. Length-random samples at the top answer's target length without filtering
on correctness. Each original length stratum has exactly two eligible candidates, one
correct and one wrong; no missing stratum or fallback occurred.

One wording correction matters: the premeasurement specification overstates
the reference-reading boundary by saying references are read only by evaluation.
The validator also reads them to check authored-fixture consistency before
collection. Selection and student training never use those fields. The original
specification and source snapshot are retained rather than silently rewritten;
this report makes the additional consistency check explicit.

## Matched examples and unmatched exposure

| Dataset | Samples | Prompts / source groups | Verifier-correct examples | Assistant + END targets per update | Targets over 80 updates |
|---|---:|---:|---:|---:|---:|
| top | 12 | 12 / 12 | 12 | 42 | 3,360 |
| random | 12 | 12 / 12 | 4 | 46 | 3,680 |
| length-random | 12 | 12 / 12 | 6 | 42 | 3,360 |

Every arm covers six copy/six reverse and six two-word/six three-word prompts.
The primary top/random comparison controls examples and prompt coverage, not
supervised-token exposure. Its sensitivity controls the latter in this pool.
Neither claims equal gradient distributions or a broad sampling-method ranking.

Global top3/top6/top12 are separate audits, not extra student fits. Their prompt
coverage is 0.25/0.5/1.0. Global top6 contains only two-word tasks because
correctness ties prefer shorter answers. One-per-prompt selection retains both
difficulty slices. Different global sample budgets are not a matched experiment.
All selected IDs, excluded prompts and lengths are saved in the
[selection artifact](../../fixtures/teacher-data/reference-journal/selection.json).

## Actual sequence students and all final outcomes

Each student is a randomly initialized 6,120-parameter causal decoder: one layer,
width24, three heads, feed-forward width48, learned positions, tied output head,
float64 CPU. The existing instruction encoder/collator and SFT loss perform
exactly one shift and supervise final-answer plus real END, not prompt, role
headers or padding. Trace and audit fields never enter the model input.

Seeds 1101/1102/1103 pair initialization across all three arms. Each fit uses
eighty full-batch AdamW updates at the frozen lr0.015, weight decay0 and clip1.
All nine fits completed: 720 updates, 31,200 valid target presentations and
141,120 padded training-forward positions. Training loss is measured on each
arm's own selected dataset, not a common held-out corpus.

Generation is unconstrained across the full 22-ID vocabulary: one greedy and
four item/sample-seeded temperature1 responses per prompt, max6 new tokens.
END is sampled or greedily chosen, not forced by a response grammar. Raw IDs,
format, errors, natural END and truncation remain in every denominator.

| Data / seed | Final demonstration NLL | Train greedy correct | Test greedy correct | Test sampled correct | Control greedy correct |
|---|---:|---:|---:|---:|---:|
| top / 1101 | 0.412686 | 8/12 | 1/4 | 0/16 | 0/4 |
| top / 1102 | 0.419879 | 8/12 | 0/4 | 0/16 | 0/4 |
| top / 1103 | 0.548256 | 3/12 | 1/4 | 1/16 | 0/4 |
| random / 1101 | 0.001450 | 4/12 | 0/4 | 0/16 | 0/4 |
| random / 1102 | 0.005791 | 4/12 | 0/4 | 0/16 | 0/4 |
| random / 1103 | 0.037047 | 4/12 | 0/4 | 0/16 | 0/4 |
| length-random / 1101 | 0.148475 | 6/12 | 1/4 | 2/16 | 0/4 |
| length-random / 1102 | 0.006477 | 6/12 | 2/4 | 7/16 | 0/4 |
| length-random / 1103 | 0.270421 | 3/12 | 0/4 | 0/16 | 0/4 |

Untrained baselines have zero greedy correctness on every slice. Trained greedy
END fractions are one, even on zero-correctness controls: stopping is not
correctness. Sampled formatting/stopping failures also remain in the raw panel.
The test/control splits contain only four prompts each, and only one selection
seed is tested. These outcomes establish neither universal top-selection
superiority nor a unique cause for imperfect fitting or combination transfer.
No recipe, seed or checkpoint was retuned after inspecting these results.

There are 1,080 actual post-training generation records and 360 unique baseline
records, totaling 1,440 executed responses. Baselines are shared by paired arm
entries; repeated storage must not be counted as fresh generation. These calls
generated 5,721 actual symbolic tokens and used 76,507 full-prefix forward
positions. Teacher work is separate: 966 serialized trace/final words, summed
attempt elapsed7.430ms, zero model forwards and zero API calls. These word units
exclude field markers and are not inference tokens or billed units.

## Verification and retained failures

The [full measurement](2026-10-04-teacher-data.json) preserves actual environment,
commands, fixed contract, all update rows, paired baselines and all generated
outputs. It ran in3.160s on Python3.12.14/PyTorch2.14.1+cpu, Linux ARM64,
float64 CPU, one thread. Actor states are identified in memory; no reloadable
student-checkpoint export or exact training-resume claim is made.

The [first compact acceptance](2026-10-04-teacher-data-acceptance.json) is
retained as failed. All nine student states, update trajectories, pool/selection
and generated outputs matched, but actual executor strings differed: original
`-m` recorded `__main__:programmatic_teacher`, while imported replay recorded
`dongxi_llms.teacher_data_lab:programmatic_teacher`. These genuine invocation
strings remain untouched in their raw journals.

The [final acceptance](2026-10-04-teacher-data-acceptance-final.json) gates a
comparison of that exact known Python alias on identical actual teacher source
bytes and frozen contract. It does not normalize arbitrary module names. Actual
elapsed durations and their content-derived digests also differ, as expected.
All attempt coordinates, raw text, IDs, errors, stops, cost counts, selected
responses, model states, 720 update rows and all generated outputs replayed
exactly under that disclosed comparison. All 1,080 post-training records were
independently regraded; 23 focused tests pass. Twenty-one source/lesson/figure
hashes stayed unchanged during the final collection. Math passed54book files,
1,201expressions and zero issues at that snapshot.

The [new notebook](../../notebooks/day-11/04_teacher_attempts_and_matched_rejection_sft.ipynb)
passed a fresh isolated CPU kernel: eight code cells and six inspected figures.
One initial preflight failed before launching a kernel because the shared Day25
registry entry temporarily disagreed with its chapter; the owner repaired that
entry. Figure inspection also found clipped schematic borders, fixed solely by
expanding plot limits. Neither repair changed teacher data, numerical source or
student recipe. The final fresh-kernel/source identities are retained in the
acceptance JSON; saved plots remain previews, not live controls or animations.

## Evidence boundary

This package executes provenance, rejection, selection and tiny sequence SFT.
It does not execute a language-model teacher, prove faithful rationales, establish
general assistant transfer, validate human judgments, compare broad ranker
families, demonstrate Mac/GPU execution, or implement production API recovery.
Existing held-out failures are the result, not a reason to silently make the
fixture easier. Follow the [lab](../../book/labs/08-instruction-data-as-an-interface.md)
with unused journal/report paths for a new measurement.
