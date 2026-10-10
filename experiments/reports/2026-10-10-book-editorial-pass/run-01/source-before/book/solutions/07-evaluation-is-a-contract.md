# Worked solutions — Evaluation Is a Contract

Read [Chapter 7](../chapters/07-evaluation-is-a-contract.md) first. The five Day 10 notebooks supply runnable reference calculations next to each prediction. These answers emphasize the claim supported by a measurement.

## 1. Operational story coherence

A defensible claim is: “On the frozen set of 60 original story openings, greedy continuations of at most 192 tokens have fewer character-identity contradictions than the baseline.” Preserve the prompt identifiers and split hash. Two blinded readers assign the prewritten 0–2 identity rubric; retain disagreement and use adjudication only according to a declared rule. Separately score causal continuity, repetition and endings. An improvement on identity does not imply every dimension improved.

Choose the item or source group as the sampling unit. If six prompts are variants of the same underlying plot, a bootstrap should resample plots rather than treating all variants as independent. The test compares behavior under this decoding budget; it does not establish unlimited-length storytelling.

## 2. Normalization policy

For an integer-answer task, a numeric parser may reasonably consider “1.0” equal to “1”. Define whether decimals, scientific notation, units and multiple answers are accepted. If the instruction demands exactly one integer token, the strings may instead differ on format while sharing numeric meaning.

Report separate answer and format scores rather than deleting formatting requirements after seeing failures. The reference normalizer intentionally preserves punctuation; its three-string example yields true, false, false. That verifies its chosen policy, not universal language equivalence.

## 3. Derive pass@3

There are $\binom{10}{3}=120$ equally sized three-candidate subsets. With two correct candidates, eight are incorrect, and $\binom{8}{3}=56$ subsets contain only incorrect answers. Therefore

$$
\widehat{\mathrm{pass@}3}=1-\frac{56}{120}=0.533333\ldots.
$$

The fixture tests the product implementation against combinatorial counts for many small $n,c,k$ combinations, including zero successes and all-success cases. The event is at least one verified success; it does not identify which candidate a user should receive.

## 4. Duplicate draws

Generating once and storing the same string ten times does not provide ten independent opportunities. Conditional on that one draw, every copy has the same outcome. The finite-pool combinatorics still describe subsets of the recorded pool, but interpreting the result as fresh independent sampling is unsupported.

Record sampling settings and candidate identities. Deduplication alone is not enough to establish independence: genuinely independent draws can coincidentally produce the same output. Independence concerns the generation process, not merely distinct strings.

## 5. Two kinds of confidence

Token confidence is a model distribution over vocabulary candidates. It is conditional on the current context and learned training distribution. It does not guarantee answer truth.

A statistical score interval describes uncertainty in an estimate over a declared input population and sampling model. Ten successes out of ten leave a nontrivial Wilson interval. Neither quantity is equivalent to a model saying “I am certain.” Verbal confidence requires its own calibration evaluation.

## 6. Paired comparison

Use the same item index for A and B, compute $d_j=b_j-a_j$, and bootstrap indices once for both scores. Shared task difficulty remains paired. Independent resampling breaks that correlation and can obscure a precise comparison.

The notebook also duplicates all items ten times. The naive binomial interval narrows, although the source information is unchanged. That deliberate failure illustrates why the unit of independence belongs in the contract. A percentile interval that crosses zero means the sign is unresolved at the chosen precision, not that the models are equivalent.

## 7. Slice report

Publish the total score and a table with task family, count, score, interval and paired change. Freeze the slice definitions before evaluating candidates. Include a rare-format slice even when its denominator is small; show the broad interval rather than hiding it.

The purpose-built forty-item fixture gives model B a better aggregate but a worse rare-format score. This verifies the accounting and shows how an aggregate conceals a regression. It supplies no measured model capability.

## 8. Exact contamination audit

Normalized prompt-plus-reference hashes detect the deliberately inserted cross-split duplicate. Grouping those two records repairs that particular exact leak.

The procedure does not detect every paraphrase, translated question, shared document or leaked solution in a reasoning trace. Preserve provenance, run near-duplicate analysis, and document what was inspected. “No exact collisions found” is a narrower and more accurate statement than “the test is uncontaminated.”

## 9. Repeatedly used tests

Once examples inform learning-rate selection, prompt engineering, parser edits or checkpoint choice, they contribute to development. Rename the split and freeze a new final suite, or explicitly describe the evaluation as exploratory. Calling a repeatedly consulted set “publication test” does not restore independence.

If no new data is available, report that limitation and avoid a final-test claim. Cross-validation can support some comparisons, but requires nesting the selection process appropriately.

## 10. Stopping rule

Before inspecting candidate outputs, declare the item count, review rubric, generation budget, seed policy and minimum useful effect. Stop after all fixed items are scored, or on an implementation failure. If uncertainty is too broad, a separately specified expansion can add independent items.

Do not stop when the first excellent example appears. Do not extend only the losing recipe's budget after reviewing its results. Changes remain legitimate experiments when their selection and new evidence are recorded.

## 11. Three mathematical equivalence decisions

Convert decimal text directly to an exact rational. `0.5` is `1/2`, but a finite
rounded third is not exactly `1/3`; numerical tolerance would require a named
metric. Interval equality preserves endpoint inclusion, so `[1,2)` differs
from `[1,2]`. Unit equality uses declared aliases without conversions in this
instrument: `seconds` matches `s`, while `100 cm` does not match `1 m`.

These decisions define grader coverage, not universal mathematics. Variables,
roots and unknown units are unsupported. Do not map every unsupported answer
to an ordinary mathematical mistake or silently expand the parser after seeing
one checkpoint's outputs. The adversarial fixtures expose each boundary.

## 12. Answer format and stopping are independent

For “return only one integer,” `Final answer: 4` can have correct arithmetic,
invalid format and a max-token stop. Record answer correctness1, format0 and
natural termination0, with truncation1. Under our declared metric,
`task_success` is correctness AND format, hence0. It does not secretly require
natural termination; a different deployment requirement needs a different metric.

Retain raw text, token IDs/counts if measured, and the actual stop reason. A
replay fixture's assigned stop is simulated evidence, not proof of an EOS event.
Unknown generation token counts and latency stay null, not character counts or
zero-cost claims.

## 13. Resample source groups but preserve the estimand

Align item/sample IDs first. Sample source groups with replacement and retain
all differences inside each drawn group. Sum their differences and divide by
the total number of retained item/sample differences. Unequal group sizes remain
unequal; this estimates a micro-average difference, not an equal-group mean.

The fraction siblings in the original fixture travel together. A comparison
with a deleted response or changed group identity fails instead of changing the
population silently. With few deliberately authored groups, even a positive
percentile interval cannot justify a broad model capability claim.

## 14. Prevent a hidden answer oracle

An answer selector receiving held-out references or correctness flags can
choose the correct candidate because it already knows the answer. That measures
oracle availability, not the deployable ranker. Preserve references in evaluator
inputs while projecting candidate records to a view containing only generation
metadata and the raw response.

The `candidate_view` test injects gold fields into a graded record, then checks
they are absent from the selection view. This protects the interface, not the
ranker's statistical quality. Inspect actual selection code and its provenance
before trusting a real inference-time comparison.

## 15. Preserve the stop action without grading its spelling

The model chose two actions: the token for `4` and EOS. Keep both IDs, their
likelihoods and generation cost2. Raw decoding preserves the EOS spelling;
the frozen grading-text policy decodes only the prefix before that final
declared stop. The answer can then satisfy a one-integer format without erasing
termination evidence. No other token or explanation is trimmed. A max-token or
context-truncated response has no final natural stop to remove.

This is a declared interface rule, not a post-hoc repair selected because one
checkpoint otherwise scores poorly. Change it only through a named new contract.

## 16. Partial execution is evidence but not recovery equivalence

One generated action and the successful first forward are known. The failing
second forward was attempted, so attempted token positions and calls exceed
completed work; its elapsed forward time still belongs to the attempt. Preserve
the partial text/IDs, error stage and recorded wall time. Append-only progress
and final error records survive an ordinary interruption, while the failure
summary lists planned item/sample coordinates that were never attempted.

A new invocation may regenerate the same coordinate under the same seed, but
that is not proof of exact mid-attempt resume, identical device kernels or an
uninterrupted timing comparison. This adapter does not implement resume. The
tiny randomly initialized CPU model verifies actual HF loading/forward/record
replay; scripted stop and failure controls are separately labeled.

## 17. Availability and a deployable decision

A correct candidate already in $C_N$ remains in every longer nested prefix, so
the any-correct event cannot become false. Majority counts can change: three
correct0 votes lose to five wrong1 votes in the actual seed10052 `train-11`
pool. A selector returning one candidate satisfies $d_N\le o_N$, but may not
find the available correct answer. An oracle seeing correctness labels is not
the same selector. The worked notebook keeps oracle results in evaluator inputs,
not the strict selection projection.

## 18. Sampling identity and answer support

Independent coordinates can produce identical strings. Their repeated frequency
is part of majority voting under the declared sampling process. Copies of one
stored sample ID are not new draws and are rejected by the candidate-pool
contract. Counting one vote per distinct answer instead removes frequency
information: with binary support both0 and1 get one vote, so the stable rule
returns the earliest eligible answer. This is a different estimator, not a
universal correction for correlated mistakes.

Inspect string diversity separately from wrong-answer and correctness agreement.
Across-item sample-ordinal error correlations have undefined values for zero
variance, stored as null. Related source siblings and tiny panel size prevent
those diagnostics from establishing general IID properties.

## 19. Charge a whole-attempt overshoot

The candidate is generated and its three actions cost work. It is not eligible
for selection under the remaining two-action cap. Retain its ID and rejection,
charge all three actions, and stop without looking for a cheaper later answer.
The accepted prefix stays unchanged; the charged prefix contains the boundary
attempt and has a one-action overshoot. This retrospective whole-attempt control
is not an exactly capped interruptible generator. Prompt/scoring positions and
time remain separate from the generation-token cap.

## 20. Likelihood and usefulness of compute

Mean conditional log probability, including EOS, measures what the fitted
policy favors under its declared support. A frequent wrong answer can have a
large value. It does not compare the candidate with an independent reference,
prove a rationale faithful or establish calibrated answer confidence.

Use the same candidate pools for competing selectors, preserve every attempt,
compare declared token caps and actual work, and report correctness plus oracle
availability on independent frozen slices. Additional attempts can enlarge
availability but reinforce wrong votes. A positive fixed-panel result does not
establish optimized serving latency, equal total compute or broad reasoning
improvement. The actual tiny lesson deliberately retains negative seeds and
source/template/family failures.

## 21. A reproducible summary does not invent a model event

The authored panel supports its actual item/source-group coverage, frozen
contract, accepted/rejected parses, per-task metrics, retained errors and paired
group resampling. Its input hashes identify the exact fixture bytes. They do
not establish a model forward, token count, device measurement, checkpoint
weight digest, training genealogy, approval or independent behavior review.
Missing measurements remain unknown, not zero.

Run the export command in [Lab 7](../labs/07-evaluation-is-a-contract.md) into a
new directory. Compare the readable card with its full replay, including the
set/JSON regression and simulated error. Repeated export from identical inputs
should agree; existing outputs must be refused rather than overwritten.
Calling the authored panel Qwen changes a name, not its provenance. Even an
actual tiny random-model record supports only that recorded model/interface,
task population and resource boundary. An attractive card is not a substitute
for a pretrained evaluation or a broader safety assessment.

## 22. Preserve proxy success and whole-response failure

The automatic SFT score1/15 is correct under the frozen phrase rubric: its
`benign-a` response contains the required strings and no forbidden string.
The whole text also contains repeated `.nasa` fragments, an unfinished final
sentence and a measured64-token cap stop. Those facts are not erased by the
pass. A separate review should inspect every retained response, distinguish
the useful fragment from a completed helpful answer and state its own review
criteria and independence limits. It is not human agreement or broad safety.

Keep the original automatic0/15 versus1/15 comparison, its grouped uncertainty
and both panels'0/15 natural-stop rates. Do not change the rubric after seeing
this checkpoint's failure or treat its score as semantic helpfulness. A stricter
next contract may be useful, but needs its own identity and selection history.
The [actual report](../../experiments/reports/2026-10-05-pretrained-evaluation-replay.md)
links complete raw records, the unchanged contract, actual local identities,
external resources and the immutable card. Instrument execution is completed;
assistant quality and the larger publication campaign are not established.

## 23. Blind records without inventing reviews

Give the reviewers complete anonymous opening/continuation pairs and the
predeclared rubric, not the checkpoint/arm codebook. Preserve all emitted
tokens, selected likelihoods, stop reasons, costs and original source groups
privately so the join remains auditable. Disclose whether raters are human or
AI, independent and previously exposed; output shuffling alone cannot establish
those properties.

The [rating lab](../labs/07-evaluation-is-a-contract.md) prepares authored
controls with two empty rating templates. Evaluating without rating files must
report awaiting ratings and null paired scores/intervals, not0, an assumed1 or
a claimed trained-model tie. If ratings later disagree, preserve each original
decision and use only the declared adjudication policy. Reject a changed text
digest, duplicate ID or score outside0/1/2 instead of quietly repairing it.

Pair checkpoints only at the same declared update/recipe and resample the
twelve source-opening groups, carrying both arms together. Sampling seeds and
raters are repeated measurements within an opening, not independent new
openings. Report any partial matched cohort explicitly. Grammar, consistency,
causality, repetition and ending retain their separate meanings; an overall
average cannot hide which one regressed. A capped response can be judged on
its actual incomplete ending without pretending it naturally terminated.
The authored fixtures verify the consumer, not real model quality or human
annotation agreement.

The later [actual rating replay](../../experiments/reports/native-story-publication-20261005-01/ratings-evaluation-01/report.json)
contains192 individually assessed continuations from each of two anonymous
AI instances, no abstentions and56 candidates with an axis disagreement.
Apply the already frozen two-rater mean; do not renegotiate an unfavorable
score after revealing the arms. At update400, the control's45/48 natural EOS
stops coexist with narrative-ending mean0.0208/2. EOS proves a stop, not a
resolved story. Its grammar mean0.0729/2 also keeps lower NLL separate from
readable generation. The half-rate arm's higher repetition mean does not
override its lower consistency/causality/grammar scores. Initial random text's
repetition1.5/2 is another warning against an unspecified overall average.

Resample twelve openings, not96 independent rater/recipe observations. Retain
all288 missing later cells and their null scores; no completed400 comparison
can stand in for absent4000/8000/14000 results. This is actual AI review of
actual model text, not human review or a new training-seed sweep.

## Notebook pathway

1. [Metrics and contracts](../../notebooks/day-10/01_metrics_and_contracts.ipynb): exact match and attempt budgets.
2. [Paired uncertainty](../../notebooks/day-10/02_paired_uncertainty.ipynb): intervals, aligned resampling and false independence.
3. [Slices and contamination](../../notebooks/day-10/03_slices_and_contamination.ipynb): aggregate regression and leakage repair.
4. [Mathematical grading and response replay](../../notebooks/day-10/04_mathematical_grading_and_response_replay.ipynb): bounded equivalence, complete ledgers and source-group pairing.
5. [Actual candidates and answer selection](../../notebooks/day-10/05_actual_candidates_and_answer_selection.ipynb): real causal samples, gold-blind choices, attempted costs and oracle gaps.

Each lesson contains prediction, reference answer, runnable computation and a labeled data-derived plot.
