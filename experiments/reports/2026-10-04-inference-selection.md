# Actual candidates separate availability from selection

The three-seed CPU experiment generates864 real autoregressive candidates and
compares selectors on identical ordered pools. Increasing a nested candidate
prefix increases or preserves any-correct availability, but does not guarantee
that majority voting or a likelihood ranker selects the correct answer. Fixed
train-only fitting reduces training loss while preserving substantial failures
on unseen sources, instructions and task families. Every failed candidate and
whole-attempt budget rejection remains in the accounting.

The [premeasurement specification](../specs/2026-10-04-inference-selection.md)
was saved before any fit or candidate collection. The
[raw result](2026-10-04-inference-selection/results.json),
[append-only candidate ledger](2026-10-04-inference-selection/responses.jsonl),
[fit and completion events](2026-10-04-inference-selection/events.jsonl) and
[input identity](2026-10-04-inference-selection/input-identity.json) retain the
actual evidence. The [verification](2026-10-04-inference-selection-verification.json)
binds the source, tests, notebook and inspected previews. Existing categorical
inference simulation and prior reasoning reports are unchanged.

## What the tiny model receives

The original shared TinyDecoder has vocabulary10, width16, four query heads,
two KV heads, one modern block, hidden32 and context8. It runs float64 on CPU
without dropout. Each input is four symbolic IDs: BOS, an instruction ID and
two operand IDs. The fixture's English strings document the task; the decoder
does not parse them. Conditional sampling admits EOS, numeral0 and numeral1
at every action, cap3, temperature1. Immediate EOS, repeated numerals and cap
truncation are genuine outcomes. No answer or stopping action is inserted.

The18 original items contain six balanced training parity sources, four
source-heldout parity items, their four grouped alias-instruction siblings,
and four unseen sum-positive-family items. Aliases use an untrained instruction
ID and share source groups with direct test items. They cannot be counted as
independent sources. Sum-positive references have three1s and one0: a constructed
constant1-plus-oracle-EOS baseline would score3/4, while constant0 would score1/4.
Balanced train and source-heldout references both give either constant answer1/2.
These reference controls are not actual generated outputs and provide no free
model termination evidence.

For each seed10051,10052,10053, compare the initial weights with update80 from
full-batch, train-only conditional cross-entropy. AdamW uses learning rate0.02,
weight decay0 and gradient-norm clipping1. Both answer and EOS targets participate
in the gradient. The architecture's embedding, attention, MLP and output head
receive actual first-update gradients. No held-out checkpoint or seed is selected.
All240 update rows, targets, train IDs and initial/final state hashes remain saved.

| Seed | Initial train loss | Final train loss |
|---:|---:|---:|
|10051|1.088906|0.000694844|
|10052|1.122570|0.116522|
|10053|1.131154|0.000111377|

These losses do not summarize held-out capability. In particular, seed10052's
incomplete training fit is retained rather than replaced by another run.

## Selectors receive no gold labels

Each candidate seed derives from the model seed, exact frozen policy identity,
stable item ID and ordinal. Changing execution order does not change its
sampling coordinate. PrefixesN=1,2,4,8 reuse the same actual eight-candidate
pool; selectors never receive freshly generated favorable pools.

The first-attempt selector returns the first record even when unusable. Majority
counts independently sampled eligible0/1 answers; unique-support voting counts
one vote per distinct answer. Ties choose the earliest eligible ordinal.
Eligibility requires one binary numeral, natural EOS and no execution error,
not agreement with a reference. Invalid candidates may be unusable votes but
still cost work. Duplicate strings from different sample coordinates remain
repeated votes; repeated record IDs are rejected.

The likelihood ranker chooses the greatest mean conditional action log
probability, EOS included, among eligible records. The fitted policy supplies
learned probabilities, but this is not a learned correctness judge or calibrated
reward model. The constructed longest-output preference is intentionally
degenerate: every eligible output has one numeral and EOS. That selector reduces
to the earliest eligible candidate, not a general test of verbosity preference.

The strict selection projection removes references, problems, correctness,
task-success labels, graded status and rubric fields, even when given a graded
record. Gold-field injection does not change decisions. The independent evaluator
computes answer correctness, format and natural termination afterward. Complete
success requires all three. Any-correct complete-path availability is saved only
as an undeployable evaluator diagnostic; it is never passed to a selector.

## More availability does not imply better decisions

This table covers all18 items, including training rows, so it is a measurement
of the fixed panel rather than a held-out benchmark. Every seed remains visible.

| Seed | Policy | FirstN1 | MajorityN8 | LikelihoodN8 | OracleN8 |
|---:|---|---:|---:|---:|---:|
|10051|Initial|0.166667|0.722222|0.611111|0.777778|
|10051|Fixed SFT|0.500000|0.555556|0.555556|0.555556|
|10052|Initial|0.111111|0.500000|0.500000|0.555556|
|10052|Fixed SFT|0.333333|0.333333|0.333333|0.388889|
|10053|Initial|0.055556|0.388889|0.388889|0.500000|
|10053|Fixed SFT|0.500000|0.555556|0.555556|0.555556|

On seed10052's actual `train-11` pool, the reference is0 and generated answers
are1,0,0,1,1,1,1,0. The correct answer is available three times. Majority selects
the five-vote wrong1, so availability1 and selected success0 coexist. The
likelihood selector also cannot be assumed to rescue an available answer.
This example follows a fixed identifiable source, not manual output authoring.

The held-out boundary is less favorable than the aggregate:

| Seed | Final majorityN8 source-heldout | Alias-template | Unseen family |
|---:|---:|---:|---:|
|10051|0.25|0.25|0.50|
|10052|0.25|0.00|0.00|
|10053|0.75|0.25|0.00|

Each slice contains four items, and direct/alias siblings share sources. These
are descriptive fixed-set observations, not population accuracy estimates or
proof that additional compute generally helps. Initial random models can have
greater candidate availability than the concentrated fitted models. A correct
chance sample is not evidence of acquired reasoning.

## Repetition and errors remain visible

There are685 actual EOS stops and179 cap stops, with no ordinary candidate
exceptions in this campaign. Grading yields245 correct,111 supported incorrect
and508 invalid records. Immediate EOS is often naturally terminated but empty;
natural stopping is not task success. Every candidate is retained, including
those508 invalid records.

Mean within-pool raw duplicate fractions rise from0.305556 to0.861111 for
seed10051,0.298611 to0.826389 for10052 and0.375000 to0.847222 for10053. These
counts concern string diversity, not independence of sample coordinates.
The saved diagnostics separately count canonical answer support, pairwise
wrong-answer agreement and correctness agreement. Across-item ordinal error
covariances/correlations use population moments of this fixed panel; zero
variance gives undefined correlationnull. Related sources and the small authored
panel preclude broad statistical claims about error dependence.

## A token cap is not the cost already spent

Token controls3,6,12,24 use the same ordered pool. A complete attempt is selectable
only if it fits the remaining generation-token cap. The first whole attempt
that crosses the boundary is charged, rejected from selection and ends the
replay. There is no peeking ahead for a cheap correct candidate. This avoids
silently granting a free generation, but allows declared overshoot: it is not
an exactly capped token-level generation policy.

For seed10052's final policy, majority success on the full panel is0.388889 at
cap6,0.333333 at cap12 and0.333333 at cap24. The corresponding actual charged
generation actions are111,190 and252 across18 items. More budget does not
monotonically improve this deployable decision. Oracle availability remains
0.388889 at all three caps. Changes in voting, not lack of a correct available
answer, explain the gap on this frozen replay.

Equal-attempt comparisons share exact candidate pools and mandatory path
rescoring. Equal-token-cap comparisons across initial/final models can differ
in accepted attempts, actual consumed actions, overshoot, prompt-forward work
and time. All arms are charged the measurement protocol's rescoring, although
an optimized first-answer service could omit it. This is a matched microscope,
not a serving implementation comparison.

## Measured work and environment

The full campaign emits1736 generated actions and rescores1736 actions, including
EOS. Full-prefix forward input positions are counted separately. A four-ID prompt
followed by three actions requires generation forwards over4,5,6 positions,
not merely three positions. Its separate path rescore forwards over six input
positions and evaluates three chosen actions. These counts are not FLOPs or
provider billing tokens.

| Seed | Policy | Generated and rescored actions | Sum observed attempt wall seconds |
|---:|---|---:|---:|
|10051|Initial|316|0.120124|
|10051|Fixed SFT|308|0.147539|
|10052|Initial|312|0.143390|
|10052|Fixed SFT|252|0.125620|
|10053|Initial|291|0.130004|
|10053|Fixed SFT|257|0.122589|

Per-record prefill time covers the first full-prefix forward, decode time covers
later full-prefix forwards, and rescoring time covers a separate model call.
Attempt wall times also include sampling/record construction. Selector wall
times are measured during retrospective replay, not new adaptive model runs.
The run function records6.970667seconds including fitting, all decisions and
journal writes; final JSON serialization and setup identity capture lie outside
that boundary. The terminal command completes in7.639seconds. Neither value is
an optimized inference latency benchmark.

Actual execution uses the isolated Linux ARM64 Python3.12.14 interpreter and
Torch2.14.1+cpu, with one CPU thread and offline flags. Process lifetime peak RSS
is754740KiB. Starting/ending observed MemAvailable are117.996704/117.910725GiB,
not a continuously monitored minimum or a CUDA allocator measurement. No GPU,
downloaded checkpoint, API, installation, server or animation is involved.

## Reproduction and evidence limits

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/tmp/dongxi-course-reproduction.ZfVaEu/venv/bin/python \
  -m dongxi_llms.inference_selection_lab \
  --items fixtures/inference-selection/items.json \
  --spec experiments/specs/2026-10-04-inference-selection.md \
  --output /tmp/NEW-selection-run
PYTHONPATH=src /tmp/dongxi-course-reproduction.ZfVaEu/venv/bin/python \
  -m unittest discover -s tests -p test_inference_selection_lab.py -v
```

Both measured commands exit0. Existing run directories are refused. Partial
ordinary generation/scoring errors retain IDs, exact stages and attempted work;
scripted controls test these failures without calling them actual campaign
outcomes. Invocation failure retains events and a failure summary instead of
creating a favorable completed result. Unknown replay cost remains null.

The [visual notebook](../../notebooks/day-10/05_actual_candidates_and_answer_selection.ipynb)
puts gold isolation, fresh causal sampling, all-seed comparisons, charged token
budgets, observed CPU times and actual error dependence next to predictions and
worked answers. The canonical implementation is
[inference_selection_lab.py](../../src/dongxi_llms/inference_selection_lab.py).
This closes a bounded selection/cost instrument. It does not establish English
reasoning, pretrained capability, calibrated confidence, a reward judge,
optimized Spark/Mac execution or learner mastery. Future model-scale evaluation
needs its separately authorized contract and hardware profile.
