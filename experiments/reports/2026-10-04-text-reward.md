# Tiny text reward and process supervision: measured CPU reference

The text encoder, supervision boundaries and frozen-scoring contract are
implemented and verified. Reward fitting succeeds on this narrow training
fixture, but format sensitivity and unknown-token collisions prevent a claim
of independent held-out calibration or arithmetic generalization. DXI-07 is
therefore a partial experimental result, not a completed pretrained reward
campaign. All original measurements and the preselected export are retained.

## Protocol and execution

The [specification](../specs/2026-10-04-text-reward.md) was written before
measurement. The independently written model receives token IDs, not known
quality/length/format features. It has one causal decoder block, width 24,
two attention heads, a 48-unit GELU MLP, learned normalized positions and a
scalar head: 8,209 trainable parameters with the 40-entry vocabulary. Pooling
uses the maximum valid array position, including EOS. Float64 arithmetic,
no dropout, at most 96 positions and one CPU thread bound the run.

The vocabulary is fitted on training texts from both supervision branches.
The fixed unlabeled formatting alphabet `Answer : ** In summary , this is a
clear answer .` was already present in the measured source; the specification
now states it explicitly. It supplies surface-form tokens, not new labels.
Unseen pieces map to the declared `<unk>` ID and are retained in every
prediction. Overlength input fails rather than truncating silently.

Twelve original color-comparison source groups train the pairwise model;
four calibration and four test groups are raw-source-disjoint. The test
contains four pairs in each of five conditions, not twenty independent
questions. The separate process fixture contains nineteen traces across five
source groups: eleven training traces with fourteen labeled steps, four
calibration traces and four test traces. It includes one-step/two-step traces
and all four terminal/local correctness combinations. All labels are
**course-authored**, not human feedback, live AI judgments or an imported
benchmark. Source-balanced losses average within each source before averaging
sources.

Seeds 1601, 1602 and 1603 were declared in advance. Preference models receive
120 full-batch AdamW updates; separate terminal and process models receive
80 each. Learning rate is 0.02 and weight decay 0.01. The observed per-seed
fit/evaluation times were 1.226, 0.511 and 0.511 seconds. These short timings
include in-process CPU work, not notebook-kernel startup or any model-scale
training estimate.

Execution used `/home/dongxi/dgx-spark-dongxi/.venv/bin/python`, Python
3.12.14, Torch 2.13.0+cu130 on Linux/aarch64. Tensors and fitting stayed on
CPU; the installed CUDA build is not evidence of a GPU run. No pretrained
backbone, external data/API, package installation or server was used.

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 \
  /home/dongxi/dgx-spark-dongxi/.venv/bin/python \
  -m dongxi_llms.text_reward_lab \
  --fixture fixtures/text-reward/records.json \
  --output experiments/reports/2026-10-04-text-reward.json \
  --frozen-export fixtures/text-reward/frozen-preference-seed1601.json
```

The command refuses to replace existing results or an export. To reproduce,
choose new output paths. The [raw JSON](2026-10-04-text-reward.json) contains
all seeds, learning histories, first-backward norms, source IDs, exact texts,
labels, margins/probabilities, unknown pieces, reliability bins and failures.

## Preference results and nuisance interventions

Every decisive test condition has four pairs from four raw sources. The
table reports uncalibrated probabilities, with observed binary NLL/Brier.

| Seed | Final training preference loss | Matched-format accuracy / NLL / Brier | Both answers headed: accuracy / NLL / Brier | Incorrect answer longer: accuracy / NLL / Brier |
|---|---:|---|---|---|
| 1601 | 0.000072741 | 1.00 / 0.000072727 / 0.00000000530 | 0.50 / 0.741273 / 0.272199 | 0.75 / 0.750655 / 0.225785 |
| 1602 | 0.000003218 | 1.00 / 0.000003430 / 0.0000000000188 | 1.00 / 0.162512 / 0.046025 | 1.00 / 0.000003481 / 0.0000000000212 |
| 1603 | 0.000105272 | 1.00 / 0.000105062 / 0.0000000110 | 0.50 / 0.520966 / 0.188053 | 1.00 / 0.000062851 / 0.00000000985 |

The correct answer can also be copied unchanged and given a heading or a
redundant sentence. These equal-substance pairs have a declared binary soft
target of 0.5, not observed human frequencies or a separately fitted tie
model. The expected Bernoulli Brier score retains its 0.25 floor at prediction
0.5. Its measured values expose large departures from neutrality:

| Seed | Same-substance heading: expected Brier | Same-substance longer: expected Brier |
|---|---:|---:|
| 1601 | 0.495909 | 0.342497 |
| 1602 | 0.397703 | 0.448739 |
| 1603 | 0.491830 | 0.392313 |

Temperatures 0.5, 1, 2 and 4 were evaluated on calibration labels only; each
seed selected 0.5. No temperature or checkpoint was selected from test labels.
Temperature sharpening does not fix a ranking inversion: seed 1601's headed
NLL worsens from 0.741273 to 0.835117 and Brier from 0.272199 to 0.304448.
Raw/calibrated reliability bins, counts and ECE are in the retained JSON.

## Encoding audit: raw disjointness is insufficient

The after-fit audit compares actual valid token-ID pairs, including separator
and EOS, not merely row/source names. Training has twelve unique encoded
baseline pairs; calibration and test each have four. No test baseline pair
duplicates a training encoding. However, **all four test baseline encodings
duplicate calibration encodings**. The unseen nouns “box” and “book” both
become `<unk>`, removing the distinction in the raw prompts. Calibration-only
label access remains procedurally true, but the encoded test is not independent
of that calibration set. The apparent baseline success cannot establish
independent held-out calibration. The
[verification record](2026-10-04-text-reward-verification.json) lists exact
colliding pair IDs.

Do not silently rename a source, expose test words to a refitted vocabulary,
or replace failed records. The original raw report and frozen file remain
byte-identical. A fixed-prior vocabulary or character/byte-tokenization
intervention needs its own specification, disjoint encoded-input checks and
fresh output identities before fitting. This criterion remains open here.

## Outcome and process results

An incorrect equality can precede a correct final answer, and a valid equality
can precede a wrong one. Terminal BCE reads final EOS; process BCE reads only
explicit step-marker states. Local step labels are not critic targets for
expected future reward. Changing the later final answer leaves the earlier
step score unchanged under the causal mask; tests and the notebook verify it.

All seeds fit the training labels. The test result is deliberately reported
without converting fit into a generalization claim:

| Seed | Terminal accuracy / NLL / Brier | Step accuracy / NLL / Brier |
|---|---|---|
| 1601 | 0.50 / 4.391133 / 0.499847 | 0.50 / 4.898875 / 0.499944 |
| 1602 | 0.50 / 2.753042 / 0.495938 | 0.50 / 4.830682 / 0.499936 |
| 1603 | 0.50 / 4.228493 / 0.499788 | 0.50 / 4.422877 / 0.499856 |

Each metric counts four terminal targets or four labeled test steps. Numbers
7 and 8 are absent from the training vocabulary and both map to `<unk>`.
Opposite labels can therefore have exactly equal encoded traces. A classifier
cannot recover a distinction already removed by its input representation.
This explains a concrete information loss; it does not establish that fixing
the vocabulary would make the model learn general arithmetic.

## Gradients, frozen identity and software verification

The first preference backward pass reaches the embedding, Q/K/V projection
and scalar-head weight. For seed 1601 their gradient norms are respectively
0.0000889674, 0.000664689 and 0.000434518. The pairwise loss cancels the shared
head bias; terminal BCE has a different intercept contract.

Seed 1601 was selected for export before testing, irrespective of nuisance
results. The complete JSON stores every numeric tensor, config, ordered
vocabulary, normalization/separators, special IDs, unknown/EOS/endpoint policy,
fixture/source groups and measured source identity. Loading validates the
payload hash, actual shapes and an independently expected interface; it
returns exact-equal detached scores with all reward gradients disabled.
The adapter is `load_frozen(path, expected_interface=...).score(prompt,
completion)` or `.score_many([(prompt, completion), ...])`. Scores are raw
reward scalars, not calibrated goodness probabilities.

The measured export remains historical evidence:

- Export file SHA256: `3e631f45c28ef7af2dae4280a9e3e6cf3bcc7bf76e3ecbd8307ae6ce9e6c8d4d`.
- Payload SHA256: `f4b3ab69c8c4dd4fe8c574a9442215794ee9c80fedd45565a12fc53a6858e285`.
- Measured module SHA256 saved in the export: `c35d8730463379aa53562f3cffef4dd44f91bfd2d7e1c3107a6d3032f781cbb5`.
- Raw observation JSON SHA256: `3eced705915ebf837a77fd6975222e6bcfdcafc9cd2aba6a7d4c5e7118b8ba17`.

After measurement, an independent acceptance review found that a rehashed
`source_groups` string could be iterated as character IDs. The loader now
requires each split to be a nonempty list of unique nonblank strings before
flattening. Seven adversarial subcases cover strings, empty/blank IDs,
nonstring IDs, duplicates and dictionaries. This validation-only hardening
does not change the original numeric state or predictions. The current source
hash and original historical hash are recorded separately.

Nineteen dedicated unit tests pass on the stated Spark CPU interpreter. They
cover last-valid positions, left/right/extra padding, all-padding rejection,
finite padded-query gradients, EOS variants, pair reversal, explicit OOVs,
no silent truncation, source collisions, step boundaries, future-text
causality, crossed labels, source weighting, full exact serialization,
freezing and corruption/interface rejection. A fresh `dgx-spark-native`
kernel executes all eight code cells of the
[new notebook](../../notebooks/day-16/03_text_reward_and_process_labels.ipynb)
and produces five inspected figures. The existing Day 16 notebook also passes
its four cells/three figures. No notebook server is started.

## What remains unproven

This is a token-sequence training and supervision microscope, not a human
preference evaluator, a robust policy reward, a pretrained process-reward
model or general arithmetic reasoning. Independent encoded-input calibration
remains open, with a separately specified tokenizer intervention as the next
experiment. A later policy may exploit this frozen proxy, so it must report
independent factual outcomes, OOV failures and nuisance changes. A pretrained
Spark extension remains separately gated and unexecuted. Prepared course
materials and execution checks do not mark learner mastery.
