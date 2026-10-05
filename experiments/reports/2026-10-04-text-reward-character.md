# Character reward inputs and heldout failures

The character intervention removes the exact input collisions diagnosed in
the historical word-token run. The real decoder and scalar heads fit all
training objectives, but held-out preference, terminal and process behavior
remains poor. This closes the narrow requirement to measure calibration on
distinct encoded inputs; it does not establish a useful general reward model
or arithmetic learning. The negative results and the original artifacts stay
visible together.

## Protocol before measurement

The [finalized specification](../specs/2026-10-04-text-reward-vocabulary-intervention.md)
was saved before any new full fit. Its SHA256 is
`bda7916d29fce3a7556e19b958b40b00b5848595b111aab4a14fa60d9ca84db1`.
The unchanged original fixture SHA256 is
`284deafebcecea747fa506f8ed6fcb245745f142a7809cedb5f9c56d7527d124`.
The predeclared seeds are 1611, 1612 and 1613; seed 1611 is exported before
its test evaluation, irrespective of the other seeds' outcomes. No test-based
recipe change or checkpoint selection occurred.

An additive [module](../../src/dongxi_llms/char_reward_lab.py) reuses the
actual `TinyTextReward`, not known quality/length/format features. Its fixed
vocabulary contains five special IDs and all 95 printable ASCII characters,
including spaces. It rejects non-ASCII/control characters before casefolding
and never emits UNK. Literal `<eos>` text is characters; EOS is added only by
the collator. Overlength or all-padding input fails, not silently truncates.

The decoder remains one causal block, width 24, two heads, a 48-unit GELU
MLP and a scalar head. Its fixed 160-position table gives 11,185 parameters;
float64 CPU, no dropout and inclusive-EOS last-valid pooling remain explicit.
Character sequences and the position table differ from the word arm, and the
environment is newer. This is not an equal-compute or single-factor tokenizer
quality comparison. The alphabet was fixed independently of labels, but the
follow-up was motivated by an already observed failure; it is not a blind
benchmark or an independent human annotation study.

Each preference model gets 120 full-batch AdamW updates; separate terminal
and process models get 80 each. Learning rate 0.02, weight decay 0.01 and
source-balanced losses match the frozen recipe. Two-update software checks
preceded measurement; they did not choose a training recipe.

## Exact input independence gates

Before fitting, the audit checks 55 records: twelve training pairs, four
calibration pairs, twenty held-out baseline/nuisance pairs and nineteen traces.
It stores each source, split, exact input hash, candidate-order-independent
pair hash, step-prefix hash and length. No encoded pair collides across
sources/splits, including candidate swaps. Full prompts, prompt/completion
inputs, terminal traces and complete prefixes through supervised STEP markers
also pass. Repeated equal-label prefixes within one source remain related
observations; conflicting labels on an equal causal prefix are rejected.

| Split | Unique encoded pairs | Unique supervised step prefixes |
|---|---:|---:|
| Train | 12 | 7 |
| Calibration | 4 | 2 |
| Test, including nuisance pairs | 20 | 2 |

Maximum actual pair length is 111 positions and trace length 56, below the
bound of 160 frozen before fitting. There are no unknown pieces. Numbers
7 and 8 have distinct character IDs. This checks exact encoded independence,
not semantic near-duplicate discovery or population-wide independence. The
four base test questions are still only four sources; their five conditions
do not create twenty independent questions.

In contrast, the retained word arm has four baseline test encodings identical
to calibration. Its historical ranking success cannot serve as an independent
calibration result. The new comparison makes that information loss visible
without replacing the original data, source, report or export.

## Actual execution

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /tmp/dongxi-course-reproduction.ZfVaEu/venv/bin/python \
  -m dongxi_llms.char_reward_lab \
  --fixture fixtures/text-reward/records.json \
  --protocol experiments/specs/2026-10-04-text-reward-vocabulary-intervention.md \
  --output experiments/reports/2026-10-04-text-reward-character.json \
  --frozen-export fixtures/text-reward/frozen-char-preference-seed1611.json
```

The command exited 0. Each seed completed every planned objective: 840 updates
total, with all histories and first-backward norms retained. Per-seed measured
fit/evaluation time is 1.709, 1.282 and 1.334 seconds. This excludes environment
installation/kernel startup and implies nothing about model-scale throughput.

The isolated interpreter is Python 3.12.14 with Torch 2.14.1+cpu on Linux ARM64.
The kernel uses the same temporary environment prefix, not Spark's shared GPU
environment. No package installation, acquisition, API, GPU, notebook server,
Git operation or animation rendering was performed for this intervention.
The [raw JSON](2026-10-04-text-reward-character.json) retains all three seeds,
132 preference prediction rows, 57 trace prediction rows, every authored
label, raw/calibrated margin/probability, reliability-bin count and failure.
Outcome versus local validity remains separately supervised, not combined
into a scalar with ambiguous meaning.

## Fitted training objectives and failed transfer

| Seed | Final preference training loss | Final terminal training loss | Final step training loss |
|---|---:|---:|---:|
| 1611 | 0.000003277 | 0.000081377 | 0.000079455 |
| 1612 | 0.000006447 | 0.000136095 | 0.000134622 |
| 1613 | 0.000005670 | 0.000166336 | 0.000095139 |

Training terminal/step accuracy is 1 for every seed. The first preference
backward pass reaches the embedding, attention Q/K/V and head; seed 1611's
norms are 0.0357164, 0.0483318 and 0.931764. This is actual neural training,
not a lookup table of known quality scores. Nevertheless, all three seeds
rank only two of four ordinary held-out pairs correctly:

| Seed | Plain matched accuracy / NLL / Brier | Both answers headed accuracy / NLL / Brier | Incorrect answer longer accuracy / NLL / Brier |
|---|---|---|---|
| 1611 | 0.50 / 6.295234 / 0.499997 | 0.75 / 0.920019 / 0.237500 | 0.50 / 6.287197 / 0.585577 |
| 1612 | 0.50 / 6.075204 / 0.499995 | 0.50 / 5.830474 / 0.551009 | 0.50 / 0.504450 / 0.200430 |
| 1613 | 0.50 / 6.035362 / 0.499994 | 0.50 / 6.038803 / 0.556842 | 0.50 / 3.504317 / 0.457624 |

Equal-substance targets are explicitly authored binary approximations to a
tie, not human preference frequencies. Expected Bernoulli Brier scores retain
their 0.25 minimum at prediction 0.5. Same-substance heading Brier is
0.375520/0.327224/0.281036 across the seeds; same-substance longer Brier is
0.393512/0.317337/0.437570. Small differences in surface form can still move
the learned score despite unchanged factual substance.

The character intervention preserves information, but does not compel the
optimizer to learn a general color-matching or arithmetic rule. These data
do not identify a unique cause of poor transfer, nor justify a test-selected
increase in updates. The frozen seed 1611 remains the policy-linked export,
not the best-looking headed-answer seed selected after testing.

## Independent calibration measurement

Temperatures 0.5, 1, 2 and 4 were frozen before fitting. Every seed chooses 4
using only four matched-format calibration labels, with the ascending-grid
tie rule declared in advance. Temperature cannot change the ranking:

| Seed | Selected temperature | Calibration NLL | Raw test NLL / Brier | Calibrated test NLL / Brier |
|---|---:|---:|---|---|
| 1611 | 4 | 1.016868 | 6.295234 / 0.499997 | 1.614901 / 0.460377 |
| 1612 | 4 | 1.541540 | 6.075204 / 0.499995 | 1.565459 / 0.456309 |
| 1613 | 4 | 0.973763 | 6.035362 / 0.499994 | 1.556254 / 0.455495 |

Softening reduces overconfidence but leaves the held-out ranking at 0.5.
NLL, Brier and ECE with counted reliability bins are now measured on distinct
encoded calibration/test inputs. This closes the encoding-collision criterion,
not calibrated human preferences, robust formatting invariance or a favorable
generalization claim. The four-source denominators remain visible.

## Terminal correctness versus local validity

Correct final answers following false steps and wrong final answers following
valid steps remain in the original authored fixture. Final EOS predicts
terminal correctness; STEP markers predict only the validity of the displayed
equality. A local process target is not a value critic's future return.
Changing a later final answer does not change an earlier marker score in a
causal forward pass.

| Seed | Heldout terminal accuracy / NLL / Brier | Heldout step accuracy / NLL / Brier |
|---|---|---|
| 1611 | 0.00 / 9.070920 / 0.999770 | 0.00 / 9.292352 / 0.999779 |
| 1612 | 0.50 / 4.620839 / 0.499903 | 0.50 / 4.687588 / 0.499915 |
| 1613 | 0.00 / 7.813466 / 0.998916 | 0.50 / 4.831744 / 0.499936 |

Each test metric counts four targets. Unknown-token aliasing is absent, so
it cannot explain these new failures. All incorrect confident predictions
are retained. General arithmetic and pretrained process-reward quality
remain unestablished.

## Frozen state and independent verification

The complete [seed 1611 character JSON](../../fixtures/text-reward/frozen-char-preference-seed1611.json)
contains numeric weights, config, exact fixed alphabet/encoding, specials,
EOS/endpoint policy, fixture/source groups, protocol/encoding-audit identity,
and source hashes. Its file SHA256 is
`f1bbb5134c1da05e3d85b8fe8e0af2890eab6686b2b6fa7797acf6657e66269f`;
the interface SHA256 is
`ec8e88cc72ed6e3939833cfa9ee3c8c6a877ba788ca8a51089bd4f966c465035`.
Live/save/reload scores are exactly equal in the locked CPU environment.
Parameters and returned scores have gradients disabled. The scorer returns
raw scalars, not goodness probabilities.

Twenty-five focused tests pass. They cover explicit rejection before casefold,
literal specials, pair/source/prefix/swap gates, last-valid left/right/extra
padding, all-padding rejection, finite gradients, EOS variants, actual three-
objective updates, future-text causality, exact freezing, complete numeric
JSON, external file identity, shape/hash/interface corruption, duplicate fields,
source-list typing, boolean/nonfinite weights, nested/oversized payloads and
retained failed CLI gates. Loading bounds bytes/nesting/elements; self-hashes
are still integrity checks, not authentication against rewritten metadata.

An independent parent-agent CPU replay passes all 25 tests and reproduces
all three full campaigns. Every saved fit history/first-backward/train ID,
temperature choice, calibration panel, raw/calibrated test panel, trace result
and encoding gate is exactly equal; runtime/date fields are not compared.
That replay takes 3.936 seconds and does not refit a preferred recipe.
The [verification JSON](2026-10-04-text-reward-character-verification.json)
records source hashes, command outcomes, actual kernel identity and inspected
figures. The measured module SHA256 is
`a152706e54f6adaeab1c9c50488ba5b017254d30567be20b243d7a4593262a91`.

## Notebook portability failure remains visible

The first fresh locked-kernel attempt failed at the old word-arm assertion
that fresh Torch 2.14.1 retraining must equal Torch 2.13 historical weights.
The [failed manifest and traceback](2026-10-04-text-reward-character-notebook-first-attempt.json)
are retained. Same-state serialization and retraining reproducibility are
different checks. The notebook now compares exact scores after loading the
same saved word state twice, prints the measured fresh-versus-historical
training delta (maximum score difference 0.0000000000201226 in this pass), and
keeps strict exact live/save/reload checks for the new
character arm in its own locked environment. No historical weights or reports
were regenerated to make the assertion pass.

The extended [Day 16 notebook](../../notebooks/day-16/03_text_reward_and_process_labels.ipynb)
retains the original panels and adds encoded-collision, training/heldout and
counted-calibration visuals with adjacent reference explanations. The new
panels show actual negative measurements. Original five PNG preview bytes
are preserved; new previews receive indices 06–08.

## Evidence boundary and next experiment

The original word report, raw JSON, export and source remain byte-identical.
The character arm adds exact-input-disjoint evaluation, real neural fitting,
full negative outcomes and a validated frozen interface. These are bounded
CPU mechanism/evidence results, not independent human preferences, a robust
reward, general reasoning or learner mastery. A linked policy experiment must
name its own frozen proxy and retain independent factual measurements. Own
balanced/confounded reward fits use separate fixtures/protocols/exports; they
do not replace this negative experiment. Pretrained/Spark validation remains
separately gated and unexecuted.
