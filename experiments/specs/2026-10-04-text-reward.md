# Tiny text reward and process supervision experiment

The experiment replaces known feature coordinates with trainable token-sequence
representations, while retaining explicit original labels and CPU-scale evidence.
No pretrained model, external dataset, API, GPU, installation or server is used.
This specification precedes fitting and evaluation.

## Frozen recipe

Use `fixtures/text-reward/records.json` and record its exact SHA256. Fit a
case-folded word/punctuation vocabulary on training pairs and training traces
only, with explicit padding, unknown, separator, step-boundary and EOS IDs.
Post-measurement documentation clarification: the measured source already
included the fixed unlabeled formatting alphabet `Answer : ** In summary ,
this is a clear answer .`; it supplies formatting tokens without preference
labels. This statement records the original construction, not a refit or a
new intervention.
Unknown words map to `<unk>` under a disclosed policy; no text is silently
truncated. The scalar text encoder is an independently written one-block causal
decoder: width 24, two heads, learned normalized positions, residual attention,
48-unit GELU MLP and a scalar head. Use float64 CPU arithmetic, dropout disabled,
maximum 96 positions and inclusive EOS pooling at the final nonpadding position.
Left/right padding must agree under normalized position IDs.

Predeclared seeds: 1601, 1602, 1603. Fit preference models for 120 AdamW updates at
learning rate 0.02, weight decay 0.01, full training-pair batches. Fit separate
outcome and process models for 80 updates under the same optimizer recipe.
Outcome supervision uses final-EOS binary targets; process supervision uses
only explicit step-marker positions, never padding or the final-answer target.
For the latter, average supervised steps within a source before averaging
sources. No future-return labels are substituted for step validity.

Report every seed. Freeze and export the preference model from seed 1601 for
the later policy microscope, regardless of held-out performance. Do not select
a checkpoint or seed using test labels. A JSON-only export stores complete
numeric weights, config, ordered vocabulary/encoding/special IDs, endpoint
semantics, fixture/source-group identities and content hashes. Reject corrupt,
shape-incompatible or tokenizer-permuted reloads. Frozen scoring returns
detached CPU values and never updates reward parameters.

## Measurements and controls

Keep source groups disjoint across train, calibration and test. Measure pair
ranking, observed binary Brier/NLL, reliability-bin counts and ECE. Fit a
temperature on calibration labels only from a predeclared grid 0.5, 1, 2, 4; show
raw and calibrated test results. Test slices include equal-length/equal-format
comparisons, matched headings, longer incorrect answers and equal-factual-content
verbosity/format preferences. Pair swaps must reverse probability correctly.
Retain all pair texts, margins, probabilities, labels, OOV counts and failures.

Measure terminal and step predictions separately on all source-assigned traces,
including correct-answer/wrong-step and wrong-answer/valid-step cases. Show
training curves and held-out predictions for every seed; negative results are
not removed. Verify causality by changing a final answer without changing
earlier step scores, and check gradients through both backbone and scalar head.

Run unit tests for pooling/padding, EOS, causal masking, swap, boundaries,
empty/all-padding input, finite gradients, exact serialization, freezing and
interface/hash rejection. Execute the Day 16 notebook in a fresh CPU kernel and
inspect its data-backed figures plus model/endpoint schematics. Prepared material
is not learner mastery or a successful pretrained reward campaign.

## Optional pretrained extension

A pretrained reward model remains unexecuted and separately gated. It would need
an approved local backbone/tokenizer, independent reviewed preference and step
labels, source-disjoint splits, fixed chat/EOS/endpoint conventions, matched
length/format controls, optimizer/resource bounds and shared checkpoint identity
before a profiled Spark pilot. CPU microscope scores do not supply its evidence.
