# Character tokenization and independent encoded reward inputs

Status: finalized premeasurement protocol. No fit or new test outcome was
observed before this specification was saved. This follows the separately
retained word-token experiment; its raw JSON, prose report and seed 1601
export must remain byte-identical. New outputs use distinct filenames.

## Question and predictions

The original tokenizer aliases unseen nouns “box”/“book” and numbers 7/8.
The intervention asks whether preserving their character distinctions gives
source-disjoint, encoding-disjoint calibration/test comparisons for the
same actual decoder/scalar-head task. It is a representation and evaluation
experiment, not a claim that character tokens are best for production LLMs.

Prediction before fitting: exact-input collisions should disappear because
printable ASCII characters have distinct IDs. The text reward may still fail
to optimize, transfer to unseen sources, calibrate probabilities, ignore format
or classify arithmetic steps. Retain every such result. Longer sequences and
a larger position table mean this is not an equal-compute tokenizer comparison.

## Fixed alphabet and encoding

The fixed, data-independent vocabulary is PAD, reserved UNK, SEP, STEP and EOS
at IDs 0–4, followed by every printable ASCII character U+0020–U+007E in
ascending order: 100 entries total. Validate the original text before
casefolding; reject non-ASCII, controls and blank/nonstring input. Do not let
Unicode normalization turn an unsupported character into an accepted ASCII
one. Preserve spaces and punctuation, then casefold ASCII letters. The UNK
entry exists only for compatible special-ID geometry and is never produced.
Literal text such as `<eos>` is encoded as characters, not a special token.

Preference input is prompt characters, SEP, completion characters, EOS.
Process input is prompt, SEP, each displayed step followed by STEP, then final
answer and EOS. Step supervision occurs only at STEP; terminal supervision
occurs only at final EOS. Last-valid pooling uses the maximum valid array
index. Reject all-padding, unsupported or overlength input; no truncation.

Use the unchanged original `fixtures/text-reward/records.json` and the same
five nuisance construction rules. Before fitting, scan exact sequence lengths
including specials. Freeze the maximum position bound at **160**, not a
test-selected variable. The original fixture/control scan maximum is 112.
Record actual maxima and fixture hash in the new report.

## Gates before any fit

Run the raw source-split contract, then build actual valid token-ID sequences
including separator/EOS. Whole preference pairs must be disjoint across train,
calibration and test, including candidate swaps. Prompt/completion identities
and full prompt identities cannot belong to different sources or splits.
Check complete causal prefixes through each supervised STEP, not arbitrary
shared prefixes such as the initial character “C”. Opposite process labels
must never attach to identical causal inputs. Repeated equal-label prefixes
within one source remain related observations, not independent examples.

Reject setup before fitting if any cross-split encoded-pair, response, full-
prompt or step-prefix collision occurs. Save the entire gate diagnostic,
including colliding record/source IDs, on failure. Preserve distinct IDs for
each number character and retain the earlier failed word-token gate as
historical evidence, not as a quietly repaired dataset.

## Frozen model and optimization

Reuse the independently written `TinyTextReward`, not a known-feature scorer:
one causal decoder block, width 24, two heads, learned normalized positions,
48-unit GELU MLP and scalar head. Use 100 vocabulary entries, 160 positions,
float64 CPU, no dropout and inclusive-EOS pooling. The original word arm's
96-position table and shorter token sequences remain unchanged.

Predeclared seeds: **1611, 1612, 1613**. Preference models receive 120 AdamW
updates; independent terminal/process models receive 80 each. Full training
batches, learning rate 0.02, weight decay 0.01 and source-balanced means match
the original recipe. Do not add steps, change the optimizer or select a model
after inspecting test results. Export seed **1611** regardless of which seed
performs best. Tiny two-update software checks are not a recipe-selection run.

Temperature candidates are 0.5, 1, 2 and 4. Minimize matched-format calibration
NLL only; choose the first value in the ascending grid on an exact tie. Report
all raw and calibrated test results. No seed, checkpoint, alphabet, length cap
or step budget is chosen from test labels. Calibration data is not used for
gradient updates. No positive-result threshold is required to retain a run.

## Measurements and export

For every seed retain all three objective histories and first-backward
embedding/QKV/head norms, exact pair/trace texts, authored labels, source IDs,
encoded input/prefix hashes, probabilities, margins, reliability counts,
ranking accuracy, NLL/Brier/ECE, raw/calibrated nuisance slices, step positions,
and failures. Five four-pair test conditions remain four source questions,
not twenty independent questions. Authored soft tie targets have expected
Bernoulli Brier semantics with a 0.25 floor; no human calibration claim.

Export `fixtures/text-reward/frozen-char-preference-seed1611.json` with a new
schema, complete numeric state, fixed alphabet/config, input/source/encoding
identities, EOS/endpoint convention, protocol SHA and predeclared selection.
Require caller-expected file/interface hashes where appropriate, disable
reward gradients and verify exact live/save/reload CPU scores. Never replace
an existing export/report. Frozen scalar scores are not goodness probabilities.

Run pooling/padding, causal-prefix, gradients, swap, explicit-rejection,
disjointness, JSON numeric/shape/hash and exact-freezing tests. Execute the
existing Day 16 extension in fresh `dongxi-course-clean` kernels using
`/tmp/dongxi-course-reproduction.ZfVaEu/venv/bin/python`, with JUPYTER_PATH under
its temporary kernel prefix. Inspect data-backed new figures and preserve the
original previews/report. No installation, download, API, GPU or server work.

Successful software execution and exact-input disjointness establish only the
bounded authored CPU experiment. They do not prove robust reward transfer,
general arithmetic, pretrained quality, policy safety, human agreement or
learner mastery. Optional pretrained/Spark validation remains separately
gated; the later critic experiment must independently measure factual outcomes.
