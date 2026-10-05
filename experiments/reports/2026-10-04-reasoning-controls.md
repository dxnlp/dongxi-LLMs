# DXI-04 — sampled learning, retained failures and reasoning controls

The bounded CPU instrument passed: all three predetermined seeds improved their
mean training probability of a correct, naturally terminated response, and all
checks passed. This is **not** three fully successful task runs. Seed 2301 solved
all four training prompts; seeds 2302 and 2303 learned the majority answer and
continued to fail the independently known-solvable zero-sum prompt. Those
failures, poor held-out results and the original negative GRPO results remain
part of the evidence.

The [final machine-readable report](2026-10-04-reasoning-controls-verification.json) contains actual
raw evaluation records, update metrics, hashes, environment identities and
verification outputs. The [specification](../specs/2026-10-04-reasoning-controls.md)
and original fixtures were recorded before measurements. No pretrained weights,
model download, GPU job, API, installation or external service was used.

## Frozen experiment and actual training

Each seed 2301/2302/2303 initialized a 2672-parameter shared TinyDecoder: one layer,
width 16, four query heads, two KV heads, hidden width 32 and vocabulary 10. It
received four symbolic training prompts and no SFT warm-start. The independently
fixed arithmetic oracle asks whether the sum of two nonnegative numerals is
positive. A response must emit the correct 0/1 and then EOS. References and
source IDs are never supplied as input token features.

The predeclared recipe ran 120 fresh sampled GRPO updates per seed, four prompts
and 16 responses per prompt, cap 2, temperature 1, AdamW learning rate 0.02,
weight decay 0, epsilon 0.2, KL beta 0.01, population group standard deviation,
per-response token mean and gradient clip 1.0. The syntax-conditioned sampler
permits 0/1 first, then 0/1/EOS. Collection and current/old/reference likelihoods
use exactly that same conditional support. EOS is sampled and learned; it is not
inserted by the verifier. This is not raw-vocabulary training or a change to the
existing Qwen full-support baseline.

The exact complete-path probability is a diagnostic; the decoder was updated
using sampled relative rewards. All three runs consumed 15360 valid response
tokens each, 46080 total, with 360 actual optimizer updates. Initial likelihood
ratios matched the collected behavior probabilities exactly in every update.
Initial nonzero gradients reached the shared token embedding, Q/K/V and output
projections, MLP, normalization and language-model head. First-update absolute
EOS-head gradient sums were 1.425122, 1.671371 and 1.349241. All decoder state hashes
changed; each paired reference remained frozen and its before/after generated
records matched the initial model.

## Observed results, with no seed or checkpoint selection

All rows use the final scheduled update 120. Probabilities average the four
training prompts; greedy successes retain all four prompts in the denominator.

| Seed | Initial answer probability | Initial complete-path probability | Final answer probability | Final complete-path probability | Initial→final greedy training success |
|---|---:|---:|---:|---:|---:|
|2301|0.474835|0.156799|0.998446|0.998364|1/4→4/4|
|2302|0.527019|0.176079|0.749954|0.749946|0/4→3/4|
|2303|0.499130|0.157151|0.749970|0.749924|0/4→3/4|

For seeds 2302/2303, `control-train-0-0` still generated `[7,2]`, decoded as
`1` plus EOS, when the independent oracle required 0. Its complete-path
probability fell to 0.000093 and 0.000041. Mean improvement therefore cannot
establish that every solvable row improved, let alone that the learned rule
generalizes. No failure was removed, retried or used to retune this recipe.

Each held-out slice has four original prompts. No source/problem identity
crosses a split. New-template rows also contain new sources, so this is not an
isolated template-only experiment. The tiny decoder receives symbolic
instruction tokens, not the authored English problem strings.

| Final greedy complete success | Training | New source | New source+template | Unseen odd-sum family |
|---|---:|---:|---:|---:|
|Seed 2301|4/4|0/4|2/4|0/4|
|Seed 2302|3/4|4/4|4/4|2/4|
|Seed 2303|3/4|4/4|4/4|2/4|
|Constructed constant 1+EOS|3/4|4/4|4/4|2/4|
|Constructed constant 0+EOS|1/4|0/4|0/4|2/4|

The source/template held-outs happen to contain only positive sums. Their
perfect scores for seeds 2302/2303 equal a fixed constant-answer baseline, not
evidence of arithmetic transfer. Seed 2301 learned the training exception but
did worse on unseen sources. The balanced odd-sum slice remains a held-out
task-family test, not a promised success.

## Stopping, support and the lookup counterexample

The following intervention results include all 48 seed×item greedy rows.
Invalid and truncated outcomes remain in their denominators.

| Delivered protocol | Answer correct | Complete success | Invalid/unsupported | Truncated |
|---|---:|---:|---:|---:|
|Trained syntax support, cap 2|32/48|32/48|0/48|0/48|
|Same syntax support, cap 1|32/48|0/48|0/48|48/48|
|Raw full vocabulary, cap 2|0/48|0/48|48/48|0/48|

A mathematically correct numeral can survive cap 1, but EOS cannot: the
complete-path reward is zero for every capped path. Removing the training-time
syntax conditioning exposes a different delivered distribution and fails on
every row here. A stop and an invalid answer can coexist; the report keeps
answer, formatting and stopping separate rather than treating them as one
success flag.

The auxiliary tabular arm is explicitly an exact-expectation lookup control,
not sampled sequence training. Forty predeclared SGD steps changed correct
answer probability from 0.5 to 0.971495 on its four observed keys. Every unseen
key stayed exactly 0.5, and the unknown row's weights stayed unchanged. This arm
supplies EOS externally and therefore cannot replace the shared-decoder proof
or establish learned sequence termination or mathematical reasoning.

## Original language panel and unexecuted wider protocol

Twenty independently authored arithmetic, linear-equation and multi-step
quantity items cover training, new sources, new-source-plus-template and an
unseen task family. Exact integer/Fraction oracle calculations validate each
reference. Eighty oracle/wrong/invalid/cap parser probes use the existing
bounded mathematical grader. These are authored grading checks, **not**
natural-language generations from the tiny decoder.

The protocol fixture separately plans base/raw, instruct/chat with thinking
disabled, and the **same** instruct checkpoint with supported thinking enabled.
It predeclares final-only cap 32 and short-work cap 128 and requires actual
weight/tokenizer/template/stop/sampling/cost identities before any row is
claimed as executed. A thinking toggle does not become a new set of weights;
correct answers and longer traces do not demonstrate rationale faithfulness.
All pretrained rows remain unexecuted and the separate Spark baseline remains
approval-gated.

## Verification and retained evidence

The durable report was collected on 2026-10-04 using Python 3.12.14,
PyTorch 2.13.0+cu130 on Linux/aarch64, explicitly CPU/float32 and one CPU thread.
CUDA was hidden from all verification subprocesses. Whole control execution
took 18.525 seconds. Linux report-process maximum RSS was 713320 KiB; this is not
a system-wide or GPU memory peak, and no throughput claim follows from it.

| Check | Result |
|---|---|
|Focused independent control tests|15 passed, 1.738s|
|Full shared unit suite at report snapshot|288 passed, 12.819s|
|Book math rendering checks|54 Markdown files, 1114 expressions, 0 issues|
|Fresh Day 23 kernel panel|All 3 notebooks passed; new route 8/8 code cells, 5 images, 0 skipped|
|Source/preview coherence|All 16 recorded source/input hashes and 5 exported preview hashes match|

Fresh manifest: `/tmp/dongxi-course-check-b1beeywl/manifest.json`, SHA256
`4cc71fe177129d0f515b35bfbaf1379443e830059a5d2e213b2ff1b04ac10af7`.
Its actual kernel executable was
`/home/dongxi/dgx-spark-dongxi/.venv/bin/python`, with
`cuda_available:false`; the kernel label alone is not the device evidence.
The transient executed notebook is recorded in the report; the committed
source notebook and five portable preview figures are the durable course
artifacts. All five images were inspected for labels, denominators, readable
axes and schematic-versus-measured distinctions.

The [first collection](2026-10-04-reasoning-controls.json) is preserved as
historical evidence, SHA256
`1740f65c15e8dcef59948239650631a419fd328369514397105f3dde2e3ca85e`.
Its `command` field used Python's rewritten `sys.argv[0]` and must **not** be
treated as a runnable direct-file invocation of a relative-import module. The
final verification records the actual `sys.orig_argv`, Python's rewritten argv
separately and a runnable `python -m dongxi_llms.reasoning_controls` invocation.
Only provenance collection changed: all three final parameter hashes and all
360 numerical training-history rows matched the first collection exactly.
All five regenerated preview hashes also match the already inspected figures.

The JSON preserves 816 evaluated decoder paths, including paired initial/frozen
controls, final greedy/seeded-sampled rows and stopping/support interventions.
Every evaluation path has raw text/tokens, contract/checkpoint/split identity,
grading, stop/truncation and cost. All 360 training-update metric rows are
preserved. **Only first and last training rollout groups** are fully archived:
384 sampled training paths with masks, collected behavior log-probabilities,
rewards, advantages and stops. Intermediate training paths are reproducible
from the frozen recipe/seeds, not fully archived. No decoder checkpoint files
are persisted by this microscope; actual parameter-byte hashes identify the
measured in-memory states.

The original
`experiments/reports/2026-10-04-grpo-diagnostics-distillation.json` is unchanged,
SHA256 `f8f8bd4c47c0dd3c08dc3fd812f83240be1e7971db4bf89ce4277aa493607e51`.
Its accompanying Markdown is also unchanged. Old failures have not been
relabelled by the new instrument.

## Reproduction and scope

Run the notebook from a fresh CPU kernel, then attach its newly generated
matching manifest. Use an unused report path to preserve historical evidence.

```bash
CUDA_VISIBLE_DEVICES= /home/dongxi/dgx-spark-dongxi/.venv/bin/python \
  scripts/verify_course_notebooks.py --days 23 --kernel dgx-spark-native

CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  /home/dongxi/dgx-spark-dongxi/.venv/bin/python \
  -m dongxi_llms.reasoning_controls \
  --report experiments/reports/UNUSED-reasoning-controls.json \
  --notebook-manifest /tmp/ACTUAL-FRESH-RUN/manifest.json \
  --export-previews
```

The specification and machine-readable report contain the complete recipe and
actual source hashes. Instrument acceptance establishes bounded sampled
learnability, likelihood/mask accounting and transparent retained failures.
It does not establish natural-language reasoning competence, generalization to
new families, faithful explanations, pretrained-model gains, GPU safety or
Spark-scale throughput. Chapter 13, its worked solutions/lab and the Chapter 15
defense bridge use precisely these boundaries.
