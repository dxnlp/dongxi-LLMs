# Lab 7 — Evaluate before changing the model

The CPU notebooks use declared fixture scores to expose the measurement mechanism. They are runnable on Mac or Spark with the course PyTorch/Matplotlib environment. No model checkpoint, external dataset, API key or GPU is needed.

1. Open [metrics and contracts](../../notebooks/day-10/01_metrics_and_contracts.ipynb). Explain each accepted/rejected string, derive the pass@3 subset count, and change the normalization policy only as a named new contract.
2. Open [paired uncertainty](../../notebooks/day-10/02_paired_uncertainty.ipynb). Compare all-success intervals at different sample sizes, inspect the paired-difference distribution, then reproduce the deliberate duplicate-item failure.
3. Open [slices and contamination](../../notebooks/day-10/03_slices_and_contamination.ipynb). Find the rare regression, detect the planted overlap, and preserve a repair record.
4. Open [mathematical grading and response replay](../../notebooks/day-10/04_mathematical_grading_and_response_replay.ipynb). Distinguish extraction, exact grammar, task format and natural stopping. Inspect the authored set/JSON regressions and the paired source-group distribution. Change a local response, not the frozen fixture files.
5. Open [actual candidates and answer selection](../../notebooks/day-10/05_actual_candidates_and_answer_selection.ipynb). Generate coordinate-seeded candidates from the original tiny decoder, compare first/vote/likelihood selectors on the same pool, and charge invalids plus rejected whole-attempt boundaries. Separate oracle availability from the deployed decision.

The reusable implementation is [evaluation_lab.py](../../src/dongxi_llms/evaluation_lab.py). Its tests compare pass@k against exact combinatorial counts and check interval/normalization boundaries. Run:

~~~bash
PYTHONPATH=src OMP_NUM_THREADS=1 python -m unittest discover -s tests -p test_evaluation_sft_labs.py -v
~~~

To move from fixture to model evidence, freeze the original instruction evaluation generated in Lab 9, its task/format slices, and the exact decoding rule. Development contains 60 original examples across 20 value groups; publication test contains 120 examples across 40 separate groups. Baseline and candidate must share item identities, serialization, parser and attempt budget. The provided training runner scores a declared eight-ID greedy development grid and full development answer NLL; it does not substitute for the publication suite.

For stories, keep a separate original prompt population and the five-dimensional rubric from Chapter 7. A copy/extraction evaluation cannot establish story coherence. Compare checkpoint outputs blind and retain each reviewer's raw decisions.

Deliver an evaluation card with capability, suite hash, checkpoint, template, decoding, verifier, denominators, uncertainty unit, error slices, selection history and limitations. Completion of this lab requires an explained instrument and audit, not a favorable score.

## Offline response replay

The [original fixtures](../../fixtures/reasoning-evaluation/README.md) contain
fifteen development items, fourteen source groups and thirty authored responses.
Neither panel is a trained model. The reusable
[reasoning grader](../../src/dongxi_llms/reasoning_evaluation.py) uses bounded
exact rational/set/interval grammar; unsupported expressions stay unsupported.
It does not invoke a symbolic fallback or execute model text. The existing
strict integer verifier remains a separate unchanged task.

```bash
PYTHONPATH=src python -m unittest discover -s tests \
  -p test_reasoning_evaluation.py -v
python scripts/evaluate_reasoning_records.py \
  --compare authored-baseline authored-candidate --draws 2000 --seed 1010
```

The CLI retains raw responses/errors and prints task/source/split slices,
known-cost totals with unknown denominators and source-group paired uncertainty.
Every comparison requires identical item/sample coverage and a frozen contract.
The [specification](../../experiments/specs/2026-10-04-reasoning-evaluation.md)
and [report](../../experiments/reports/2026-10-04-reasoning-evaluation.md) record
the actual checks and their limits. Token counts, generation cost and the
simulated stop/error fixtures are not measured model events. The adapter below
has actual tiny random CPU evidence; approved pretrained checkpoints and
independently reviewed behavioral labels are still needed for a model claim.

## Export a card without inventing a model run

The [card exporter](../../scripts/export_evaluation_model_card.py) reads an
existing frozen contract and complete saved ledger, then writes a new exclusive
bundle. It performs no generation, model loading, hardware probe or download.
Use the authored fixture first:

```bash
python scripts/export_evaluation_model_card.py \
  --items fixtures/reasoning-evaluation/items.json \
  --contract fixtures/reasoning-evaluation/contract.json \
  --records fixtures/reasoning-evaluation/responses.jsonl \
  --compare authored-baseline authored-candidate --draws 2000 --seed 1010 \
  --output /path/to/NEW-authored-evaluation-card
```

Supply an unused destination. Inspect `model-card.md`, its canonical JSON and
the full `replay.json`. Check that the raw errors, set/JSON regressions, task
slices and source-group pairing survive the summary. Token/resource fields
remain unknown, and the record labels do not identify trained weights. Reusing
an existing destination must refuse; do not overwrite the original ledger to
make export convenient.

For an actual saved local-adapter run, pass its exact items/contract/response
paths and `--run-identity /path/to/RUN/input-identity.json`. Supplied identities
must internally match the record input hashes, settings and interface. The
exporter does not independently reload checkpoint bytes or authenticate the
producer. A missing identity stays unverified. A paired comparison requires
aligned item/sample coverage; partial panels remain explicitly incomplete.
Even a successful export leaves pretrained behavior, training genealogy,
independent review and publication approval unestablished unless separate
evidence supports them. Chapter 7 exercise 21 asks you to explain this boundary.

## Inspect actual pretrained evidence without loading a model

The [Base/short-SFT replay report](../../experiments/reports/2026-10-05-pretrained-evaluation-replay.md)
links thirty actual raw responses under the same fifteen-item frozen instrument.
Inspect the card, complete replay and two producer identity/resource bundles.
Compare correctness, format, stopping and truncation separately; account for
all1,920 generated tokens rather than keeping only eligible answers.

Read the SFT `benign-a` text beside its rubric. Explain why an automatic pass
and an unfinished, artifact-filled answer are compatible observations. Then
compare the independent review rather than replacing the original metric.
This is a CPU-readable evidence session, not permission to rerun generation.
The later [original120-item publication comparison](../../experiments/reports/2026-10-05-native-sft400-comparison.md)
now measures Base, full400 and merged LoRA400. Read its strict whole-answer,
natural-ending and cap counts beside the older15-item development instrument;
do not pool the populations. Forty held-out lexical groups within three shared
task templates are not120 independent tasks. The
[actual model card](../../docs/model_cards/native-assistant-interface-study.md)
keeps those limits visible. The original authored fixture still labels an
instrument check, not a model event. Exercise22 supplies a worked answer.

## Blind story ratings without model loading

The [story-rating CLI](../../scripts/evaluate_story_ratings.py) consumes the
original frozen story contract. Its
[authored fixture packet](../../fixtures/story-rubric/README.md) demonstrates
the interface, not a real generated story comparison or human annotation.
The five rubric dimensions are deliberately separate from the copy/extraction
and mathematical instruments above.

From the repository root, choose new output directories:

```bash
python3 scripts/evaluate_story_ratings.py prepare --contract fixtures/story-rubric/contract.json --checkpoints fixtures/story-rubric/checkpoints.json --records fixtures/story-rubric/authored-records.jsonl --raters fixtures/story-rubric/raters.json --output outputs/story-ratings-packet-01
python3 scripts/evaluate_story_ratings.py evaluate --bundle outputs/story-ratings-packet-01 --output outputs/story-ratings-empty-01 --compare authored-baseline-14000 authored-candidate-14000
```

Inspect `packet.json` and the two empty rating templates. Keep
`private-codebook.json` away from reviewers. The second command deliberately
has no ratings: its comparison must remain null/awaiting ratings. Missing
review is not a score0, default midpoint or agreement. No training or model
generation occurs in either command.

To explore the authored controls, reviewers may fill separate templates with
explicit integer0/1/2 dimension scores or abstain with a reason. Supply them
with repeated `--ratings` arguments to a new evaluation output. Those practice
decisions remain ratings of authored demonstrations, not model results or a
certified independent human panel. Attempt a changed text hash, duplicate ID
and out-of-range score; each must be refused rather than silently corrected.

Keep greedy and sampled recipes separate. Inspect coverage at the declared
checkpoints and compare aligned source openings. Link the paired uncertainty
to the [source-group notebook](../../notebooks/day-10/02_paired_uncertainty.ipynb),
not an independent-row bootstrap over48 repeated outputs. Exercise23 and its
adjacent solution explain the missing-data, blinding and adjudication limits.
Real model records additionally require actual interface/token/likelihood/stop/
cost identities; filling an authored fixture cannot satisfy that requirement.

## Local checkpoint generation

The [generation CLI](../../scripts/generate_reasoning_records.py) has no model
download path. Supply a local full or explicitly merged HF checkpoint containing
its saved fast tokenizer and safetensors weights. Do not pass a remote model name.
An external tokenizer must match the checkpoint's saved encoding semantics.
Raw/chat modes, exact template, thinking-template setting, stops and decoding
are frozen before generation, not patched after reading the outputs.

The settings JSON contains these complete fields; adapt IDs/template/context
only after inspecting the intended local checkpoint:

```json
{
  "template_id": "reviewed-raw-v1",
  "thinking_mode": "not-applicable",
  "decoding": {"mode": "sample", "seed": 1010, "temperature": 1.0, "top_k": null, "top_p": 1.0},
  "stopping": {"eos_token_ids": [1], "turn_stop_token_ids": [], "pad_token_id": 1},
  "max_new_tokens": 64,
  "generation": {
    "input_mode": "raw", "template": null, "context_window": 512,
    "samples": 2, "device": "cpu", "dtype": "float32", "add_special_tokens": false,
    "max_run_seconds": 300, "scoring_text": "decode_without_terminal_stop"
  }
}
```

The IDs above are examples, not Qwen defaults. A chat contract instead uses
`input_mode: chat`, the exact template string, no second special-token insertion,
and `thinking_mode: template-default`, `enabled` or `disabled`; an explicit toggle
requires the template's `enable_thinking` keyword. Its saved semantics matter
more than its short template label.

Freeze a new contract using only the tokenizer, then explicitly generate and
replay. All paths below are reader-supplied examples, not a launch instruction:

```bash
python scripts/generate_reasoning_records.py \
  --items /path/to/frozen-items.json --settings /path/to/reviewed-settings.json \
  --checkpoint /path/to/local-hf-checkpoint --freeze-contract /path/to/NEW-contract.json
python scripts/generate_reasoning_records.py \
  --items /path/to/frozen-items.json --contract /path/to/NEW-contract.json \
  --checkpoint /path/to/local-hf-checkpoint --output /path/to/NEW-response-run
python scripts/evaluate_reasoning_records.py \
  --items /path/to/frozen-items.json --contract /path/to/NEW-contract.json \
  --records /path/to/NEW-response-run/responses.jsonl
PYTHONPATH=src python -m unittest discover -s tests -p test_reasoning_generation.py -v
```

Inspect `input-identity.json`, `observed-interface.json`, `events.jsonl`, every
`responses.jsonl` row and `evaluation.json`. `failure.json` retains failed setup
or interruption stages and missing planned coverage; do not delete it to make a
run look complete. The transparent no-cache loop counts full-prefix forward
positions separately from generated tokens. Its deadline is checked between
forwards, not enforced by a hard external supervisor. CUDA remains an additional
explicit gate and has not been executed by this adapter verification; model-scale
Spark work requires its separately approved profiling/supervision protocol.

The [specification](../../experiments/specs/2026-10-04-reasoning-generation-adapter.md)
and [measured adapter report](../../experiments/reports/2026-10-04-reasoning-generation-adapter.md)
separate actual random-model CPU execution from authored and scripted controls.
They do not close pretrained evaluation or independent behavioral-review gates.

## Actual story rating comparison

For a completed real-story example, inspect the
[supplied rating report](../../experiments/reports/native-story-publication-20261005-01/ratings-evaluation-01/report.json)
and its [input/source receipt](../../experiments/reports/native-story-publication-20261005-01/ratings-evaluation-01/receipt.json).
The public packet and both actual AI submissions are retained beside the
private codebook. Do not show that codebook to a new reviewer before collecting
their judgments. The authored empty templates remain empty deliberately.

Predict whether natural EOS implies a good ending before reading the scores.
The measured answer is no: control400 stops naturally45/48 times but has
ending mean0.0208/2. Then inspect the56 disagreement candidates, per-reader
scores and half-rate-minus-control paired deltas. Check that later missing
checkpoints have null means, not zero. The companion Chapter7 solution23
explains why twelve opening groups remain twelve after adding recipes and
raters. Reading or replaying this report needs no checkpoint, GPU or server.

## Actual candidate selection microscope

The [original18-item panel](../../fixtures/inference-selection/README.md) includes
six balanced train sources and source/template/family failure slices. English
descriptions are not parsed by the symbolic decoder. The
[measured experiment](../../experiments/reports/2026-10-04-inference-selection.md)
retains864 real candidates, all240 train-only updates,1736 generated and1736
rescored actions, caps, invalids and aligned selector decisions. It does not
replace the earlier categorical simulator or negative reasoning reports.

```bash
PYTHONPATH=src python -m unittest discover -s tests \
  -p test_inference_selection_lab.py -v
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python \
  -m dongxi_llms.inference_selection_lab \
  --items fixtures/inference-selection/items.json \
  --spec experiments/specs/2026-10-04-inference-selection.md \
  --output /tmp/NEW-selection-run
```

Choose a new run directory. Inspect the append-only responses/events, complete
results and source/input identity. A failed invocation keeps a failure summary;
partial forward/scoring failures remain records rather than disappearing from
denominators. The notebook recomputes a full fixed80-update fit and four fresh
coordinates, then uses the frozen all-seed measured panel for its five figures.
Changing a local selector or budget never edits that historical evidence.

All selector arms pay mandatory likelihood rescoring in this protocol; an
optimized first-answer baseline could avoid it. Its time plots sum observed
uncached CPU attempts plus retrospective selection and are not an optimized
Spark service benchmark. Source siblings remain grouped. Neither a positive
candidate-availability score nor low training loss establishes English reasoning,
pretrained performance, human preference or learner completion.
