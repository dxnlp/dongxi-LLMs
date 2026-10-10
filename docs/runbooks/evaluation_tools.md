# Evaluation, evidence reading and CPU reference tools

The [guided evaluation lab](../../book/labs/07-evaluation-is-a-contract.md)
defines the learning route. This reference keeps exact replay, export and
local-generation commands together with their original evidence boundaries.
Use the isolated environment from [Appendix D](../../book/appendices/d-reproduction-and-environments.md).
A historical output directory is an input to read; new verification uses a new
destination. Saved fixture responses, historical model outputs and fresh CPU
mechanisms remain distinct sources of evidence.


## Focused CPU checks

The full unit suite is registered in Appendix D. These focused commands inspect
the evaluation and policy mechanisms without loading pretrained weights.

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 python -m unittest discover -s tests -p test_evaluation_sft_labs.py -v
PYTHONPATH=src python -m unittest discover -s tests -p test_preference_policy_labs.py -v
```

## Evaluation replay and collection

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
all 1,920 generated tokens rather than keeping only eligible answers.

Read the SFT `benign-a` text beside its rubric. Explain why an automatic pass
and an unfinished, artifact-filled answer are compatible observations. Then
compare the independent review rather than replacing the original metric.
This is a CPU-readable evidence session, not permission to rerun generation.
The later [original 120-item publication comparison](../../experiments/reports/2026-10-05-native-sft400-comparison.md)
now measures Base, full400 and merged LoRA400. Read its strict whole-answer,
natural-ending and cap counts beside the older 15-item development instrument;
do not pool the populations. Forty held-out lexical groups within three shared
task templates are not 120 independent tasks. The
[actual model card](../model_cards/native-assistant-interface-study.md)
keeps those limits visible. The original authored fixture still labels an
instrument check, not a model event. Exercise 22 supplies a worked answer.

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
review is not a score 0, default midpoint or agreement. No training or model
generation occurs in either command.

To explore the authored controls, reviewers may fill separate templates with
explicit integer 0/1/2 dimension scores or abstain with a reason. Supply them
with repeated `--ratings` arguments to a new evaluation output. Those practice
decisions remain ratings of authored demonstrations, not model results or a
certified independent human panel. Attempt a changed text hash, duplicate ID
and out-of-range score; each must be refused rather than silently corrected.

Keep greedy and sampled recipes separate. Inspect coverage at the declared
checkpoints and compare aligned source openings. Link the paired uncertainty
to the [source-group notebook](../../notebooks/day-10/02_paired_uncertainty.ipynb),
not an independent-row bootstrap over 48 repeated outputs. Exercise 23 and its
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
The measured answer is no: control 400 stops naturally 45/48 times but has
ending mean 0.0208/2. Then inspect the 56 disagreement candidates, per-reader
scores and half-rate-minus-control paired deltas. Check that later missing
checkpoints have null means, not zero. The companion Chapter 7 solution 23
explains why twelve opening groups remain twelve after adding recipes and
raters. Reading or replaying this report needs no checkpoint, GPU or server.

## Actual candidate selection microscope

The [original 18-item panel](../../fixtures/inference-selection/README.md) includes
six balanced train sources and source/template/family failure slices. English
descriptions are not parsed by the symbolic decoder. The
[measured experiment](../../experiments/reports/2026-10-04-inference-selection.md)
retains 864 real candidates, all 240 train-only updates,1736 generated and 1736
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
denominators. The notebook recomputes a full fixed 80-update fit and four fresh
coordinates, then uses the frozen all-seed measured panel for its five figures.
Changing a local selector or budget never edits that historical evidence.

All selector arms pay mandatory likelihood rescoring in this protocol; an
optimized first-answer baseline could avoid it. Its time plots sum observed
uncached CPU attempts plus retrospective selection and are not an optimized
Spark service benchmark. Source siblings remain grouped. Neither a positive
candidate-availability score nor low training loss establishes English reasoning,
pretrained performance, human preference or learner completion.


## Standard-library pretraining evidence replay

## 1. Which clock are we reading?

Before running: why might “tokens per second” have two correct values for the
same experiment?

```python
import json
from pathlib import Path

path = Path('experiments/reports/2026-09-14-tinystories-learning-result.json')
run = json.loads(path.read_text())
m = run['metrics_summary']
print('Valid-position fraction:', m['valid_targets'] / m['processed_positions'])
print('Update-timer targets/s:', m['valid_targets'] / m['sum_update_seconds'])
print('Full-run targets/s:', m['valid_targets'] / run['completion']['seconds'])
```

Reference interpretation: about 21.29% of positions were scored targets. The
two rates are about 4249/s and 4035/s. Both use valid targets; their denominators
cover different work. Neither includes the preceding full data preparation.
Do not infer that changing padding would produce an exactly proportional
speedup: attention, memory traffic, kernels and batching also affect runtime.

## 2. Does prediction improve on the same problem?

Before running: what needs to stay fixed before comparing checkpoint losses?

```python
curve = run['validation_curve']
contract = {
    (x['windows'], x['valid_targets'], x['selection_seed'])
    for x in curve
}
assert contract == {(512, 107264, 909)}
assert all(not x['complete_prepared_split'] for x in curve)
assert all(b['loss'] < a['loss'] for a, b in zip(curve, curve[1:]))
for x in curve:
    if x['update'] in {0, 400, 4000, 8000, 14000}:
        print(x['update'], round(x['loss'], 4))
```

Reference interpretation: the declared development unit is stable and its
recorded loss decreases. These checks corroborate the stored contract; counts
alone cannot independently prove identical target IDs or absence of leakage.
The source identities and implementation record provide additional context.
Nothing in this snippet scores a generated plot.

## 3. Read past the pleasant opening

Before running: does emitting EOS imply that the model resolved its plot?

```python
final = run['samples'][-1]
assert final['update'] == 14000
for sample in final['samples']:
    print('\nMETHOD:', sample['method'], 'TEMPERATURE:', sample['temperature'])
    print('PROMPT:', sample['prompt'])
    print(sample['continuation'])
    print('EOS:', sample['ended_with_eos'],
          'GENERATED TOKENS:', sample['generated_token_count'])
```

Reference interpretation: both final archived completions reach EOS. The greedy
story nevertheless changes speaker roles without explaining them; the sampled
story includes malformed constructions. Ending, sentence fluency, entity
continuity and causal continuity should be assessed separately.

Next, replace `run['samples'][-1]` with each earlier entry. Read both decoding
modes and keep settings visible. The structural checkpoint grid is not a
guarantee of smooth, monotonic improvement in every story property.

## 4. Design the missing comparison

Question: should the next intervention target computation or story coherence?
Write the intended claim first; those are not interchangeable experiments.

Reference systems design: compare separate-document padding with a declared
length-aware policy. Keep model, precision and target exposure fixed; record
data ordering and any context changes. Measure useful throughput, memory and
evaluation invariants. A faster run that changes document visibility is not a
clean demonstration of equivalent computation.

Reference behavior design: freeze a small previously unused prompt panel and
decoding settings, annotate entity/event consistency, then compare a separately
authorized training-budget intervention. Track development loss and complete
generations together. More tokens may help; allow the result to show they did
not fix the target error.

Neither design starts a run. A trained comparison still needs a bounded
specification, approval, and its own result record.

## 5. Can a budget leave unused targets?

Before running: should the trainer remove one target to fit an 8-target group
into 7 remaining places? Inspect the separately collected CPU boundary evidence:

```python
import json
from pathlib import Path

boundary = json.loads(Path(
    'experiments/reports/2026-10-05-story-valid-target-budget.json'
).read_text())
case = next(row for row in boundary['boundary_controls'] if row['cap'] == 16)
assert case['final_cumulative_targets'] == 9
assert case['final_cumulative_physical_positions'] == 16
assert case['refusal']['remaining_valid_targets'] == 7
assert case['refusal']['next_update_valid_targets'] == 8
assert case['rejected_update_forward_calls'] == 0
assert case['refusal_left_state_bitwise_equal']
```

The report retains all cap 8/9/16/17 controls, not just the largest successful
one. Read its `boundary_controls` and find the cap 16 refusal, then compare completed
valid targets, physical positions and remaining allowance. Why are refused
targets not counted as completed learning exposure?

Reference: cap 16 completes 9 valid targets in 16 input positions; it refuses the
next 8-target group with 7 allowance unused. There are zero rejected-update forward
calls, and before/after full-state identities match. Cap 17 completes both groups,
17 valid targets in 32 positions. Changing labels to spend the remaining allowance
would be a different objective. These are actual tiny CPU updates, not additional
TinyStories training or a GPU speed estimate. The source collector and all
controls are linked from the [report](../../experiments/reports/2026-10-05-story-valid-target-budget.md).

## 6. Does restoring a checkpoint restore unused work?

Before running: a 9-target update succeeds, an admitted 8-target attempt fails,
then the old checkpoint is restored. How much allowance does retrying 8 need?
Inspect actual tiny CPU results without loading PyTorch or training again:

```python
base = Path('experiments/reports/2026-10-05-story-work-lesson/run-01')
retry = json.loads((base / 'native-work-example.json').read_text())
refusal = json.loads((base / 'lower-cap-retry-refusal.json').read_text())
assert (retry['successful_targets'], retry['reserved_targets']) == (17, 25)
assert retry['exact_numerical_recovery'] and retry['original_record_equal']
assert retry['later_failed_work_retained']
assert (refusal['successful_targets'], refusal['reserved_targets']) == (9, 17)
assert refusal['retry_refused'] and refusal['next_refusal_forward_calls'] == 0
assert refusal['refusal_numerical_state_unchanged']
assert refusal['refusal_work_prefix_unchanged']
```

Reference: completed exposure rewinds to 9, but the same journal retains the
failed attempt's 8 reserved places. At cap 25 the complete retry fits, producing
17 successful targets and 25 reservations. At cap 24 it does not:7 places remain,
so the whole 8-target group is refused without changing labels or state. A
reservation is not proof that every operation ran; inspect `completed`,
`known_partial` and `uncertain_upper` separately. The failure really follows a
native forward/backward, but precedes the second forward and optimizer call.
See the [lesson report](../../experiments/reports/2026-10-05-story-work-lesson.md)
for the source-bound notebook visual and limits. This teaches recovery semantics,
not story coherence or a physical resource quota.

## 7. Inspect the fresh matched first tranche

Read the [actual two-arm first 400 report](../../experiments/reports/2026-10-05-native-story-first400-comparison.md)
and its source-bound figure. Both arms complete 400 updates and present the same
1,389,548 valid targets in 6,553,600 training positions. Keep those measurements
separate from reservations for validation, generation and other model work.

Explain why `requested_stop_reached=true` and `schedule_complete=false` are
compatible. Trace the plotted learning rates to the original 14,000-update
horizon and 200-update warmup. Then identify which conclusions require the
separate fixed-checkpoint NLL panel and which require independent story ratings.
Missing later checkpoints must not become zeros, interpolated quality or a
claim of convergence. Chapter 6 exercise 21 and its adjacent solution provide
the worked reasoning.

From the repository root, inspect the retained measurements without loading
weights or generating another sample:

```python
import json
import hashlib
from collections import Counter
from pathlib import Path

panel = Path("experiments/reports/native-story-publication-20261005-01")
archive_dir = Path("experiments/reports/native-story-publication-20261005-01-archive")
accepted = json.loads((archive_dir / "acceptance.json").read_text())
raw_archive = (archive_dir / "archive.json").read_bytes()
assert accepted["status"] == "passed" and not (archive_dir / "failure.json").exists()
assert hashlib.sha256(raw_archive).hexdigest() == accepted["archive"]["sha256"]
archive = json.loads(raw_archive)
coverage = json.loads((panel / "coverage.json").read_text())
records = [json.loads(line) for line in (panel / "records.jsonl").read_text().splitlines()]
ratings = json.loads((panel / "ratings-evaluation-01/report.json").read_text())

assert len(records) == coverage["actual_records"] == 192
assert coverage["missing_cells"] == 288
assert coverage["generation_failures"] == 0
for checkpoint in sorted({row["checkpoint_id"] for row in records}):
    rows = [row for row in records if row["checkpoint_id"] == checkpoint]
    scores = ratings["checkpoint_summaries"][checkpoint]["overall"]
    print(checkpoint, len(rows), Counter(row["stop_reason"] for row in rows),
          scores["mean_supplied_scores"]["ending"])
    nll = archive["raw"]["measurements"][checkpoint]["fixed_nll"]
    print("NLL:", {split: (value["nll"], value["valid_targets"], value["windows"])
                   for split, value in nll.items()})

paired = next(row for row in ratings["paired_comparisons"]
              if row["matched_update"] == 400)
print(paired["complete_pairs"], paired["equal_recipe_mean"]["causal_continuity"])
print("Disagreeing candidates:", ratings["disagreement_candidates"])
for row in ratings["paired_comparisons"]:
    if row["matched_update"] > 400:
        assert row["status"] == "incomplete-coverage"
        assert row["equal_recipe_mean"] is None
```

Reference: each available checkpoint has 48 continuations. The two initial
checkpoints reach the token cap in all 48; control at 400 reaches natural EOS
45 times and half-rate 25 times. Their ending means nevertheless remain
0.020833 and 0. Inspect the rubric and a complete continuation before confusing
an EOS token with a resolved plot. Neither mean is a percentage of coherent
stories. There are 56 disagreeing candidates, whose original reader judgments
remain visible rather than being replaced by asserted consensus.

The paired causal-continuity delta is half-rate minus control, −0.104167; its
source-opening percentile interval is approximately [−0.177083, −0.010417].
The consumer uses 800 draws and seed 1010, resampling twelve openings with all
four decoding recipes together. Do not reinterpret 48 paired recipe cells as
48 independent story sources. The equal-recipe mean mixes one greedy and
three temperature-0.8 sampled recipes; their separate strata remain in the
report. Two separate AI readers are not human raters or independently verified
underlying models.

Compare those scores with the report's fixed NLL table: control development
NLL 3.388772 versus half-rate 3.652813, on 64 windows/13,132 valid labels.
The 64 training-source windows contain 13,285 valid labels; neither selection
is its entire split or proof that all training-source windows were presented
in the tranche. These NLL measurements use publication auto-SDPA/BF16 forward
autocast, not the MATH-only deterministic training entry. The
[measurement archive](../../experiments/reports/native-story-publication-20261005-01-archive/archive.json)
retains raw NLL and child/supervisor documents with before/after byte bindings,
so this lab no longer needs ignored native NLL files. Model weights and corpus
bodies remain outside it. Reading the archive is not new inference, training,
an actual Mac execution or full model/corpus rehashing.

This is a CPU-readable evidence session. It does not authorize more training,
infer human consensus or future quality, diagnose overfitting from repetition,
or reopen the original September run. The later 288 planned cells stay missing.


## Preference and reward CPU reproduction

Reproduce from the course root:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 python scripts/run_preference_policy_cpu.py
PYTHONPATH=src python -m dongxi_llms.preference_audit --fixture fixtures/preference-audit/pairs.json
PYTHONPATH=src OMP_NUM_THREADS=1 python -m dongxi_llms.text_reward_lab --fixture fixtures/text-reward/records.json
python scripts/verify_course_notebooks.py --days 15 16 --kernel dgx-spark-native --export-figures
```

The runner prints a deterministic JSON evidence object; it does not overwrite
the historical report. The [predeclared specification](../../experiments/specs/2026-10-04-preference-policy-cpu.md)
and [report](../../experiments/reports/2026-10-04-preference-policy-cpu.md) describe
the actual fixture and measurement. Reading or running these material checks
does not assess learner mastery. A neural reward-model Spark extension should
start with a small independently labeled pair set, frozen source groups,
endpoint-mask tests, and an explicit rubric; no such model-scale run is claimed.

For the collection audit, the [specification](../../experiments/specs/2026-10-04-preference-audit.md)
and [measured report](../../experiments/reports/2026-10-04-preference-audit.md)
retain 484 observations, including every malformed verdict and simulated
transport failure. `--output NEW.json` exports a new full raw ledger and refuses
to overwrite old evidence. Six exact source groups, their family/sibling split
contracts and equal-source pair weights are explicit; no semantic near-duplicate
detector is claimed. Optional human or live-AI collection requires separate
authority, content/privacy terms and independently reviewed held-out labels.

The [text-model specification](../../experiments/specs/2026-10-04-text-reward.md)
and [three-seed report](../../experiments/reports/2026-10-04-text-reward.md)
retain ordinary and nuisance pair predictions, calibration bins, terminal/step
predictions and unknown-token failures. The predeclared seed 1601
[frozen preference export](../../fixtures/text-reward/frozen-preference-seed1601.json)
is a complete JSON-only historical tiny checkpoint, not
a pretrained model. `load_frozen` validates identity/shapes and offers detached
CPU `score(prompt, completion)` and `score_many` without reward gradients.
Use new `--output`/`--frozen-export` paths to retain a new run; old evidence
cannot be overwritten by the runner.

The separately [specified character arm](../../experiments/specs/2026-10-04-text-reward-vocabulary-intervention.md)
preserves the original word artifacts and runs pre-fit pair/swap/source/STEP-
prefix gates. Its [measured report](../../experiments/reports/2026-10-04-text-reward-character.md)
retains all three seeds' failed heldout predictions despite fitted training
objectives. Distinct inputs close the encoding-collision requirement, not a
robust reward-learning claim. For a new run, choose fresh output/export paths:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 python -m dongxi_llms.char_reward_lab \
  --fixture fixtures/text-reward/records.json \
  --protocol experiments/specs/2026-10-04-text-reward-vocabulary-intervention.md \
  --output NEW_CHARACTER_RESULTS.json --frozen-export NEW_CHARACTER_REWARD.json
```

`load_char_reward` validates the fixed ASCII encoding and numeric state and
can enforce an externally expected file/interface hash. The seed 1611 export
is an imperfect detached scalar proxy, not a factual-quality promise. The
extended notebook has twelve complete code cells and eight explanatory
previews. It distinguishes exact saved-state reload from cross-version
retraining; the first historical-word equality failure remains recorded.
Fresh execution preserves source/learner cells. Use an isolated locked kernel,
not a notebook server; the new character previews are additive indices 06–08.


## Critic and frozen text-reward reproduction

## A neural critic and an actual text reward

The exact-value control above stays unchanged. `critic_policy_lab.py` trains a
causal token actor and separate causal value head. Trace reward/TD/GAE tensors,
break the cap mask, then run the adjacent small actor/critic reference with
verified saved rewards. Inspect every fixed balanced/confounded seed. The
[specification](../../experiments/specs/2026-10-04-critic-policy.md) precedes
fitting; the [report](../../experiments/reports/2026-10-04-critic-policy.md)
retains all sampled paths, value errors and quality failures.

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH=src python -m unittest discover -s tests -p test_critic_policy_lab.py -v
CUDA_VISIBLE_DEVICES= PYTHONPATH=src python -m dongxi_llms.critic_policy_lab --report /tmp/NEW-critic-policy.json --exports /tmp/NEW-frozen-rewards
```

Use unused output paths. Conditional syntax/cap-four continuation are explicit,
not pretrained critic evidence, production PPO, human feedback or demonstrated
color-rule transfer. Saved plots are measured previews; rerun for interventions.

