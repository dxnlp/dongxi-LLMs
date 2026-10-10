# Lab — Read a completed run without retraining

Companion to [Chapter 6, sections 6.14–6.21](../chapters/06-pretraining-as-a-controlled-system.md).
This is a lightweight evidence-reading lab, not a new training experiment or
an already-built Day 9 notebook. It needs only Python's standard library and
the repository's compact report, so it can run on Mac or Spark without PyTorch,
model weights, Jupyter, or a dashboard server.

Run the snippets from the repository root in a Python session. Form a prediction,
then inspect the adjacent reference and explanation.

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

Reference interpretation: about21.29% of positions were scored targets. The
two rates are about4249/s and4035/s. Both use valid targets; their denominators
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

Before running: should the trainer remove one target to fit an8-target group
into7 remaining places? Inspect the separately collected CPU boundary evidence:

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

The report retains all cap8/9/16/17 controls, not just the largest successful
one. Read its `boundary_controls` and find the cap16 refusal, then compare completed
valid targets, physical positions and remaining allowance. Why are refused
targets not counted as completed learning exposure?

Reference: cap16 completes9 valid targets in16 input positions; it refuses the
next8-target group with7 allowance unused. There are zero rejected-update forward
calls, and before/after full-state identities match. Cap17 completes both groups,
17 valid targets in32 positions. Changing labels to spend the remaining allowance
would be a different objective. These are actual tiny CPU updates, not additional
TinyStories training or a GPU speed estimate. The source collector and all
controls are linked from the [report](../../experiments/reports/2026-10-05-story-valid-target-budget.md).

## 6. Does restoring a checkpoint restore unused work?

Before running: a9-target update succeeds, an admitted8-target attempt fails,
then the old checkpoint is restored. How much allowance does retrying8 need?
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

Reference: completed exposure rewinds to9, but the same journal retains the
failed attempt's8 reserved places. At cap25 the complete retry fits, producing
17 successful targets and25 reservations. At cap24 it does not:7 places remain,
so the whole8-target group is refused without changing labels or state. A
reservation is not proof that every operation ran; inspect `completed`,
`known_partial` and `uncertain_upper` separately. The failure really follows a
native forward/backward, but precedes the second forward and optimizer call.
See the [lesson report](../../experiments/reports/2026-10-05-story-work-lesson.md)
for the source-bound notebook visual and limits. This teaches recovery semantics,
not story coherence or a physical resource quota.

## 7. Inspect the fresh matched first tranche

Read the [actual two-arm first400 report](../../experiments/reports/2026-10-05-native-story-first400-comparison.md)
and its source-bound figure. Both arms complete400 updates and present the same
1,389,548 valid targets in6,553,600 training positions. Keep those measurements
separate from reservations for validation, generation and other model work.

Explain why `requested_stop_reached=true` and `schedule_complete=false` are
compatible. Trace the plotted learning rates to the original14,000-update
horizon and200-update warmup. Then identify which conclusions require the
separate fixed-checkpoint NLL panel and which require independent story ratings.
Missing later checkpoints must not become zeros, interpolated quality or a
claim of convergence. Chapter6 exercise21 and its adjacent solution provide
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
