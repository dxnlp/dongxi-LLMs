# Lab — Read a completed run without retraining

Companion to [Chapter 6, sections 6.14–6.20](../chapters/06-pretraining-as-a-controlled-system.md).
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
