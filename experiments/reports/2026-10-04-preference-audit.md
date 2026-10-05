# Preference collection and judge audit results

This experiment verified an offline collection instrument on original authored
texts, authored reference labels and deterministic simulated judges. No language
model, human annotation study, paid service, credential or GPU was used. The
[specification](../specs/2026-10-04-preference-audit.md) preceded reference
execution. [Full JSON evidence](2026-10-04-preference-audit.json) preserves all
484 raw observations, exact presentations, review labels, identities and failures.

## Inputs and collection

The original [fixture](../../fixtures/preference-audit/README.md) has eight
base pairs covering six source groups. Three story turns share one source and
conversation. Train, calibration and test source/family groups remain disjoint.
Left-answer verbosity and untrusted-instruction variants expand the fixture
to twenty-four pairs. Each of five explicitly simulated rules supplies two
repeats in both blind display orders: 96 observations per rule. Three malformed
verdicts and one simulated transport failure form a separate control identity.

Authored reference labels were specified separately from the toy rules; they
are not independent human feedback. Perturbation labels remain unchanged under
the factual-answer rubric by construction. In a live study, independently
review the variants rather than assuming an intended nuisance preserves quality.
The content-rule judge uses a declared finite keyword list, making it a narrow
positive instrument control rather than an intelligent evaluator.

## Measured behavior

Outcome agreement includes reviewed ties/abstentions and counts invalid
observations as disagreements. Decisive agreement includes only comparisons
where both review and judge choose a side. Each ordinary rule has 72 decisive
observations out of 96, with 12 ties and 12 abstentions. Order consistency compares
48 matched AB/BA observation pairs per ordinary judge. Repeat disagreement
compares 48 fixed-pair/fixed-order repeat pairs. Each nuisance change rate has 32
matched baseline/variant observations; the authored reference never changes.

| Simulated rule | All-outcome agreement | Decisive agreement | Order consistency | Repeat disagreement | Verbosity change | Injection change |
|---|---:|---:|---:|---:|---:|---:|
| Content rule | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| First slot | 0.6250 | 0.5000 | 0.2500 | 0.0000 | 0.0000 | 0.0000 |
| Longer answer | 0.4583 | 0.2778 | 1.0000 | 0.0000 | 0.3750 | 0.2500 |
| Injection sensitive | 0.8750 | 0.8333 | 1.0000 | 0.0000 | 0.0000 | 0.3750 |
| Repeat unstable | 0.6250 | 0.5000 | 1.0000 | 0.7500 | 0.0000 | 0.0000 |

The first-slot rule selects A on every decisive observation, giving a
first-position choice rate 1. Its order consistency is not zero because reviewed
ties/abstentions are handled identically in both directions. The longer-answer
and injection-sensitive rules illustrate why passing an order test is not
sufficient. Repeat instability is visible even when each individual display
order is handled consistently.

The longer-answer rule also changes on 25% of injection variants: adding the
instruction increases text length. An observed injection-change rate alone
therefore does not establish that a live judge followed the instruction. These
toy rules expose their decision logic, but a future causal study needs a
matched-length benign-text control in addition to independent factual review.
This is a limitation of attribution, not a reason to discard the failures.

All four failure controls are invalid: three preserve parse-stage errors and
their original raw strings, and one preserves a simulated transport-stage
timeout. None is reclassified as a tie or abstention. Metrics with no eligible
denominator are null, not fabricated zeros. Invalid-control repeat disagreement
is zero because their canonical category is consistently invalid; it must not
be interpreted as successful evaluator reliability.

## Weighting and grouping

The box-story source contributes nine expanded pair rows; each receives
weight 1/9. Each one-turn source contributes three rows with weight 1/3. Total
pair weight per source is exactly one before global normalization. The
notebook's separate authored demonstration has row-weighted outcome agreement
0.5 and source-balanced agreement 1/3 on the same observations. It correctly
judges every box turn and correctly abstains on the unobservable source. The
metric difference is a declared population-weighting difference, not changed
raw predictions.

The auditor rejects source/family split collisions, cross-source conversational
siblings, ambiguous base/condition mappings, content-ID collisions, duplicate
record IDs and a judge grading against its own reference identity. It also
replays retained raw evidence: editing cached outcomes, verdict strings or
presentation mappings without matching raw proof is rejected. Exact IDs do not
discover semantic near duplicates; that needs a separate contamination rule.

## Verification and visuals

Actual reference execution used
`/home/dongxi/dgx-spark-dongxi/.venv/bin/python` on Linux/aarch64, Python 3.12.14,
with CPU-only fresh `dgx-spark-native` kernels. The eighteen independent
`test_preference_audit.py` tests pass. They cover blind/canonical swaps,
strict/duplicate-field parsing, transport retention, provenance, ordinal
rating/ranking conversion, split/sibling/candidate identity, equal-source
weights, duplicate/forged evidence, missing denominators and simulated controls.

The [new notebook](../../notebooks/day-15/03_preference_collection_and_judges.ipynb)
executes eight complete code cells and emits three plots: the metric heatmap,
full outcome counts and source-weighting comparison. Each was visually
inspected. A count-label/legend overlap in the first generated count chart was
repaired and rerendered; the final PNGs keep labels readable. Existing Day 15
notebooks also passed in the same verifier invocation: 16 code cells and 7 images
across all three sessions, with no skipped exercises or execution failures.
Source notebooks were not overwritten with executed copies.

Commands from the course root:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p test_preference_audit.py -v
PYTHONPATH=src python -m dongxi_llms.preference_audit --fixture fixtures/preference-audit/pairs.json
python scripts/verify_course_notebooks.py --days 15 --kernel dgx-spark-native --export-figures
python scripts/check_book_math.py
```

The [verification JSON](2026-10-04-preference-audit-verification.json) records
actual source/output hashes, interpreter/packages, notebook manifest, measured
reference-replay duration and result equality. Reported memory is sampled host
availability before notebook execution, not continuous minimum or peak memory.

## Evidence boundary

This establishes collector and audit correctness on declared original controls.
It does not establish human agreement, AI judge quality, causal attack success
on a live model, reward-model improvement, statistical population confidence
intervals, Mac execution or learner mastery. No model-scale or live collection
campaign ran. The learner remains at Day 9; these are prepared course companions.
