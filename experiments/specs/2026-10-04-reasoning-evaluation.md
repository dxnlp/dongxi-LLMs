# Offline mathematical grading and response replay specification

Date:2026-10-04. Mode: CPU instrument; no model generation. Placement: Chapter7
and Day10. Hypotheses are fixed before the bounded replay and final notebook
verification; previous construction-time unit checks are development checks.

## Questions and hypotheses

Exact rational arithmetic should equate a half with its finite decimal while
rejecting a rounded third as an exact third. Set order should not matter, but
interval boundaries and units should. Unsupported expressions and ambiguous
answer markers should remain visible, rather than be rescued by picking the
last number. An authored aggregate gain should coexist with the deliberately
constructed set/JSON regressions.

## Inputs and controls

Original source: `fixtures/reasoning-evaluation/`. Fifteen development items,
fourteen source groups and thirty authored responses. Two fraction items share
a source group. Twenty-three adversarial grading cases cover exact arithmetic,
nested fractions, sets, intervals, units, unsupported syntax and input bounds.
No third-party text, dataset or model weights are copied or downloaded.

Parser version: `bounded-rational-set-interval-v1`. Rubric version:
`explicit-marker-fixture-v1`. Response schema: `dongxi-response-record-v1`.
Raw response cap16,384 characters; extracted answer cap2,048 characters;
arithmetic token cap256; depth12; collection cap64; literal digit cap64;
intermediate rational numerator/denominator cap4,096 bits. No `eval`, `exec`,
SymPy, arbitrary code execution, unit conversion or fuzzy semantic fallback.

Both authored panels use the identical suite/settings hash, attempt IDs and
source groups. Metrics separate answer correctness, required format, natural
termination and truncation. Errors/unsupported/ambiguous rows remain in the
all-record denominator. Tokens/costs are explicitly unknown, not inferred.
`task_success` is answer correctness AND required format; natural termination
is reported separately rather than silently added to that metric.

## Execution and acceptance

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 python -m unittest discover -s tests \
  -p test_reasoning_evaluation.py -v
python scripts/evaluate_reasoning_records.py \
  --compare authored-baseline authored-candidate --draws 2000 --seed 1010
python scripts/verify_course_notebooks.py --days 10 \
  --kernel dgx-spark-native --export-figures
python scripts/check_book_math.py
```

Use the existing verified interpreter; no installation or GPU process is part
of the specification. The notebook verifier writes executed copies in a new
temporary directory, not into learner source notebooks.

Acceptance: all reviewed grammar cases and schema/hash failures behave as
declared; the existing strict-integer verifier is unchanged; raw/error records
are retained; selectors receive no gold labels through `candidate_view`; paired
comparison rejects mismatched IDs and resamples complete source groups. Use
2,000 draws, seed1010, paired source-group resampling and micro-average B−A as
the estimand. Freeze source identity and actual execution results in the report.
Do not discard any response when parsing fails.

## Evidence boundary

This verifies a bounded instrument and an authored regression panel. It does
not measure generation latency, model accuracy, benchmark contamination,
general mathematical equivalence, rationale faithfulness or broad safety.
The local generation adapter and actual pretrained checkpoint comparison are
pending; this specification does not complete the entire improvement package.
