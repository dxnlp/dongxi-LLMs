# Offline mathematical grading and response replay results

Executed2026-10-04. This is a CPU instrument test on original authored fixtures,
not an evaluation of a trained model. The [specification](../specs/2026-10-04-reasoning-evaluation.md)
defines the questions and bounds; the [complete JSON record](2026-10-04-reasoning-evaluation.json)
retains environment, commands, current source hashes, every replayed row, raw
errors and notebook verification. Base Git revision:
`ac203efca4ca4a5b42cb7e1a04dbe3a923559ab4`; source edits were uncommitted during the checks.

## What was executed

The offline CLI exited0 after grading thirty authored records from fifteen
development items and fourteen source groups. It used2,000 paired source-group
bootstrap draws, seed1010. The observed CLI invocation took
0.030331 seconds on this local CPU path; that is replay
latency, not model generation latency. CUDA was hidden for the captured CLI/test
invocation, and no model or API was used.

All14 targeted unit tests passed, including23 reviewed grammar/adversarial
cases, strict-integer regression, nested boxed extraction, ambiguity,
set/interval/unit identities, resource limits, schema/hash/stop/cost failures,
JSON duplicate keys, non-oracle candidate views and grouped pairing. The numeric
case outcomes were8 correct-equivalent,5 supported unequal,5 invalid and5
unsupported. The parser does not execute generated text or use SymPy.

All four Day10 notebooks passed fresh-kernel verification using
`dgx-spark-native`; the new response-replay notebook executed6 code cells and
produced4 data/schematic PNGs. Its four saved reference figures were inspected
at full size: labels and item IDs are readable, fixed reference plots are
distinguished from live outputs, and the process schematic is labeled as such.
The existing three notebook sources were preserved. The verification manifest
is identified in the JSON report; final course-wide integration verification is
a separate check. A local kernel transport warning was emitted during notebook
execution; no notebook web server was started by this experiment.

The book-math source check passed with no issues after the Chapter7 additions.
The checker does not establish GitHub's live rendering.

## Observed authored panel scores

| Metric | Authored baseline | Authored candidate |
|---|---:|---:|
| Answer correctness | 7/15 | 13/15 |
| Task success: answer AND required format | 6/15 | 13/15 |
| Required format valid | 13/15 | 14/15 |
| Assigned natural termination | 13/15 | 15/15 |
| Assigned max-token truncation | 1/15 | 0/15 |

These are computed grades of intentionally written responses, not learned
capabilities. Candidate set membership regresses from1/1 to0/1. Candidate JSON
validity regresses from1/1 to0/1. The aggregate improves despite these retained
failures. The baseline also retains one ambiguous answer, one unsupported
expression and one simulated execution error. The candidate retains one invalid
JSON response. No row is discarded to improve the denominator.

The paired source-group micro-average task-success difference is
0.4666666667; the percentile interval is[0.125,0.8]. Related fraction variants
travel together during resampling. A positive interval on a deliberately
constructed fourteen-group development panel does not demonstrate a real model
gain or justify a population-level capability claim.

Every row has unknown token IDs, token counts and generation/scoring costs.
The summary's known subtotal0 is accompanied by15 unknown rows per panel; it
does not claim zero-cost inference. The truncated and error rows are assigned
fixtures, not an observed EOS, OOM or actual generation event.

## Mechanism checks and limits

One balanced boxed answer can contain nested rational expressions. Multiple
explicit answer markers are ambiguous, even when their values agree. Exact
finite decimals are stored as fractions; a rounded third is not exactly a
third. Finite sets ignore order and duplicates. Interval boundaries and named
units survive canonicalization. Unit aliases match, but conversions do not:
`100 cm` and `1 m` are unequal under this limited contract.

Whitespace cannot merge separate numeric atoms: `1 2` is invalid rather than
silently becoming12. Variable algebra, roots, unknown units and arbitrary code
remain unsupported. An unsupported/ambiguous answer does not automatically fail
an unconstrained format criterion; parser coverage and task interface are scored
independently. Size, nesting, token, element, digit and rational-magnitude
bounds limit accepted parsing work. Unsupported answers expose grader coverage
and are not equivalent to a proven mathematical error.

The required/forbidden phrase rubrics distinguish only the authored positive
and negative examples for benign process instructions and disagreement with a
false arithmetic premise. They cannot certify general helpfulness, truthfulness,
over-refusal behavior or safety. The non-oracle candidate projection excludes
reference and grade fields; this is an interface check, not proof that a future
ranker is statistically valid.

The shared frozen record/replay schema is ready for later integration. A
separately invoked local generation adapter, actual checkpoint identities,
measured token/stop/cost observations and independently reviewed model responses
remain pending. The whole frozen-evaluation improvement package is therefore
**partial**, not complete.

## Reproduce and study

Use the commands in the [lab](../../book/labs/07-evaluation-is-a-contract.md)
and [fixture guide](../../fixtures/reasoning-evaluation/README.md). Read
[Chapter7 sections7.10–7.11](../../book/chapters/07-evaluation-is-a-contract.md)
and its [worked solutions11–14](../../book/solutions/07-evaluation-is-a-contract.md),
then open the [fourth Day10 notebook](../../notebooks/day-10/04_mathematical_grading_and_response_replay.ipynb).
The exercise changes a local response and reveals its answer/format distinction
without changing the frozen fixture suite or claiming a new model run.
