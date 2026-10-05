# Original response replay fixtures

This is an authored instrument test, not generated output from Qwen, DongxiGPT,
or another model. `authored-baseline` and `authored-candidate` are two deliberately
constructed response panels. Neither identifies trained weights. Every item is
development material, because its purpose is to inspect and change the grader.

`items.json` contains fifteen original capability/interface items from fourteen
source groups. Two fraction variants share a group. `settings.json` freezes the
template, thinking declaration, non-generation mode, stop definitions and cap.
`responses.jsonl` contains thirty authored responses; its contract ID is the
hash of that suite and those settings. Do not silently rewrite this ID after
editing inputs. Create a named new contract and record the migration.

`contract.json` is the explicit frozen serialization of those same unchanged
items/settings for offline card export. Its identity is the existing response
identity; adding this file does not create a generation run or new results.

The simulated error and max-token stop are deliberately assigned fixtures, not
an actual OOM or measured generation event. Token IDs, token counts and costs
are unknown (`null`), not estimated from string length. The replay measures
grading and aggregation only.

`grading_cases.json` adds twenty-three reviewed exact-equivalence and adversarial
cases. Numeric answers use exact rationals, including decimal literals and
nested fractions. Finite sets ignore order and duplicate elements; intervals
preserve open/closed boundaries. Named unit aliases are matched, not converted:
`3 seconds` and `3 s` agree; `100 cm` and `1 m` do not under this contract.
Variables, roots, unknown units and arbitrary code are explicitly unsupported.

The benign-process and false-arithmetic-premise rubrics have positive and
negative responses. Their required/forbidden phrase checks teach how a rubric
can become executable. They cannot determine nuanced helpfulness, detect all
over-refusal or sycophancy, or certify safety. A general evaluation requires
independently reviewed labels and a stronger instrument.

The candidate has better aggregate fixture scores but deliberately regresses on
set membership and JSON validity. Preserve both failures. The apparent gain and
its paired interval describe these authored panels, never model capability.

Replay without a model, GPU, API key or network:

```bash
python scripts/evaluate_reasoning_records.py \
  --compare authored-baseline authored-candidate --draws 2000 --seed 1010
```

The CLI prints a complete graded ledger, task/source/split slices, unknown costs,
missing-item IDs and source-group paired uncertainty. The original raw answers
and errors remain in every graded row. A separate local-only generation adapter
has random-model CPU evidence; actual pretrained-model evaluation remains
pending. [Lab 7](../../book/labs/07-evaluation-is-a-contract.md) also exports an
evidence-limited card from this exact authored contract and ledger. Neither
instrument replay nor card formatting creates a pretrained-model event.
