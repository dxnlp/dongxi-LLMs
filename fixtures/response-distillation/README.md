# Original response sequence distillation

The original18-item fixture teaches a symbolic parity response with an explicit
printed total and final binary answer. English strings are documentation, not
inputs parsed by the tiny model. Train sources are balanced; direct/alias test
siblings stay related. Sum-positive test references are intentionally imbalanced.

The [frozen protocol](../../experiments/specs/2026-10-04-response-distillation.md)
requires actual neural teacher/student generation. Authored truth-table probes
are separately labeled and never inserted into teacher training selections.
Verified printed arithmetic is not causal reasoning faithfulness. No upstream
prose/code/data is copied and no pretrained result is implied.
