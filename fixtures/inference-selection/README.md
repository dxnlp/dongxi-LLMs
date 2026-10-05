# Original answer selection fixtures

`items.json` will contain independently authored parity and sum-positive tasks.
Their English prompts describe the task; the tiny decoder receives only a
declared four-token symbolic representation. This is not natural-language
reasoning data. Training sources are balanced and held-out template siblings
remain grouped. The sum-positive failure slice has a three-to-one answer
imbalance, so constant-answer baselines must remain visible.

The [premeasurement protocol](../../experiments/specs/2026-10-04-inference-selection.md)
fixes the model, seeds, training budget, sampling, selectors and cost boundaries.
Original fixtures and logic are not copied from either reference repository.
