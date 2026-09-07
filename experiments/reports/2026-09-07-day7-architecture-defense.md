# Day 7 architecture-defense notebook — verification report

## Result

Added the core architecture-defense notebook alongside the existing optional
recurrence notebook, preserving the older path and all previous notebook cells.
The new notebook has 37 cells, including 16 code cells and four saved figures.
The expanded Chapter 5 suite passed all twelve fresh-kernel executions:
168 code cells and 52 figures. Runner elapsed time was 32.99 seconds, including
kernel startup and the existing fixed training sanity check, not model latency.
All 53 repository tests passed in 3.595 seconds. Both processes exited 0.

Specification: `experiments/specs/2026-09-07-day7-architecture-defense.md`.
Adjacent JSON records environment identity, source/notebook/test hashes,
per-notebook execution results, new asset hashes, and the fixed training trace.
Executed copies: `/tmp/chapter5-reference-hmfh39x8` (temporary reproduction
outputs; not a dependency for the portable notebook).

## Observations

- Actual module traces confirm Q output `[2,6,16]`, K output `[2,6,8]` before
  head splitting, MLP expansion `[2,6,32]`, and logits `[2,6,16]`.
- The modern model contains 4,960 unique parameters; its compact full cache is
  3,072 float64 bytes. Estimated dense forward matmul FLOPs: 125,952.
- Its untrained mean loss is 2.798485622783327. All 24 parameter tensors have
  connected finite gradients on this fixture; no optimizer step was taken and
  weight values remain unchanged.
- Reference: future-invariance and cached-suffix maximum errors both 0.0.
- Wrong rotation offset: future error 0.0, cached-suffix error
  0.032522888924928016. Finite and causal does not imply correct caching.
- Full-time centering: future error 0.00034718220923955767, cache error
  0.0003740826436335297. All logits remain finite despite the noncausal operation.
- Existing one-batch training sanity check remains unchanged: loss
  2.774792432785034 to 0.0007856183219701052, accuracy 1.0, seed 505,
  160 AdamW steps. This belongs to the previous batch-fitting contract, not
  the new notebook's unexecuted recurrence proposal.

## Review and limits

New tests cover unique-parameter accounting, trace cleanup on success/error,
unchanged parameters/RNG, expected diagnostic signatures, audit input checks,
actual bar widths, and pass/fail tiles. Saved budget and diagnosis figures were
visually inspected; architecture maps reuse the established modern-model
schematics. All 52 preview assets were exported from verified executed copies.
Source notebooks were not overwritten and original notebook hashes remain intact.

Final review checked all four new images, 139 local links/figure paths, and
`git diff --check`. Previously committed `.ipynb` files have no changes.

This is a CPU mechanism audit, not learner mastery, a quality evaluation, a
measured serving benchmark, or a trained recurrence comparison. The final
worksheet is a proposal requiring concrete data/budget/measurement choices
before execution. No new large model, training campaign, article publication,
or animation production occurred. Day 6 prose changes remain preserved.

The kernels emitted local TCP-encryption warnings; no security property follows
from passing numerical checks. The interactive server was not restarted or stopped.
