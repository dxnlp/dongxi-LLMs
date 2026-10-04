# Days 3–4 visual extension verification

On 2026-10-04, append-only prediction, reference interpretation and plotting
cells were added to all six existing Chapter 3/4 notebooks. Original lesson
cells and learner answers were retained. Day 3 Session 1 now distinguishes
its standard-library mathematical core from the Matplotlib visual extension.

The fresh-kernel verifier executed all six reference paths successfully and
exported seven actual PNG figures. The adjacent JSON preserves source hashes,
code-cell counts, generated-image hashes and the CPU environment identity.
Incomplete tagged learner scaffolds were skipped by the verifier; their nearby
reference solutions were executed. This establishes reference readiness, not
completion of the learner's exercises.

The figures show probability/target/gradient competition, causal-shift copying
failure, prompt backward paths, 70/30 entropy convergence, future-intervention
attention failures, detached projection routes, scaling saturation, and KV
payload/work accounting. Their values come from the existing exact tensors,
recorded experiment checkpoints or declared paired simulations. Work counts
are not hardware speed benchmarks.

Saved previews are linked in the appended notebook Markdown. Live plot cells
regenerate them from current reference state. Two representative full-size
images—the shift/gradient plot and causal intervention plot—were visually
inspected for readable labels and unclipped content. Other figures passed
execution/export checks; whole-course figure review is recorded separately.

Reproduction:

```bash
OMP_NUM_THREADS=1 python scripts/verify_course_notebooks.py \
  --days 3 4 --kernel dgx-spark-native --export-figures
```

No training service, GPU campaign or animation was launched. Animation
production remains separately approved Mac Studio work.
