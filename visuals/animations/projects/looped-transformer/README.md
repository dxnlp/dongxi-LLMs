# ANIM-LOOP-001 — Fixed recurrent depth

User approved the fixed-recurrence first act on 2026-09-09. Production is local
Mac Studio, branch `codex/visuals/looped-transformer`, from main after a clean
fast-forward sync. Exact base revision is retained in metadata.json.

Source: Chapter 5 section 5.20; Day 7 `01_recurrent_depth.ipynb`;
`src/dongxi_llms/decoder_lab.py`; CAND-ANIM-011. Follow the packet in
LEARNING_MEMORY.md and the shared animation style guide.

## Predeclared CPU smoke check

Use the notebook's seed 505, default DecoderConfig, CPU float64, and state shape
[2,6,16]. Reuse one DecoderBlock for three applications. Predict: parameters
stay identical, hidden states change, equal-valued independent copies give the
same result, and their parameter gradients sum to the shared parameter gradient.
Stored block parameter count is constant; dense matrix multiply FLOPs grow with
applications for fixed shapes. No training or timing comparison is attempted.

Display batch 0 as a 6×16 signed activation heatmap with a common scale across
all states. Blue/purple denote negative/positive values; brightness is magnitude.
No color indicates quality. Animate one packet through a stable Excalidraw block;
Manim supplies naturally kerned text, counters, and movement. Keep production
metadata outside the learner-facing animation and HTML.

Scope excludes adaptive routing, article drafting, model-quality comparisons,
GPU execution, publishing, commit and push. MP4s remain ignored and local.

## Reproduce

Base revision: `8e39e98cd45e024416041aed3015495f02d6a240`.
Install the pinned exporter from this folder with
`pnpm install --frozen-lockfile`. Node, Chrome and FFmpeg must be available.
From the repository root, run these steps sequentially:

```bash
uv run --project visuals/animations --with torch==2.10.0 python visuals/animations/projects/looped-transformer/build_trace.py
uv run --project visuals/animations python visuals/animations/projects/looped-transformer/build_sketches.py
node visuals/animations/projects/looped-transformer/export.mjs
uv run --project visuals/animations python visuals/animations/projects/looped-transformer/render.py
node visuals/animations/projects/looped-transformer/verify.mjs
```

The extra Torch dependency is isolated by uv; the course's base animation lock
is unchanged. `--preview` renders 960×540/15fps; the default is 1920×1080/30fps.
Wait for sketch export to finish before rendering. An initial smoke render
started before export finished and failed on the missing SVG; the renderer now
checks those prerequisites before starting.

Excalidraw 0.18.0 supplies seeded, editable MIT-licensed geometry; Manim 0.21.0
supplies typography and motion. No external artwork was copied. Numerical data
comes directly from the canonical decoder implementation, with a source hash.
`metadata.json` records environment, source and output hashes; `qa.json` records
repeat-export and browser playback checks. The review page shows teaching content
only. Production metadata and caveats about the renderer stay in this README.

## Measured smoke result

- 2,160 unique stored block parameters; three independent copies store 6,480.
- 53,760 dense matrix-multiply FLOPs per full-batch application, or 161,280
  for three. No runtime measurement is inferred from this arithmetic.
- Shared versus equal-valued independent-copy forward difference: exactly zero.
- Maximum gradient-sum discrepancy: 1.1102230246251565e-16.
- State-change norms: 4.9691, 5.8123, 6.4700; these are changes, not quality scores.
- All eight declared numerical checks passed on CPU float64, seed 505.

Known boundaries: only batch 0 is displayed; no training, output-token generation,
adaptive depth, quality comparison, latency benchmark or memory experiment. The
gray internal Attention/FFN shapes abbreviate the full pre-norm residual block.
The repeated parameter set remains fixed through every animated forward pass.

## Return evidence

Final render: 41.333333 seconds, 1920×1080, 30fps, H.264/yuv420p. All eight
numerical checks pass; six SVG/editable-scene files reproduce byte-identically.
All manifest hashes match. Eight geometry checkpoints, full/half stills and nine
transition samples were inspected. Chrome played and sought the video; the page
has no browser errors or mobile horizontal overflow. The first fixed-delay
playback check ran too early; the final check waits for actual media advancement.
No production metadata appears in the final video or learner-facing page.

Review entry: `review.html`. Source is ready for user feedback and remains
uncommitted. The article and adaptive-recursion act are separate future work.
