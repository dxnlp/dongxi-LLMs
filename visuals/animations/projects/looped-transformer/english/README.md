# English recurrent-depth animation series

User approved multiple English animations after the Raschka article comparison
on 2026-09-09. This expands ANIM-LOOP-001 into three standalone explanations:

1. One block, three applications: translate/refine the existing introduction.
2. Three distinct blocks, two stack passes: reuse A/B/C in the same order.
3. Unfold six applications and pair A/A, B/B, C/C by parameter identity.

Predeclared CPU float64 checks, seed 505: three distinct blocks store 6,480
parameters; two passes execute six applications; six independently allocated
copies store 12,960 parameters, initially compute equal outputs, and paired copy
gradients sum to each shared block gradient. Reuse the canonical decoder module.
Do not claim trained quality, wall-clock performance, or universal optimal depth.

Keep the Chinese original and all its rendered artifacts intact. Media stays local
and MP4s stay ignored. One English review page links all three individual players.
Excalidraw assets are reused from the parent project with native Arial typography;
tool names, internal identifiers, and production status belong only in this file
and manifests. No public article is drafted or published by this request.

## Reproduction

Use the parent project's pinned Manim environment and installed Node exporter
dependencies. The existing parent Excalidraw block SVG is reused by all clips;
no new SVG export is required for this English series.

From the repository root:

```bash
uv run --project visuals/animations --with torch==2.10.0 python visuals/animations/projects/looped-transformer/english/build_trace.py
uv run --project visuals/animations python visuals/animations/projects/looped-transformer/english/render.py
node visuals/animations/projects/looped-transformer/english/verify.mjs
```

Add `--preview` to the renderer for the 540p preview series. Default output is
1080p/30fps. `review.html` lets the reader select any of the three videos;
`01-one-block.html`, `02-repeat-stack.html`, and `03-unfold-sharing.html` are
standalone players. Each video has a GIF, full/half still and contact sheet.

The first clip uses the parent's verified single-block trace. The other two use
the local three-block trace. All use CPU float64, seed 505; the two fixtures have
different initial hidden states because initializing three blocks consumes a
different random sequence. Colors use a common scale within each fixture.

The renderer hashes every Chinese original source/output it may encounter and
asserts preservation after all English renders. `metadata.json` records sources,
media, versions and checks. `qa.json` records selector, playback, individual-player,
mobile layout, English-language and production-label checks.
Use `--only 03-unfold-sharing` (or another clip ID) to revise one existing clip
without rerendering the other two; its manifest is refreshed for the whole series.

## Mechanism evidence

Three distinct parameter sets: 6,480 values. Six independent application copies:
12,960. Both execute six block applications, totaling 322,560 dense matrix FLOPs
for this batch. With identical copied weights, their forward outputs agree
exactly; paired gradients sum to the shared gradients with maximum discrepancy
2.7755575615628914e-17. All nine new fixture checks pass. This is an initialization
identity, not a trained equivalence or quality result.

The unrolled drawings are six applications of three blocks. Lines and matching
colors preserve A/A, B/B, C/C identity; creating an extra drawing allocates no new
model weights. Labels 1–6 count applications, not unique parameter sets.

Editorial reference: Sebastian Raschka's discussion of shared blocks, repeated
stacks and unrolled computation in
[GPT-6 Astra, Looped Transformers, and Hidden Reasoning](https://magazine.sebastianraschka.com/p/gpt-6-astra-looped-transformers-and).
The drawings are independently authored from our course example. No paper figure
was copied, and no vendor architecture or benchmark claim appears in the series.

Visual revision: the unfolded second-pass drawings now travel above the first
pass before settling into place, preventing the label crossings seen in the
initial transition. All three final clips have verified browser playback,
standalone players, and English-only presentation pages. Final approval remains
with the learner; files are local and uncommitted.
