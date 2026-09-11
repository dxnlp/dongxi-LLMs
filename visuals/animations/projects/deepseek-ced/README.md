# DeepSeek CED and shared-KV films

Two English animations approved for implementation and rendering on 2026-09-10.
Task packets: ANIM-CED-001 and ANIM-KV-002 in `LEARNING_MEMORY.md`.

Review refinement, 2026-09-10: the learner requested the full term
“Causal Encoder–Decoder” and fewer words. The first film now uses that full title;
both films omit all sentence-style bottom captions, shorten operation headings,
and retain only a compact toy-state qualifier. The model head count moved to
expandable page notes. Numerical traces and tensor motion are unchanged except
for removed caption/head-count presentation time. Exact current durations are
recorded in `metadata.json` and `qa.json`.

Typography refinement, 2026-09-10: all scene labels now opt into 16x internal
whole-string shaping and uniform vector downscaling. The original tiny Pango
layout rounded glyph advances visibly, especially in small labels and decimals.
Arial, normal weight, text content, timing, motion and the original architecture
poster are retained. `check_typography.py` produces a before/after comparison and
`typography-qa.json`, comparing glyph advances against a 32x reference. The
shared helper is opt-in so no other animation is silently restyled.

- [Series player](review.html).
- [Where global KV comes from](01-global-kv.html).
- [One KV representation, two roles](02-shared-roles.html).
- [Approved design and precision boundaries](DESIGN.md).

## Reproduce on Mac

Both players use the original report's Figure 3 as their shared architecture
cover. This is an HTML poster; adding it did not alter either MP4. The diagram is cropped
from PDF page 7, uniformly resized and centered on white, without redrawing.
Attribution is displayed beneath each player. The pinned source, PDF hash,
extraction coordinates and repository MIT license are retained in `assets/`.

To regenerate the cover in a Python environment with `pypdfium2` and Pillow:

```bash
python3 visuals/animations/projects/deepseek-ced/build_poster.py
uv run --project visuals/animations python visuals/animations/projects/deepseek-ced/render.py --pages-only
```

The extractor accepts `--pdf /path/to/report.pdf` for an existing copy of the
pinned report. `--pages-only` verifies existing media hashes before refreshing
players and their manifest; it never renders or rewrites the videos.

From the repository root:

```bash
uv run --project visuals/animations python visuals/animations/projects/deepseek-ced/build_trace.py
uv run --project visuals/animations python visuals/animations/projects/deepseek-ced/check_typography.py
uv run --project visuals/animations python visuals/animations/projects/deepseek-ced/render.py
node visuals/animations/projects/deepseek-ced/verify.mjs
```

Use `render.py --preview` for a 960x540/15fps draft without GIF generation.
Default output is 1920x1080/30fps. After a full render, use
`render.py --only 02-shared-roles` to replace only that clip and retain the first.
Do not mix preview/full modes with `--only`.

The renderer needs the existing animation uv environment and FFmpeg. Browser QA
uses the existing sibling looped-transformer Playwright installation and local
Chrome. `verify.mjs` must pass against a full-resolution render; it checks source
and media hashes, both selectors, both standalone players, playback/seek, mobile
overflow and absence of production labels. Browser screenshots are ignored.

`trace.json` records the exact NumPy fixture and nine checks; no DeepSeek weights
or GPU resources are used. `metadata.json` records package/host identity, Git base,
frame dimensions/codec checks, checkpoint geometry checks, hashes and durations.
`qa.json` is produced only after browser verification succeeds.

## Numerical evidence

The predeclared checks passed: one shared global storage object is unchanged
across three upper-layer reads; the tiny architecture preserves causal prefixes;
layer outputs differ; toy weights normalize; independent equal-valued K/V copies
agree with one-buffer arithmetic; changing V breaks that agreement; all results
are finite and the three queries favor rows 1, 2 and 3 respectively.

The second film uses C = [[1,0],[0,1],[-1,0]] and queries [2,0], [0,2], [-2,0].
Their outputs are approximately [0.722530,0.186694], [0,0.672842], and
[-0.722530,0.186694]. The first query's weights are approximately
[0.767918,0.186694,0.045388]. Display rounding does not feed back into arithmetic.

The first film uses magnitude-only colored cells on a fixed mapping; it does not
show signs, DeepSeek activations, or semantic meaning. Layer numbers correspond
to the architecture but only three reduced toy layers are computed. The source
change redraws cache values; the sharing transition is not a numeric merge.

## Deliberate simplifications

- CED and CSA2 sharing are introduced separately; the intermediate fan-out view
  is a conceptual decomposition, not the released storage configuration.
- The shared buffer in the second film is the selected core-attention pool.
  Actual selected global and local entries enter the same core operation; their
  persistent caches remain distinct. Other indexer/cache state is not eliminated.
- The toy excludes RoPE, attention sinks, sparse routing, quantization, output
  projection and full Hyper-Connections. It demonstrates the score/payload roles,
  not exact DeepSeek model outputs. The architecture has 64 query heads; the toy
  has three 2D queries, identified by the compact example label and page notes.
- Prefill speed, cache byte counts, bounded replay and quality are not animated
  or measured. The source report's local replay shortcut is approximate.

## Files and scope

Editable scene geometry lives in `scenes.py`, numerical logic in `build_trace.py`,
and media/player generation in `render.py`. Animation diagrams and code are
original; the shared poster reproduces Figure 3 under the source repository's
MIT license, retained in `assets/LICENSE.deepseek.txt`. Primary pinned sources
are listed in `DESIGN.md` and poster provenance is in `assets/provenance.json`.

Each clip returns a local MP4, GIF, full/half still and checkpoint contact sheet.
MP4s are ignored by repository policy. The renderer verifies hashes of the
existing looped-transformer project before/after rendering to prevent accidental
changes to earlier work. This task does not authorize commit, push, or publication.
