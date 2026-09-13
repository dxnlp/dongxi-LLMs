# Prefill → decode

`ANIM-KV-001`, approved 2026-09-12 as the first GIF for `X-ATTN-KV-001`.
Local Mac production only; no model run, X upload, publication, commit or push.

## Learning objective

Make the cache grow from three prompt positions to four processed positions.
Selecting “blue” leaves the cache at three; feeding it back creates its K/V,
then a fresh Q reads all four positions and the remaining model predicts “.”.
Earlier K/V objects retain their exact geometry throughout decode.

Canonical sources: Chapter 4 §4.10 and
`learning_artifacts/day-04-attention-and-causal-information-boundary/why-cache-keys-and-values.md`.
The shared style comes from `../../STYLE_GUIDE.md` and `../../manim_style.py`:
white canvas, normal Arial shaped at 16× internal scale, semantic color, stable
anchors and mechanism-carrying motion. Labels stay in English in both articles.

## Review and outputs

- [MP4 player](review.html)
- [GIF](rendered/prefill-decode.gif): 960×540, about 8.94 seconds, infinite loop
- [MP4](rendered/prefill-decode.mp4): 1920×1080, H.264, 30 fps, ignored by Git
- [Still](rendered/prefill-decode-still.png) and half-size fallback
- [Contact sheet](rendered/prefill-decode-contact-sheet.png)
- `metadata.json`: versions, exact commands, source/output hashes, media checks
- `timeline.json`: ordered events, cached-position counts, geometry assertions

The article builder copies the GIF and half-size still into its assets, verifies
their hashes, and uses them for slot 1 in English and Chinese. A local button
switches to the still when paused; reduced-motion preferences default to the still.
The X insertion plan contains the GIF itself. Actual X editor upload/playback
has not been tested in this task.

## Reproduce

From the repository root:

```bash
uv run --project visuals/animations python visuals/animations/projects/kv-prefill-decode/render.py --preview
uv run --project visuals/animations python visuals/animations/projects/kv-prefill-decode/render.py
uv run --project visuals/animations --with mistune==3.1.3 python publications/x-articles/x-attn-kv-001/build.py
node publications/x-articles/x-attn-kv-001/verify.mjs
```

Inspect the preview contact sheet before final export. Final rendering separately
checks event order, cache counts, unchanged past geometry, MP4 codec and dimensions,
GIF frame diversity, duration, infinite-loop flag, size, and source/output hashes.
The bilingual browser audit checks visible GIF advancement, pause/resume, reduced
motion, image order, mobile layout, and the existing companion players.

An extra mid-motion inspection frame can be regenerated with:

```bash
ffmpeg -y -loglevel error -ss 6.4 -i visuals/animations/projects/kv-prefill-decode/rendered/prefill-decode.mp4 -frames:v 1 -update 1 visuals/animations/projects/kv-prefill-decode/rendered/read-flow.png
```

## Precision boundaries

- Token pieces and continuations are illustrative, not tokenizer/model results.
- One layer is shown; only the final prompt Q is drawn. Prefill actually computes
  all prompt queries with causal visibility.
- Equal vector glyphs are symbolic. Flow paths show matching and value retrieval,
  without asserting numeric attention weights or equal weighting.
- “Rest of model” summarizes remaining layer operations and vocabulary projection.
- No selected token enters the cache until it is processed as input.
- The final fade is an editorial loop reset, not a depiction of cache eviction.
- Append/edit and global-sharing loops remain separate, unapproved future work.
