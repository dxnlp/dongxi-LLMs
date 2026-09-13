# Mathematical animations

These animations are editable companion material for the book. The original
previews use Matplotlib and Pillow so they can render on the DGX Spark without a
browser, LaTeX, or FFmpeg. Signature animations may also use Manim Community
Edition when continuity of motion is part of the explanation.

New Manim work should follow the approved
[`STYLE_GUIDE.md`](STYLE_GUIDE.md) visual system.

## Folder and workflow

`visuals/animations/` is the dedicated home for course animations:

- `PROPOSALS.md` is the two-way inbox for user- and agent-suggested ideas.
- `STYLE_GUIDE.md` defines the reusable visual and motion system.
- editable `.py` files are the source of existing animations;
- `rendered/` contains review and approved media plus metadata;
- `pyproject.toml` and `uv.lock` preserve the rendering environment.

New ideas begin in [`PROPOSALS.md`](PROPOSALS.md). Once Dongxi approves a
proposal, promote it to an `ANIM-*` task packet in `LEARNING_MEMORY.md` before
production. The packet must link the canonical chapter or learning artifact,
state the evidence boundary, define precision risks, and identify acceptance
checks. Approval of an animation concept never implies approval to publish it.

The current flat source layout remains valid. If the directory becomes crowded,
new signature animations may use `projects/<task-id>/`, while shared style code
stays at the top level.

## Render

One document, three requests: [`projects/kv-prefix-abc/`](projects/kv-prefix-abc/)
distinguishes an edit instruction from edited input, with stable retained K/V
and recomputed suffixes. The 19-second social-post companion retains editable
source, local 1080p MP4, looping GIF, stills and verification metadata.

One decode step: [`projects/kv-decode-step/`](projects/kv-decode-step/) follows
new-state projection, cache append, Q/K matching, softmax and a weighted V sum;
editable source, checked toy computation, reusable local GIF and MP4 are retained.

Cache memory growth: [`projects/kv-memory-growth/`](projects/kv-memory-growth/)
contains the approved position-growth and KV-head configuration comparisons,
with editable source, local GIF/MP4, stills and checked calculated payloads.

From the repository root:

```bash
uv run --project visuals/animations \
  python visuals/animations/bpe_byte_merges.py

uv run --project visuals/animations \
  python visuals/animations/cross_entropy_curve.py
```

The scripts write to `visuals/animations/rendered/` by default. Pass `--output`
to select another destination.

### Manim BPE refinement

The Manim scene uses a minimal, motion-first layout: token objects split and
fuse while a three-stage scale anchors byte, character, and phrase
representations. Text is limited to structural labels and the final conceptual
distinction. On macOS, install Manim's system prerequisites once:

```bash
brew install cairo pkg-config
```

Render the editable source to a temporary Manim media directory, then preserve
the deliverables under `visuals/animations/rendered/`:

```bash
manim_media=$(mktemp -d)

uv run --project visuals/animations manim render \
  -r 1920,1080 --fps 30 --renderer cairo --progress_bar none \
  --media_dir "$manim_media" -o bpe-byte-merges-manim \
  visuals/animations/manim_bpe_byte_merges.py BPEByteMerges

cp "$manim_media/videos/manim_bpe_byte_merges/1080p30/bpe-byte-merges-manim.mp4" \
  visuals/animations/rendered/bpe-byte-merges-manim.mp4

ffmpeg -y -loglevel error \
  -i visuals/animations/rendered/bpe-byte-merges-manim.mp4 \
  -vf "fps=15,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer" \
  visuals/animations/rendered/bpe-byte-merges-manim.gif

ffmpeg -y -loglevel error -ss 16.8 \
  -i visuals/animations/rendered/bpe-byte-merges-manim.mp4 -frames:v 1 \
  visuals/animations/rendered/bpe-byte-merges-manim-still.png
```

Record dependency revisions with:

```bash
uv run --project visuals/animations manim --version
uv run --project visuals/animations python --version
ffmpeg -version | head -n 1
```

## Learning objectives

- [`projects/kv-article-loops/`](projects/kv-article-loops/README.md): two short
  article GIFs contrast append/reuse with edit/recompute and explain encoder-derived
  global KV, cross-layer reuse, and the same selected vector's scoring/value roles.
  Distinct local state remains visible; no new model or benchmark claims.

- [`projects/kv-prefill-decode/`](projects/kv-prefill-decode/README.md): a compact
  prefill/decode loop for the KV-cache article. The cache grows only after the
  selected token is fed back and processed; old K/V remain unchanged. Manim source,
  MP4/GIF/PNG exports, event-order checks and bilingual article integration.

- [`projects/deepseek-ced/`](projects/deepseek-ced/README.md): two English films
  distinguishing encoder-derived global KV from cross-layer sharing, and the
  same selected KV representation's score and payload roles. Independent NumPy
  toy traces, editable Manim scenes and a [series player](projects/deepseek-ced/review.html).
  No DeepSeek model run or serving-speed claim.

- [`projects/looped-transformer/`](projects/looped-transformer/README.md): one
  shared causal block applied three times, with actual hidden-state matrices,
  constant stored parameters, and increasing matrix-operation counts.
  `ANIM-LOOP-001` uses editable Excalidraw structure and continuous Manim motion;
  its HTML player contains teaching content only.
  The [English series](projects/looped-transformer/english/review.html) separates
  one-block recurrence, two passes through a three-block stack, and unfolded
  shared-weight applications into three standalone videos.

- `bpe_byte_merges.py`: distinguish universal byte coverage from learned BPE
  compression, using `数` and `数据库` as the running example.
- `manim_bpe_byte_merges.py`: show the same mechanism as one continuous frozen
  encoding pass, using spatial motion rather than a dense merge-rank table.
- `cross_entropy_curve.py`: connect the probability assigned to the correct
  token with its negative-log-likelihood penalty. This is a preview for Day 3
  and should be integrated only after the derivation is complete.
- [`projects/meteor-bpe/`](projects/meteor-bpe/README.md): count BPE pairs in the
  user's five Chinese corpus entries, disclose the tie leading to `流星雨`,
  retain constituent tokens, assign the illustrative 1–5 subset, and replay
  learned merges during encoding. Includes a local video-review page.

The first Manim attempt failed on an earlier ARM host because required Cairo/X11
development components were absent. The current Mac workflow uses the Cairo
renderer with pinned Manim Community Edition dependencies; the Matplotlib
versions remain the lower-dependency baseline and are not overwritten.
