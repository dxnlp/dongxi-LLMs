# Mathematical animations

These animations are editable companion material for the book. The original
previews use Matplotlib and Pillow so they can render on the DGX Spark without a
browser, LaTeX, or FFmpeg. Signature animations may also use Manim Community
Edition when continuity of motion is part of the explanation.

## Render

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

The Manim scene keeps the encoder rulebook visible while token objects move
through byte, character, and phrase representations. On macOS, install Manim's
system prerequisites once:

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

ffmpeg -y -loglevel error -ss 18.0 \
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

- `bpe_byte_merges.py`: distinguish universal byte coverage from learned BPE
  compression, using `数` and `数据库` as the running example.
- `manim_bpe_byte_merges.py`: show the same mechanism as one continuous frozen
  encoding pass, with the offline merge ranks visible throughout.
- `cross_entropy_curve.py`: connect the probability assigned to the correct
  token with its negative-log-likelihood penalty. This is a preview for Day 3
  and should be integrated only after the derivation is complete.

The first Manim attempt failed on an earlier ARM host because required Cairo/X11
development components were absent. The current Mac workflow uses the Cairo
renderer with pinned Manim Community Edition dependencies; the Matplotlib
versions remain the lower-dependency baseline and are not overwritten.
