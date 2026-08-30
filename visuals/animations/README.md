# Mathematical animations

These animations are editable companion material for the book. They use
Matplotlib's animation API and Pillow's GIF writer so they can render on the
DGX Spark without a browser, LaTeX, or FFmpeg.

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

## Learning objectives

- `bpe_byte_merges.py`: distinguish universal byte coverage from learned BPE
  compression, using `数` and `数据库` as the running example.
- `cross_entropy_curve.py`: connect the probability assigned to the correct
  token with its negative-log-likelihood penalty. This is a preview for Day 3
  and should be integrated only after the derivation is complete.

The first attempted renderer was Manim. Its OpenGL dependency could not build
on the current ARM host because the X11 development headers are unavailable.
This fallback preserves editable source and deterministic local rendering with
a substantially smaller dependency surface.
