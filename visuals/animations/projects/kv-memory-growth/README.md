# Cache memory growth

Approved task: `ANIM-KV-005`, Mac Studio production for article slot 5.

Two comparisons connect the cache dimensions to calculated tensor memory:
4096→8192 retained positions at eight KV heads gives 192→384 MiB; a separate
architecture comparison at 4096 positions, eight→four KV heads, gives 192→96 MiB.
Source: Chapter 5 §5.18 and the original KV article's memory diagram.

Each column groups 1024 positions, each row is one KV head; paired purple/green
glyphs represent separate K/V. Batch, layers and feature dimensions are omitted
visually but included in the calculation. Other factors: batch 1, 24 layers,
head width 64, two bytes per element. MiB means 2^20 bytes. These are calculated
payloads, excluding weights, temporary buffers, metadata and allocator overhead.
The second comparison does not depict runtime head removal. Resets are editing,
not eviction. There is no model run or claim about speed or quality.

## Reproduce

```bash
uv run --project visuals/animations python visuals/animations/projects/kv-memory-growth/render.py --preview
uv run --project visuals/animations python visuals/animations/projects/kv-memory-growth/render.py
```

The renderer reuses the existing article-loop exporter, hashes its source, and
checks that both earlier animation projects remain unchanged. Arithmetic,
bar-length ratios, unchanged geometry, bounds, codec, GIF diversity, looping and
size checks run automatically. Inspect preview contact sheets before full export.

Outputs in `rendered/memory-growth/`: reusable 1080p `memory-growth.mp4`,
960×540 `memory-growth.gif`, PNG fallback and contact sheet. MP4 is Git-ignored.
Exact commands, environment versions, source/media hashes and limits are recorded
in `metadata.json`. The bilingual article supplies playback/reduced-motion controls.
