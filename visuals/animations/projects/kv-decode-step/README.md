# One decode step

Approved `ANIM-KV-006`: replace the article's code example with an animated
close-up, keeping the existing prefill/decode overview unchanged.

The new hidden state passes through Q/K/V projections. New K/V move into the
cache before the current Q reads past and self. Scaled matching gives softmax
weights, then a weighted V sum; temporary query/output state clears while K/V
remain. One head/layer is shown. This ends before multi-head combination, output
projection and remaining model operations, and does not show next-token logits.

`example.py` preserves the executable NumPy example and defines the exact toy
fixture. Weight bars and flow stroke widths use calculated weights. The vectors
are didactic inputs, not observed model activations; visual vector bars have a
small nonzero baseline for visibility. Padding/position operations are omitted.
No physical allocator behavior, latency, model run or training claim is made.

## Reproduce and verify

```bash
uv run --project visuals/animations python visuals/animations/projects/kv-decode-step/render.py --preview
uv run --project visuals/animations python visuals/animations/projects/kv-decode-step/render.py
uv run --project visuals/animations python publications/x-articles/x-attn-kv-001/test_example.py
```

`render.py` reuses and hashes the existing article-loop exporter. It verifies
event order, toy arithmetic, self inclusion, unchanged earlier cache geometry,
canvas bounds, codec, GIF diversity/looping/size, and unchanged earlier projects.
`metadata.json` records environment, exact commands, hashes and known limits.

`rendered/decode-step/` contains the reusable 1080p MP4 (Git-ignored), 960×540 GIF,
PNG fallback and contact sheet. The article contains the GIF in slot 4 with
pause/resume and reduced-motion support. No X upload or publishing is included.
