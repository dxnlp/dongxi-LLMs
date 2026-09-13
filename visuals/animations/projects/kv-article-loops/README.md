# Cache validity and global KV sharing

User-approved follow-ups to the prefill/decode article loop, 2026-09-12.
Task packets: `ANIM-KV-003` and `ANIM-KV-004` in `LEARNING_MEMORY.md`.
The previous first-loop-only approval is superseded for these two concepts.
Production runs on the Mac Studio. No new model or GPU experiment is involved.

## Animations

- **Append or edit?** (`append-edit`): append E while retaining cached A–D;
  change B to X, preserve A, invalidate and recompute the dependent suffix.
  GIF: 960×540, 7.67 seconds, 1,074,254 bytes, infinite loop.
- **Causal Encoder–Decoder** (`global-sharing`): encoder output supplies one
  global bank, created at decoder layer 21; layers 21, 22, and 40 illustrate reuse
  while retaining distinct Q/local KV. Magnify one selected vector to illustrate
  its query-matching and weighted-accumulation roles.
  GIF: 960×540, 7.87 seconds, 1,039,763 bytes, infinite loop.

[Open both videos](review.html). Each `rendered/<clip>/` folder contains the
1080p H.264 MP4, 960×540 GIF, full/half-size PNG still, and complete-sequence
contact sheet including mid-motion samples. MP4s remain ignored by Git.

## Sources and boundaries

- Chapter 4 §4.10 and `why-cache-keys-and-values.md` establish causal cache
  invariance and prefix identity.
- `learning_artifacts/day-06-modern-architecture/deepseek-v41-causal-encoder-decoder.md`
  records the pinned report/reference-code evidence for the second schematic.
- `../../STYLE_GUIDE.md` and `../../manim_style.py` provide the approved white
  canvas, semantic palette, high-resolution Arial shaping and geometry rules.
- Colored glyphs are schematic cache entries, not activations or numeric weights.
- Suffix invalidation is a conservative reuse rule. Recomputed values need not
  all change; in particular, first-layer projections have narrower dependencies.
- Global sharing does not imply query sharing, equal attention weights, or removal
  of local caches/indexer state. Intervening decoder layers and sparse-selection
  mechanics are omitted; arrows represent selected-entry access, not dense reads.
- The lower selected-vector detail is magnification, not an extra bank allocation.
- The two vector roles are illustrated separately; no extra physical K/V copy is
  implied. Other model operations and compression/indexer details are not shown.
- Fade-out resets the loop editorially; it is not cache eviction or reversed inference.

## Reproduction and checks

From the repository root:

```bash
uv run --project visuals/animations python visuals/animations/projects/kv-article-loops/render.py --preview
uv run --project visuals/animations python visuals/animations/projects/kv-article-loops/render.py
uv run --project visuals/animations --with mistune==3.1.3 python publications/x-articles/x-attn-kv-001/build.py
node publications/x-articles/x-attn-kv-001/verify.mjs
```

The preview contact sheets are inspected before final render. The final renderer
checks ordered state events, preservation of reusable geometry, one global bank,
distinct local objects, and both roles' reference to the same entry. It verifies
MP4 dimensions/codec, GIF dimensions/duration/loop/frame diversity/size and hashes.
It also hashes the entire existing `kv-prefill-decode/` project before and after
rendering to ensure it is not modified. Sources, commands and versions are in
`metadata.json`; individual scene event records are in `*-timeline.json`.

The article uses GIFs in slots 1, 3 and 6; equations and the memory chart remain
static. Its browser audit verifies each GIF's motion, pause/resume and still
fallback, plus reduced-motion handling and the insertion order in both languages.
No X editor upload or public publishing action has been performed.
