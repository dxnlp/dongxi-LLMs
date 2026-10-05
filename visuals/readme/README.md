# README mechanism previews

Three English GIF loops introduce the book's mechanisms using its white-canvas,
blue/purple/green visual system. They are authored for the repository README;
they do not replace the full chapter animations or present trained-model results.

| Preview | Learning objective | Canonical source |
|---|---|---|
| [Next-token decoding](next-token.gif) · [still](next-token.png) | Follow prefix → decoder → vocabulary probabilities → greedy selection → appended token. | [Chapter 3](../../book/chapters/03-learning-the-next-token.md); CAND-ANIM-001 |
| [Causal attention](causal-attention.gif) · [still](causal-attention.png) | Follow each query's allowed keys, normalized weights and weighted value output while future weights remain zero. | [Chapter 4](../../book/chapters/04-attention-and-the-causal-information-boundary.md); CAND-ANIM-008 |
| [Group-relative rewards](group-relative-rewards.gif) · [still](group-relative-rewards.png) | Compare population-normalized advantages for a mixed binary-reward group and a constant-reward group. | [Chapter 13](../../book/chapters/13-group-relative-policy-optimization.md); CAND-ANIM-027 |

## Reproduction

From the repository root, with the existing animation environment and FFmpeg:

```bash
uv run --project visuals/animations python visuals/readme/render.py
```

The renderer writes GIFs, PNG stills and `manifest.json` here. Decoded-GIF
contact sheets go to the system temporary directory (`dongxi-readme-qa`);
use `--qa-dir /path/to/folder` to change that location. It creates no video.
The current render used the repository's `.venv/bin/python`; exact Python,
Pillow, FFmpeg, platform and font identity are recorded in the manifest.
Arial is selected on macOS; Liberation Sans or DejaVu Sans can be used on Linux,
or supply `--font /path/to/font.ttf`. Font changes alter raster output; the
manifest records the selected font's hash. Fonts are not bundled.

## Numerical and presentation boundaries

- The decoding loop uses fixed illustrative logits over seven toy vocabulary
  entries, scores all seven, displays the top three, and selects the maximum.
  Decoder internals are schematic; no model forward pass or activations were
  measured. The three context words are teaching token boundaries.
- The attention loop calculates scaled Q/K scores and causally masked softmax
  for five positions, with three-dimensional Q/K and two-dimensional V.
  It shows one head and omits position encoding, other heads, output projection
  and the rest of the model. The bar weights and value mixtures are computed,
  not a claim about a real model's interpretation of these words.
- The reward loop shows two independently specified groups for the same prompt.
  Rewards are 1 for answer `4` and 0 otherwise. Advantages use population standard
  deviation and epsilon `1e-8`, matching `src/dongxi_llms/grpo_lab.py`.
  Constant groups give zero advantages. Policy ratios, clipping, KL and optimizer
  state are omitted; zero reward advantages do not guarantee zero weight updates.
- Reveals and moving markers communicate ordering, not measured latency.
  White fades reset the editorial loop rather than reversing a model operation.

The renderer verifies probability normalization, greedy choices, zero future
attention weights, invariance to changed future values, population normalization
and constant-group zero advantages. It decodes every exported GIF and checks
dimensions, duration, infinite looping, changing frames and a 3 MiB per-file
ceiling. Source, font and output SHA-256 hashes remain in `manifest.json`.
Visual review uses the decoded contact sheets and full/half-size stills.
The GIFs have no production credits or internal task IDs on their canvas.

Current review, 2026-10-05: all three decoded contact sheets were inspected at
half scale, alongside full-size representative frames. The local rendered
Markdown preview loads all six root-README images and plays the GIFs; its
default 1280px viewport has no page overflow. All 59 local links in the root
and media READMEs resolve. GIF hashes agree with the manifest, `git diff --check`
passes, and the book-math checker reports zero issues across 54 Markdown files
and 1,294 expressions. Aggregate GIF size is 1,179,142 bytes (1.13 MiB).

Production packet: `ANIM-README-001` in `LEARNING_MEMORY.md`. User requested README
GIFs and authorized new animations on 2026-10-05. Publication and Git push are
separate actions.
