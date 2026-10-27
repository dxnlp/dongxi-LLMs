# Visual plan — X-POST-003 (LoRA and QLoRA)

Approved by the learner on 2026-10-23 (GIF plan and cover), `ANIM-POST-003`. Follows the series
conventions in `../README.md`: animated GIFs by default, Dongxi Manim style, English labels
only (shared by both versions), one idea per GIF, few large labels, motion carries the
meaning, 1280×720 20 fps GIF encoded from a lossless render, 1080p MP4 kept local.
Planned source: `visuals/animations/projects/post-training-lora/`.

The thread through every figure is the **pattern**: one column times one row (a rank-1
matrix). Each pattern is drawn the same way in every GIF, as a column bar and a row bar
whose product fills a grid, so readers recognize it from figure to figure.

| # | Figure | Motion | Evidence |
|---|---|---|---|
| 01 | One GPU, 65B model (GIF) | A 48 GB GPU outline; a bar for full fine-tuning grows far past it (~1 TB, labeled estimate); the QLoRA bar settles inside it. Stat chips: 24 h, 99.3% of ChatGPT on Vicuna. | QLoRA paper's published claims; the ~1 TB bar is a labeled byte-count estimate |
| 02 | A pattern: column × row (GIF) | Column [10, 20, 30] and row [1, 2, 2, 1] multiply into a 3×4 sales grid (rank 1); a second pattern (promotion) adds in and the grid changes (rank 2). Counter: 7 numbers → 12 entries. | Illustrative |
| 03 | Keep the strongest patterns (GIF) | A public-domain photo rebuilt from r = 1, 5, 20, 50 patterns; a strength bar list (singular values) shrinks as weaker patterns are dropped. | Illustrative; no error percentages on screen |
| 04 | Low-rank approximation (GIF) | A d×d matrix splits into a tall-thin × short-wide pair; parameter counters 16.8M → 65,536 (d = 4096, r = 8). | Arithmetic |
| 05 | LoRA (GIF) | Frozen W (grey, lock icon) plus a B·A branch built from r column × row patterns (blue); only the branch lights up during training; at the end B·A merges into W. Starts with B = 0. | Illustrative |
| 06 | Quantizing a block (GIF) | Eight weights in a block are divided by the block scale, snap to the nearest of 16 levels, and become 4-bit codes; then rebuilt as level × scale. | Illustrative values (from the learning note's worked example) |
| 07 | Uniform vs NF4 levels (GIF) | A bell-shaped weight histogram; 16 evenly spaced levels vs 16 NF4 levels placed at normal quantiles, dense near 0. | Illustrative; NF4 levels are the published table |
| 08 | Low rank vs quantization (GIF) | One matrix's pattern strengths as bars. Low rank: the tail of weak patterns is cut away. Quantization: every bar stays, and a thin uniform noise floor appears. | Illustrative |
| 09 | QLoRA forward and backward (GIF) | 4-bit W block is converted to bf16 just in time, multiplied with the input, plus the LoRA branch; the backward arrow passes through W to A and B only. | Illustrative |
| 10 | Memory for a 7B model (GIF) | Bars grow: full ~100 GB, LoRA ~15 GB, QLoRA ~5 GB (weights, gradients, optimizer; no activations). | Labeled byte-count estimate |
| 11 | The pattern lens (static PNG) | Summary card: rank, SVD, low-rank approximation, LoRA, quantization, QLoRA, each with its one-line pattern reading. | Summary |

## Cover

Series style (white Avenir Next Heavy title, gold "POST-TRAINING SERIES · 03" and "DONGXI
LABS", deep ink background, engraved vector lines, stipple dust), with a new motif for this
article, distinct from covers 1 and 2:

- A large, dim engraved matrix grid fills the lower half: the frozen pretrained weights.
- Across it, a few glowing patterns: each a bright column line crossing a bright row line,
  their product lighting a soft rectangle in the grid (LoRA's added patterns, in blue).
- Toward the right edge the grid's dots snap onto a few discrete horizontal levels
  (quantization to 4 bits), drawn as stepped stipple.
- Title: "LoRA & QLoRA". Small labels in the art: "frozen W", "+ patterns", "4-bit".
