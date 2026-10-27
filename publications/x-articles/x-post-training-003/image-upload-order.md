# Image Upload Order

Cover (upload separately, 5:2): `assets/cover-post-training-03.png`. All figures are English and shared by both versions.

1. `assets/01-one-gpu.gif` — QLoRA (2023) fine-tuned a 65B model on one 48 GB GPU and preserved 16-bit fine-tuning performance; Guanaco reached 99.3% of ChatGPT on Vicuna after 24 hours. The full fine-tuning bar is a byte-count estimate.
2. `assets/02-pattern.gif` — A pattern is a column times a row. The sales table is one pattern (rank 1); adding a promotion pattern makes it rank 2.
3. `assets/03-strongest-patterns.gif` — SVD orders a matrix's patterns by strength. A photo rebuilt from its 1, 5, 20 and 50 strongest patterns.
4. `assets/04-low-rank.gif` — A d×d matrix approximated as tall-thin × short-wide: 16.8 million numbers become 65,536 at d = 4096, r = 8.
5. `assets/05-lora.gif` — LoRA freezes W and trains only B·A, a sum of r patterns, starting from B = 0. After training, B·A can be merged into W.
6. `assets/06-quantize-block.gif` — 4-bit quantization of one block: divide by the block scale, snap to the nearest of 16 levels, store the 4-bit code.
7. `assets/07-nf4-levels.gif` — Bell-shaped weights: 16 evenly spaced levels vs NF4's 16 levels at normal quantiles, dense near 0.
8. `assets/08-lowrank-vs-quant.gif` — Low rank removes the weak patterns; quantization keeps every pattern and adds a small, spread-out error.
9. `assets/09-qlora.gif` — QLoRA: 4-bit frozen weights converted to bf16 just in time, plus a 16-bit LoRA branch; gradients reach only A and B.
10. `assets/10-memory-7b.gif` — Approximate memory for a 7B model (weights, gradients, optimizer; no activations): full ~100 GB, LoRA ~15 GB, QLoRA ~5 GB. Estimates.
11. `assets/11-pattern-lens.png` — Summary: every concept in this article read through patterns.
