# Source map — X-POST-003

The public body has no links. Every factual claim maps to a source here.

| Article claim | Source |
|---|---|
| QLoRA (2023, University of Washington): 65B model fine-tuned on a single 48 GB GPU; preserves full 16-bit fine-tuning task performance; Guanaco reaches 99.3% of ChatGPT on the Vicuna benchmark with 24 hours of fine-tuning on one GPU; NF4, double quantization, paged optimizers | Dettmers et al. (2023), "QLoRA: Efficient Finetuning of Quantized LLMs", arXiv:2305.14314 (abstract checked 2026-10-23). The ">780 GB" figure often quoted is not in the abstract and is not used. |
| ~16 bytes per parameter for full fine-tuning (fp32 weight, gradient, two Adam moments); ~100+ GB for 7B, ~1 TB for 65B; activations extra | Byte arithmetic; learning note topic 5 and Predict 13 (`learning_artifacts/day-11-instruction-data-engineering/learning-instruction-tuning.md`). Labeled as an estimate in the text. |
| Pattern = column × row = rank-1 matrix; matrix = sum of patterns; rank = number of independent patterns; read one input direction, write one output direction | Learning note "Key idea: patterns connect the concepts" and Predict 4 discussion (learner's own insight, 2026-10-22) |
| Sales-table example (rhythm [1,2,2,1] × size [10,20,30]; promotion [0,0,5,5] × [1,0,2]) | Learning note Predict 4 follow-up |
| SVD orders patterns by strength, orthogonal; top-r is the best rank-r approximation (Eckart–Young) | Learning note SVD follow-up; standard linear algebra |
| Photo: 1 pattern gives bands, 20 recognizable, 50 close to the original | Learning note Predict 5 (qualitative only; no error percentages in the article) |
| Low-rank parameter count: d² vs 2dr; 16,777,216 vs 65,536 at d = 4096, r = 8 (~0.4%) | Learning note topic 4 (arithmetic) |
| LoRA: W + B·A with B: d×r, A: r×d; freezes W; trains A, B; B·A is a sum of r patterns | SLP3 §8.2 (eq. 8.1, written there as W + AB); Hu et al. (2021) arXiv:2106.09685; learning note topic 4 (notation differences noted there) |
| Pretrained W spreads strength over hundreds of patterns, not low-rank compressible | Learning note Predict 6 (stated qualitatively) |
| LoRA's premise: task adaptation changes the weights along few patterns; LoRA paper's empirical study of rank-deficiency | Hu et al. (2021) abstract ("empirical investigation into rank-deficiency in language model adaptation"); learning note Predict 10 (not quoted) |
| B = 0, A random: output unchanged at start; B gets gradient first | Learning note Predict 8 follow-up |
| Merge B·A into W, no extra inference latency; swap adapters per task | Hu et al. (2021) abstract ("no additional inference latency"); SLP3 §8.2 |
| LoRA on GPT-3 175B: ~10,000× fewer trainable parameters, ~3× less GPU memory vs full fine-tuning with Adam | Hu et al. (2021) abstract (checked 2026-10-23) |
| 65B weights alone ~130 GB at 16 bits | Arithmetic (65e9 × 2 bytes) |
| Bits and values; 4 bits = 16 levels; codes 0–15; block size 64; absmax scale; NF4 levels at normal quantiles with exact 0; double quantization ≈ 4.13 bits/weight; dequantize to bf16 for compute | Learning note "Key concept: what 4-bit quantization means in QLoRA", topic 5, Predicts 11–13; QLoRA abstract for NF4 and double quantization |
| Low rank removes patterns; quantization keeps all patterns, strengths move slightly, adds noise-like error spread over all directions | Learning note "can the pattern concept explain quantization too?" (stated qualitatively) |
| Pretrained weights: hard to low-rank compress but quantize well | Learning note "does pretrained weights can't be compressed conflict with QLoRA?" |
| QLoRA structure: 4-bit NF4 frozen W, 16-bit LoRA A/B, dequantize in forward, gradients through dequantized W to A and B; paged optimizers | QLoRA abstract; learning note topic 5 |
| 7B memory: ~100 GB full, ~15 GB LoRA, ~5 GB QLoRA (weights, gradients, optimizer; no activations) | Learning note topic 5 and Predict 13 (computed estimates). Labeled as estimates. |
| QLoRA costs: slower (dequantization), quantization error (starts slightly worse), merging requires a 16-bit copy | Learning note topic 5 trade-offs and Predict 14 discussion (stated qualitatively) |

## Evidence boundary

Published results come from the LoRA and QLoRA abstracts. Memory figures are byte-count
estimates, labeled as such. Concept examples (sales table, photo) are illustrations. No
numbers from our own notebook runs are reported in the article.
