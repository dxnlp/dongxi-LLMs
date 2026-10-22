# Day 12 — Supervised Fine-Tuning

These are complete CPU lessons for Chapter 9. Material readiness is separate from learner practice and real-model evidence.

- [Can ignored prompt positions still learn?](01_sft_objective_and_gradient_paths.ipynb).
- [Does splitting a batch preserve the update?](02_accumulation_and_checkpoint_identity.ipynb).
- [Instruction tuning end to end: template → labels → LoRA → QLoRA](03_instruction_tuning_end_to_end.ipynb) Optional, **Spark GPU** (`spark-gpu`): real Qwen3-0.6B, from the chat template and loss mask through rank and low-rank approximation to full fine-tuning vs. LoRA vs. hand-built QLoRA. Needs the cached Qwen3-0.6B tokenizer and weights.
- [Instruction tuning workbook: write the key lines yourself](04_instruction_tuning_workbook.ipynb) Optional, **Spark GPU**, companion to 03: fourteen exercises where you write 1–4 key lines (template IDs, labels, shifted loss, top-r SVD, energy, the LoRA layer, freezing, a training step, block quantization, 4-bit packing, memory). Each has a ✅/❌ check from `dongxi_llms.sft_workbook`, then a reference solution.

Each notebook contains predictions, adjacent explanations and runnable solutions, and plots from the actual lesson tensors or declared illustrative fixtures. Use the course kernel with PyTorch and Matplotlib. No downloads or GPU work are required.

Reader route: [chapter](../../book/chapters/09-supervised-fine-tuning.md), [solutions](../../book/solutions/09-supervised-fine-tuning.md), [lab](../../book/labs/09-supervised-fine-tuning.md).
