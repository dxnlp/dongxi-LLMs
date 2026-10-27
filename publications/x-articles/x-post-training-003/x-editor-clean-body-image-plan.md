# X Article Inline Image Plan

Use this after pasting the clean body into X Articles.

Rule: insert images from bottom to top so earlier block positions do not shift.

- Title: `LoRA and QLoRA: Fine-Tuning a Large Model on One GPU`
- Clean body HTML: `x-editor-clean-body.html`
- Clean body Markdown: `x-editor-clean-body.md`
- Total clean body blocks: `54`
- Images: `11`

## Reverse Insertion Order

### IMAGE 11

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-003/assets/11-pattern-lens.png`
- Exists: `True`
- Insert after block index: `47`
- Target text: Summary
- Next text: A pattern is a column times a row. A matrix is a sum of patterns, and its rank is the number of independent patterns.

### IMAGE 10

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-003/assets/10-memory-7b.gif`
- Exists: `True`
- Insert after block index: `45`
- Target text: A rough byte-count estimate for a 7B model, counting weights, gradients and optimizer state but not activations: about...
- Next text: The costs of QLoRA: converting weights before every computation makes training slower than 16-bit LoRA; quantization ad...

### IMAGE 09

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-003/assets/09-qlora.gif`
- Exists: `True`
- Insert after block index: `44`
- Target text: Paged optimizers move optimizer state to CPU memory during memory spikes, so training doesn't run out of GPU memory.
- Next text: A rough byte-count estimate for a 7B model, counting weights, gradients and optimizer state but not activations: about...

### IMAGE 08

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-003/assets/08-lowrank-vs-quant.gif`
- Exists: `True`
- Insert after block index: `38`
- Target text: This explains an apparent contradiction. Pretrained weights are hard to compress with low rank, because they contain ma...
- Next text: QLoRA: low rank and quantization where each fits

### IMAGE 07

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-003/assets/07-nf4-levels.gif`
- Exists: `True`
- Insert after block index: `31`
- Target text: NF4. QLoRA uses a 4-bit format called NF4 (4-bit NormalFloat). Trained weights are roughly bell-shaped, so NF4 places i...
- Next text: Storage and compute. Each weight takes 4 bits, plus a share of the scale used by its block of 64. Quantizing the scales...

### IMAGE 06

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-003/assets/06-quantize-block.gif`
- Exists: `True`
- Insert after block index: `30`
- Target text: Block-wise scaling. The 16 levels lie between −1 and 1, while real weights are small, often within about ±0.05. So the...
- Next text: NF4. QLoRA uses a 4-bit format called NF4 (4-bit NormalFloat). Trained weights are roughly bell-shaped, so NF4 places i...

### IMAGE 05

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-003/assets/05-lora.gif`
- Exists: `True`
- Insert after block index: `25`
- Target text: After training, B·A has the same shape as W, so it can be added into W with no extra cost at inference. Alternatively,...
- Next text: LoRA has limits. The rank r bounds what the update can express: a change that needs more than r patterns can't be learn...

### IMAGE 04

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-003/assets/04-low-rank.gif`
- Exists: `True`
- Insert after block index: `17`
- Target text: A low-rank approximation represents a large matrix with only r patterns, written as a tall, thin matrix times a short,...
- Next text: LoRA: add a few patterns to frozen weights

### IMAGE 03

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-003/assets/03-strongest-patterns.gif`
- Exists: `True`
- Insert after block index: `16`
- Target text: Take a grayscale photo. With 1 pattern, you get a grid of horizontal and vertical bands. With 20, the face is recogniza...
- Next text: A low-rank approximation represents a large matrix with only r patterns, written as a tall, thin matrix times a short,...

### IMAGE 02

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-003/assets/02-pattern.gif`
- Exists: `True`
- Insert after block index: `11`
- Target text: Now add a promotion: days 3 and 4 get an extra [0, 0, 5, 5], with participation [1, 0, 2] across the three stores. The...
- Next text: A matrix is a sum of patterns, and its rank is the number of independent patterns in it.

### IMAGE 01

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-003/assets/01-one-gpu.gif`
- Exists: `True`
- Insert after block index: `2`
- Target text: For comparison, full fine-tuning in the usual way needs roughly 16 bytes of GPU memory per parameter, about 1 TB for a...
- Next text: QLoRA combines two parts: LoRA reduces the number of parameters that are trained, and 4-bit quantization reduces the nu...
