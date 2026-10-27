# X Article Inline Image Plan

Use this after pasting the clean body into X Articles.

Rule: insert images from bottom to top so earlier block positions do not shift.

- Title: `LoRA 与 QLoRA：用一块 GPU 微调大模型`
- Clean body HTML: `x-editor-clean-body.html`
- Clean body Markdown: `x-editor-clean-body.md`
- Total clean body blocks: `54`
- Images: `11`

## Reverse Insertion Order

### IMAGE 11

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-003/assets/11-pattern-lens.png`
- Exists: `True`
- Insert after block index: `47`
- Target text: 小结
- Next text: 模式是一列乘一行得到的矩阵。矩阵是若干模式之和，秩是其中独立模式的个数。

### IMAGE 10

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-003/assets/10-memory-7b.gif`
- Exists: `True`
- Insert after block index: `45`
- Target text: 按字节数粗略估算一个 7B 模型，只计权重、梯度和优化器状态，不含激活值：全参数微调约 100 GB，LoRA 约 15 GB，QLoRA 约 5 GB。
- Next text: QLoRA 的代价：每次计算都要先还原权重，训练速度比 16 位的 LoRA 慢；量化带来少量误差，量化后的模型起点略差，需要由 LoRA 在训练中弥补；训练好的 A、B 不能直接加到 4 位权重上，需要合并到还原后的 16 位权重，或者...

### IMAGE 09

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-003/assets/09-qlora.gif`
- Exists: `True`
- Insert after block index: `44`
- Target text: paged optimizers（分页优化器） 在显存出现峰值时，把优化器状态临时移到 CPU 内存，避免显存不足。
- Next text: 按字节数粗略估算一个 7B 模型，只计权重、梯度和优化器状态，不含激活值：全参数微调约 100 GB，LoRA 约 15 GB，QLoRA 约 5 GB。

### IMAGE 08

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-003/assets/08-lowrank-vs-quant.gif`
- Exists: `True`
- Insert after block index: `38`
- Target text: 这也解释了一个看似矛盾的现象：预训练权重很难做低秩压缩，因为它的模式很多，删去弱模式会损失大量信息；它却可以较好地量化，因为全部模式都被保留，只增加了少量噪声。
- Next text: QLoRA：低秩和量化各用在合适的位置

### IMAGE 07

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-003/assets/07-nf4-levels.gif`
- Exists: `True`
- Insert after block index: `31`
- Target text: NF4。 QLoRA 使用的 4 位格式称为 NF4（4-bit NormalFloat）。训练好的权重大致呈钟形分布，NF4 按正态分布的分位数放置 16 个级别：靠近 0 处密集，两端稀疏，并且包含精确的 0。大多数权重落在 0 附近...
- Next text: 存储和计算。 每个权重占 4 位，再加上每 64 个权重共享的缩放因子。缩放因子本身再量化为 8 位，称为 double quantization（双重量化），之后平均每个权重约占 4.13 位，约为 bf16 的四分之一。计算时，每一块...

### IMAGE 06

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-003/assets/06-quantize-block.gif`
- Exists: `True`
- Insert after block index: `30`
- Target text: 分块缩放。 16 个级别分布在 −1 到 1 之间，而真实的权重通常很小，大约在 ±0.05 的范围内。因此把权重每 64 个分成一块，每块记录一个缩放因子，即块中绝对值最大的权重。存储时，先把权重除以缩放因子，再取最近的级别，只保存编号...
- Next text: NF4。 QLoRA 使用的 4 位格式称为 NF4（4-bit NormalFloat）。训练好的权重大致呈钟形分布，NF4 按正态分布的分位数放置 16 个级别：靠近 0 处密集，两端稀疏，并且包含精确的 0。大多数权重落在 0 附近...

### IMAGE 05

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-003/assets/05-lora.gif`
- Exists: `True`
- Insert after block index: `25`
- Target text: 训练完成后，B·A 和 W 形状相同，可以直接加到 W 上，推理时没有额外开销；也可以为不同任务保存不同的 A、B，按需切换。LoRA 论文报告，在 GPT-3 175B 上，与用 Adam 做全参数微调相比，可训练参数减少到约万分之一，...
- Next text: LoRA 也有限制。秩 r 限制了更新的表达能力，需要超过 r 个模式才能表示的变化无法完整学到。另外，它减少的是需要训练的状态，原有权重仍以 16 位存放在显存中：65B 模型的权重本身就需要约 130 GB。

### IMAGE 04

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-003/assets/04-low-rank.gif`
- Exists: `True`
- Insert after block index: `17`
- Target text: 低秩近似 就是只用 r 个模式表示一个大矩阵，写成一个细高矩阵乘以一个扁宽矩阵。一个 d×d 的矩阵需要 d² 个数；用秩 r 表示时，只需要 2dr 个数。d = 4096、r = 8 时，数的个数从约 1,680 万减少到 65,53...
- Next text: LoRA：在冻结的权重上增加少量模式

### IMAGE 03

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-003/assets/03-strongest-patterns.gif`
- Exists: `True`
- Insert after block index: `16`
- Target text: 以一张灰度照片为例：只保留 1 个模式，只能得到横竖交错的条纹；保留 20 个，已经能认出人脸；保留 50 个，和原图相当接近。少数强模式决定整体结构，大量弱模式只贡献细节。
- Next text: 低秩近似 就是只用 r 个模式表示一个大矩阵，写成一个细高矩阵乘以一个扁宽矩阵。一个 d×d 的矩阵需要 d² 个数；用秩 r 表示时，只需要 2dr 个数。d = 4096、r = 8 时，数的个数从约 1,680 万减少到 65,53...

### IMAGE 02

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-003/assets/02-pattern.gif`
- Exists: `True`
- Insert after block index: `11`
- Target text: 再加入一次促销：第 3、4 天额外增加 [0, 0, 5, 5]，三家店的参与程度分别为 [1, 0, 2]。新的销售表是两个模式相加，各行不再是同一个行向量的倍数，秩为 2。
- Next text: 矩阵可以看作若干模式相加的结果，秩就是其中相互独立的模式的个数。

### IMAGE 01

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-003/assets/01-one-gpu.gif`
- Exists: `True`
- Insert after block index: `2`
- Target text: 作为对比，按常规方式做全参数微调，每个参数大约需要 16 字节显存，65B 模型约需 1 TB，还不包括激活值。这是按字节数推算的数量级估计。
- Next text: QLoRA 由两部分组成：LoRA 减少需要训练的参数数量，4 位量化 减少每个参数占用的位数。理解这两部分，需要三个基本概念：rank（秩）、low-rank approximation（低秩近似）和 quantization（量化）。...
