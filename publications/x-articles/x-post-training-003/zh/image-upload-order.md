# Image Upload Order

封面（单独上传，5:2）：`../assets/cover-post-training-03.png`。所有配图均为英文，中英文版共用。

1. `../assets/01-one-gpu.gif` — QLoRA（2023）在一块 48GB GPU 上微调 65B 模型，并保持了 16 位微调的表现；Guanaco 训练 24 小时，在 Vicuna 评测中达到 ChatGPT 的 99.3%。全参数微调的柱子是按字节数估算的。
2. `../assets/02-pattern.gif` — 模式是一列乘一行。销售表是一个模式（秩为 1）；加入促销这个模式后，秩变为 2。
3. `../assets/03-strongest-patterns.gif` — SVD 按强度排列矩阵的模式。用最强的 1、5、20、50 个模式重建一张照片。
4. `../assets/04-low-rank.gif` — 把 d×d 矩阵近似为细高矩阵乘扁宽矩阵：d = 4096、r = 8 时，约 1,680 万个数减少到 65,536 个。
5. `../assets/05-lora.gif` — LoRA 冻结 W，只训练由 r 个模式组成的 B·A，训练从 B = 0 开始；训练完成后，B·A 可以合并进 W。
6. `../assets/06-quantize-block.gif` — 一块权重的 4 位量化：除以缩放因子，取最近的 16 个级别之一，存储 4 位编号。
7. `../assets/07-nf4-levels.gif` — 钟形分布的权重：均匀分布的 16 个级别与按正态分位数放置的 NF4 级别对比，后者在 0 附近更密。
8. `../assets/08-lowrank-vs-quant.gif` — 低秩近似删去弱模式；量化保留全部模式，并加入分散的细小误差。
9. `../assets/09-qlora.gif` — QLoRA：4 位冻结权重在计算前临时还原为 bf16，再加上 16 位的 LoRA 分支；梯度只更新 A 和 B。
10. `../assets/10-memory-7b.gif` — 7B 模型的显存估算（权重、梯度、优化器状态，不含激活值）：全参数微调约 100 GB，LoRA 约 15 GB，QLoRA 约 5 GB。
11. `../assets/11-pattern-lens.png` — 小结：用模式解读本文的每个概念。
