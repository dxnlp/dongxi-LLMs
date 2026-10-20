# Image Upload Order

封面（单独上传，5:2）：`../assets/cover-post-training.png`，中英文版共用英文封面。动画标签为英文，中英文版共用。

1. `../assets/01-same-prompt-two-models.gif` — 同一个请求，两个模型。基座模型最可能写出的下一个词，开启的是另一道题；经过后训练，开头变成了回答。图中概率和回答为示意，参照 OpenAI 公布的 GPT-3 与 InstructGPT 对比，未在真实模型上测量。
2. `../assets/02-two-stages.gif` — 同一个模型，两个阶段。预训练灌入海量原始文本；后训练给同一个模型喂三股小而精的数据：示范、比较、可检查的答案。数据量未按比例绘制。
3. `assets/03-pretraining-vs-post-training.png` — 预训练与后训练对照：目标、数据、信号、成本，以及对模型的改动幅度。
4. `../assets/03-instruction-tuning.gif` — 指令微调。指令只当上下文；模型只在回答的词上接受评分。为便于观看，图中逐个揭示回答的词；实际训练一次就能给所有回答位置评分。
5. `../assets/04-preference-alignment.gif` — 偏好对齐。更受偏好的回答变得更可能，另一个变得更不可能；“牵引绳”让模型留在原模型附近。此处把“裁判”路线（RLHF）和 DPO 简化进同一幅图。
6. `../assets/05-rlvr.gif` — RLVR。模型尝试几种答案，检查器逐个判对错，正确答案的概率一轮轮上升。图中检查器只看最终答案。
