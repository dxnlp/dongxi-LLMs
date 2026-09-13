# Image Upload Order

01. `../assets/prefill-decode.gif` — 先从“The sky is”预测“blue”，再处理“blue”、追加其 K/V，预测“.”。只展示一个层，向量图形为示意；prefill 仅画出最后一个位置的 Q。“Rest of model”包含剩余层和 vocabulary projection。
02. `../assets/math-attention.png` — 当前 query 与截至位置 t 的 keys 计算分数，经缩放和 softmax 后，对相应 values 加权求和。这里展示单个 attention head。
03. `../assets/append-edit.gif` — 追加 E，已有 A–D 的 cache 保持有效。把 B 改成 X，A 仍可复用，从修改处起需要重新计算相关后缀。向量图形表示状态身份和有效性，未使用实测 activations；重新计算不意味着每个数值都必然改变。
04. `../assets/decode-step.gif` — 新 hidden state → Q/K/V 投影 → 追加新 K/V → Q 与过去及当前位置的 K 匹配 → softmax → 对 V 加权求和。权重条来自教学用 toy example，未使用实测模型 activations；省略位置处理和模型其他部分。
05. `../assets/math-memory.png` — 两份 tensor × batch size B × 层数 L × 保留位置数 T × KV heads 数 × head width d × 每个元素的字节数 b。系数 2 对应 K 和 V。
06. `../assets/memory-growth.gif` — 位置翻倍：192 → 384 MiB。单独比较架构配置，固定 4,096 个位置，将 KV heads 从 8 个减为 4 个：192 → 96 MiB。每列代表 1,024 个位置，每行代表一个 KV head 的 K/V；其他条件不变，未在运行时删除 head。
07. `../assets/global-sharing.gif` — 第 21 层从 encoder output 生成 global KV，decoder 各层共享它，同时保留自己的 Q 和 local KV。下方放大一个被选中的向量，展示匹配评分与加权求和两种用途；没有新增一份 cache，中间层和 sparse indexer 细节已省略。
