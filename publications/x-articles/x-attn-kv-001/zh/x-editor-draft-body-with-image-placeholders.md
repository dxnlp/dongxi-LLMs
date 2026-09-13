# AI 每生成一个 token，都要把前文重新算一遍吗？

你把一份很长的项目文档交给 AI，让它总结风险。

回答开始出现，一个 token 接着一个 token。生成下一段内容时，模型仍然需要利用前面的文档和已经生成的回答。

难道每多写一个 token，都要把越来越长的全文从头再算一遍？

**KV cache 让模型保留了一部分已经完成的计算。** 但它为什么保留 K 和 V？同样参与 attention 的 Q，为什么通常可以用完就放下？

我们从一次很短的续写开始，再回到这份长文档。

## 每生成一个 token，都要从头再算吗？

把输入缩短为“The sky is”，假设模型接着生成“ blue”，再生成“.”。这里的 token 划分和续写只是示意，没有实际运行 tokenizer 或模型。

模型先处理整个 prompt，建立各层的 KV cache。这一步叫 **prefill**。最后一个 prompt 位置的输出经过 vocabulary head，得到用来选择第一个输出 token 的 logits。

选出“ blue”后，模型把它作为新输入，继续预测“.”。这一步进入逐 token 的 **decode**。

看下面的动画：**“blue”出现前后，哪些方块留在原位，哪些方块是新加入的？**

[IMAGE 01] 先从“The sky is”预测“blue”，再处理“blue”、追加其 K/V，预测“.”。只展示一个层，向量图形为示意；prefill 仅画出最后一个位置的 Q。“Rest of model”包含剩余层和 vocabulary projection。

已有 K/V 留在原位，新位置的 K/V 加在后面。新 token 仍然要经过模型计算，只是省去了对有效历史状态的重复计算。

留意顺序：**选出“blue”时，它的 K/V 还没有生成；把它送回模型、预测后续 token 时，才会计算它的 K/V。**

## 为什么留下 K/V，却通常不留下 Q？

在一个 attention 层里，每个位置都有一个输入向量，也叫 hidden state。三个可学习的投影把它变成 query（Q）、key（K）和 value（V）。

处理“blue”这个位置时，它的新 Q 会与“The”“sky”“is”以及“blue”自身的 K 计算匹配分数。

分数经过缩放和 softmax，变成 attention weights。再按这些权重对对应的 V 加权求和，得到这个 head 的输出。

**下一位置会产生自己的 Q，但仍可能需要读取这些位置的 K/V。** 历史 Q 已经完成了对应位置的计算，普通 decode 无需再次使用它。

把这个过程写成公式，就是下面这一行：

[IMAGE 02] 当前 query 与截至位置 t 的 keys 计算分数，经缩放和 softmax 后，对相应 values 加权求和。这里展示单个 attention head。

其中，t 是当前位置，d 是 query/key 的宽度，“⊤”表示转置。“≤ t”包含过去的位置和当前位置自身。

这解释了 **KV cache** 这个名字：K/V 继续供后续位置读取；Q 随新位置重新产生。特殊实现可能保留更多状态，这里讨论普通 autoregressive attention。

## Context 变长了，旧计算为什么仍然有效？

回到项目文档。开头写着“预算为 100 万元”，后面是进度、人员和交付计划。

现在比较两个操作：

- 在文档末尾追加：“请总结风险。”
- 把开头的“100 万元”改成“50 万元”。

**哪一种操作会让旧 cache 失效？先想一下，再看动画。** 图中的字母代表 token 位置。

[IMAGE 03] 追加 E，已有 A–D 的 cache 保持有效。把 B 改成 X，A 仍可复用，从修改处起需要重新计算相关后缀。向量图形表示状态身份和有效性，未使用实测 activations；重新计算不意味着每个数值都必然改变。

区别来自 **causal attention 的信息方向**：一个位置只能使用它自己和前面的输入，无法读取后来的内容。

所以，末尾追加问题，不会反过来改变文档位置已经完成的计算。修改开头的预算，则可能影响后续位置的表示，不能直接沿用原来的整份 cache。

这个结论要求模型权重、位置处理和推理设置保持一致，且修改前的实际 token prefix 不变。它可以沿 causal attention 和逐位置 MLP 等操作逐层成立。

Cache 对应的是**这段确定输入在特定层的计算状态**。换一篇文档，即使再次出现“预算”这个词，也不能仅凭词相同就复用它的 K/V。

## 新 token 到来时，究竟计算了什么？

现在放大“blue”被送回模型的那一步。只看一个 attention head：**历史 K/V 不动，新位置产生 Q/K/V，新的 Q 读取可用的 K/V。**

[IMAGE 04] 新 hidden state → Q/K/V 投影 → 追加新 K/V → Q 与过去及当前位置的 K 匹配 → softmax → 对 V 加权求和。权重条来自教学用 toy example，未使用实测模型 activations；省略位置处理和模型其他部分。

新 K/V 先加入 cache，再参与 attention，所以当前 Q 也能关注自身位置。这个 head 的输出被后续计算使用后，K/V 仍然保留，供下一位置读取。

图中的 Head output 还需要经过层内后续计算和模型其他部分，才能得到 next-token logits。在输入和执行条件兼容时，cached 与 uncached 结果应在适当的数值误差范围内一致。

## 省下重复计算，为什么又增加了显存开销？

项目文档越长，需要保留的位置就可能越多。**使用随序列增长的完整 KV cache 时，新加入的位置会继续占用存储空间。**

可以按维度数一遍：多少条序列、多少层、每层保留多少位置、多少个 KV heads、每个 head 多宽、每个元素占多少 bytes。

假设 K/V 分开存储、宽度相同、各层结构一致，乘起来就是：

[IMAGE 05] 两份 tensor × batch size B × 层数 L × 保留位置数 T × KV heads 数 × head width d × 每个元素的字节数 b。系数 2 对应 K 和 V。

给这份文档安排一个假设配置：24 层、1 条序列、8 个 KV heads、head width 为 64，每个元素占 2 bytes。

- 保留 **4,096** 个位置：KV tensor 共 **192 MiB**。
- 保留 **8,192** 个位置：增加到 **384 MiB**。

看动画里的两次变化：先增加位置，再回到原来的长度，比较 KV heads 更少的另一种配置。

[IMAGE 06] 位置翻倍：192 → 384 MiB。单独比较架构配置，固定 4,096 个位置，将 KV heads 从 8 个减为 4 个：192 → 96 MiB。每列代表 1,024 个位置，每行代表一个 KV head 的 K/V；其他条件不变，未在运行时删除 head。

这些是 **tensor 存储量的计算值**，未测量 GPU 实际分配量，也不包含模型权重、临时 buffer、metadata 和 allocator 开销。不同层若使用不同 window 或 KV 表示，需要分别计算。

KV heads 为什么可以更少？**Grouped-query attention（GQA）允许多个 query heads 共享 K/V heads。**

例如，Q₁、Q₂ 读取同一组 K/V，Q₃、Q₄ 读取另一组。四个 query heads 保留各自的 attention weights 和输出，只需要两组 K/V。

这是模型架构选择。存储量可以据此计算，质量和实际速度仍要另行评估。

## 同一份文档，换个问题，还能接着用吗？

你刚让 AI 总结完风险，又想知道截止日期。假设两条请求这样组织：

**请求 A：** [同一份项目文档] + “总结风险。”

**请求 B：** [同一份项目文档] + “列出截止日期。”

如果文档部分的实际 token prefix 完全一致，模型与执行设置兼容，而且系统仍保留这部分 cache，就可以复用它，再分别计算两个问题及其续写。这叫 **prefix caching**。

三个词可以这样区分：**prefill 是处理 prompt 的计算阶段；KV cache 是保存下来的 K/V 状态；prefix caching 是复用已有前缀状态的策略。** 即使没有启用 prefix caching，普通生成也可以使用 KV cache。

vLLM 官方的典型例子是：针对同一份年报或软件手册反复提问。上面的项目文档就是同一种用法：第一次通过 prefill 建立缓存，后续命中共同前缀时，直接复用相应 K/V，省去这部分重复计算。

新的问题仍需继续做 prefill，并读取文档的 K/V；新的回答仍需逐 token decode。**Prefix caching 主要节省重复的 prefill 工作，不会省掉新回答的生成计算。**

如果请求 B 在文档前多放了一段不同的说明，共同 prefix 就可能缩短。即使后面的文档文字相同，也不能直接认定它的 causal states 相同。

一次回答内部的逐 token 复用，与跨请求复用共同 prefix，是两种策略。Tokenization、位置处理、模型权重、adapter 或多模态输入变化，都可能影响复用。

**聊天记录还在，也不保证 KV cache 还在。** 如果 cache 已被释放或 evict，系统需要重建所需状态。是否跨请求保留，由 serving system 的缓存策略决定。

## 如果重新设计 KV 的来源和共享方式呢？

前面从“每层保存自己的 K/V”出发。理解这些计算依赖，就能看懂 DeepSeek-V4.1-Flash 的几个变化。

**先看来源：Causal Encoder–Decoder。** Decoder 的 global KV 来自 causal encoder 的最终输出。各 decoder 层仍然生成自己的 Q，以及 local sliding-window KV。

**再看跨层：Cross-layer sharing。** 在所讨论的配置中，第 21 层利用 encoder output 生成一份 decoder global KV bank，供第 21–40 层使用。

可以盯住第 21、22 层看：它们读取共享的 global entries，但 Q 和 local states 各自保留。共享条目不会强制产生相同的 attention weights；sparse selection 决定读取哪些 global entries。

**最后看 K/V 的表示：Shared KV representation。** 在 reference core-attention kernel 中，同一个被选中的向量既参与 Q 的匹配评分，也参与输出的加权求和。

看动画：一份 global KV 如何连接多个层？放大一个向量后，它又参与了哪两项计算？

[IMAGE 07] 第 21 层从 encoder output 生成 global KV，decoder 各层共享它，同时保留自己的 Q 和 local KV。下方放大一个被选中的向量，展示匹配评分与加权求和两种用途；没有新增一份 cache，中间层和 sparse indexer 细节已省略。

模型仍保留 local caches、indexer state 等数据。Shared KV representation 与把两份独立 K/V 数组相邻存放，有明确区别。

回到项目文档：假设这次输入有 **10 万个 token**。报告中的 serving 方案可以这样处理：

1. Encoder 处理完整 prompt，提供构建 decoder global KV 的表示。
2. Decoder 随后处理最后 **128 个 prompt tokens**，重建 local state。更早的历史仍可通过 global KV 访问。

这说明 global KV 的来源会影响 prefill 中哪些计算可以调整。生成后续 token 时，新输入仍然经过 causal encoder 和 decoder。

**这里的 SWA Bounded Replay 是近似计算。** 重建的 local state 与完整 decoder forward pass 不完全相同。

报告称其评估中的回答质量影响很小；我们没有独立复现，公开的易读 reference implementation 也没有 benchmark 这一 serving 优化。

## 再回到开头的那份文档

现在可以沿着同一场景，把 KV cache 连起来：

- **继续写回答：** 新位置产生自己的 Q，读取仍然有效的历史 K/V。
- **修改文档开头：** 从实际 token prefix 的变化处重新检查复用范围。
- **换一个问题：** 检查共同 prefix、执行条件，以及 cache 是否仍被保留。
- **读一份更长的文档：** 关注保留位置带来的存储增长，以及架构如何组织这些状态。

再看到一种 KV-cache 优化，可以用四个问题来理解：

**保存什么？** 普通 KV cache 保存各层已处理位置的 K/V，供后续位置读取。具体架构也可能采用共享的 KV 表示。

**从哪里生成？** 普通 self-attention 从本层输入的 hidden states 生成 K/V；本文的 Causal Encoder–Decoder 则从 encoder 最终输出生成 decoder global KV。

**什么条件下能复用？** 对普通精确复用，需要相同的实际 token prefix、兼容的模型和执行设置，以及仍然可用的缓存状态。追加内容保持过去的 causal states 有效；修改前文则要重新检查复用范围。

**减少了哪一项开销？** 普通 KV cache 减少历史位置的重复计算，但占用存储空间。减少 KV heads 或跨层共享可以减少相应的存储；Bounded Replay 调整 prefill 计算，并引入近似。实际速度与质量仍需单独评估。
