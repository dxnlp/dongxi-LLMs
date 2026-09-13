# LLM 为什么缓存 K 和 V，却通常不缓存 Q？

生成一个新 token，为什么可以省去前面许多 token 的重复计算？沿着这个问题，可以把 attention、长对话的显存开销，以及重新设计 KV cache 的新架构串起来。

## 一个 prompt，两次预测

假设输入是未完成的句子“The sky is”，模型接着生成“ blue”，然后生成“.”。这里的 token 划分和续写仅用于说明机制，没有实际运行 tokenizer 或模型。

模型先处理 prompt，这个阶段叫 **prefill**。最后一个 prompt 位置的输出经过 vocabulary head 得到 logits，用来选择第一个输出 token：“ blue”。

随后，模型将“ blue”作为新输入，预测下一个 token。此时仍然需要利用“The sky is”的信息。如果每次都重新计算整个增长中的序列，会重复大量工作。KV cache 让其中可复用的结果保留下来。

注意先后顺序：选出“ blue”时，还没有计算这个新 token 的 K/V；将它送回模型、预测后续 token 时，才会产生对应的 K/V。

[IMAGE 01] Prefill 预测“blue”；将它送回模型后，先生成并追加其 K/V，再由新的 Q 读取全部四个位置，预测“.”。图中展示一个层；“Rest of model”包含剩余层和 vocabulary projection。Token 划分和向量图形仅为示意；prefill 只画出最后一个位置的 Q。

## 为什么叫 KV cache？

在某个 attention 层，每个位置都有一个进入本层的 hidden state。三个可学习的投影分别产生 query（Q）、key（K）和 value（V）。

当前 Q 与可访问的 K 计算匹配分数。屏蔽不可访问的位置并归一化后，得到 attention weights，再按这些权重对 V 加权求和。

对于新位置 t 的一个 attention head，核心计算可以写成：

[IMAGE 02] 单个 head 的 attention：当前 query 与截至位置 t 的 keys 计算分数，除以 d 的平方根，经 softmax 归一化后，对对应的 values 加权求和。

d 表示 query/key 的宽度。下标“≤ t”包含历史位置以及当前位置 t；上标“⊤”表示转置。单 token decode 中，如果提供的条目全部属于过去或当前，且没有 padding 等额外限制，就不存在需要屏蔽的未来位置。一次处理多个新 token 时，mask 需要正确计入 cached prefix 的长度。

历史 Q 已经完成了对应位置的 attention 计算。新位置会生成自己的 Q，其计算无需再次使用历史 Q。历史 K/V 则仍然是新 Q 可以读取的信息来源。

因此，标准 autoregressive attention 保留 K/V。“普通 decode 不需要历史 Q”有明确适用范围；特殊实现可能保留更多中间状态。

## 表示依赖 context，为什么还能复用？

一个自然的疑问是：token 的表示会受 context 影响，对话不断增长，旧表示为什么还能继续使用？

关键在于 **causal attention 限制了信息的影响方向**。

位置 j 只能使用截至 j 的输入。追加位置 j + 1，无法向 j 传递新信息。在权重固定、位置处理一致、推理设置不变的条件下，过去的计算仍然有效。这个性质可以沿着 causal attention 和逐位置 MLP 等操作，逐层传递。

所以，各层能够保留自己先前计算的 K/V。

[IMAGE 03] 追加 E，已有 A–D 的 cache 保持有效。将 B 改为 X，A 仍可复用，依赖该修改的后缀需要重新计算。向量图形表示 cache 的身份和有效性，没有使用实测 activations；重新计算也不意味着每个数值都会改变。

KV cache 对应特定 prefix、位置、层和模型执行状态。不能为词表中的“bank”保存一对通用 K/V，再在所有句子里重复使用。

这里还有一个容易忽略的细节：若输入 embedding 和位置处理相同，即使前文不同，第一层最初的 Q/K/V 投影也可能相同。经过 attention 混合前文后，更深层的 hidden states 才可能不同。Context 的影响需要沿实际计算过程分析。

## Prefill 与 decode：保留什么，新增什么？

**Prefill 建立 prompt 的 KV cache。** 各层可以并行处理 prompt 的不同位置，同时遵守 causal mask。

**Decode 每次增加一个位置。** 新位置仍然经过模型的每一层。把镜头放大到一个 attention head：只对这个位置的新 hidden state h 计算 Q、K、V 投影。

[IMAGE 04] 新 h → Q/K/V 投影 → 追加新 K/V → Q 读取过去和当前位置的 K → softmax weights → 对 V 加权求和。已有 K/V 保持不变。权重条来自明确的 toy example 计算，没有使用实测模型 activations；省略位置处理和模型其他部分。

注意这个顺序：**新 K/V 先加入 cache，再计算 attention**，因此当前 Q 也能关注自身位置。这个 head 的输出被下游计算使用后，K/V 继续保留，供后续位置读取；普通 decode 无需保留历史 Q。

图中的 Head output 对应一个 head 的 attention output。后续层内计算和模型其他部分仍需运行，才能得到 next-token logits。在输入和执行条件兼容时，cached 与 uncached 结果应在适当的数值误差范围内一致；cache 减少重复计算，不会增加模型学到的知识。

## KV cache 的内存开销

假设 K/V 分开存储、宽度相同、各层结构一致，每条序列保留 T 个位置，那么逻辑存储量为：

[IMAGE 05] 独立存储 K/V 时的 payload（bytes）：两份 tensor × batch size × 层数 × 保留位置数 × KV heads 数 × head width × 每个元素的字节数。

各因子依次为 batch size B、层数 L、保留位置数 T、每层 KV heads 数、head width d，以及每个元素的字节数 b。系数 2 对应 K 和 V。若不同层使用不同 window 或 KV representation，就需要按实际 tensor 分别计算。

以一个假设配置为例：24 层、1 条序列、4,096 个位置、8 个 KV heads、head width 为 64，每个元素占 2 bytes。结果为 201,326,592 bytes，即 **192 MiB**。

保留位置增加到 8,192，得到 384 MiB。在 4,096 个位置下，将 KV heads 改为 4 个，得到 96 MiB。这些都是计算值，没有测量 GPU 实际分配量，也没有包含模型权重、临时 buffer、metadata 和 allocator 开销。

[IMAGE 06] 保留位置从 4,096 增加到 8,192，KV tensor memory 从 192 增加到 384 MiB。随后单独比较两种架构配置：固定 4,096 个位置，KV heads 从 8 个减为 4 个，对应 192 → 96 MiB。每列代表 1,024 个位置，每行代表一个 KV head 的 K/V。其他条件保持不变；数值为 tensor payload 计算值。Head 数量的变化表示架构配置比较，不涉及运行时删除 head。

Grouped-query attention（GQA）让多个 query heads 共享 K/V heads。例如，4 个 query heads 可以使用 2 个 KV heads：Q₁、Q₂ 读取 KV₁，Q₃、Q₄ 读取 KV₂，同时保留各自的 attention weights 和输出。减少 KV heads 属于架构选择，对质量和速度的影响需要单独评估，存储量计算无法直接给出这些结论。

## 另一条请求，能复用同一份 cache 吗？

假设两条请求以同一份长文档开头，末尾的问题不同。Serving system 可以在条件兼容时，复用共同 prefix 的计算，再分别处理不同的后缀。

匹配依据是实际模型输入和执行条件。Tokenization、位置处理、模型权重、adapter 或多模态输入发生变化，都可能使复用失效。

> 请求 A：[同一份文档] + “总结风险。”
>
> 请求 B：[同一份文档] + “列出截止日期。”

如果文档 prefix 的 token IDs 完全一致，执行设置也兼容，就可以共享它对应的 cached states，随后分别处理两个问题。如果相同文本前面接着不同内容，仅凭这段文本相同，还不足以判断其 causal states 可以复用。

一次 generation 内的复用，与跨请求的 prefix caching，属于两种不同策略。界面里保留着对话记录，不代表对应 KV 始终驻留内存。如果状态被 evict，运行时需要重新构建所需状态。

请求结束后，request-local cache 可以释放；allocator 可能将空闲内存保留给后续工作，prefix-caching 系统也可能有意保留可复用的 blocks。仅凭 reserved memory 较高，无法判断旧请求是否仍然活跃。

## DeepSeek 如何改变这些依赖关系？

每层各存一份 K/V，是理解 cache 的基础模型。实际架构可以采用不同布局。DeepSeek-V4.1-Flash 展示了三个需要分别理解的设计。

**Causal Encoder–Decoder 改变来源。** Decoder 的 global KV 由 causal encoder 的最终输出生成。各 decoder 层仍然计算自己的 main Q 和 local sliding-window KV。Prompt 和后续生成的 token 都会经过 causal encoder。

**Cross-layer sharing 改变跨层复用方式。** 已发布配置中，第 21 层生成 decoder global KV bank，供第 21–40 层使用。各层保留不同的 Q 和 local states，所以共享条目不会强制产生相同的 attention weights。Sparse selection 决定读取哪些 global entries。

**Shared KV representation 改变 K/V 的表示方式。** 在 reference core-attention kernel 中，同一个被选中的向量参与 Q 的匹配评分，也参与输出的加权求和。这与将独立 K/V 数组相邻存放有明确区别。模型仍保留 local caches、indexer state 等数据。

[IMAGE 07] 第 21 层利用 encoder output 生成一份 global KV bank，供 decoder 各层复用；每层保留自己的 Q 和 local KV。下方放大同一个被选中的向量，展示它如何参与 Q matching 和加权求和；该细节图不代表新增一份 cache。中间层和 sparse indexer 的细节已省略。

这种依赖关系支持不同的 prefill 策略。假设输入是 10 万个 token 的项目历史，encoder 处理完整 prompt，并提供构建 decoder global KV 的表示。报告中的 serving 方案随后只让 decoder 处理最后 128 个 prompt tokens，重建 local state。更早的历史仍可通过 global KV 访问。

这里的 **SWA Bounded Replay 是近似计算**。重建的 local state 与完整 decoder forward pass 不完全相同。DeepSeek 报告称，其评估中对回答质量的影响很小；我们没有独立复现该结论。公开的易读 reference implementation 也没有对这一 serving 优化进行 benchmark。

## 最后，记住四个问题

看到任何 KV-cache 优化，先问：**保存了什么？从哪里生成？什么条件下可以复用？减少了哪一项开销？**

普通 causal attention 中，历史 K/V 保持有效，新位置产生自己的 Q。新架构则可以改变表示、共享方式或状态重建策略。将这些选择分开，就能更清楚地区分精确复用、架构调整，以及明确引入的近似计算。
