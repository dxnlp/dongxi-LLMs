# Token 到底是什么？从 Unicode 字节到 BPE 与 Token ID

*Token 是 Tokenizer 输出的离散单元。Tokenization 会依次执行规范化、预切分、BPE 等 Tokenization 算法与后处理，把原始文本转换成 Token ID 序列。*

同一个 Tokenizer，把中文里的 `下一个` 编成了一个 Token，却把瑞典语中的单个单词 `språkmodellen` 编成五个子词 Token。

这个结果来自固定版本的 Qwen3 Tokenizer，不能当作语言规律。Token 的边界与 ID 取决于具体词表、Tokenization 算法和配置。

想真正理解 Token，需要从读者看到的文字一路追到模型收到的整数。

## “文本”里面藏着六种单位

先看家庭 Emoji：`👨‍👩‍👧‍👦`。读者看到一个符号；机械拆开后，它包含一个扩展字素簇、七个 Unicode 码点，以及 25 个 UTF-8 字节。

再看汉字 `数`。它是一个可见字符、一个 Unicode 码点，UTF-8 会把它存成三个字节：`E6 95 B0`。

瑞典语单词 `språkmodellen` 在正字法上是一个词，在我们固定的 Tokenizer 里却对应五个子词 Token。

[IMAGE 01] 正字法词、字素簇、Unicode 码点、UTF-8 字节、子词或字节 Token 与 Token ID 是六种不同单位。

这六种单位各自回答不同问题：

- **正字法词**是特定语言书写系统中的词单位。
- **字素簇**是读者感知到的字符，一个字素簇可以由多个码点组成。
- **Unicode 码点**是分配给抽象字符的整数。
- **UTF-8 字节**属于 Unicode 文本的一种可变长编码。
- **子词或字节 Token**是 Tokenization 算法输出的离散单元。
- **Token ID**是该 Token 在词表中的整数索引。

这些单位之间没有普遍的一一对应关系。因此，“这句话有多少个 Token？”还缺少关键信息：使用哪个 Tokenizer、哪个修订版本、哪套配置？

## Tokenizer 决定模型的输入表示

模型不会直接收到 `token`、`下一个` 或 ` spr` 这些可读 Token。Tokenizer 会先把它们变成 `3950`、`108725` 一类整数。Embedding 查表再用每个 ID 选择输入 Embedding 矩阵中的一行。

假设 Tokenizer A 把 ID 42 分配给 `cat`，Tokenizer B 把同一个 ID 分配给一个中文 Token。把 A 的 ID 送进配套 B 的模型，张量形状依然合法，程序也可能继续运行，模型实际采用的 Token 到 ID 映射已经错了。

模型训练时建立了下面这组对应关系：

> **词表 Token ↔ Token ID ↔ 输入 Embedding 矩阵的行索引**

普通文本会经过完整的 Tokenization 流程：

> **原始文本 → 规范化 → 预切分 → BPE / WordPiece / Unigram → 后处理 → Token ID**

Chat 模型还会先序列化消息：

> **消息列表 → Chat Template 序列化 → 格式化文本与控制 Token → Tokenization → Token ID**

每个阶段都会改变最终的 Token ID 序列。

## BPE 在离线阶段学习合并规则

把训练与使用分开后，Byte-pair Encoding 会清楚很多。

在 **Tokenizer 训练**阶段，BPE 从基础符号出发，统计允许合并的相邻符号对，选择一对，把合并后的新 Token 加入词表，重写语料，再在有限词表预算内继续循环。

到了 **运行时编码**阶段，Tokenizer 不会从当前 Prompt 里继续学习。它会执行已配置的规范化与预切分，再按照训练得到的 Merge Rank 生成 Token，并映射到词表 ID；后处理还可能加入特殊 Token。

我们用一个刻意缩小的加权语料，把训练阶段写成了可执行程序：

```text
hug     × 5
hugs    × 3
hugging × 2
```

实测得到三轮合并：

```text
h + u   → hu    计数 10
hu + g  → hug   计数 10
hug + s → hugs  计数 3
```

[IMAGE 02] 微型语料上的三轮 BPE 实测；第一轮最高频计数并列，因此训练规则必须声明如何打破平局。

第一轮还有一个容易忽略的细节：`h+u` 与 `u+g` 都出现十次。“选择最高频符号对”无法唯一确定结果。我们的训练器预先声明按字典序打破平局，所以选择了 `h+u`。

训练结束后，编码器会按这些 Merge Rank 把 `hugs` 编成 `[hugs]`，把 `hugging` 编成 `[hug] [g] [i] [n] [g]`。更长 Token 只有在对应合并路径成立时才会出现。运行时编码也不能简化成“从词表里找最长字符串”。

这条轨迹用来说明 BPE 机制，没有重建 Qwen Tokenizer 当年的训练历史。

## 字节覆盖与压缩效率需要分开

Byte-level BPE 的基础词表包含全部 256 个字节。即使词表没有学到更长 Token，任何有效 UTF-8 文本仍然可以表示为字节 Token。

例如 `数`：

```text
数 → E6 95 B0 → [E6] [95] [B0]
```

语料中的重复暴露会让同一输入逐渐变短。有序合并可以先形成多字节 Token，再跨字符形成多字符 Token：

```text
[E6] + [95]       → [E6 95]
[E6 95] + [B0]    → [数]
[数] + [据]        → [数据]
[数据] + [库]      → [数据库]
```

[IMAGE 03] 基础字节词表提供覆盖能力，学习到的 BPE 合并把重复模式压缩成更大的 Token。

BPE 并不知道 `数据库` 的语义。它学到的是某些相邻模式反复出现，并且值得占用有限的词表容量。

下面三项结论必须分别成立：

> **基础字节提供覆盖能力。学习到的合并提升压缩效率。模型训练形成语言能力。**

一个只看过英文的 Byte Tokenizer 仍能完整保存中文 UTF-8 字节，同时可能把中文切得很碎。模型可以收到合法的字节 ID，对中文的理解却依然很弱。

还有一些 Tokenizer 没有完整的 Byte Fallback。BPE 只是 Tokenization 算法之一。WordPiece 使用不同的评分过程学习子词词表，常见编码器会贪心选择合法 Token；Unigram 从大量候选 Token 开始，逐步剪枝，并对多种切分方案进行概率评分。仅凭算法名称，无法判断它是否具备完整字节覆盖。

## 预处理会改变“同一段”文本

Unicode 允许视觉上等价的文本拥有不同底层序列。

NFC `café` 使用预组合字符 `é`，共有四个码点、五个 UTF-8 字节。NFD `café` 使用 `e` 加组合重音，共有五个码点、六个字节。两者都包含四个字素簇。

固定的 Qwen3 Tokenizer 把两种形式都映射到同一组 ID：`[924, 58858]`。

只有 NFC 源串按码点精确往返。NFD 输入解码后变成 NFC；两段字符串保持 Unicode 规范等价，精确源序列已经改变。返回的 Offset 也停在原始组合重音之前。

[IMAGE 04] 规范化可能改变精确源序列；Chat Template 序列化还会在用户内容之外加入控制 Token。

这项结果提醒我们先说清“往返”的判定标准。视觉相似、规范等价、码点完全相同、Token ID 相同、解码后源串相同，是五种不同性质。

空格同样会影响预切分与词表查找。在同一个 Tokenizer 下：

```text
token  → ID 5839
 token → ID 3950
```

两个输入都占一个 Token，前置空格却改变了词表 Token 与对应的 Token ID。只看数量会漏掉这项差异。

## Chat Template 把消息序列化为模型输入

原始文本 `Hello` 只占一个 Token：ID `9707`。

把相同内容渲染成一条用户消息，并加入 Generation Prompt 后，实际文本变成：

```text
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
```

这段序列占用九个 Token 位置。

新增位置来自模板字面量、角色标记、换行和控制 Token。当前消息没有触发 BPE 学习新词表。在这份 Tokenizer 快照中，`<|im_end|>` 是 EOS Token，`<|endoftext|>` 用作 PAD。两种特殊 Token 承担不同任务，即使它们都可能出现在序列边界附近。

因此，估算上下文窗口或调用成本时，可见消息不能代表送入模型的完整 Token ID 序列。

## 一次固定 Tokenizer 配置下的多语言意外

我们在固定的 `Qwen/Qwen3-0.6B` Tokenizer 修订版本下，对语义大致对应的中文、英文和瑞典语句子进行编码，并关闭新增特殊 Token。

原先的预测顺序是瑞典语最少、中文居中、英文最多。实测顺序完全反了过来：

```text
中文      9 个 Token
英文     11 个 Token
瑞典语   20 个 Token
```

[IMAGE 05] 固定 Qwen3 示例中，中文使用 9 个 Token、英文 11 个、瑞典语 20 个；这组结果无法代表语言排名。

中文切分包含多个多字符 Token：

```text
[小型] [语言] [模型] [学习] [预测] [下一个] [词] [元] [。]
```

瑞典语复合词从内部被切开：

```text
[ spr] [å] [k] [mod] [ellen]
```

正字法词结构无法决定 Token 效率，真正起作用的是训练得到的词表与 Merge Rank。

这组结果只描述三个固定字符串和一个固定修订版本，无法推出中文整体上比英文或瑞典语更节省 Token。语言级结论需要冻结的代表性语料、匹配内容、明确抽样规则、足够的领域覆盖，以及分布统计；每种语言各选一句远远不够。

即使完成这些测量，较少的 Token 也只直接反映压缩或上下文占用；它无法直接度量理解力、生成质量或公平性。

## Token 数量很重要时，应该记录什么

一项可复现的 Tokenization 结果至少要记录：

- Tokenizer 仓库与不可变修订版本；
- 规范化与 Pre-tokenization 行为；
- 是否加入特殊 Token；
- 完整 Chat Template 与 Generation Prompt 策略；
- 原始输入，以及比较时使用的分母；
- ID、可读 Token 和精确 Encode/Decode 检查。

最实用的心智模型可以浓缩成一句话：

> Token 是特定 Tokenizer 配置输出的离散单元；Token ID 是词表索引，Embedding 查表会用它选择模型参数。单词、字符与语义还需要分别讨论。

这些地址进入 Embedding 表后，下一层问题才开始：共享向量如何经过上下文计算，变成位置相关的表示？这会把我们带到模型内部。

## 可复现来源

- [第 2 章：Text, Tokens, and Embeddings](https://github.com/dxnlp/dongxi-LLMs/blob/main/book/chapters/02-text-tokens-and-embeddings.md)
- [透明 Tokenizer 机制实验报告](https://github.com/dxnlp/dongxi-LLMs/blob/main/experiments/reports/2026-08-31-tokenizer-mechanics.md)
- [固定 Qwen3 多语言 Tokenization 报告](https://github.com/dxnlp/dongxi-LLMs/blob/main/experiments/reports/2026-08-30-qwen3-multilingual-tokenization.md)
- [微型可执行 BPE 训练器](https://github.com/dxnlp/dongxi-LLMs/blob/main/src/dongxi_llms/tiny_bpe.py)
- [Hugging Face Tokenizers：Tokenization 流程](https://huggingface.co/docs/tokenizers/main/pipeline)
- [Hugging Face Transformers：编写 Chat Template](https://huggingface.co/docs/transformers/main/chat_templating_writing)
- [Unicode 标准附录 #29：Unicode 文本切分](https://www.unicode.org/reports/tr29/)
