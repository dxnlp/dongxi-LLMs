# X Article Inline Image Plan

Use this after pasting the clean body into X Articles.

Rule: insert images from bottom to top so earlier block positions do not shift.

- Title: `AI 每生成一个 token，都要把前文重新算一遍吗？`
- Clean body HTML: `x-editor-clean-body.html`
- Clean body Markdown: `x-editor-clean-body.md`
- Total clean body blocks: `82`
- Images: `7`

## Reverse Insertion Order

### IMAGE 07

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-attn-kv-001/assets/global-sharing.gif`
- Exists: `True`
- Insert after block index: `63`
- Target text: 看动画：一份 global KV 如何连接多个层？放大一个向量后，它又参与了哪两项计算？
- Next text: 模型仍保留 local caches、indexer state 等数据。Shared KV representation 与把两份独立 K/V 数组相邻存放，有明确区别。

### IMAGE 06

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-attn-kv-001/assets/memory-growth.gif`
- Exists: `True`
- Insert after block index: `41`
- Target text: 看动画里的两次变化：先增加位置，再回到原来的长度，比较 KV heads 更少的另一种配置。
- Next text: 这些是 tensor 存储量的计算值，未测量 GPU 实际分配量，也不包含模型权重、临时 buffer、metadata 和 allocator 开销。不同层若使用不同 window 或 KV 表示，需要分别计算。

### IMAGE 05

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-attn-kv-001/assets/math-memory.png`
- Exists: `True`
- Insert after block index: `37`
- Target text: 假设 K/V 分开存储、宽度相同、各层结构一致，乘起来就是：
- Next text: 给这份文档安排一个假设配置：24 层、1 条序列、8 个 KV heads、head width 为 64，每个元素占 2 bytes。

### IMAGE 04

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-attn-kv-001/assets/decode-step.gif`
- Exists: `True`
- Insert after block index: `31`
- Target text: 现在放大“blue”被送回模型的那一步。只看一个 attention head：历史 K/V 不动，新位置产生 Q/K/V，新的 Q 读取可用的 K/V。
- Next text: 新 K/V 先加入 cache，再参与 attention，所以当前 Q 也能关注自身位置。这个 head 的输出被后续计算使用后，K/V 仍然保留，供下一位置读取。

### IMAGE 03

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-attn-kv-001/assets/append-edit.gif`
- Exists: `True`
- Insert after block index: `25`
- Target text: 哪一种操作会让旧 cache 失效？先想一下，再看动画。 图中的字母代表 token 位置。
- Next text: 区别来自 causal attention 的信息方向：一个位置只能使用它自己和前面的输入，无法读取后来的内容。

### IMAGE 02

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-attn-kv-001/assets/math-attention.png`
- Exists: `True`
- Insert after block index: `17`
- Target text: 把这个过程写成公式，就是下面这一行：
- Next text: 其中，t 是当前位置，d 是 query/key 的宽度，“⊤”表示转置。“≤ t”包含过去的位置和当前位置自身。

### IMAGE 01

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-attn-kv-001/assets/prefill-decode.gif`
- Exists: `True`
- Insert after block index: `9`
- Target text: 看下面的动画：“blue”出现前后，哪些方块留在原位，哪些方块是新加入的？
- Next text: 已有 K/V 留在原位，新位置的 K/V 加在后面。新 token 仍然要经过模型计算，只是省去了对有效历史状态的重复计算。
