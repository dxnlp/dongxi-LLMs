# X Article Inline Image Plan

Use this after pasting the clean body into X Articles.

Rule: insert images from bottom to top so earlier block positions do not shift.

- Title: `Token 到底是什么？从 Unicode 字节到 BPE 与 Token ID`
- Clean body HTML: `x-editor-clean-body.html`
- Clean body Markdown: `x-editor-clean-body.md`
- Total clean body blocks: `106`
- Images: `5`

## Reverse Insertion Order

### IMAGE 05

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-bpe-001/zh/assets/05-multilingual-measurement.png`
- Exists: `True`
- Insert after block index: `79`
- Target text: 瑞典语 20 个 Token
- Next text: 中文切分包含多个多字符 Token：

### IMAGE 04

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-bpe-001/zh/assets/04-interface-surprises.png`
- Exists: `True`
- Insert after block index: `59`
- Target text: 只有 NFC 源串按码点精确往返。NFD 输入解码后变成 NFC；两段字符串保持 Unicode 规范等价，精确源序列已经改变。返回的 Offset 也停在原始组合重音之前。
- Next text: 这项结果提醒我们先说清“往返”的判定标准。视觉相似、规范等价、码点完全相同、Token ID 相同、解码后源串相同，是五种不同性质。

### IMAGE 03

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-bpe-001/zh/assets/03-byte-coverage-compression.png`
- Exists: `True`
- Insert after block index: `49`
- Target text: [数据] + [库] → [数据库]
- Next text: BPE 并不知道 数据库 的语义。它学到的是某些相邻模式反复出现，并且值得占用有限的词表容量。

### IMAGE 02

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-bpe-001/zh/assets/02-tiny-bpe-training.png`
- Exists: `True`
- Insert after block index: `37`
- Target text: hug + s → hugs 计数 3
- Next text: 第一轮还有一个容易忽略的细节：h+u 与 u+g 都出现十次。“选择最高频符号对”无法唯一确定结果。我们的训练器预先声明按字典序打破平局，所以选择了 h+u。

### IMAGE 01

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-bpe-001/zh/assets/01-six-units.png`
- Exists: `True`
- Insert after block index: `7`
- Target text: 瑞典语单词 språkmodellen 在正字法上是一个词，在我们固定的 Tokenizer 里却对应五个子词 Token。
- Next text: 这六种单位各自回答不同问题：
