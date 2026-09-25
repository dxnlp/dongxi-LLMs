# X Article Inline Image Plan

Use this after pasting the clean body into X Articles.

Rule: insert images from bottom to top so earlier block positions do not shift.

- Title: `Prefill and Decode`
- Clean body HTML: `x-editor-clean-body.html`
- Clean body Markdown: `x-editor-clean-body.md`
- Total clean body blocks: `55`
- Images: `7`

## Reverse Insertion Order

### IMAGE 07

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-prefill-decode-001/assets/07-handoff.gif`
- Exists: `True`
- Insert after block index: `51`
- Target text: 交接也可以边算边进行：前面某层的 KV 准备好，先发出去，同时继续计算后面的层。这样，部分传输与计算可以重叠；末尾尚未传完的数据、格式转换，仍然要花时间。
- Next text: 这是一项第三方演示，我们没有做本地测速。能否受益，要看具体芯片、模型、软件与网络，不能推导出任意 Spark + Mac 组合都更快，也不能把演示当作当前版本的一键部署承诺。

### IMAGE 06

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-prefill-decode-001/assets/06-scheduling.gif`
- Exists: `True`
- Insert after block index: `42`
- Target text: 另一个办法是分配不同的资源：一组 worker 专门处理输入，另一组接着生成回答。这叫 prefill/decode disaggregation，两边可以分别调整容量和调度策略。
- Next text: 这时必须完成一次交接。后一组要拿到文档对应的 KV，以及继续生成所需的状态，例如 token 位置、已经选出的新 token，才能从原来的地方接着算。

### IMAGE 05

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-prefill-decode-001/assets/05-workloads.gif`
- Exists: `True`
- Insert after block index: `32`
- Target text: B：“请写一篇两千字的演讲稿。” 输入短、输出长，反复进行的 decode 可能占据大部分等待时间。
- Next text: 两个阶段使用同一个 decoder-only Transformer、同一套参数，硬件面对的工作却不同。

### IMAGE 04

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-prefill-decode-001/assets/04-decode-loop.gif`
- Exists: `True`
- Insert after block index: `23`
- Target text: 图里把这两个新 token 简写成 y1、y2。沿着外侧回路看：上一步选出的输出，就是下一步送入模型的输入。 实际 token 可以是字、词片段或其他文本单元，未必是完整词。
- Next text: 不妨数一下：如果原来有四个输入 token，选出 y1 时，cache 仍有四个位置；把 y1 送回模型处理后，才增加到五个，并得到预测 y2 所需的输出。

### IMAGE 03

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-prefill-decode-001/assets/03-first-token.gif`
- Exists: `True`
- Insert after block index: `19`
- Target text: 留意下面的变化：第一个新 token 刚出现时，cache 里仍只有输入位置的 KV。 新 token 自己的 KV，要等它被送回模型才会产生。
- Next text: Decode：刚写出来的内容，为什么又要送回去？

### IMAGE 02

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-prefill-decode-001/assets/02-parallel.gif`
- Exists: `True`
- Insert after block index: `14`
- Target text: 关键在那条斜线：第二个位置需要 K₁/V₁，它们从这一层的输入就能算出来，无须等待第一个位置的 attention 输出 O₁。 因此，所需 Q/K/V 准备好后，两行 attention 可以批量、并行计算。
- Next text: “并行”不要求 GPU 在同一个瞬间完成所有操作。Causal mask 仍限定每个位置只能读取自己和前文的 K/V；读取范围有先后边界，计算输出无需按位置排队。

### IMAGE 01

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-prefill-decode-001/assets/01-lifecycle.gif`
- Exists: `True`
- Insert after block index: `6`
- Target text: 这里还有排队、输入处理和网络的影响，界面也可能合并显示多个 token。仅凭屏幕停顿，无法确定瓶颈；理解 prefill 和 decode，能帮助我们把问题问得更具体。
- Next text: Prefill：三条结论之前，要先处理几十页输入
