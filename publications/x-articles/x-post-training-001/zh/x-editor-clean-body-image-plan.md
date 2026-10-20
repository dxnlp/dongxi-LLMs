# X Article Inline Image Plan

Use this after pasting the clean body into X Articles.

Rule: insert images from bottom to top so earlier block positions do not shift.

- Title: `预训练之后，为什么还要后训练？`
- Clean body HTML: `x-editor-clean-body.html`
- Clean body Markdown: `x-editor-clean-body.md`
- Total clean body blocks: `55`
- Images: `6`

## Reverse Insertion Order

### IMAGE 06

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-001/assets/05-rlvr.gif`
- Exists: `True`
- Insert after block index: `49`
- Target text: 所以 RLVR 很适合那些基座模型不擅长的长链条问题。需要提醒的是：检查器的可靠程度取决于它的规则。如果它只看最终答案，就无法说明中间每一步都对。
- Next text: 小结

### IMAGE 05

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-001/assets/04-preference-alignment.gif`
- Exists: `True`
- Insert after block index: `44`
- Target text: 没有这根牵引绳，模型可能一味追分，离出发点越来越远，把预训练学到的东西忘掉不少。
- Next text: 3. RLVR：让检查器来判对错

### IMAGE 04

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-001/assets/03-instruction-tuning.gif`
- Exists: `True`
- Insert after block index: `37`
- Target text: 它也叫 supervised fine-tuning（SFT，监督微调），因为每条指令都配有一份可以模仿的正确回答。在许多种任务上训练之后，模型听指令的能力会整体变强，连没见过的任务也能做得更好。
- Next text: 2. 偏好对齐：通过比较来学

### IMAGE 03

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-001/zh/assets/03-pretraining-vs-post-training.png`
- Exists: `True`
- Insert after block index: `27`
- Target text: 粗略地说：预训练决定模型知道什么，后训练决定它怎样运用这些知识。
- Next text: 数据少，改变却可以很大。有一项研究在大型指令数据里只混入 500 条安全示例，模型给出有害回答的情况就明显减少了。

### IMAGE 02

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-001/assets/02-two-stages.gif`
- Exists: `True`
- Insert after block index: `21`
- Target text: 后训练并没有新建一个模型，它接着训练同一个模型。变化的是数据，以及告诉模型“怎样算更好”的那个信号。
- Next text: 目标。 预训练：猜准文本里的下一个词。后训练：按人们希望的方式行事，听懂指令、保证安全、把题答对。

### IMAGE 01

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-001/assets/01-same-prompt-two-models.gif`
- Exists: `True`
- Insert after block index: `10`
- Target text: 这篇文章讲三件事：为什么需要后训练，它和预训练有什么区别，以及它通常包含的三种技术。
- Next text: 只有预训练，还差什么？
