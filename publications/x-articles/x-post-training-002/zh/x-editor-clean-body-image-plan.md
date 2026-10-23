# X Article Inline Image Plan

Use this after pasting the clean body into X Articles.

Rule: insert images from bottom to top so earlier block positions do not shift.

- Title: `指令微调：同一个损失，不同任务`
- Clean body HTML: `x-editor-clean-body.html`
- Clean body Markdown: `x-editor-clean-body.md`
- Total clean body blocks: `80`
- Images: `10`

## Reverse Insertion Order

### IMAGE 10

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-002/assets/10-token-share.gif`
- Exists: `True`
- Insert after block index: `64`
- Target text: 组合数据时还要注意长度。损失按计算损失的 token 求平均，回答越长，在损失中的占比越大：一条 500 个 token 的回答，权重约为一条 2 个 token 回答的 250 倍。因此，组合指令数据时应按 token 数量平衡，只看示...
- Next text: 指令微调的能力与局限

### IMAGE 09

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-002/assets/09-one-position-shift.gif`
- Exists: `True`
- Insert after block index: `56`
- Target text: 如果没有错开这一位，形状仍然匹配，代码可以运行，但每个预测都和当前位置的 token 对齐，模型学习的是复制当前 token。
- Next text: 训练前，建议把计算损失的 token 解码出来检查一遍：结果应当只包含各个回答及其 <|im_end|>。

### IMAGE 08

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-002/assets/08-masked-not-untouched.gif`
- Exists: `True`
- Insert after block index: `50`
- Target text: 损失掩码只决定哪些位置的预测计入损失，不影响 attention（注意力） 读取哪些 token。回答依赖于对提示的读取，因此回答的损失产生的梯度会通过注意力传回提示中的 token，并更新处理这些 token 的参数。
- Next text: 也就是说，被掩码的 token 不计算损失，但仍然参与训练。

### IMAGE 07

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-002/assets/07-what-gets-loss.gif`
- Exists: `True`
- Insert after block index: `45`
- Target text: <|im_end|> 之后的换行：不计算。 此时生成已经结束。
- Next text: 写成公式，mₜ = 1 表示位置 t 计算损失，否则为 0：

### IMAGE 06

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-002/assets/06-three-forms-summary.png`
- Exists: `True`
- Insert after block index: `36`
- Target text: 下图把三种形态放在一起对照：
- Next text: 哪些 token 计算损失

### IMAGE 05

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-002/assets/05-three-forms.gif`
- Exists: `True`
- Insert after block index: `35`
- Target text: 训练形态：损失掩码标出序列中参与计算损失的 token。模型读取整个序列，但只有被标出的 token 计入损失。
- Next text: 下图把三种形态放在一起对照：

### IMAGE 04

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-002/assets/04-recipe.gif`
- Exists: `True`
- Insert after block index: `30`
- Target text: 评测：在训练中没有出现过的任务上评估效果。
- Next text: 第 2、3 步涉及较多细节，下面分别说明。

### IMAGE 03

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-002/assets/03-flan.gif`
- Exists: `True`
- Insert after block index: `22`
- Target text: FLAN（2021） 把 60 多个 NLP 任务改写成自然语言指令，用来微调一个 137B 的模型。在训练中没有出现过的任务上，它在 25 项测试中有 20 项超过了零样本的 175B GPT-3。
- Next text: 这说明指令微调学到的是遵循指令的通用能力，可以泛化到训练中没有出现过的任务。因此，指令微调是后训练的第一步：先让模型能够按指令作答，再由偏好对齐和强化学习提高回答质量。

### IMAGE 02

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-002/assets/02-lima.gif`
- Exists: `True`
- Insert after block index: `21`
- Target text: LIMA（2023） 只用 1,000 条精选示例微调 65B 的 LLaMA，没有使用强化学习。在人工评测中，43% 的回答与 GPT-4 相当或更好。作者据此认为，模型的知识几乎全部来自预训练，少量指令数据主要决定回答的方式。
- Next text: FLAN（2021） 把 60 多个 NLP 任务改写成自然语言指令，用来微调一个 137B 的模型。在训练中没有出现过的任务上，它在 25 项测试中有 20 项超过了零样本的 175B GPT-3。

### IMAGE 01

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-002/assets/01-alpaca-recipe.gif`
- Exists: `True`
- Insert after block index: `3`
- Target text: 这些示例由 OpenAI 的 text-davinci-003 自动生成，花费不到 500 美元。训练在 8 块 GPU 上用了 3 个小时，花费不到 100 美元。团队做了一次小规模盲测，Alpaca 和 text-davinci-00...
- Next text: Alpaca 也会给出错误答案。问它坦桑尼亚的首都，它回答达累斯萨拉姆，正确答案是多多马。
