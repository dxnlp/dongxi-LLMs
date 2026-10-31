# X Article Inline Image Plan

Use this after pasting the clean body into X Articles.

Rule: insert images from bottom to top so earlier block positions do not shift.

- Title: `Reward model：人的选择怎样变成一个分数`
- Clean body HTML: `x-editor-clean-body.html`
- Clean body Markdown: `x-editor-clean-body.md`
- Total clean body blocks: `117`
- Images: `12`

## Reverse Insertion Order

### IMAGE 12

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-005/assets/11-three-uses.gif`
- Exists: `True`
- Insert after block index: `106`
- Target text: best-of-n。 采样几个回答，留下分数最高的一个。同样的排序也可以用来为指令微调挑选数据。
- Next text: 本系列后面的文章会讲另一半：chat model 怎样用 PPO 和 GRPO 针对 reward model 训练，以及 DPO 怎样直接从同样的偏好数据学习，省掉单独训练的 reward model。

### IMAGE 11

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-005/assets/10-proxy.gif`
- Exists: `True`
- Insert after block index: `100`
- Target text: 所以 reward model 只是人们真正想要的东西的一个代理（proxy），是从人们的选择中学来的。一旦有模型针对它做优化，代理的缺陷就会暴露出来。为了提高分数而训练的 chat model，会找到分数和质量不一致的地方，比如更长、更...
- Next text: reward model 学到的是标注者的选择，连同他们的习惯。针对它过度优化，就会找到它的错误。

### IMAGE 10

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-005/assets/09-noise-vs-bias.gif`
- Exists: `True`
- Insert after block index: `98`
- Target text: 系统性偏差。 假设一部分标注者偏爱自信的 Yes. OpenAI is open.，而不选谨慎的 Partly.；或者偏爱更长的回答，或者偏爱附和用户的回答。这类错误每次都朝同一个方向，在梯度里不断累积，偏差就成了 reward 的一部分...
- Next text: 两种情况在实践中都出现过。一项研究考察了几种设置下的 RLHF，其中两种设置里，chat model 获得的 reward 大部分来自把回答写得更长。另一项研究发现，人和 reward model 有时会更偏爱附和用户错误看法、写得很有说...

### IMAGE 09

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-005/assets/08-pair-softmax.gif`
- Exists: `True`
- Insert after block index: `78`
- Target text: reward model：两个回答之间的 softmax。 A 和 B 各跑一遍，各得一个分数：2.0 和 1.0。两个分数做 softmax，得到 0.731 和 0.269，正是 σ(1)。这个 softmax 只出现在训练的损失里：...
- Next text: 对照下来，两者的区别在于：哪些候选放在一起比较，以及 softmax 在什么时候计算。

### IMAGE 08

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-005/assets/07-chat-softmax.gif`
- Exists: `True`
- Insert after block index: `77`
- Target text: chat model：覆盖整个词表的 softmax。 回答的第一个 token，logits 是 Part 2.0、Yes 1.0、No −1.0，在这三个上做 softmax，得到 0.705、0.259、0.035。真实模型里，so...
- Next text: reward model：两个回答之间的 softmax。 A 和 B 各跑一遍，各得一个分数：2.0 和 1.0。两个分数做 softmax，得到 0.731 和 0.269，正是 σ(1)。这个 softmax 只出现在训练的损失里：...

### IMAGE 07

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-005/assets/06-last-token.gif`
- Exists: `True`
- Insert after block index: `71`
- Target text: 在 Qwen3 的 tokenizer 里，两个回答都是 8 个 token，只在 Some 和 All 处不同。在这个 token 之前，两次运行的向量和数完全相同；从这个 token 开始直到结尾，都不相同。只有最后一个位置读完了整个...
- Next text: 训练时回答成批处理，较短的行用填充 token 补到同样长度。代码必须读最后一个真实 token 上的分数，不能读到填充位置上。

### IMAGE 06

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-005/assets/05-score-head.gif`
- Exists: `True`
- Insert after block index: `63`
- Target text: reward model：score head。 reward model 去掉 LM head，换上一个 score head（打分层）：只有一行 1,024 个数，随机初始化。这时输入是问题加回答 A，假设它的 h 同样是 (0.5,...
- Next text: 每个 logit 都是一次点积：最终隐藏向量乘以输出层的一行。chat model 的输出层有 151,936 行；reward model 的只有一行，这唯一的 logit 就是分数。

### IMAGE 05

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-005/assets/04-chat-head.gif`
- Exists: `True`
- Insert after block index: `62`
- Target text: chat model：LM head。 chat model 的最后一层是 LM head，一张 151,936 行的表，词表里每个条目一行，每行 1,024 个数。每一行都和 h 逐项相乘再相加，即点积：Part 那一行得 2.0，Ye...
- Next text: reward model：score head。 reward model 去掉 LM head，换上一个 score head（打分层）：只有一行 1,024 个数，随机初始化。这时输入是问题加回答 A，假设它的 h 同样是 (0.5,...

### IMAGE 04

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-005/assets/03-gradient-push.gif`
- Exists: `True`
- Insert after block index: `54`
- Target text: 第 2 步。 A 得 0.5 分，B 得 −0.5 分，分差 1.0，σ = 0.731，损失 0.313。推力降到 0.269，因为这一对已经基本排对了。
- Next text: 每一步都把被选中回答的分数抬高，把被淘汰回答的分数压低，幅度相同。在大量数据上，梯度主要来自模型还排错的那些对。

### IMAGE 03

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-005/assets/02-bradley-terry.gif`
- Exists: `True`
- Insert after block index: `35`
- Target text: 分数可以从选择中反推出来。 假设 10 个人比较 A 和 B，7 个人选了 A。最能解释这个结果的分差满足 σ(分差) = 0.7，即分差 = log(7/3) ≈ 0.85。国际象棋的 Elo 等级分用的是同一个思路：用同样的 S 形曲...
- Next text: reward model 是一个网络 r(x, y)，给出问题 x 下回答 y 的分数。训练要回答的问题是：对于标注者实际做出的选择，模型给了多大的概率？A 被选中时，这个概率是 σ(r_A − r_B)，损失就是它的负对数：

### IMAGE 02

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-005/assets/01-one-click.gif`
- Exists: `True`
- Insert after block index: `7`
- Target text: 这次选择存成一行数据：问题、被选中的回答（chosen）、被淘汰的回答（rejected）。成千上万个问题、成千上万行这样的数据，就组成了 reward model 的训练集。
- Next text: 本文从这一行数据出发，一步步讲到最后的分数：

### IMAGE 01

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-005/assets/00-previous-scores.gif`
- Exists: `True`
- Insert after block index: `1`
- Target text: 上一篇里，judge 读完 Is OpenAI open?（OpenAI 开放吗？）的各个回答，给每个回答打了分。OpenAI is open. It's in the name.（OpenAI 是开放的，名字里就写着）得 2 分，以 p...
- Next text: 上一篇留下了一个问题：这些分数从哪里来？
