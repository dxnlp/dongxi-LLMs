# X Article Inline Image Plan

Use this after pasting the clean body into X Articles.

Rule: insert images from bottom to top so earlier block positions do not shift.

- Title: `你的 LLM 其实是一个 reward model：DPO 是怎么工作的`
- Clean body HTML: `x-editor-clean-body.html`
- Clean body Markdown: `x-editor-clean-body.md`
- Total clean body blocks: `240`
- Images: `13`

## Reverse Insertion Order

### IMAGE 13

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-kl-dpo/assets/11-what-to-watch.gif`
- Exists: `True`
- Insert after block index: `224`
- Target text: DPO 论文用 GPT-4 评判生成的回复，并把过度优化在 DPO 中如何表现留作开放问题。
- Next text: 分差变大，只说明 chosen 回答比 rejected 回答领先更多。模型现在写什么，要另外检查。

### IMAGE 12

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-kl-dpo/assets/10-answers-to-tokens.gif`
- Exists: `True`
- Insert after block index: `205`
- Target text: Tülu 3 用的一个变体更进一步：把每个回答的 log ratio 除以 token 数，减轻剩余的长度偏差。
- Next text: 在真实模型上看什么

### IMAGE 11

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-kl-dpo/assets/09-ideal-test.gif`
- Exists: `True`
- Insert after block index: `189`
- Target text: 数据每增加十倍，距离大约缩小到三分之一。这是从样本估计平均值的常见速度。
- Next text: 这个测试还给出两条警告。

### IMAGE 10

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-kl-dpo/assets/08-where-probability-comes-from.gif`
- Exists: `True`
- Insert after block index: `173`
- Target text: 错误的回答上升了。没有配对提到它，所以没有更新把它压下去。
- Next text: 现在这两个回答的分差是 1，和最优模型相同。比例却差得很远：About 10 minutes 是 15.5%，目标是 30.3%；错误的回答是 8.2%，目标是 1.3%。

### IMAGE 09

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-kl-dpo/assets/07-the-push.gif`
- Exists: `True`
- Insert after block index: `153`
- Target text: 参考模型是冻结的，不接收任何更新。
- Next text: Hyperparameter（超参数）。 β 和 learning rate 都是 hyperparameter：由人选定，训练不会改变。两者分工不同。

### IMAGE 08

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-kl-dpo/assets/06-loss-on-one-pair.gif`
- Exists: `True`
- Insert after block index: `134`
- Target text: 这里只看变化。小幅移动给 chosen 回答 22%，给 rejected 回答 25%，分差仍然是正的，因为它相对参考模型抬高了 chosen 回答。
- Next text: 更新力度

### IMAGE 07

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-kl-dpo/assets/05-implicit-reward.gif`
- Exists: `True`
- Insert after block index: `97`
- Target text: 两个分数都减了 0.808，差不变，都得到 σ(1) = 0.731。
- Next text: 所以，最优模型和真实 reward 对人的选择给出相同的预测。

### IMAGE 06

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-kl-dpo/assets/04-no-move-helps.gif`
- Exists: `True`
- Insert after block index: `79`
- Target text: 每个回答都保留了一些比例，只是按 reward 重新加权。β 越小，加权越强。
- Next text: 有两点后面会用到。

### IMAGE 05

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-kl-dpo/assets/03-why-pay-for-distance.gif`
- Exists: `True`
- Insert after block index: `48`
- Target text: reward 下降了，gold 上升了。
- Next text: DPO 论文指出了 KL penalty 的作用：让模型留在 reward model 准确的那些回答附近，保留回答的多样性，防止模型塌缩到单个高分回答。

### IMAGE 04

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-kl-dpo/assets/02-three-signals.gif`
- Exists: `True`
- Insert after block index: `41`
- Target text: DPO 改动前两个：存好的配对取代 reward model，β 和参考模型直接写进 loss。gold 仍在训练之外。
- Next text: 为什么要有 KL penalty

### IMAGE 03

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-kl-dpo/assets/01-secretly-a-reward-model.gif`
- Exists: `True`
- Insert after block index: `21`
- Target text: 训练就是在大量配对上降低这个 loss。不需要采样回答，不需要训练 reward model，也没有 RL 循环。
- Next text: 这里有两个问题：log ratio 为什么能当 reward？为什么降低这个 loss，得到的正是 RL 要找的模型？

### IMAGE 02

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-kl-dpo/assets/00b-change-is-the-score.gif`
- Exists: `True`
- Insert after block index: `9`
- Target text: 然后拿人比较过的两个回答，套用 reward model 的 Bradley–Terry loss。
- Next text: 看 KL 那篇的鸡蛋问题。问 How long should I boil an egg?，参考模型的五种回答比例如下：

### IMAGE 01

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-kl-dpo/assets/00a-two-routes.gif`
- Exists: `True`
- Insert after block index: `5`
- Target text: DPO 不需要这个外部的 reward model：LLM 本身就是 reward model。它相对起点把一个回答的概率调高或调低了多少，这个量就是该回答的分数。
- Next text: 具体做法是：留一份起点模型的冻结副本，叫 reference（参考模型），然后这样给每个回答打分：
