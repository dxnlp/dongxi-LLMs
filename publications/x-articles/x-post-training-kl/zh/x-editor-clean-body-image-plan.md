# X Article Inline Image Plan

Use this after pasting the clean body into X Articles.

Rule: insert images from bottom to top so earlier block positions do not shift.

- Title: `KL：用一个数衡量模型变了多少`
- Clean body HTML: `x-editor-clean-body.html`
- Clean body Markdown: `x-editor-clean-body.md`
- Total clean body blocks: `137`
- Images: `12`

## Reverse Insertion Order

### IMAGE 12

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-kl/assets/12-implicit-reward.gif`
- Exists: `True`
- Insert after block index: `123`
- Target text: 这就是 DPO（direct preference optimization，直接偏好优化）的想法。它省掉单独的 reward model：对人选出的每一对回答，用 β × log ratio 当分数，代入同样的 Bradley–Terr...
- Next text: KL 里的 log ratio 衡量一个回答移动了多少，也可以当作这个回答的 reward。

### IMAGE 11

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-kl/assets/11-reading-a-run.gif`
- Exists: `True`
- Insert after block index: `115`
- Target text: KL 3.61：reward 2.67，gold −0.76，错误回答已经占了 69%。
- Next text: reward 一路上升，gold signal 先升、到顶、再降。Gao、Schulman 和 Hilton 在 2022 年用真实的 reward model 报告过这个规律。同一篇论文也提醒，只看 KL 无法判断过度优化到了什么程度。...

### IMAGE 10

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-kl/assets/10-tokens.gif`
- Exists: `True`
- Insert after block index: `105`
- Target text: DeepSeekMath 的 GRPO 把 KL 直接加进损失，β = 0.04。它对每个 token 的估计是 x − log x − 1，其中 x = π_ref / π。这个式子同样来自前面那条直线，对任何 token 都不会是负数。
- Next text: 训练时怎样读 KL

### IMAGE 09

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-kl/assets/09-from-samples.gif`
- Exists: `True`
- Insert after block index: `101`
- Target text: 样本。 KL 是新模型所有回答上的平均，而平均可以用样本估计。让新模型写一批回答，算出每条的 log ratio，再求平均。在鸡蛋例子上模拟一次：10 条回答得到 0.096，100 条得到 0.061，10,000 条得到 0.048，...
- Next text: token。 一条回答的概率，是它每个 token 的概率连乘，每个 token 的概率都以前面的 token 为条件。log 把连乘变成求和，所以一条回答的 log ratio，就是它每个 token 的 log ratio 之和。看一...

### IMAGE 08

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-kl/assets/08-which-way.gif`
- Exists: `True`
- Insert after block index: `98`
- Target text: 对调后的顺序出现在 pretraining（预训练）里。它用的损失 cross-entropy（交叉熵）等于数据的 entropy（熵）加上 KL(数据 ‖ 模型)。数据的 entropy 和模型无关，所以降低损失就是降低这个 KL，这时...
- Next text: 真实模型：样本和 token

### IMAGE 07

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-kl/assets/07-rare-moves.gif`
- Exists: `True`
- Insert after block index: `91`
- Target text: 这正是训练需要的。参考模型几乎不写的回答，reward model 也很少见到，它的分数在那里最不可靠。KL 恰好在 reward 信号最弱的地方收费最高。
- Next text: 哪个模型放在上面

### IMAGE 06

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-kl/assets/06-no-direction.gif`
- Exists: `True`
- Insert after block index: `83`
- Target text: 最后一行的 reward 最高，总分却很低。训练会停在 reward 和代价之间的折中点，β 决定折中点的位置。随着价格升高，折中点上最好回答的比例依次是：不收费时 1.00，β = 0.5 时 0.71，β = 1 时 0.40，β =...
- Next text: reward 决定移动的方向，β × KL 限制移动的距离。

### IMAGE 05

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-kl/assets/05-never-negative.gif`
- Exists: `True`
- Insert after block index: `66`
- Target text: 如果某个回答大幅下降呢？让含糊的回答从 0.40 降到 0.01，最好的回答升到 0.51。含糊回答的 log ratio 是 −3.689，是个很大的负数。可是它的权重是新比例 0.01，这一项只有 −0.037。模型几乎不说的回答，权...
- Next text: 没有变化时，两个模型的距离已经是最小值。所以 KL 从 0 开始，只会增大。

### IMAGE 04

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-kl/assets/04-evidence.gif`
- Exists: `True`
- Insert after block index: `58`
- Target text: KL 2.12。 一个永远给出最好回答的模型。求和只剩一项：1 × log(1 / 0.12)。只看一条回答，就有 8.33 比 1 的把握。
- Next text: KL 为什么不会是负数

### IMAGE 03

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-kl/assets/03-kl-term-by-term.gif`
- Exists: `True`
- Insert after block index: `51`
- Target text: Σ：对所有回答加总。
- Next text: 这个数是什么意思

### IMAGE 02

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-kl/assets/02-why-log.gif`
- Exists: `True`
- Insert after block index: `32`
- Target text: 比值靠近 0 时，它不断下降，没有下限。 log(0.1) 约为 −2.3，log(0.01) 约为 −4.6，log(0) 没有定义。
- Next text: 比值取 log 之后的数叫 log ratio。在这次变化里，含糊的回答是 −0.288，最好的回答是 +0.606，没变的三个回答是 0。

### IMAGE 01

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-kl/assets/01-two-splits.gif`
- Exists: `True`
- Insert after block index: `8`
- Target text: 现在把模型训练一小会儿。含糊的回答从 40% 降到 30%，最好的回答从 12% 升到 22%，另外三个回答不变。
- Next text: 模型变了多少？我们想要一个数：什么都没变时是 0，变得越多，数越大。最常用的度量是 KL divergence，简称 KL，由 Solomon Kullback 和 Richard Leibler 在 1951 年提出，名字取自两人的姓。
