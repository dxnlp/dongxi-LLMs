# 图片上传顺序

封面单独上传（5:2）：`../assets/cover-post-training-05.png`。所有图都是英文，带 DONGXI LABS 标记，中英文版共用，位于 `../assets/`。第 1 张取自上一篇（X-POST-004 的第 5 张 GIF，截到 V 线出现之前，并换了标题）。05–06、08–09 两两成对，前后相邻。

1. `../assets/00-previous-scores.gif` — 回顾上一篇：judge 给 Is OpenAI open? 的十个回答打分，以 open 开头的回答分数低，两个 partly 回答得 10 分和 8 分。示意值。
2. `../assets/01-one-click.gif` — 一次点击 A 或 B，变成一行数据：问题、chosen 回答、rejected 回答。很多行叠成训练集。
3. `../assets/02-bradley-terry.gif` — 两个分数、分差、sigmoid、A 被选中的概率和损失。两个分数同时平移，结果不变。
4. `../assets/03-gradient-push.gif` — gradient descent 的每一步：一对排错时推力 1 − σ(分差) 很大，排对后逐渐变小。
5. `../assets/04-chat-head.gif` — chat model：h 与 LM head 逐列对齐，每个 logit 是 h 乘以一行；softmax 选出 token，进入下一轮循环。
6. `../assets/05-score-head.gif` — reward model：同一个主体，输出层只有一行，h 乘以这一行就是分数。A 得 2.0，B 单独跑一遍得 1.0。
7. `../assets/06-last-token.gif` — 每个 token 上都有一个数。Some 换成 All，只改变后面的位置；分数取最后一个真实 token。
8. `../assets/07-chat-softmax.gif` — chat model 的 softmax 覆盖整个词表，概率总和为 1，从中采样下一个 token。
9. `../assets/08-pair-softmax.gif` — reward model 的 softmax 只覆盖两个回答，等于 σ(1)，只用于训练损失；使用时原始分数就是 reward。
10. `../assets/09-noise-vs-bias.gif` — 两个学到的权重：准确和长度。随机噪声让两者同比缩小；偏向长回答的系统性偏差让长度权重变大。示意图。
11. `../assets/10-proxy.gif` — 更长但没有新内容的回答得分更高；针对代理优化，分数持续上升，真实质量先升后降。示意图。
12. `../assets/11-three-uses.gif` — 一个冻结的 reward model，三种用途：best-of-n、给新数据标注、RLHF 中的 reward。
