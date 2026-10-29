# X Article Inline Image Plan

Use this after pasting the clean body into X Articles.

Rule: insert images from bottom to top so earlier block positions do not shift.

- Title: `写给 LLM 读者的强化学习术语：一个分数如何分到每个 token`
- Clean body HTML: `x-editor-clean-body.html`
- Clean body Markdown: `x-editor-clean-body.md`
- Total clean body blocks: `83`
- Images: `13`

## Reverse Insertion Order

### IMAGE 12

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-004/assets/12-term-map.gif`
- Exists: `True`
- Insert after block index: `74`
- Target text: 整个循环可以画在一张图里：state、policy 的分支、action、reward、V、Q、advantage，以及重新调整分支的更新。
- Next text: (end of body)

### IMAGE 13

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-004/assets/13-term-cards.gif`
- Exists: `True`
- Insert after block index: `74`
- Target text: 整个循环可以画在一张图里：state、policy 的分支、action、reward、V、Q、advantage，以及重新调整分支的更新。
- Next text: LLM 就是 policy。prompt 加上已写出的 token 是 state，下一个 token 是 action，整段回答是一条 trajectory。

### IMAGE 11

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-004/assets/11-kl-penalty.gif`
- Exists: `True`
- Insert after block index: `71`
- Target text: KL penalty。 judge 只对它熟悉的那类回答打分可靠。假设 judge 喜欢恭维：它给 partly. Some open weights. 打 9 分，给 partly. What a brilliant question!...
- Next text: 这个恭维的回答是 reward hacking 的一个小例子：policy 靠迎合 judge 的偏好来提高分数，回答本身并没有变好。KL penalty 限制了它能偏离多远。

### IMAGE 10

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-004/assets/10-ratio-clip.gif`
- Exists: `True`
- Insert after block index: `70`
- Target text: ratio 与 clipping。 policy gradient 假设回答来自当前的 policy，这样的数据叫作 on-policy 数据。采样成本很高，所以 PPO 一类的训练会用当前模型采一批回答，保存一份冻结的副本 π_old，...
- Next text: KL penalty。 judge 只对它熟悉的那类回答打分可靠。假设 judge 喜欢恭维：它给 partly. Some open weights. 打 9 分，给 partly. What a brilliant question!...

### IMAGE 09

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-004/assets/09-exploration.gif`
- Exists: `True`
- Insert after block index: `67`
- Target text: 以温度 1 做 sampling，模型有时会写出 closed 或 partly。这些样本显示出其他分支的分数更高，policy 正是靠它们改变的。这就是 exploration：policy 必须尝试概率较低的 token，所以强化学习...
- Next text: 小步更新：ratio、clipping 与 KL

### IMAGE 08

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-004/assets/08-policy-gradient.gif`
- Exists: `True`
- Insert after block index: `63`
- Target text: 每个 token 有了 advantage，更新规则一句话就能说清：advantage 为正的 token，提高它的对数概率；advantage 为负的 token，降低它的对数概率；调整幅度与 advantage 成正比。这就是 pol...
- Next text: 按期望梯度精确地更新两步（步长是示意值），概率从 0.50 / 0.30 / 0.20 变成 0.15 / 0.36 / 0.49，再变成 0.04 / 0.11 / 0.86；V 从 4.3 升到 6.5，再升到 8.3。第一步中 cl...

### IMAGE 07

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-004/assets/07-credit-assignment.gif`
- Exists: `True`
- Insert after block index: `59`
- Target text: credit 归于预期分数发生变化的那个 token。
- Next text: 这要求 V 是准确的，而实际中的 V 是学出来的估计值。GAE（generalized advantage estimation）引入了一个取值在 0 到 1 之间的参数 λ。λ = 0 时，每个 token 的 advantage 就是...

### IMAGE 06

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-004/assets/06-baseline.gif`
- Exists: `True`
- Insert after block index: `51`
- Target text: 例子里 judge 的分数都是正的。如果从 0 算起，每个采样回答都会被往上推，open 也不例外。它只是被推得比其他回答少，正确的方向要等大量样本相互抵消后才能显现。从 V 算起，每个样本本身就指向正确的方向：低于平均的回答往下压，高于...
- Next text: 只要 baseline 与采到哪个 token 无关，减去它就不会改变平均的更新，改变的只是噪声。在这个例子的模拟中，以 0 为 baseline 和以 V 为 baseline，平均更新完全相同；以 V 为 baseline 时，方差约...

### IMAGE 05

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-004/assets/05-value-q-advantage.gif`
- Exists: `True`
- Insert after block index: `40`
- Target text: advantage A(s, a) = Q(s, a) − V(s)：在 state s 下写 token a 的平均分，减去不指定 token 时的平均分，也就是写 a 比模型平时的水平高多少或低多少。例如 A(open) = Q(op...
- Next text: 图里把三者画在一起。十个采样回答落在各自的分数上：同一分支上的点收拢到它们的平均，得到这条分支的 Q；十个点一起平均，得到 V。state 画在 V = 4.3 的高度上，从 V 线指向每个 Q 的箭头就是 advantage。

### IMAGE 04

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-004/assets/04-reward-return.gif`
- Exists: `True`
- Insert after block index: `29`
- Target text: γ = 1 且只有结尾一个 reward 时，每个位置的 return 都等于最终分数：OpenAI 是 2，is 是 2，open 是 2，name 也是 2。
- Next text: 所以只看 return，分辨不出哪个 token 好、哪个 token 差。这段回答里每个 token 拿到的都是同一个 2。后面的术语，都是为了把这个分数分到不同的 token 上。

### IMAGE 03

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-004/assets/03-state-action.gif`
- Exists: `True`
- Insert after block index: `24`
- Target text: LLM 就是一个 policy：prompt 加上已写出的 token 是 state，下一个 token 是 action。
- Next text: reward 与 return

### IMAGE 02

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-004/assets/02-rl-loop.gif`
- Exists: `True`
- Insert after block index: `18`
- Target text: 循环的一轮是这样的：environment 给出一个 prompt，模型写出回答，judge 返回分数，分数用来更新 policy。下一轮用更新后的模型继续。
- Next text: state、action、trajectory

### IMAGE 01

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-004/assets/01-one-score.gif`
- Exists: `True`
- Insert after block index: `1`
- Target text: 2025 年 1 月，DeepSeek 发布了 DeepSeek-R1-Zero。它直接在基座模型上做 reinforcement learning（RL，强化学习）训练，中间没有经过监督微调。它的 reward 来自规则：一条规则检查最...
- Next text: 规则检查只给整段回答一个分数，而一段回答可能有几千个 token。训练时要调整的，却是每一个 token 的概率。LLM 的强化学习，归根到底在回答一个问题：
