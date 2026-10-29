# Image Upload Order

封面（单独上传，5:2）：`../assets/cover-post-training-04.png`。所有配图均为英文，中英文版共用。

1. `../assets/01-one-score.gif` — DeepSeek-R1-Zero：规则只检查最终答案，RL 训练中 AIME 2024 的 pass@1 从 15.6% 升到 71.0%（论文公开数据）。token 方格是示意。
2. `../assets/02-rl-loop.gif` — 强化学习的循环：environment 给出 prompt，模型（agent）写出回答，judge 返回分数，分数用来更新 policy。
3. `../assets/03-state-action.gif` — state 是 prompt 加上已写出的 token；policy 给出下一个 token 的概率；采样到的 token 是 action；写完的回答是一条 trajectory。概率为示意值。
4. `../assets/04-reward-return.gif` — judge 给写完的回答打分：每个 token 的 reward 是 0，最后一个是 2。γ = 1 时，每个位置的 return 都是 2。
5. `../assets/05-value-q-advantage.gif` — 十个采样回答落在各自的分数上。十个一起平均得到 V = 4.3，每条分支的平均是它的 Q（2、5、9），A = Q − V 为 −2.3、+0.7、+4.7。数字为示意值。
6. `../assets/06-baseline.gif` — 从 0 算起，每个回答都被往上推；从 baseline V = 4.3 算起，只有好于平时的回答被往上推。
7. `../assets/07-credit-assignment.gif` — V 在 open 之后下降，TD error −2.3 把责任归给 open。λ = 0 时责任留在 open 上；λ = 1 时扩散到 OpenAI 和 is。
8. `../assets/08-policy-gradient.gif` — policy gradient 走两步：分支粗细从 50/30/20% 变成 15/36/49%，再变成 4/11/85%，V 从 4.3 升到 8.3；closed 从高于平均变成低于平均。步长为示意值。
9. `../assets/09-exploration.gif` — greedy decoding 只走 open：其他分支的分数始终未知，A = 0，什么都不变。sampling 会尝试每条分支，policy 随之改变。
10. `../assets/10-ratio-clip.gif` — 这批数据由 π_old 采样。A = +4.7 把 π(partly) 从 20% 推到 24%；ρ = π / π_old 到达 1 + ε（ε = 0.2）后不再推动。
11. `../assets/11-kl-penalty.gif` — 喜欢恭维的 judge 给恭维回答 9.6 分。policy 从 π_ref 漂向恭维时，judge 平均分升到 9.54，β·KL 升到 2.38，R − β·KL 从 9.03 降到 7.16（β = 1，示意）。
12. `../assets/12-term-map.gif` — 整个循环放在一张图里：state、policy、action、reward、V、Q、A，以及重新调整分支的更新。
13. `../assets/13-term-cards.gif` — 速查表：本文的全部术语，按循环的步骤分组。
