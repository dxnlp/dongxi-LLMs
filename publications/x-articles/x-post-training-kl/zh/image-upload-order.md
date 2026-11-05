# 图片上传顺序

封面单独上传（5:2）：`../assets/cover-post-training-kl.png`。所有图都是英文，带 DONGXI LABS 标记，中英文版共用，位于 `../assets/`。2026-11-04 由 `visuals/animations/projects/post-training-kl/` 渲染。

1. `../assets/01-two-splits.gif` — 参考模型的五个比例；0.10 从含糊的回答移到最好的回答。示意值。
2. `../assets/02-why-log.gif` — 比值和 log 曲线：1 对应 0，翻倍和减半落在 +0.693 和 −0.693，靠近 0 时曲线不断下降。
3. `../assets/03-kl-term-by-term.gif` — 比值、log、按新模型的比例加权、相加：−0.086 + 0.133 = 0.047。公式在表格上方逐步点亮。
4. `../assets/04-evidence.gif` — 每条回答把自己的 log ratio 累加到证据上。KL 0.047 时，平均约 64 条回答达到 3.0，也就是 20 比 1。一次模拟。
5. `../assets/05-never-negative.gif` — 变化量加起来是 0，每一项都不小于变化量，所以总和不小于 0。降到 0.01 时 log ratio 是 −3.689，这一项只有 −0.037。
6. `../assets/06-no-direction.gif` — 朝最好的回答和背离它都有 KL 代价（0.047 和 0.076）。reward 区分两者（0.638 和 0.209），β 决定最好回答的比例能升多高。
7. `../assets/07-rare-moves.gif` — 同样挪动 0.10，分别给常见回答和罕见回答：挪动量都是 0.10，KL 是 0.031 和 0.233。
8. `../assets/08-which-way.gif` — KL 的两种顺序：小变化是 0.047 和 0.042；丢掉一个回答，一个方向是 0.102，另一个方向是无穷大；新增回答则相反。
9. `../assets/09-from-samples.gif` — 新模型写出回答，每条有一个 log ratio，累计平均值落在 0.047 附近。一次模拟。
10. `../assets/10-tokens.gif` — 一条回答的 log ratio 拆到每个 token：0.223 + 0 + 0.182 = 0.405，以及 PPO 和 GRPO 各把它放在哪里。token 概率是示意值。
11. `../assets/11-reading-a-run.gif` — KL 增大时，reward 一路上升，gold 在 KL 0.81 处到顶（1.10）后下降。玩具例子，每个点是更小 β 下的折中点。
12. `../assets/12-implicit-reward.gif` — 在折中点，每个 log ratio 都等于 reward 减 0.808。比较时这个常数消掉，两种算法都得到 0.731。
