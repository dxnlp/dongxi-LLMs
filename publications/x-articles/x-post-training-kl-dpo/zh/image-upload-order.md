# 图片上传顺序

封面单独上传（5:2）：`../assets/cover-post-training-dpo.png`。所有图都是英文，带 DONGXI LABS 标记，中英文版共用。文件名保留原编号：`00a` 是第 1 张，`00b` 是第 2 张，之后的文件都往后顺延两位。`05-implicit-reward.gif` 是 KL 文章的 GIF 12 去掉标签后重新渲染的版本。十三张都已于 2026-11-06 渲染。

1. `../assets/00a-two-routes.gif` — RLHF 从人工配对经过 reward model 和 RL 才到 chat model；DPO 用存好的配对和一个 loss 直接到达。
2. `../assets/00b-change-is-the-score.gif` — 模型的比例离开虚线的参考模型时，每个回答的分数跟着变化：调高的回答高于 0，调低的回答低于 0，没动的回答是 0。示意值。
3. `../assets/01-secretly-a-reward-model.gif` — 左边是带打分头的单独 reward model；右边是训练中的模型和冻结的参考模型读同一个回答，比值、log、β 得到分数：小幅移动后最好的回答得 0.606。示意值。
4. `../assets/02-three-signals.gif` — reward 和 KL 进入参数更新，gold 只评判 checkpoint。DPO 用存好的配对取代 reward model。
5. `../assets/03-why-pay-for-distance.gif` — 只看 reward 时，比例全部流向被高估的回答：reward 3，gold −2。β = 1 时它只占 4.3%：reward 1.22，gold 1.00。玩具例子。
6. `../assets/04-no-move-helps.gif` — 五根 net value 柱子拉平到 0.808，然后分三步：乘 e^reward，加出 Z = 2.243，再除。
7. `../assets/05-implicit-reward.gif` — 在最优模型上，每个 log ratio 都等于 reward 减 0.808。比较时常数抵消：两边都是 0.731。
8. `../assets/06-loss-on-one-pair.gif` — 分差从 −3 扫到 3，loss 曲线逐渐画出，鸡蛋这个配对标在 0.693、0.435 和 0.313。
9. `../assets/07-the-push.gif` — 先做 nudge test，再用箭头长度表示更新力度：0.953、0.500、0.047。一个配对逐步训练，更新越来越小，但不会停下。
10. `../assets/08-where-probability-comes-from.gif` — 一步只改动配对的两个 logit，五个比例都变了。错误的回答从 8.0% 升到 8.2%，柱子也和 π* 不同。
11. `../assets/09-ideal-test.gif` — 谁赢了的记录逐条进入，reward 对训练隐藏。在 1,000 条记录上训练后，比例从参考模型走向目标，距离从 0.33 变为 0.05。模拟数据，柱子是一次运行的比例。
12. `../assets/10-answers-to-tokens.gif` — 回答的 token 和 end-of-turn token 相加，问题的 token 不计入。回复越长总和越低，所以每个回答只和自己的参考值比较。示意值。
13. `../assets/11-what-to-watch.gif` — 四件要看的事：训练集和留出集准确率从 50% 开始，chosen 和 rejected 的概率，两者都下降的情况，从样本测出的 KL。曲线是虚构的，没有测量值。
