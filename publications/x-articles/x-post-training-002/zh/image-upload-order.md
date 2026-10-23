# Image Upload Order

封面（单独上传，5:2）：`../assets/cover-post-training-02.png`。所有配图均为英文，中英文版共用。

1. `../assets/01-alpaca-recipe.gif` — 斯坦福的 Alpaca（2023）：一个 AI 模型生成了 52,000 条指令与回答示例，LLaMA 7B 由此学会按指令作答。数字来自斯坦福团队的公开报告。
2. `../assets/02-lima.gif` — LIMA（2023）：在 LLaMA 65B 上只用 1,000 条精选示例、不做强化学习，43% 的回答被评为与 GPT-4 相当或更好。论文公开结果。
3. `../assets/03-flan.gif` — FLAN（2021）：137B 模型在 60 多个改写成指令的任务上微调，在 25 项未训练过的任务中有 20 项超过零样本的 175B GPT-3。论文公开结果；哪几格为灰色仅为示意。
4. `../assets/04-recipe.gif` — 指令微调的五个步骤：收集、格式化、掩码、训练、评测。
5. `../assets/05-three-forms.gif` — 一条示例的三种形态：对话、带模板标记的 token 序列、损失掩码。标记做了简化。
6. `../assets/06-three-forms-summary.png` — 三种形态对照，使用 Qwen 模板的真实标记。
7. `../assets/07-what-gets-loss.gif` — 在多轮对话中，只有助手的回答及其结尾标记计算损失。
8. `../assets/08-masked-not-untouched.gif` — 提示不计算损失，但回答产生的梯度仍会通过注意力传回提示。示意图。
9. `../assets/09-one-position-shift.gif` — 每个位置预测下一个 token。没有错开一位时，每个预测都和当前位置的 token 对齐，模型学会的是复制。示意图。
10. `../assets/10-token-share.gif` — 损失按计算损失的 token 求平均，因此 500 个 token 的回答权重是 2 个 token 回答的 250 倍。token 数为示意。
