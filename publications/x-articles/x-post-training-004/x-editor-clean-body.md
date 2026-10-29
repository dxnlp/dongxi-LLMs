## One score, thousands of tokens

In January 2025, DeepSeek reported DeepSeek-R1-Zero, a model trained with reinforcement learning (RL) directly on a base model, without supervised fine-tuning first. Its reward came from rules: one rule checks whether the final answer is correct, another checks that the reasoning is placed between think tags. Over thousands of RL steps, its pass@1 on the AIME 2024 math competition rose from 15.6% to 71.0%, and to 86.7% with majority voting. Its responses also grew longer during training: the model learned to spend more tokens on reasoning.

A rule-based check produces one number for a whole response, and the response can be thousands of tokens long. Training, however, changes the probability of each token. One question sits under all of RL for LLMs:

> A judge gives one score to a whole answer. Which tokens should become more likely, and which less?

RL has its own vocabulary: policy, state, action, reward, return, value, advantage, baseline, TD error, GAE, clipping, KL. Each term is one piece of the answer to that question. This article goes through them in order on a single running example, so that the next two articles, on preference alignment and on RL with verifiable rewards, can use them without stopping to define them. It is the fourth article in our post-training series; the previous one covered LoRA and QLoRA.

## The running example

The prompt is *Is OpenAI open?* The name says open. The accurate answer, as of this writing, is partly: OpenAI has released open-weight models, gpt-oss-120b and gpt-oss-20b under the Apache 2.0 license, while its flagship models are available only through its API and apps.

Suppose the model has written *OpenAI is* and must choose the next token. Its favorite is the naive one:

- *open*, probability 0.5, continued as *It's in the name.* The judge gives 2 out of 10 on average.
- *closed*, probability 0.3, which misses the open-weight models. The judge gives 5 on average.
- *partly*, probability 0.2, continued as *some open weights.* The judge gives 9 on average.

All numbers in this article are illustrative. The arithmetic on them is exact.

## The loop: agent, environment, reward

In RL, an **agent** acts, an **environment** responds, and a **reward** measures how well the agent did. For an LLM:

- The agent is the model. Its next-token distribution is the **policy**, written π_θ, where θ are the weights. Training changes θ and so changes the policy.
- The environment is everything outside the model: the prompts it receives and the judge that scores its answers.
- The **judge** is the part of the environment that scores a finished answer. It can be a reward model trained on human preference pairs (RLHF, the topic of the next article), a rule-based verifier like R1-Zero's (RLVR, the article after), unit tests that run generated code, or another LLM grading against a rubric. This article writes judge so that every term applies to all of them.

The judge belongs to the environment because the model cannot change it. A reward model is frozen during RL, like the prompt set.

One round of the loop: the environment gives a prompt, the model writes an answer, the judge returns a score, and the score updates the policy. The next round starts with the updated model.

## State, action, trajectory

Inside one answer, RL works token by token.

- The **state** is everything the model sees before it writes the next token: the prompt plus the tokens written so far. Here, *Is OpenAI open?* plus *OpenAI is*.
- An **action** is the next token. The policy gives every candidate a probability (open 0.5, closed 0.3, partly 0.2), and sampling picks one.
- The chosen token extends the state, and the process repeats until the answer ends. The full answer is a **trajectory**, also called a rollout or an episode.

> An LLM is a policy: the prompt plus the tokens so far is the state, and the next token is the action.

## Reward and return

Say the model sampled *open* and finished with *OpenAI is open. It's in the name.* The judge reads the whole answer and gives 2.

The **reward** is the score attached to each step. In LLM RL, every token gets reward 0 except the last one, which gets the judge's score. This is a **sparse**, or delayed, reward. A prefix has no correct score: *OpenAI is* can still end well or badly. Scoring every step would need a process reward model trained on step-level labels. The DeepSeek-R1 report lists the problems its authors found with that approach: a step is hard to define in general reasoning, judging an intermediate step is hard to automate and hard to label by hand at scale, and a learned step judge invites reward hacking. A score for the finished answer is cheap, harder to game, and leaves the model free to find its own reasoning path. R1-Zero found longer reasoning on its own.

The **return** G at a token is the sum of all rewards from that token to the end. RL in general multiplies later rewards by a discount factor γ below 1, which matters for tasks that run on for a long time. An LLM answer ends, and its score belongs to the whole answer, so LLM RL uses γ = 1, no discount. With γ below 1, an early token would receive less of the score even when it decides the answer, as *open* does here, and shorter answers would be favored.

With γ = 1 and one reward at the end, the return at every position equals the final score: 2 for *OpenAI*, 2 for *is*, 2 for *open*, 2 for *name*.

The return alone therefore cannot tell which token was good or bad. Every token of this answer carries the same 2, and the rest of the vocabulary exists to split it.

Sparse rewards have a second cost. A judge that checks only the outcome cannot see how the outcome was reached. *OpenAI is partly. It's in the name.* has the right verdict for the wrong reason, and a judge that reads only the verdict would give it full marks. Lucky guesses, correct numbers from flawed derivations and drift into unreadable text pass the same way. DeepSeek reported that R1-Zero's reasoning had poor readability and mixed languages; DeepSeek-R1 added a small amount of supervised cold-start data and a language-consistency reward. Training still tends toward sound reasoning, because each update follows the average over many samples and sound reasoning succeeds more often than luck. That is a statistical tendency, and readability is rewarded only if the reward asks for it.

## Value, Q and advantage

Return to the state *OpenAI is*. Let the model continue from there many times and have the judge score every finished answer. The figure shows ten of them. Following the policy's probabilities, 5 start with *open*, 3 with *closed* and 2 with *partly*. Answers with the same first token still continue differently, so their scores vary:

- the 5 answers that start with *open* score 3, 1, 2, 1 and 3;
- the 3 answers that start with *closed* score 5, 4 and 6;
- the 2 answers that start with *partly* score 10 and 8.

Three quantities follow from these ten scores. In the notation, s is the state, here *OpenAI is*, and a is the action, the next token: here *open*, *closed* or *partly*.

- The **action value** Q(s, a) is the average score when the model, in state s, writes token a first and then writes as usual. Group the answers by their first token and average each group: Q(open) = (3 + 1 + 2 + 1 + 3) / 5 = 2, Q(closed) = (5 + 4 + 6) / 3 = 5, Q(partly) = (10 + 8) / 2 = 9. The scores of 2, 5 and 9 in the running example are these averages.
- The **value** V(s) is the average score when the model, in state s, writes as it usually does, with no token fixed. Average all ten scores: (3 + 1 + 2 + 1 + 3 + 5 + 4 + 6 + 10 + 8) / 10 = 4.3. The same number is the probability-weighted average of the three Q values: V = 0.5 × 2 + 0.3 × 5 + 0.2 × 9 = 4.3.
- The **advantage** A(s, a) = Q(s, a) − V(s) is the average score after writing token a in state s, minus the average with no token fixed: how much better or worse writing a is than what the model usually does. For example, A(open) = Q(open) − V = 2 − 4.3 = −2.3: answers that start with *open* score 2.3 below the average. Likewise A(closed) = 5 − 4.3 = +0.7 and A(partly) = 9 − 4.3 = +4.7.

The figure puts the three in one picture. Ten sampled answers land at their own scores. The dots of one branch fall to their average, which is that branch's Q, and all ten averaged give V. The state sits at the height of V = 4.3, and the arrow from the V line to each Q is the advantage.

The advantage is the number training acts on. Four of its properties matter.

**It answers what a raw score cannot.** The judge gave *open* a 2. Whether 2 is good depends on what the model usually scores at this point. The reward is an absolute score; the advantage compares it with the model's own expectation at this state.

**Its sign is the direction, its size the amount.** A above 0 means make this token more likely; A below 0 means make it less likely. A larger |A| means a larger change.

**It is relative to the current model.** *closed* gets +0.7 although it is a weaker answer than *partly*, because it beats the model's current average of 4.3. Once the model improves and its average passes 5, the same *closed* gets a negative advantage.

**Its probability-weighted average is zero.** 0.5 × (−2.3) + 0.3 × 0.7 + 0.2 × 4.7 = −1.15 + 0.21 + 0.94 = 0. An update cannot make every token more likely. It moves probability from below-average tokens to above-average ones.

> The advantage is how much better this token is than what the model usually writes here.

Algorithms differ in how they obtain V. PPO trains a second network, the critic, to predict it. GRPO, the algorithm behind R1-Zero, samples a group of answers to the same prompt and uses their average score. The next article covers both.

## Baseline

Training never sees Q directly. It sees one sampled answer and its score R, and estimates that answer's advantage as R minus a **baseline**, usually V. The five *open* answers in the figure scored 3, 1, 2, 1 and 3, so R − V gives −1.3, −3.3, −2.3, −3.3 and −1.3. Their average is −2.3, which is Q − V.

The judge's scores here are all positive. Measured from 0, every sampled answer is pushed up, *open* included. It is pushed up less than the others, and the right direction appears only after many samples offset each other. Measured from V, each sample already points the right way: below-average answers go down and above-average answers go up.

As long as the baseline does not depend on which token was sampled, subtracting it leaves the average update unchanged; it changes only the noise. In a simulation of this example, the average update was the same with a baseline of 0 and with V, and its variance was about five times smaller with V.

> A baseline changes the noise of the update, not its direction.

GRPO's group average and PPO's critic are both baselines in this sense.

## Credit assignment: TD error and GAE

The advantage so far belongs to one decision, the token after *OpenAI is*. A real answer has many tokens, and each needs its own share of the score. Deciding which tokens earned it is **credit assignment**.

Follow V along the sampled answer. Before *open*, V = 4.3. Once *open* is written, the expected score is 2, because answers that say *open* score about 2. After that nothing changes until the end.

The **TD error** (temporal-difference error) measures this change at each token: δ = r + V(next state) − V(current state). It is 0 everywhere except at *open*, where it is 0 + 2 − 4.3 = −2.3. The token where the expected score dropped takes the blame. *It's in the name* takes none: by then the outcome was set.

> Credit goes to the token where the expected score changed.

This depends on an accurate V, and in practice V is a learned estimate. **GAE** (generalized advantage estimation) adds a dial, λ, between 0 and 1. With λ = 0, each token's advantage is its own TD error, and the blame stays on *open*. With λ = 1, each token's advantage is the final score minus V at that token, so *OpenAI*, *is* and *open* all get −2.3. That spreads blame onto tokens that did nothing wrong, since every answer starts with *OpenAI is*; over many samples this noise averages out. λ = 0 trusts V, λ = 1 trusts only the final score, and PPO commonly uses a value near 0.95.

GRPO has no per-token V. It gives every token of an answer the same advantage, the answer's score compared with its group. In our answer, all ten tokens would get the same negative value, *It's in the name* included.

## Policy gradient

With an advantage for every token, the update can be stated in one line: raise the log-probability of tokens with A above 0 and lower it for tokens with A below 0, in proportion to A. This is the **policy gradient**. At our state, it moves each next-token logit by an amount proportional to π × A.

Two exact update steps, with an illustrative step size, take the probabilities from 0.50 / 0.30 / 0.20 to 0.15 / 0.36 / 0.49, then to 0.04 / 0.11 / 0.86. V rises from 4.3 to 6.5 to 8.3. In the first step *closed* rises, since 5 beats 4.3. Once V passes 5, *closed* is below average and falls. The advantage is measured against the current model, so the bar keeps rising as the model improves.

## Exploration

An update can use only the answers the model actually wrote. A model that always takes its most likely token, greedy decoding, writes *open* every time. It sees only scores near 2, its average is 2, the advantage of *open* is 0, and nothing changes. It never finds out that *partly* scores 9.

Sampling at temperature 1 sometimes writes *closed* or *partly*. Those samples show that the other branches score higher, and they are what move the policy. This is **exploration**: the policy has to try less likely tokens, so RL training samples its answers.

## Small steps: ratio, clipping and KL

Two more mechanisms keep each update small.

**Ratio and clipping.** The policy gradient assumes the answers came from the current policy, which is called on-policy data. Sampling is expensive, so PPO-style training samples a batch with the current model, keeps a frozen copy π_old, and takes several update steps on that batch. After a few steps the model π has moved away from π_old. The **importance ratio** ρ = π / π_old measures how far, for each sampled token, and reweights its contribution. **Clipping** stops the update once ρ leaves the window from 1 − ε to 1 + ε. With ε = 0.2 and π_old(partly) = 0.2, the advantage of +4.7 can raise π(partly) to 0.24 on this batch and no further. Further progress needs fresh samples.

**KL penalty.** A judge is reliable only near the kind of answers it was built on. Suppose the judge likes flattery: it gives *partly. Some open weights.* a 9 and *partly. What a brilliant question!* a 9.6. The reference model π_ref, the SFT model before RL, writes the flattering version 5% of the time. With nothing to stop it, RL would push the policy toward flattery. The **KL penalty** subtracts β × KL(π‖π_ref) from the reward, where KL measures how far the policy's distribution has moved from the reference. With β = 1, as the policy's share of flattery drifts from 5% to 90%, the judge's average rises only from 9.03 to 9.54, while the KL term reaches 2.38 and R − β × KL falls from 9.03 to 7.16. Drifting toward flattery lowers the objective, so the policy stays near the reference. Real systems compute KL token by token and use a much smaller β; β = 1 here makes the effect visible.

The flattering answer is a small case of **reward hacking**: the policy raises its score by exploiting the judge's preference instead of improving the answer. The KL penalty limits how far that can go.

## Summary

The whole loop fits on one picture: the state, the policy's branches, the actions, the rewards, V, Q, the advantages, and the update that reshapes the branches.

- The LLM is the **policy**. The prompt plus the tokens so far is the **state**, the next token is the **action**, and the whole answer is a **trajectory**.
- The **judge** scores only the finished answer. The **reward** is sparse, and with γ = 1 the **return** at every token is that one score.
- **V** is the average score from a state, **Q** the average after one specific token, and the **advantage** A = Q − V says whether that token beat the model's usual choice.
- A **baseline** such as V or a group average keeps the direction of the update and reduces its noise.
- The **TD error** and **GAE** assign credit to the tokens where the expected score changed.
- The **policy gradient** raises tokens with positive advantage and lowers tokens with negative advantage. **Exploration** supplies the samples it learns from.
- The **ratio** and **clipping** limit the updates on one batch, and the **KL penalty** keeps the policy near the reference, where the judge can be trusted.

The next article covers preference alignment: how a reward model is trained from human preference pairs, and how PPO, GRPO and DPO use it. In this article's terms, PPO learns V with a critic, GRPO replaces V with a group average, and DPO skips the RL loop and learns from preference pairs directly.
