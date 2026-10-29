# X Article Inline Image Plan

Use this after pasting the clean body into X Articles.

Rule: insert images from bottom to top so earlier block positions do not shift.

- Title: `RL Terms for LLMs: How One Score Reaches Every Token`
- Clean body HTML: `x-editor-clean-body.html`
- Clean body Markdown: `x-editor-clean-body.md`
- Total clean body blocks: `83`
- Images: `13`

## Reverse Insertion Order

### IMAGE 12

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-004/assets/12-term-map.gif`
- Exists: `True`
- Insert after block index: `74`
- Target text: The whole loop fits on one picture: the state, the policy's branches, the actions, the rewards, V, Q, the advantages, a...
- Next text: (end of body)

### IMAGE 13

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-004/assets/13-term-cards.gif`
- Exists: `True`
- Insert after block index: `74`
- Target text: The whole loop fits on one picture: the state, the policy's branches, the actions, the rewards, V, Q, the advantages, a...
- Next text: The LLM is the policy. The prompt plus the tokens so far is the state, the next token is the action, and the whole answ...

### IMAGE 11

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-004/assets/11-kl-penalty.gif`
- Exists: `True`
- Insert after block index: `71`
- Target text: KL penalty. A judge is reliable only near the kind of answers it was built on. Suppose the judge likes flattery: it giv...
- Next text: The flattering answer is a small case of reward hacking: the policy raises its score by exploiting the judge's preferen...

### IMAGE 10

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-004/assets/10-ratio-clip.gif`
- Exists: `True`
- Insert after block index: `70`
- Target text: Ratio and clipping. The policy gradient assumes the answers came from the current policy, which is called on-policy dat...
- Next text: KL penalty. A judge is reliable only near the kind of answers it was built on. Suppose the judge likes flattery: it giv...

### IMAGE 09

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-004/assets/09-exploration.gif`
- Exists: `True`
- Insert after block index: `67`
- Target text: Sampling at temperature 1 sometimes writes closed or partly. Those samples show that the other branches score higher, a...
- Next text: Small steps: ratio, clipping and KL

### IMAGE 08

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-004/assets/08-policy-gradient.gif`
- Exists: `True`
- Insert after block index: `63`
- Target text: With an advantage for every token, the update can be stated in one line: raise the log-probability of tokens with A abo...
- Next text: Two exact update steps, with an illustrative step size, take the probabilities from 0.50 / 0.30 / 0.20 to 0.15 / 0.36 /...

### IMAGE 07

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-004/assets/07-credit-assignment.gif`
- Exists: `True`
- Insert after block index: `59`
- Target text: Credit goes to the token where the expected score changed.
- Next text: This depends on an accurate V, and in practice V is a learned estimate. GAE (generalized advantage estimation) adds a d...

### IMAGE 06

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-004/assets/06-baseline.gif`
- Exists: `True`
- Insert after block index: `51`
- Target text: The judge's scores here are all positive. Measured from 0, every sampled answer is pushed up, open included. It is push...
- Next text: As long as the baseline does not depend on which token was sampled, subtracting it leaves the average update unchanged;...

### IMAGE 05

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-004/assets/05-value-q-advantage.gif`
- Exists: `True`
- Insert after block index: `40`
- Target text: The advantage A(s, a) = Q(s, a) − V(s) is the average score after writing token a in state s, minus the average with no...
- Next text: The figure puts the three in one picture. Ten sampled answers land at their own scores. The dots of one branch fall to...

### IMAGE 04

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-004/assets/04-reward-return.gif`
- Exists: `True`
- Insert after block index: `29`
- Target text: With γ = 1 and one reward at the end, the return at every position equals the final score: 2 for OpenAI, 2 for is, 2 fo...
- Next text: The return alone therefore cannot tell which token was good or bad. Every token of this answer carries the same 2, and...

### IMAGE 03

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-004/assets/03-state-action.gif`
- Exists: `True`
- Insert after block index: `24`
- Target text: An LLM is a policy: the prompt plus the tokens so far is the state, and the next token is the action.
- Next text: Reward and return

### IMAGE 02

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-004/assets/02-rl-loop.gif`
- Exists: `True`
- Insert after block index: `18`
- Target text: One round of the loop: the environment gives a prompt, the model writes an answer, the judge returns a score, and the s...
- Next text: State, action, trajectory

### IMAGE 01

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-004/assets/01-one-score.gif`
- Exists: `True`
- Insert after block index: `1`
- Target text: In January 2025, DeepSeek reported DeepSeek-R1-Zero, a model trained with reinforcement learning (RL) directly on a bas...
- Next text: A rule-based check produces one number for a whole response, and the response can be thousands of tokens long. Training...
