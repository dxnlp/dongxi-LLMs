# X Article Inline Image Plan

Use this after pasting the clean body into X Articles.

Rule: insert images from bottom to top so earlier block positions do not shift.

- Title: `KL Divergence: One Number for How Far a Model Has Moved`
- Clean body HTML: `x-editor-clean-body.html`
- Clean body Markdown: `x-editor-clean-body.md`
- Total clean body blocks: `137`
- Images: `12`

## Reverse Insertion Order

### IMAGE 12

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-kl/assets/12-implicit-reward.gif`
- Exists: `True`
- Insert after block index: `123`
- Target text: That is the idea behind DPO, direct preference optimization. It skips the separate reward model: for each human-chosen...
- Next text: The log-ratio inside KL measures how far an answer moved, and it can also serve as that answer's reward.

### IMAGE 11

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-kl/assets/11-reading-a-run.gif`
- Exists: `True`
- Insert after block index: `115`
- Target text: KL 3.61: reward 2.67, gold −0.76, and the wrong answer now takes 69% of the replies.
- Next text: The reward keeps rising the whole way, and the gold signal rises, peaks and falls. Gao, Schulman and Hilton reported th...

### IMAGE 10

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-kl/assets/10-tokens.gif`
- Exists: `True`
- Insert after block index: `105`
- Target text: GRPO in DeepSeekMath adds the KL to the loss, with β = 0.04. Its per-token estimate, with x = π_ref / π, is x − log x −...
- Next text: How to read KL during training

### IMAGE 09

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-kl/assets/09-from-samples.gif`
- Exists: `True`
- Insert after block index: `101`
- Target text: Samples. KL is an average over the new model's replies, and averages can be estimated from samples. Let the new model w...
- Next text: Tokens. A reply's probability is the product of its tokens' probabilities, each given the tokens before it. The log tur...

### IMAGE 08

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-kl/assets/08-which-way.gif`
- Exists: `True`
- Insert after block index: `98`
- Target text: The swapped order appears in pretraining. The loss used there, cross-entropy, equals the entropy of the data plus KL(da...
- Next text: Real models: samples and tokens

### IMAGE 07

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-kl/assets/07-rare-moves.gif`
- Exists: `True`
- Insert after block index: `91`
- Target text: This is what the training needs. An answer the reference almost never writes is an answer the reward model has had litt...
- Next text: Which model goes on top

### IMAGE 06

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-kl/assets/06-no-direction.gif`
- Exists: `True`
- Insert after block index: `83`
- Target text: The last line has the highest reward and a poor score. Training settles at a compromise, and β sets where. The best ans...
- Next text: The reward chooses the direction. β × KL limits the distance.

### IMAGE 05

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-kl/assets/05-never-negative.gif`
- Exists: `True`
- Insert after block index: `66`
- Target text: What about a huge drop? Take the vague answer from 0.40 down to 0.01, with the best answer rising to 0.51. The log-rati...
- Next text: You cannot get closer to a model than being the same model. KL starts at 0 for no change and only goes up.

### IMAGE 04

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-kl/assets/04-evidence.gif`
- Exists: `True`
- Insert after block index: `58`
- Target text: KL 2.12. A model that always gives the best answer. The sum has one term, 1 × log(1 / 0.12). One reply gives odds of 8....
- Next text: Why KL is never negative

### IMAGE 03

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-kl/assets/03-kl-term-by-term.gif`
- Exists: `True`
- Insert after block index: `51`
- Target text: Σ means add this up over all answers.
- Next text: What the number means

### IMAGE 02

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-kl/assets/02-why-log.gif`
- Exists: `True`
- Insert after block index: `32`
- Target text: It falls without limit near 0. log(0.1) is −2.3, log(0.01) is −4.6, and log(0) does not exist.
- Next text: The log of the ratio is called the log-ratio. For our move it is −0.288 for the vague answer, +0.606 for the best answe...

### IMAGE 01

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-kl/assets/01-two-splits.gif`
- Exists: `True`
- Insert after block index: `8`
- Target text: Now train the model a little. The vague answer falls from 40% to 30%, the best answer rises from 12% to 22%, and the ot...
- Next text: How much did the model change? We want one number: 0 when nothing changed, larger when more changed. The standard answe...
