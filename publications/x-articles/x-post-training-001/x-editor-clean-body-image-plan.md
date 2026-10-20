# X Article Inline Image Plan

Use this after pasting the clean body into X Articles.

Rule: insert images from bottom to top so earlier block positions do not shift.

- Title: `Why Does a Language Model Need Post-Training?`
- Clean body HTML: `x-editor-clean-body.html`
- Clean body Markdown: `x-editor-clean-body.md`
- Total clean body blocks: `55`
- Images: `6`

## Reverse Insertion Order

### IMAGE 06

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-001/assets/05-rlvr.gif`
- Exists: `True`
- Insert after block index: `49`
- Target text: That makes RLVR well suited to the long, multi-step problems where base models struggle. One caution: a checker is only...
- Next text: In short

### IMAGE 05

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-001/assets/04-preference-alignment.gif`
- Exists: `True`
- Insert after block index: `44`
- Target text: Without the leash, the model could chase high scores, drift far from its starting point, and forget much of what it lea...
- Next text: 3. RLVR: learn from a checker

### IMAGE 04

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-001/assets/03-instruction-tuning.gif`
- Exists: `True`
- Insert after block index: `37`
- Target text: This is also called supervised fine-tuning (SFT), because every instruction comes with a correct answer to imitate. Tra...
- Next text: 2. Preference alignment: learn from comparisons

### IMAGE 03

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-001/assets/03-pretraining-vs-post-training.png`
- Exists: `True`
- Insert after block index: `27`
- Target text: Roughly: pretraining gives the model what it knows; post-training shapes how it uses it.
- Next text: Small data can still change behavior a lot. In one study, mixing just 500 safety examples into a large instruction data...

### IMAGE 02

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-001/assets/02-two-stages.gif`
- Exists: `True`
- Insert after block index: `21`
- Target text: Post-training doesn't build a new model. It keeps training the same one. What changes is the data and the signal that t...
- Next text: Goal. Pretraining: predict the next word in text. Post-training: behave the way people want, which means following inst...

### IMAGE 01

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-001/assets/01-same-prompt-two-models.gif`
- Exists: `True`
- Insert after block index: `10`
- Target text: This article explains why we need it, how it differs from pretraining, and the three techniques it usually involves.
- Next text: Why pretraining isn't enough
