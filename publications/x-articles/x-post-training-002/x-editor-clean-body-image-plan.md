# X Article Inline Image Plan

Use this after pasting the clean body into X Articles.

Rule: insert images from bottom to top so earlier block positions do not shift.

- Title: `Instruction Tuning: Same Loss, a Different Task`
- Clean body HTML: `x-editor-clean-body.html`
- Clean body Markdown: `x-editor-clean-body.md`
- Total clean body blocks: `80`
- Images: `10`

## Reverse Insertion Order

### IMAGE 10

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-002/assets/10-token-share.gif`
- Exists: `True`
- Insert after block index: `64`
- Target text: Length matters when mixing data. The loss is averaged over loss tokens, so longer responses take a larger share: a 500-...
- Next text: What instruction tuning can and cannot do

### IMAGE 09

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-002/assets/09-one-position-shift.gif`
- Exists: `True`
- Insert after block index: `56`
- Target text: Without the shift, the shapes still match and the code runs, but each prediction is paired with the token at its own po...
- Next text: Before training, decode the loss tokens and read them. They should contain only the responses and their <|im_end|> mark...

### IMAGE 08

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-002/assets/08-masked-not-untouched.gif`
- Exists: `True`
- Insert after block index: `50`
- Target text: The loss mask decides which predictions count toward the loss. It doesn't change which tokens attention reads. The resp...
- Next text: Masked tokens get no loss, but they still take part in training.

### IMAGE 07

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-002/assets/07-what-gets-loss.gif`
- Exists: `True`
- Insert after block index: `45`
- Target text: Line break after <|im_end|>: no loss. Generation has already ended.
- Next text: As a formula, with mₜ = 1 if position t contributes to the loss and 0 otherwise:

### IMAGE 06

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-002/assets/06-three-forms-summary.png`
- Exists: `True`
- Insert after block index: `36`
- Target text: The figure below sets the three forms side by side:
- Next text: Which tokens get a loss

### IMAGE 05

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-002/assets/05-three-forms.gif`
- Exists: `True`
- Insert after block index: `35`
- Target text: Training form: the loss mask marks which tokens of the sequence contribute to the loss. The model reads the whole seque...
- Next text: The figure below sets the three forms side by side:

### IMAGE 04

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-002/assets/04-recipe.gif`
- Exists: `True`
- Insert after block index: `30`
- Target text: Evaluate on tasks that did not appear in training.
- Next text: Steps 2 and 3 contain most of the details, covered below.

### IMAGE 03

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-002/assets/03-flan.gif`
- Exists: `True`
- Insert after block index: `22`
- Target text: FLAN (2021) rewrote more than 60 NLP tasks as natural-language instructions and fine-tuned a 137B model on them. On tas...
- Next text: Instruction tuning therefore teaches a general ability to follow instructions, which transfers to tasks not seen in tra...

### IMAGE 02

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-002/assets/02-lima.gif`
- Exists: `True`
- Insert after block index: `21`
- Target text: LIMA (2023) fine-tuned a 65B LLaMA model on only 1,000 curated examples, with no reinforcement learning. In the authors...
- Next text: FLAN (2021) rewrote more than 60 NLP tasks as natural-language instructions and fine-tuned a 137B model on them. On tas...

### IMAGE 01

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-post-training-002/assets/01-alpaca-recipe.gif`
- Exists: `True`
- Insert after block index: `3`
- Target text: OpenAI's text-davinci-003 generated the examples, for less than $500. Training took 3 hours on 8 GPUs, for less than $1...
- Next text: Alpaca also gave wrong answers. Asked for the capital of Tanzania, it answered Dar es Salaam. The capital is Dodoma.
