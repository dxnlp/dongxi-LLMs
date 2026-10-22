# Learning instruction tuning

A running record of the learner's discussions of instruction tuning (SFT) and
parameter-efficient fine-tuning (LoRA), opened 2026-10-22 while reading
[SLP3 Chapter 8 §8.1–8.2](https://web.stanford.edu/~jurafsky/slp3/8.pdf). Everything on
these topics is kept in this one file.
Each topic is added at the learner's request, to be reused in later course material
or articles. Canonical course treatment: [Chapter 8](../../book/chapters/08-instruction-data-as-an-interface.md)
and [Chapter 9](../../book/chapters/09-supervised-fine-tuning.md).

## Planned: one coherent instruction-tuning notebook

Requested 2026-10-22. The key instruction-tuning knowledge is currently spread over
several notebooks: loss masks (Day 3 · 02, Day 11 · 01–03), the real Qwen chat template
(Day 11 · 05), gradients through masked prompts (Day 12 · 01) and LoRA (Chapter 9 §9.8).
The plan is one end-to-end notebook that brings it together in a single flow.

**Environment: DGX Spark GPU first** (learner instruction, 2026-10-22). Design and verify
on the Spark GPU kernel (`Python (DGX Spark Native)`, CUDA). Use the GPU for model work: the
real Qwen3-0.6B, full vs. LoRA SFT steps and QLoRA. Keep small concept cells, such as the
3×3 rank example, cheap enough to run anywhere. A CPU fallback is a convenience, not the
design target. Note: the course manifest validator currently accepts only
`execution: "cpu-offline"`, so registering a GPU-first notebook needs a new declared
execution type.


1. A conversation rendered with the real Qwen3 chat template.
2. `-100` labels built and checked: answers plus `<|im_end|>` only, and the shift.
3. Loss computed on those labels, and what the mask does and doesn't stop (prompt
   tokens still influence gradients).
4. LoRA concepts first, each with a small hands-on experiment (following topic 4):
   - **Rank:** build the 3×3 rank-1 example as column × row, check its rank with
     `torch.linalg.matrix_rank`, then add a second independent pattern and watch the
     rank go to 2.
   - **Approximation:** keep only the largest patterns of a matrix (SVD, top-`r`
     singular values) and measure the error as `r` grows. Optionally use a small
     grayscale image, so "fewer patterns" is visible as a blurrier picture.
   - **Low-rank approximation:** write a big matrix as tall-thin × short-wide, count
     parameters (`d×d` vs. `2·d·r`), and plot error vs. parameter count.
   - **Why it fits fine-tuning:** compare the rank structure of a full fine-tuning
     update `ΔW` from a tiny model with a random matrix, to test the bet that
     updates are dominated by a few directions.
5. The same SFT step done twice: full fine-tuning vs. LoRA (frozen `W` plus trainable
   low-rank `AB`). Compare trainable parameters, which tensors get gradients, and
   the update's rank.
6. QLoRA (following topic 5), concept first, then a small experiment:
   - **Quantization by hand:** quantize a weight matrix to 4 bits with one global
     scale vs. block-wise scales (blocks of 64), with and without an injected outlier.
     Measure reconstruction error, so "why blocks" becomes visible.
   - **Why NF4:** compare 16 evenly spaced levels with 16 levels placed at normal
     quantiles on bell-shaped weights, and plot where each puts its levels and the
     resulting error.
   - **Memory accounting:** bytes for weights + gradients + optimizer state under full
     fine-tuning, LoRA and QLoRA, for the toy model and scaled to 0.6B and 7B.
   - **Real QLoRA on the Spark GPU:** `bitsandbytes` NF4 + `peft` on Qwen3-0.6B, a few SFT
     steps on the same masked data as step 5. Measure GPU memory against 16-bit LoRA,
     and confirm only the adapters change. First check that `bitsandbytes` supports the
     GB10/aarch64/CUDA 13 stack.
   - **Hand-built fallback:** if `bitsandbytes` isn't available, freeze the weights as
     hand-quantized 4-bit blocks (dequantized on the fly in the forward pass), attach LoRA
     adapters and train the same steps on the GPU with plain PyTorch. This also works as a
     transparent "what the library does" companion.

Status: **built 2026-10-22** as [Day 12 · 03 "Instruction tuning end to end"](../../notebooks/day-12/03_instruction_tuning_end_to_end.ipynb),
the course's first `spark-gpu` notebook: 14 predictions on real Qwen3-0.6B, about 2 minutes on
the Spark GPU. Differences from the plan:

- QLoRA is hand-built (NF4 from its definition, block scales, packed codes, double-quantization
  arithmetic), because `bitsandbytes` isn't installed in the shared Spark environment. An
  optional cell uses it if it's ever installed.
- Building it surfaced a real lesson, now part of Predict 14. A naive quantized layer kept the
  dequantized weights for backward, so peak memory didn't drop. Re-dequantizing in backward
  (as QLoRA's kernels do) fixed it.
- Reference-run results: rank-8 Q/V LoRA trains 1,146,880 parameters (0.19%, matches `peft`).
  The full fine-tuning ΔW put about 70% of its energy in its top 8 directions (random matrix: 3%).
  Peak memory was about 19.2 GiB (full) vs. 10.6 (LoRA) vs. 10.0 (QLoRA); model storage was
  2.22 vs. 1.11 vs. 0.52 GiB.

## Topics

1. [What an instruction-tuning example looks like](#1-what-an-instruction-tuning-example-looks-like): stored record → chat template → loss mask (2026-10-22)
2. [Instruction tuning as continued training, and why it's called "supervised"](#2-instruction-tuning-as-continued-training-and-why-its-called-supervised) (2026-10-22)
3. [The loss mask](#3-the-loss-mask): which predictions get graded, and how it differs from the attention mask (2026-10-22)
4. [LoRA: rank, approximation, and low-rank approximation](#4-lora-rank-approximation-and-low-rank-approximation) (2026-10-22)
5. [Low rank vs. quantization, and QLoRA](#5-low-rank-vs-quantization-and-qlora) (2026-10-22)

---

## 1. What an instruction-tuning example looks like

Source passage: SLP3 §8.1 ("we call this method supervised fine-tuning because …
each instruction has a supervised objective"). Course: [Chapter 8 §8.2–8.4](../../book/chapters/08-instruction-data-as-an-interface.md).

### Learner's question

What does an instruction-dataset example actually look like?

### Explanation: one example, three stages

An example looks different at three stages: how it's stored, how it's turned into
one token sequence, and which tokens get the loss.

#### 1. As stored: a record of messages

Older datasets like Alpaca (the source of SLP3 Fig. 8.2) used three fields:

```json
{
  "instruction": "Translate the sentence into English.",
  "input": "No tenía lavandería.",
  "output": "It didn't have a laundry facility."
}
```

Modern datasets store a conversation as a list of role-tagged messages, so the same
format handles multi-turn chat:

```json
{
  "messages": [
    {"role": "system",    "content": "You are a helpful assistant."},
    {"role": "user",      "content": "Give me one word for a very happy feeling."},
    {"role": "assistant", "content": "Elated."},
    {"role": "user",      "content": "Another one?"},
    {"role": "assistant", "content": "Joyful."}
  ]
}
```

#### 2. As the model sees it: one string built with a chat template

The model only reads a flat token sequence, so a **chat template** wraps each
message in special role markers. With Qwen's ChatML-style template (the course's
model family), the conversation becomes:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Give me one word for a very happy feeling.<|im_end|>
<|im_start|>assistant
Elated.<|im_end|>
<|im_start|>user
Another one?<|im_end|>
<|im_start|>assistant
<think>

</think>

Joyful.<|im_end|>
```

This is the actual Qwen3-0.6B rendering (verified 2026-10-22). The last assistant turn
gets an empty `<think></think>` block because Qwen3 supports a thinking mode. Earlier
assistant turns don't. Older ChatML templates such as Qwen2.5's render every turn without
it.

#### 3. As it's trained: a loss mask over that string

This is where the "supervised" part comes in. The whole sequence is fed through
the model, but cross-entropy is counted only on the assistant's response tokens:

```
<|im_start|>system ... <|im_end|>                    ✗ no loss (context)
<|im_start|>user Give me one word ... <|im_end|>     ✗ no loss (context)
<|im_start|>assistant                                ✗ no loss (marker)
Elated.<|im_end|>                                    ✓ loss
<|im_start|>user Another one?<|im_end|>              ✗ no loss
<|im_start|>assistant <think></think>                ✗ no loss (marker + empty think block)
Joyful.<|im_end|>                                    ✓ loss
```

In code, the masked positions get the label `-100`, which PyTorch's cross-entropy
ignores.

Two details are easy to miss:

- **The closing `<|im_end|>` is trained.** That's how the model learns to *stop*.
  Mask it out and the model gives the right answer and then keeps rambling.
- **The user's text is never trained.** The model reads it as context, so it can
  attend to it, but it isn't taught to *produce* user messages. This is the
  difference from pretraining, where every token is a prediction target.

**In one line:** the objective is still next-token prediction, but the targets are
limited to a correct response, and that's what makes it "supervised".

### Precision notes for reuse

- Exact template text varies by model and version. Some templates insert a
  default system message or reasoning-mode markers (Qwen3's empty think block above).
  Always render the template with the model's own tokenizer rather than writing the
  markers by hand.
- Hands-on version: [Day 11 notebook 05](../../notebooks/day-11/05_qwen_chat_template_and_labels.ipynb)
  builds these labels from the real Qwen3 template. It also shows that the library's
  built-in assistant mask silently returns zero tokens for this template.
- Whether earlier assistant turns are trained, or only the final one, is a dataset
  design choice. This example trains every assistant turn.
- Loss masking is separate from attention visibility: masked tokens are still read
  as context (Chapter 8 §8.4).

### Reuse ideas (proposed only)

- Article or course figure: the same record shown in all three stages side by side,
  with ✓/✗ coloring on the final row.
- Animation candidate: tokens of the rendered string light up as loss targets, with
  the closing `<|im_end|>` highlighted as "learn to stop". This overlaps the existing
  Day 11 candidate "loss versus visibility masks".

---

## 2. Instruction tuning as continued training, and why it's called "supervised"

Source passage: SLP3 §8.1 ("Even though it is trained to predict the next token
(which we traditionally think of as self-supervised), we call this method supervised
fine-tuning …"). Course: [Chapter 8](../../book/chapters/08-instruction-data-as-an-interface.md),
[Chapter 9 §9.2](../../book/chapters/09-supervised-fine-tuning.md).

### Learner's statement

> Instruction tuning can be regarded as continued training of the model, because it
> uses the same next-token prediction and the same cross-entropy loss, just on an
> instruction-tuning dataset. Compared with pretraining, humans are involved in
> creating this supervised dataset, and that is why it is called SFT.

### Assessment

Mostly right. One part needed correcting, and one difference was missing.

**"Continued training": correct.** Instruction tuning starts from the pretrained
weights and keeps training with the same next-token prediction and the same
cross-entropy loss. SLP3 says the instruction data "is simply treated as additional
training data."

**Missing difference: the loss mask.** In pretraining, *every* token is a target. In
instruction tuning, only the response tokens are; the instruction is read as context
but never predicted ("same loss, different targets"; see topic 3). A smaller learning
rate and far fewer tokens are typical too, but the mask is the essential difference.

**Corrected: "supervised" is not about human involvement.** What makes it supervised
is that the data comes as **input → target pairs**: this instruction, and this
correct response to it.

- **Pretraining is self-supervised.** The labels come from the text itself: the next
  word is just the next word of the document, and nobody designs a target.
  Pretraining text is human-written too (web pages, books), so human involvement
  can't be what separates the two.
- **SFT is supervised** because someone decided "for *this* input, *this* is the
  right output." That someone is often a model, not a person. Most modern SFT data is
  LLM-generated, and it's still called supervised.

### Corrected statement

> Instruction tuning is continued training of a pretrained model with the same
> next-token cross-entropy, but on instruction → response pairs, with the loss only
> on the response. It's called *supervised* fine-tuning because each input comes
> with a designated correct output, a labeled target, unlike pretraining, where the
> targets are just the text's own next tokens.

### Misconception to watch for

"Supervised = humans wrote the labels." The defining property is the designated
input → target pairing, whoever produced it.

---

## 3. The loss mask

Course: [Chapter 8 §8.4](../../book/chapters/08-instruction-data-as-an-interface.md)
(loss vs. attention masks) and §8.8 (token share).

### Learner's question

Explain the loss mask.

### Explanation

A **loss mask** is a per-token on/off switch that decides which positions count
toward the loss. The model still reads every token. The mask only decides which of
its predictions get graded.

**Start from pretraining.** At every position, the model predicts the next token and
pays a penalty of $`-\log p(\text{correct next token})`$. The loss is the average over
*all* positions. Every token is a target.

**Instruction tuning adds the switch.** A tiny example in the spirit of Chapter 8
(`copy red` → `red`):

```
position:   1       2      3      4        5      6
token:     User:   copy   red   Assistant:  red   <END>
```

At each position the model predicts the *next* token. The mask says whether that
prediction is graded:

| At position | Model reads up to | Must predict | Graded? (mask) |
|---|---|---|---|
| 1 | `User:` | `copy` | 0, user text |
| 2 | `… copy` | `red` | 0, user text |
| 3 | `… red` | `Assistant:` | 0, template marker |
| 4 | `… Assistant:` | `red` | **1**, the answer |
| 5 | `… red` | `<END>` | **1**, learn to stop |

So the loss is

```math
L=-\frac{1}{\sum_t m_t}\sum_t m_t\,\log p_\theta(y_{t+1}\mid y_{\le t}),
```

where $`m_t\in\{0,1\}`$ is the mask. Masked positions contribute zero to the sum,
and the average is taken over graded positions only. Here that's 2, not 5.

**In code.** You don't usually build $`m_t`$ separately. You copy the token IDs into
`labels` and overwrite the ungraded positions with `-100`:

```python
input_ids = [User, copy, red, Assistant, red, END]
labels    = [-100, -100, -100,  -100,    red, END]   # aligned to the token being predicted
```

PyTorch's `cross_entropy` skips every target equal to `-100`, so those positions
produce no loss and no gradient.

**Why mask at all?**

- **Don't teach the wrong job.** Without the mask, the model is also trained to
  *write user messages*: to invent questions, imitate users, and predict template
  markers. You want it to learn to *answer*.
- **Put the training signal where it matters.** Prompts are often much longer than
  answers, for example a whole document followed by a one-line summary. Unmasked,
  most of the gradient would go to re-predicting the document, which pretraining
  already covered.
- **Learn to stop.** Keeping `<END>` (or `<|im_end|>`) graded is how the model
  learns to end its turn.

### The most common confusion: loss mask ≠ attention mask

| | Attention mask | Loss mask |
|---|---|---|
| Question it answers | Which tokens can this position *read*? | Which predictions get *graded*? |
| Effect on the user's prompt | still fully readable | not graded |
| Effect on padding | hidden | not graded |

The prompt is *masked from the loss but visible to attention*. The answer at
position 4 can only be right because the model reads `copy red`. Masking the prompt
from attention would make the task impossible.

### One subtle consequence

Because the average is over graded tokens, long answers contribute more gradient
than short ones. A 500-token answer counts about 250 times more than a 2-token one.
That's the token-share effect of Chapter 8 §8.8, and it's why balancing tasks in an
SFT mixture has to be thought about in tokens, not just examples.

### Precision notes for reuse

- In the table, the "Graded?" column is attached to the position whose *next* token
  is the target, which is why `labels` is aligned to the token being predicted. Libraries
  differ on whether the one-position shift happens before or inside the loss call.
  Check this once per codebase: an off-by-one shift silently trains the wrong tokens.
- Token-mean (shown here) is one normalization choice. An example-mean objective
  weights every example equally regardless of answer length (Chapter 9).

---

## 4. LoRA: rank, approximation, and low-rank approximation

Course: [Chapter 9 §9.8 "LoRA constrains an update"](../../book/chapters/09-supervised-fine-tuning.md).
Source passage: SLP3 §8.2, "with LoRA we freeze these layers and instead update a
low-rank approximation that has fewer parameters".

### Learner's questions

What is low-rank? What is approximation? What is low-rank approximation? (Asked for a
plain, easy-to-understand explanation.)

### Explanation

#### Rank

The **rank** of a matrix is how many truly independent patterns it contains. A matrix
can hold many numbers but still be built from only a few underlying patterns.

```
1  2  3
2  4  6
3  6  9
```

This has 9 numbers, but every row is a multiple of the first row: row 2 is 2× row 1,
and row 3 is 3× row 1. There's only **one** real pattern, `[1 2 3]`, repeated at
different strengths, so this matrix has **rank 1**. It can be rebuilt from far fewer
numbers, a column times a row:

```
[1]                 1  2  3
[2] × [1  2  3]  =  2  4  6
[3]                 3  6  9
```

Here 3 + 3 = 6 numbers describe all 9. For a 4096×4096 matrix, a rank-1 version needs
4096 + 4096 = 8,192 numbers instead of about 16.8 million.

**Low-rank** means the rank is small compared with the matrix's size: a few patterns,
not thousands.

#### Approximation

An **approximation** is a simpler version of something that's close but not exact.
3.14 approximates π. A low-resolution photo approximates the original: you still
recognize the face, but fine detail is gone. You accept a little error in exchange for
something much simpler or smaller.

#### Low-rank approximation

Describe a big matrix using only its few most important patterns and drop the rest.
Most real matrices aren't exactly rank 1 or rank 8. They have a few strong patterns and
many weak ones, and a low-rank approximation keeps the strong ones, like photo
compression keeping the main shapes. In matrix form:

```
   big matrix         ≈       tall, thin   ×   short, wide
  (4096 × 4096)              (4096 × r)        (r × 4096)
```

The inner width `r` is the chosen rank: a small `r` means fewer patterns and fewer numbers.

#### How LoRA uses it

LoRA doesn't approximate the pretrained weight `W` itself: `W` stays frozen and complete.
It makes **the change** low-rank. Full fine-tuning learns an update `ΔW` as big as `W`.
LoRA bets that adapting to a new task needs only a handful of patterns, so it builds the
change from two thin matrices (SLP3 notation):

```math
W + \Delta W \;\longrightarrow\; W + AB,\qquad A:[k\times r],\; B:[r\times d],\; r \ll \min(k,d)
```

With `k = d = 4096` and `r = 8`:

- full update: 4096 × 4096 = **16,777,216** trainable numbers
- LoRA update: 4096 × 8 + 8 × 4096 = **65,536** trainable numbers, about **0.4%**

### Precision notes for reuse

- LoRA never computes a full `ΔW` and then compresses it. It trains `A` and `B` directly,
  so the update is **restricted** to rank `r` from the start. "Approximation" describes the
  bet that a rank-`r` update is close enough to what full fine-tuning would learn. The LoRA
  paper's experiments suggest it often is, but it isn't guaranteed for every task.
- SLP3's phrase "low-rank decomposition of W" is slightly loose: the low-rank object is the
  *update* to `W`, not `W` itself.
- Notation differs between sources: SLP3 writes `W + AB` with `A:[k×r]`, `B:[r×d]`. The
  course (Chapter 9 §9.8) writes `W₀ + (α/r)·BA` with the letters swapped and a scaling
  factor `α/r`.

### Reuse ideas (proposed only)

- Figure: the rank-1 3×3 example as "column × row = grid", then the 4096 × r × 4096
  "thin sandwich" with parameter counts.
- Animation candidate: a big update matrix fading in as the sum of `r` rank-1 layers
  (column × row each), showing how a few patterns build the whole change.

---

## 5. Low rank vs. quantization, and QLoRA

Course: none yet. QLoRA isn't covered in the chapters, and this is the first course
treatment. Primary source: Dettmers et al., 2023, "QLoRA: Efficient Finetuning of
Quantized LLMs".

### Learner's questions

Is the low-rank idea related to model compression and quantizing LLMs? Then: explain
QLoRA.

### Low rank and quantization compress different things

| | Low-rank factorization | Quantization |
|---|---|---|
| What it reduces | **how many** numbers you store | **how many bits** each number uses |
| How | replace a big matrix with tall-thin × short-wide | store each weight in 8 or 4 bits instead of 16 |
| Shape of the weights | changes (two thin matrices) | unchanged (same matrix, coarser values) |
| Analogy | describe a picture with fewer brush strokes | the same picture with fewer colors in the palette |

Both are approximations (topic 4): give up a little accuracy for something smaller.

**LoRA is not model compression.** The model doesn't shrink, because the full `W` is still
there, frozen. LoRA shrinks the **training**: the numbers updated, the stored gradients and
the optimizer state. Compressing pretrained `W` itself with a low-rank approximation usually
hurts more, because pretrained weights carry many meaningful patterns. LoRA's bet is only
that the *change* during fine-tuning is low-rank.

Other low-rank uses in LLMs: DeepSeek's Multi-head Latent Attention stores a small low-rank
latent per token instead of full K/V (KV-cache compression). Methods such as LoftQ use a
low-rank correction to recover accuracy lost to quantization.

### QLoRA in plain terms

**Problem:** LoRA makes training cheap, but the frozen base still sits in GPU memory in full
precision: about 14 GB for a 7B model in 16-bit, about 130 GB for 65B.

**Idea:** store the frozen base in 4 bits, keep the LoRA adapters in 16 bits, and train only
the adapters. The base is never updated, so it only needs to be good enough to compute with.

```
frozen base W   →  stored in 4 bits    (never updated, so coarse storage is OK)
LoRA A, B       →  stored in 16 bits   (these learn, so they need precision)
output          =  x·dequant(W) + x·A·B
```

**How it works:**

1. **NF4 ("4-bit NormalFloat").** 4 bits allows 16 values per weight. Trained weights
   roughly follow a bell curve, so NF4 places its 16 levels densely near zero and sparsely in
   the tails: more levels where most weights live means less rounding error.
2. **Block-wise quantization.** Weights are split into blocks of 64, each with its own scale
   (its largest absolute value), so one outlier only distorts its own block.
3. **Double quantization.** The per-block scales are themselves quantized to 8 bits, saving
   about 0.4 bits per parameter.
4. **Compute in 16-bit.** Each 4-bit block is dequantized just in time for the matrix
   multiply. Storage is 4-bit; arithmetic is 16-bit.
5. **Gradients flow through, only adapters update.** Backpropagation passes through the
   dequantized frozen weights into `A` and `B`. Nothing is written back to the 4-bit base.
6. **Paged optimizers.** Optimizer state can move to CPU memory during memory spikes
   instead of failing with out-of-memory.

**Rough memory for a 7B model** (weights and optimizer only; activations extra):

| Method | Base weights | Trainable state | Roughly |
|---|---|---|---|
| Full fine-tuning (16-bit, Adam) | updated | gradients + 2 Adam moments for all 7B | ~100+ GB |
| LoRA | 16-bit, frozen: ~14 GB | tiny adapters | ~15 GB |
| QLoRA | 4-bit, frozen: ~3.5–4 GB | tiny adapters | ~5 GB |

The paper fine-tuned a 65B model on one 48 GB GPU, with quality close to 16-bit LoRA on its
benchmarks.

**Trade-offs:** slower training (dequantizing on every forward pass); some quantization
error in the base model; awkward merging, since `AB` can't simply be added into 4-bit
weights, so you merge into a dequantized 16-bit copy or keep the adapter separate.

**In one line:** quantization compresses what's frozen, LoRA keeps what's trained small, and
together they let one GPU fine-tune a model that full fine-tuning would need a cluster for.

### Precision notes for reuse

- Memory figures are rough order-of-magnitude estimates for weights and optimizer state,
  not measurements. Real usage adds activations, the KV cache during generation, and
  framework overhead.
- "Quality close to 16-bit LoRA" is the paper's reported result on its benchmarks, not a
  guarantee for every model or task. The course hasn't reproduced it.
- The "about 0.4 bits per parameter" saving from double quantization assumes the paper's
  setup: block size 64, FP32 scales reduced to 8-bit with a second level of blocks.

### Reuse ideas (proposed only)

- Figure: the "fewer strokes vs. fewer colors" comparison of low rank and quantization.
- Animation candidate: a weight block's bell-curve histogram snapping to 16 NF4 levels
  vs. 16 evenly spaced levels, then a LoRA adapter lighting up as the only part that
  changes during training.

