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

**Start here:** [Workbook progress](#workbook-progress) · [Key idea: patterns connect the concepts](#key-idea-patterns-connect-the-concepts) · [Key concept: what "4-bit" means in QLoRA](#key-concept-what-4-bit-quantization-means-in-qlora)

**Key rules:** [Rule 1 · only the answer and its closing `<|im_end|>` are graded; the header is not](#rule-1--only-the-answer-and-its-closing-im_end-are-graded-the-header-is-not) · [Rule 2 · the one-position shift](#rule-2--the-one-position-shift-logits-1-against-labels1)

1. [What an instruction-tuning example looks like](#1-what-an-instruction-tuning-example-looks-like): stored record → chat template → loss mask (2026-10-22)
2. [Instruction tuning as continued training, and why it's called "supervised"](#2-instruction-tuning-as-continued-training-and-why-its-called-supervised) (2026-10-22)
3. [The loss mask](#3-the-loss-mask): which predictions get graded, and how it differs from the attention mask (2026-10-22)
4. [LoRA: rank, approximation, and low-rank approximation](#4-lora-rank-approximation-and-low-rank-approximation) (2026-10-22)
5. [Low rank vs. quantization, and QLoRA](#5-low-rank-vs-quantization-and-qlora) (2026-10-22)
6. [Notebook session: Day 12 · 03](#6-notebook-session-day-12--03) (started 2026-10-22): `build_labels`, Predicts 1–4, then what a "pattern" is (sales-table example), SVD, and LoRA as a search for patterns

## Workbook progress

[Day 12 · 04 workbook](../../notebooks/day-12/04_instruction_tuning_workbook.ipynb), worked in the
learner's copy `learner_work/notebooks/day-12/`.

| Part | Exercise | Status (2026-10-22) |
|---|---|---|
| A · Data | 1 · token IDs from the chat template | ✅ finished |
| A · Data | 2 · find one answer span | ✅ finished (learner wrote `start = i + len(HEADER)` and the think-block skip) |
| A · Data | 3 · build the labels | ✅ finished (`labels[start:end + 1] = ids[start:end + 1]`) |
| A · Data | 4 · the shifted loss | ✅ finished (`F.cross_entropy(logits[:-1], labels[1:], ignore_index=-100)`) |
| B · Patterns | 5–8 | not started |
| C · LoRA | 9–11 | not started |
| D · QLoRA | 12–14 | not started |

Notebook 03: Predicts 1–13 discussed; Predict 14 (QLoRA by hand) not yet answered.

**Session ended 2026-10-22 after Exercise 4.** Resume at Exercise 5 (one pattern) in the
learner's copy, and answer notebook 03's Predict 14 when convenient.

## Key idea: patterns connect the concepts

The learner's own insight (2026-10-22): **"pattern" is the key that connects the session's
concepts.** A pattern is one column × row building block, a rank-1 matrix `u vᵀ`. As data, every
row is the same shape `v` scaled by `u[i]`. As a calculation, it **reads** one input direction `v`
and **writes** along one output direction `u`. Matrices are built by **adding** patterns together,
not by applying operations one after another (sales-table example, topic 6).

| Concept | Through the pattern lens |
|---|---|
| Rank | how many independent patterns a matrix contains |
| SVD | finds the patterns, strongest first and non-overlapping (σ₁ ≥ σ₂ ≥ …) |
| Approximation | keep the strongest patterns, drop the weak ones (photo: recognizable at r ≈ 20 of 512) |
| Pretrained `W` | strength spread over hundreds of patterns, so not low-rank compressible (97.6% error at rank 8) |
| One example's gradient | each token adds one pattern `δ_t h_tᵀ`, so rank ≤ number of tokens (26 tokens → rank 23) |
| LoRA `W + (α/r)·B·A` | add r new learned patterns on top of a frozen `W`; training searches for the patterns that lower the task loss |
| `B = 0`, `A` random | start with "write nothing" patterns that can still learn (A gives read directions, B learns where to write) |
| Full fine-tuning ΔW | was dominated by a few patterns anyway: 70% of its energy in the top 8 (random: 3%), which is LoRA's bet, measured |
| Quantization (contrast) | **keeps every pattern** (strengths move ~0.24%) and adds a thin, noise-like error spread over all directions |
| QLoRA | quantization where it works (frozen `W`: keep all patterns, fewer bits) + low rank where it works (the update: few patterns) |

Where the lens stops: the loss mask (which positions are graded) and the one-position shift are
about *positions*, not patterns (Rules 1–2).

## Key concept: what "4-bit" quantization means in QLoRA

**Bits and values.** A number stored in *b* bits can take 2ᵇ different values. bf16 uses 16 bits
(65,536 possible values) per weight. **4 bits give only 2⁴ = 16 possible values per weight**, so
each weight is stored as a **code from 0 to 15** (binary `0000`–`1111`) that points to one of 16
**levels**.

**Levels (NF4).** QLoRA's 16 levels lie between −1 and +1, placed by normal-distribution quantiles:
dense near 0, where most weights are, and sparse in the tails, with an exact 0.

```
0: -1.000   1: -0.696   2: -0.525   3: -0.395   4: -0.284   5: -0.185   6: -0.091   7: +0.000
8: +0.080   9: +0.161  10: +0.246  11: +0.338  12: +0.441  13: +0.563  14: +0.723  15: +1.000
```

**Block scale.** Levels live in [−1, 1], but real weights are tiny (around ±0.05). Weights are
grouped into **blocks of 64**, and each block stores one **scale**: its largest absolute weight
(`absmax`). Each weight is divided by the scale, snapped to the nearest level, and only that level's
code is stored.

**Worked example** (8 weights shown for readability; real blocks hold 64), computed on the Spark:
scale = absmax = 0.062.

| Weight | ÷ scale | Nearest level | Code (binary) | Rebuilt (level × scale) | Error |
|---:|---:|---:|---|---:|---:|
| +0.012 | +0.194 | +0.161 | 9 (`1001`) | +0.0100 | −0.0020 |
| −0.034 | −0.548 | −0.525 | 2 (`0010`) | −0.0326 | +0.0014 |
| +0.051 | +0.823 | +0.723 | 14 (`1110`) | +0.0448 | −0.0062 |
| −0.008 | −0.129 | −0.091 | 6 (`0110`) | −0.0056 | +0.0024 |
| +0.000 | +0.000 | +0.000 | 7 (`0111`) | +0.0000 | 0 |
| +0.027 | +0.435 | +0.441 | 12 (`1100`) | +0.0273 | +0.0003 |
| −0.062 | −1.000 | −1.000 | 0 (`0000`) | −0.0620 | 0 |
| +0.019 | +0.306 | +0.338 | 11 (`1011`) | +0.0210 | +0.0020 |

Stored: eight 4-bit codes (32 bits) plus one scale, instead of 8 × 16 = 128 bits in bf16. Two codes
are packed per byte (workbook Exercise 13).

**Storage per weight.** 4 bits for the code, plus the scale shared by 64 weights: 32 / 64 = 0.5 bits
with an fp32 scale (4.5 bits total), or ≈ 0.127 bits with **double quantization**, where the scales
themselves are stored in 8 bits (≈ 4.13 bits total). Against bf16's 16 bits, that's ~3.9× smaller.

**Using a quantized weight.** Nothing computes in 4 bits. Each block is **dequantized** just in time
(level × scale → bf16), used in the matrix multiply, and discarded, and backward rebuilds it again
(notebook 03, Predict 14). Storage is 4-bit; arithmetic is bf16.

**In QLoRA:** the frozen base `W` is stored this way and never changes, since its codes are identical
after training. Only the LoRA adapters (bf16/fp32) learn. **What's lost:** each weight is slightly
rounded (~9% relative error on a real Qwen3 weight), so the quantized model starts a bit worse
(held-out loss 3.5 vs. 2.6 in the reference run), and the adapters learn the task on top of it.
**What it doesn't change:** the number of parameters (596M stays 596M). Only the bytes shrink.

## Key rules (marked important by the learner)

### Rule 1 · Only the answer and its closing `<|im_end|>` are graded; the header is not

Recorded 2026-10-22 at the learner's request, during workbook Exercise 3.

**The `<|im_start|>assistant\n` header is *not* graded. Only the answer itself and the closing
`<|im_end|>` are.** The `labels` list has one entry per input token; graded entries hold the
token ID, and all others are `-100`.

| Position (answer 1) | Token | Label |
|---:|---|---|
| 26 | `<|im_start|>` | `-100`: header |
| 27 | `assistant` | `-100`: header |
| 28 | `\n` | `-100`: header |
| 29 | `El` | **graded** |
| 30 | `ated` | **graded** |
| 31 | `.` | **graded** |
| 32 | `<|im_end|>` | **graded**: learn to stop |
| 33 | `\n` | `-100`: after the turn has ended |

**The principle behind it: grade exactly what the model must produce at inference.**

| Not graded | Why |
|---|---|
| `<|im_start|>assistant\n` header | the inference prompt already ends with it (`add_generation_prompt=True`) |
| empty think block (thinking off) | the template writes it into the prompt |
| `\n` after `<|im_end|>` | generation has already stopped |
| system and user turns | context the model reads, never writes |

In code: `start = i + len(HEADER)` (skip the header, then any empty think block), and
`labels[start:end + 1] = ids[start:end + 1]`, where `end + 1` includes `<|im_end|>`.

### Rule 2 · The one-position shift: `logits[:-1]` against `labels[1:]`

Recorded 2026-10-22 at the learner's request, during workbook Exercise 4.

**The logit at each position guesses the *next* token.** So logit *t* is graded against label
*t + 1*. With a 4-token sequence `A B C D`:

```
position:     0      1      2      3
token:        A      B      C      D
logit there:  guess  guess  guess  guess
              ↓      ↓      ↓      ↓
guesses:      B      C      D      ???
```

| Logit at position | Has read | Guesses | Correct answer (label) |
|---:|---|---|---|
| 0 | `A` | token 1 | `B` ✓ |
| 1 | `A B` | token 2 | `C` ✓ |
| 2 | `A B C` | token 3 | `D` ✓ |
| 3 | `A B C D` | token 4 | **doesn't exist** ✗ |

**Two things are left over, so they're dropped:**

1. **The last logit** (position 3) guesses what comes after `D`, but the sequence ends there:
   nothing to check it against. `logits[:-1]` keeps logits 0, 1, 2.
2. **The first label** (`A`): no logit guesses it, because no position comes before the first
   token. `labels[1:]` keeps labels 1, 2, 3 (`B`, `C`, `D`).

**After dropping, both have length T − 1 and line up one-to-one:**

```
logits[:-1]:  logit0  logit1  logit2
labels[1:]:     B       C       D
```

In code: `F.cross_entropy(logits[:-1], labels[1:], ignore_index=-100)`. Forgetting the shift
still runs, because the shapes match, but it grades each logit on the token it's already reading:
a copying task (Predict 2). Hugging Face models shift internally, so pass them *unshifted* labels.

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

---

## 6. Notebook session: Day 12 · 03

Learner working through [Day 12 · 03 "Instruction tuning end to end"](../../notebooks/day-12/03_instruction_tuning_end_to_end.ipynb)
on the Spark GPU kernel. Notes are added as the discussion goes, grouped by notebook part.

### Part A · setup cell: what `build_labels` does

**Learner's question:** what does the setup cell (`conversation`, `render_ids`, `HEADER`,
`EMPTY_THINK`, `IM_END`, `build_labels`) do?

**Explanation given:** it turns a chat into the `labels` list that grades only the assistant's
answers.

- `render_ids` runs Qwen3's own chat template and returns token IDs.
- `HEADER`, `EMPTY_THINK` and `IM_END` are the token patterns to search for:
  `<|im_start|>assistant\n` (3 tokens), the empty `<think>\n\n</think>\n\n` block
  (4 tokens) and the end-of-turn marker (1 token).
- `build_labels` starts with every label at `-100` and scans the IDs. At each assistant header,
  the answer starts right after it (skipping an empty think block if present) and runs up to and
  including the next `<|im_end|>`. Those IDs are copied into `labels`; everything else stays
  `-100`.
- It works on token IDs, not text, because labels must line up one-to-one with the model's
  input positions.
- In this conversation: answer 1 is positions 29–32 (`El`, `ated`, `.`, `<|im_end|>`); answer 2
  is positions 49–52, after the think block at 45–48 is skipped. That gives 8 graded tokens.
- Labels are *unshifted* (aligned with the input token at the same position). Hugging Face
  shifts them inside the loss; Predict 2 checks this.

### Predict 1 · what the model sees and what it's graded on

**Learner's prediction:** 7 graded tokens; unsure whether the `\n` after each `<|im_end|>`
is graded.

**Reference result:** 8 graded tokens (`El`, `ated`, `.`, `<|im_end|>`, `Joy`, `ful`, `.`,
`<|im_end|>`). The newline is **not** graded.

**Discussion:** the count was one off, an easy slip since `Elated` splits into two tokens (`El`
+ `ated`) and each answer also ends with its own graded `<|im_end|>`. The newline is ungraded
because generation stops at `<|im_end|>`: at inference the model never produces the `\n` after
it, so training it would teach a token the model never has to write. The newline is template
formatting between turns, like the role headers.

Open edge for the learner: guiding principle "grade exactly what the model must generate at
inference", which applies again to the think block in Predict 4 of Day 11 · 05.


### Predict 2 · the one-position shift

**Learner's question:** what does "one-position shift" mean?

**Explanation given:** the model's output at position *t* is a prediction of the token at
position *t + 1*. To grade it, line logit *t* up with label *t + 1*: drop the last logit
(nothing comes after the end) and the first label (nothing predicts the very first token).

| Position | Model reads up to | Its logits predict | Graded against |
|---:|---|---|---|
| 28 | … `assistant` `\n` | next token | label 29 = `El` |
| 29 | … `El` | next token | label 30 = `ated` |
| 32 | … `<|im_end|>` | next token | label 33 = `\n` → `-100`, not graded |

In code: `cross_entropy(logits[:-1], labels[1:])`. "Forgetting the shift" means
`cross_entropy(logits, labels)`: logit *t* is graded on the token at *t*, the one it's already
reading. That's a copying task. Shapes still match, so there's no error, only a wrong number.
Hugging Face models shift internally, so you pass the unshifted labels from `build_labels`.

**Learner's prediction:** a number, no error, "because shapes still match". **Correct, with
the right reason.**

**Reference result** (Spark GPU, Qwen3-0.6B): 8 graded positions. Shifted loss 5.2878 equals
the manual average and Hugging Face's internal loss (also 5.2878). The unshifted loss is 17.8932.
The wrong version is not just slightly off: it grades a different task.

**Follow-up point:** shape checks catch *dimension* bugs, not *meaning* bugs. A wrong shift, a
wrong mask or a double shift all keep valid shapes, so they need value checks: decode the
graded tokens, or compare against a hand-computed loss, as this cell does.

### Predict 3 · masked does not mean untouched

**Learner's prediction:** the embedding rows of prompt tokens receive a gradient; the logits at
masked positions don't. **Correct on both.**

**Reference result:** max |dL/dlogit| at masked positions = 0.0. Input-gradient norms at prompt
tokens `Give` ≈ 3.17, ` happy` ≈ 3.23, ` feeling` ≈ 2.46 (nonzero).

**Why, as discussed:** the loss is a sum over graded positions only, so a masked position's
logits never appear in it, and their gradient is exactly zero. But the graded answer `Elated.`
is computed *from* the prompt through attention: change the prompt's representation and the
answer's probability changes. So the gradient flows backward through attention into the prompt
positions, and into every shared weight that processed them. **The mask controls what is
graded, not what is trained.**

**Subtlety noted:** the notebook measures the gradient on the *input vectors* at prompt
positions (`inputs_embeds`), which isolates the attention path. The embedding *table* in
Qwen3-0.6B is tied to the output head, so every vocabulary row gets some gradient through the
softmax anyway, even tokens absent from the input (Day 12 · 01, Predict 3). Measuring at input
positions avoids mixing those two paths.

### Part B · Predict 4 · rank as a count of patterns

**Learner's prediction:** rank 1 for `[[1,2,3],[2,4,6],[3,6,9]]`. **Correct.** (The second part,
adding an independent pattern, was answered in the discussion: the rank becomes 2.)

**Reference result:** rank 1, rebuilt from 3 + 3 = 6 numbers as column `[1,2,3]` × row
`[1,2,3]`. After adding `[1,0,-1]ᵀ × [0,1,0]`, the rank is 2.

**Link made to LoRA:** a rank-*r* matrix is a sum of *r* column × row patterns. LoRA's update
`(α/r)·B·A` with `B: d_out × r` and `A: r × d_in` is exactly a sum of *r* such patterns: column *i*
of `B` times row *i* of `A`. With r = 8, the update can contain at most 8 independent patterns.

**Learner's follow-up:** what exactly is a "pattern" here? A calculation being applied?

**Explanation given:** a pattern is a **rank-1 matrix**, `u vᵀ`: a column `u` times a row `v`.
Each entry is a product, `M[i][j] = u[i] · v[j]`, so every row is the same row `v` scaled by
a different amount `u[i]`. It can be read two ways:

- *As data* (like the photo in Predict 5): a grid with one repeated shape, where `v` sets the
  shape across columns and `u` sets the strength down rows.
- *As a calculation* (how a weight matrix is used, `y = M x`): `(u vᵀ) x = u · (v · x)`. The
  pattern **reads** one direction `v` from the input (how much does `x` point along `v`?) and
  **writes** that amount along one direction `u`. One detector, one output direction.

A rank-*r* matrix is a sum of *r* such read/write pairs. In SVD the patterns are `σₖ uₖ vₖᵀ`,
ordered by strength `σₖ`. So LoRA's rank-8 update adds 8 new "read direction → write
direction" pairs to a frozen layer.

**Learner's follow-up:** wants easier examples of one, two and many patterns. Learner's
working model: "there is an original plain matrix, and multiple calculations or patterns are
applied to it."

**Correction given:** patterns aren't operations applied to a matrix one after another. They're
**building blocks that are added together**. A matrix *is* the sum of its patterns, starting
from an empty grid of zeros. (For LoRA the learner's picture is nearly right: there the
starting matrix is the frozen `W`, and LoRA *adds* a few new patterns on top of it.)

**Example used: a shop sales table** (3 stores × 4 days):

- **One pattern.** Every store follows the same daily rhythm `[1, 2, 2, 1]`, scaled by store
  size `[10, 20, 30]`. Rows `10 20 20 10 / 20 40 40 20 / 30 60 60 30`: each row is a multiple
  of one shape, so rank 1.
- **Two patterns.** Add a promotion on days 3–4, shape `[0, 0, 5, 5]`, with strength `[1, 0, 2]`
  per store (store B skipped it). Sum: `10 20 25 15 / 20 40 40 20 / 30 60 70 40`. The rows are
  no longer multiples of one shape, so rank 2, and still exactly 2 patterns added together.
- **Many patterns.** A real table (1,000 stores × 365 days) also has holidays, weather, local
  events and noise, each another pattern. Usually a few strong ones (weekly rhythm, store
  size) explain most of it, and the rest are small. Keeping only the strong ones is a low-rank
  approximation.

**Mapping to LoRA:** `W_new = W_frozen + pattern₁ + … + pattern₈`, the frozen weight plus 8 learned
column × row patterns.

Status: the learner reports understanding patterns "a lot better" after the sales example.
Their mental model moved from "applied calculations" to "added building blocks"; not yet checked
with a new example.

**Learner's follow-up:** so SVD is used to find the most essential patterns?

**Explanation given:** yes. SVD (singular value decomposition) splits *any* matrix into a sum of
patterns, `M = σ₁u₁v₁ᵀ + σ₂u₂v₂ᵀ + …`, with two guarantees:

1. **Sorted by strength:** σ₁ ≥ σ₂ ≥ … (the singular values), so the first patterns carry the
   most.
2. **No overlap:** the patterns are mutually independent (orthogonal), so none repeats part of
   another, and nothing is double-counted.

Because of both, **keeping the top r patterns gives the best possible rank-r approximation**
(Eckart–Young theorem), which is what Predicts 5–6 use.

**Nuance:** on the sales table SVD would find 2 patterns, but not necessarily the exact
"rhythm" and "promotion" we built. It returns orthogonal combinations of them that span the
same thing. SVD finds the essential *directions*, not human-named causes.

**Where SVD appears for LoRA:** as an analysis tool (Predicts 6 and 10 measure rank and energy
with it). LoRA training itself doesn't run SVD; it learns `A` and `B` by gradient descent.
Some variants (e.g. PiSSA) initialize LoRA from the SVD of `W`, but that's an extension, not
standard LoRA.

**Learner's synthesis:** "LoRA training can be regarded as finding the best patterns to apply
to the parameters in order to perform some task?"

**Assessment:** essentially right. Refinements given:

1. **Added, not applied:** the patterns are *added* to frozen weights, `W + (α/r)·B·A`.
2. **"Best" means lowest training loss found by gradient descent**, within a fixed budget: r
   patterns per adapted layer (here r = 8 on `q_proj` and `v_proj` in each of 28 layers). It's
   the best the optimizer reaches, not a guaranteed global optimum, and "best on training data"
   isn't automatically best on new data.
3. **Unlike SVD, there's no target matrix to decompose.** SVD describes a matrix you already have.
   LoRA's target change is unknown; it's defined only through the loss on the task data. The
   gradient tells each `A` and `B` which read/write directions would lower the loss.
4. **Jointly:** all 28 × 2 adapters are learned together, so a pattern in one layer works with
   patterns in the others.

**Refined statement:** LoRA training searches, by gradient descent on the task's loss, for a
small number of column × row patterns per chosen layer which, *added* to the frozen weights,
make the model do the task well.

### Predict 5 · approximation: keep the strongest patterns

**Learner:** didn't know and asked for the answer (no prediction made).

**Reference result** (Grace Hopper photo, 600 × 512, so at most 512 patterns): relative error
49.9% at r = 1, 33.0% at r = 5, 18.5% at r = 20, 10.6% at r = 50, 0% at r = 512. The face is
recognizable from about r = 20 and clearly detailed by r = 50, under 10% of the patterns.

**Explanation given:**
- r = 1 is a grid of horizontal and vertical bands. One column × row pattern can only make
  "plaid".
- r = 5 shows a blurry figure; r = 20 a recognizable face (glasses, cap); r = 50 looks close to
  the original.
- The error curve drops fast, then flattens: a few strong patterns carry the overall shape,
  and hundreds of weak ones carry fine detail (texture, edges, noise).
- Link back to the sales table: the strong patterns are the "weekly rhythm" of the photo.
- Link forward: Predict 6 asks whether a *pretrained weight matrix* is as compressible as this
  photo. It isn't, which is why LoRA makes only the *update* low-rank.

### Predict 6 · is a pretrained weight as compressible as the photo?

**Learner's prediction:** "maybe not fully compressible". **Right direction; the real answer
is more extreme.** It's barely compressible at all.

**Reference result** (layer 10 `v_proj`, 1024 × 1024 = 1,048,576 numbers):

| r | numbers stored | share of full | rank-r error |
|---:|---:|---:|---:|
| 8 | 16,384 | 1.6% | 97.6% |
| 64 | 131,072 | 12.5% | 86.3% |
| 256 | 524,288 | 50% | 57.8% |
| 512 | 1,048,576 | 100% | 28.9% |

For comparison, the photo was at 10.6% error with 50 patterns. On the plot, the pretrained
weight's curve sits close to a random matrix's.

**Explanation given:** pretraining packs many independent features into each weight matrix,
learned from a huge variety of text, so its strength is spread over hundreds of patterns rather
than concentrated in a few. Compressing `W` itself to rank 8 would destroy almost everything.
Hence LoRA's design: keep `W` complete and frozen, and make only the **update** low-rank. The bet
is that *adapting* to one task needs few new patterns, even though `W` holds many. Predicts 7
and 10 test that bet.

### Learner insight: "pattern" as the session's key idea

**Learner's observation:** the pattern concept could serve as a key for understanding many
pieces of this session.

**Agreed, with the map below:**

| Notebook step | Read through "patterns" |
|---|---|
| Predict 4 · rank | rank = how many independent patterns a matrix contains |
| Predict 5 · photo | approximation = keep the strongest patterns, drop the weak ones |
| SVD | the tool that finds patterns, sorted by strength and non-overlapping |
| Predict 6 · pretrained `W` | strength spread over hundreds of patterns, so not compressible |
| Predict 7 · one example's gradient | each token contributes one pattern (`δ_t h_tᵀ`), so rank ≤ token count |
| LoRA `(α/r)·B·A` | add r new learned patterns to a frozen `W` |
| Predict 10 · full fine-tuning ΔW | measures whether the real change was dominated by a few patterns (70% in the top 8, in the reference run) |

**Where the lens stops:** it doesn't explain Part A's loss mask (which positions are graded)
or Part D's quantization. Quantization reduces the *bits per number*, not the number of
patterns, as in topic 5's "fewer colors vs. fewer brush strokes". QLoRA combines both: fewer
bits for the frozen `W`, and a few patterns for the update.

**Reuse idea (proposed only):** an article or chapter opener, "One idea behind LoRA:
patterns". It would follow the sales table → photo → pretrained weight → gradient → LoRA, with
quantization as the deliberate contrast.

### Predict 7 · why one example's gradient is low-rank

**Learner:** asked for help answering and understanding it (no independent prediction).

**Explanation given, in pattern terms:**
- For a linear layer, each token position t computes `y_t = W h_t` with the same `W`.
- Backpropagation gives one input-side vector `h_t` (what came in) and one output-side vector
  `δ_t = ∂L/∂y_t` (how the output should change) per token. That token's share of the gradient is
  `δ_t h_tᵀ`, one column × row **pattern**.
- The full gradient is the sum over tokens: `∂L/∂W = Σ_t δ_t h_tᵀ`. So **n tokens give at most n
  patterns, and rank ≤ n.**
- Answer: at most **26** for a 26-token example. The reference run measured **23**, fewer than 26:
  some tokens add nothing (positions after the last graded prediction can't affect the loss, so
  their `δ_t` is zero) or add patterns that overlap others.
- Scale: 23 patterns out of a possible 1,024, so one example nudges `W` in only a few
  directions. That's the first evidence for LoRA's bet.

**Limits stated:** a batch of 65 examples × ~30 tokens could in principle reach full rank
(> 1,024 patterns), Adam rescales each entry, and many steps add up. So this makes the bet
*plausible*, not guaranteed. Predict 10 measures a real full fine-tuning change. Technical
detail: the check runs in fp32, because bf16 rounding adds tiny spurious patterns (in bf16 it
read rank 32).

### Part C · Predict 8 · trainable parameters, and LoRA at step 0

**Learner's prediction:** adding LoRA doesn't change the output before training, "because B
starts at zero". **Correct, with the right reason.**

**Reference result:** full fine-tuning trains 596,049,920 parameters; LoRA (r = 8 on
`q_proj`/`v_proj`) trains 1,146,880 = 28 × 40,960 (0.192%), and `peft` counts exactly the same.
Only 112 tensors get gradients (28 layers × q/v × A/B). Max logit change from adding LoRA = 0.0.

**Follow-up discussed: why is `B` zero but `A` random, rather than both zero?**
- With `B = 0`, `B·A = 0`, so the model starts exactly as pretrained.
- Gradients at step 0: `∂L/∂B ∝ (upstream gradient) · (A h)ᵀ`, which is nonzero because `A` is
  random. `∂L/∂A ∝ Bᵀ · …`, which is zero because `B = 0`. So `B` moves first, and then `A`
  starts moving too.
- If both were zero, both gradients would be zero forever and LoRA would never learn. Random `A`
  breaks the symmetry; zero `B` keeps the start exact.
- Day 14 · 01 tests this asymmetry directly ("A gradient is zero, B nonzero").

### Predict 9 · full fine-tuning vs. LoRA on the same masked data

**Learner's prediction:** full fine-tuning uses more memory, "maybe twice as much". **Very
close.**

**Reference result** (40 steps, same batch):

| Run | Trainable | Peak GPU memory | Model storage | Held-out loss | Time |
|---|---:|---:|---:|---|---:|
| Full fine-tuning (fp32, lr 1e-5) | 596,049,920 | 19.17 GiB | 2.22 GiB | 2.613 → 2.0e-5 | 38 s |
| LoRA (bf16 frozen base, lr 1e-3) | 1,146,880 | 10.59 GiB | 1.11 GiB | 2.601 → 4.2e-6 | 12 s |

The ratio is about 1.8×, matching the learner's "twice".

**Why not 10× more, given the parameter counts?** Weight-and-optimizer memory differs hugely:
roughly 16 bytes × 596M ≈ 9 GiB for full fine-tuning against ~1.1 GiB of bf16 base plus a tiny
adapter state for LoRA. But both runs share a large **activation** cost, about 9 GiB here. The
output layer scores all 151,936 vocabulary entries at every position of all 65 examples, and
backward needs those activations. LoRA shrinks the trainable state, not the activations. So the
peak ratio is ~1.8×, not ~10×.

**Held-out loss:** both fall close to zero. The task is easy and the runs are short, so this is
mechanics, not a quality comparison. LoRA was also ~3× faster per run here.

### Predict 10 · was the full fine-tuning update low-rank?

**Learner's prediction:** around 50% of ΔW's energy in its top 8 patterns. **Good instinct
(far above random's 3%); the measured value was higher.**

**Reference result** (layer 10 `v_proj`, after 40 full fine-tuning steps): top-8 energy
**70.2%** for full fine-tuning ΔW, 100% for LoRA's ΔW (rank 8 by construction), **3.0%** for a
random matrix of the same shape. LoRA ΔW numerical rank = 8.

**Reading discussed:**
- Full fine-tuning was free to change all ~1 million numbers in any direction, yet ~70% of its
  change landed in just 8 patterns. On the spectrum plot its singular values drop steeply after a
  handful, while the random matrix's stay almost flat.
- That is LoRA's bet measured rather than assumed: the update needed for this task was dominated
  by a few patterns, even though `W` itself (Predict 6) is not.
- The other ~30% is what a rank-8 update *can't* express, so rank remains a real constraint.
- "Energy" means the sum of squared singular values: the share of the change's total size
  captured by those patterns.

**Limits:** one layer, one tiny dataset, 40 steps. Evidence about this run, not a general law.

### Learner question: does "pretrained weights can't be compressed" conflict with QLoRA?

**Learner's question:** earlier (Predict 6) the claim was that a pretrained LLM weight can't
be compressed. Doesn't quantization, and QLoRA, contradict that?

**Answer: no conflict, but the earlier wording was too broad and needed correcting.** Predict 6
showed that pretrained weights can't be compressed **by low rank**, i.e. by keeping fewer
patterns. Quantization compresses a *different kind* of redundancy:

| | Low-rank (fewer patterns) | Quantization (fewer bits per number) |
|---|---|---|
| What it keeps | only the strongest r patterns | all patterns, every number |
| What it gives up | the weaker patterns entirely | precision: each number is rounded |
| Assumption | strength sits in a few patterns | each number doesn't need 16 bits |
| True for pretrained `W`? | **no**: strength is spread out | **mostly yes**: small rounding is tolerated |

**Same-budget comparison** (layer 10 weights, reference run): storing ~25% of the original
size costs **~75% relative error** as a low-rank approximation (between r = 64 at 86% and
r = 256 at 58%), but only **~9% error** as 4-bit NF4 (Predict 12). Same space, completely
different damage.

**Analogy:** low rank keeps a few brush strokes and drops the rest of the painting. Quantization
keeps every stroke but uses a smaller palette of colors. The painting has too many important
strokes to drop, but survives a smaller palette.

**Not free:** the 4-bit model started from a higher held-out loss than bf16 (3.5 vs. 2.6,
Predict 14), so rounding does cost some accuracy. That's why QLoRA trains adapters on top.

**Correction recorded:** "pretrained weights aren't compressible" should read "pretrained weights
aren't *low-rank* compressible".

### Learner question: does quantization decrease the model size?

**Answer: yes in bytes, no in parameters.** "Model size" means two different things:

| "Size" | Changed by 4-bit quantization? |
|---|---|
| Number of parameters (596M for Qwen3-0.6B) | **No**: same weights, same architecture |
| Bytes in memory / on disk | **Yes**: about 2 bytes → about 0.5 bytes per quantized weight |

**From this notebook** (reference run): the 196 linear layers go from 840 MiB in bf16 to 210 MiB
of 4-bit codes + 26 MiB of block scales. The whole model goes from 1.11 GiB to 0.52 GiB, less
than 4×, because the tied embedding (~155M parameters) stays bf16.

**What it doesn't automatically shrink:**
- **Compute:** the hand-built version dequantizes to bf16 for every matrix multiply, so it ran
  *slower* (16.7 s vs. 12.0 s for bf16 LoRA). Speedups need hardware or kernels that compute in
  low precision directly (e.g. int8/fp8 paths).
- **Activations:** unchanged; they dominated the peak memory in Predict 9.

Common use: 4-bit checkpoints (e.g. GGUF/NF4 files) let large models fit on smaller GPUs or
laptops. Same model, fewer bytes, slight accuracy cost.

### Learner question: can the pattern concept explain quantization too?

**Answer: yes, as a contrast**, not as the mechanism itself (quantization is a per-number rounding
rule). Write the quantized weight as `W_q = W + E`, with `E` the rounding error. Then ask the
pattern question about each part.

**Measured on the Spark, 2026-10-22** (layer 10 `q_proj`, NF4, blocks of 64; an ad-hoc check run
during the discussion, not a notebook cell):

| Quantity | Value |
|---|---|
| Size of the rounding error, ‖E‖ / ‖W‖ | 9.3% |
| Top-8 energy: `W` / `E` / random matrix | 7.9% / 4.3% / 2.2% |
| Top-64 energy: `W` / `E` / random matrix | 32.6% / 22.3% / 15.9% |
| Median relative change of each of W's pattern strengths (singular values) | 0.24% |

**Reading, in pattern terms:**
- **Low rank** removes whole patterns: keep 8, delete about 1,016. Big, concentrated damage.
- **Quantization keeps every pattern almost exactly.** Each pattern's strength moves by ~0.24% in
  the median. What it adds is `E`: a small error **spread thinly over all directions, like noise**.
  Its energy profile sits close to the random matrix's, with no dominant pattern.
- So quantization isn't "fewer patterns". It's "all patterns, plus a faint layer of noise".
- **Link to QLoRA:** LoRA adds a few patterns to the frozen, slightly noisy base. Those few
  patterns can't cancel noise spread over every direction. They learn the *task*, and partly
  steer around the damage that matters for it (the 4-bit model's higher starting loss, 3.5 vs.
  2.6, falls just as far after training in Predict 14).

### Part D · Predict 11 · why quantize in blocks?

**Learner's prediction:** the global scale is hurt more by the outlier. **Correct.**

**Reference result** (layer 10 `q_proj`, 16 evenly spaced levels):

| | Global scale | Blocks of 64 |
|---|---:|---:|
| Clean matrix | 46.8% | 10.3% |
| With one outlier (value 50) | 7349.2% | 40.8% |

**Why, as discussed:**
- A scale is set by the largest weight it covers (`absmax`). The 16 levels are spread between
  −scale and +scale, so the step between levels is about scale / 7.5.
- **Global:** one outlier of 50 sets the step for all ~2 million weights to about 6.7, far larger
  than typical weights (~0.02). Every small weight snaps to the levels nearest zero. Because 16
  evenly spaced levels have no exact zero (closest are ±1/15 of the scale ≈ ±3.3), every tiny weight
  is pushed to about ±3.3, which gives the exploding error.
- **Blocks of 64:** the outlier only sets the scale of its own block. The other ~32,000 blocks keep
  their fine steps. The remaining 40.8% comes almost entirely from that one block's 63 neighbors
  (pushed to ±3.3) and from the error measure including the outlier's size.
- Even the clean matrix shows it: 46.8% vs. 10.3%. Ordinary larger-than-usual weights already
  coarsen a global scale.
- Cost of blocks: one stored scale per 64 weights (+0.5 bits per weight at fp32, or ~0.127 with
  double quantization, Predict 13).

### Predict 12 · why NF4 levels?

**Learner's prediction:** NF4 has lower error, with its levels denser near zero. **Correct on
both.**

**Reference result:** NF4 built from its definition matches bitsandbytes' published table to within
1.8 × 10⁻⁷. Relative error on layer 10 `q_proj` (blocks of 64): uniform levels 10.27%, NF4 9.27%.
The histogram shows block-normalized weights piled up near 0, with NF4's levels clustered there
and spread out toward ±1.

**Discussion points:**
- How NF4 is built: take the bell curve (normal distribution) and choose levels at evenly spaced
  *probability* steps (quantiles), not evenly spaced *values*. Each level then covers roughly
  the same number of weights.
- It's asymmetric: 8 positive levels, 7 negative and an exact 0, so 16 in total, and zero is
  representable (fixing the "no exact zero" problem of Predict 11's uniform grid).
- The improvement is real but modest here (≈1 percentage point, ~10% relative). Block-wise
  scaling (Predict 11) is the bigger win; NF4 is a refinement on top.

### Predict 13 · memory accounting

**Learner's prediction (7B):** full ≈ 100 GiB, LoRA ≈ 15, QLoRA ≈ 5. **Essentially exact.**
These match the rough figures from the QLoRA discussion (topic 5), now computed.

**Reference result** (weights + gradients + AdamW state only):

| Model | Full fine-tuning | LoRA | QLoRA |
|---|---:|---:|---:|
| Qwen3-0.6B | 8.9 GiB | 1.13 GiB | 0.52 GiB |
| 7B (illustrative, 90% quantizable) | 104.3 GiB | 13.34 GiB | 4.63 GiB |

Bits per quantized weight: NF4 + fp32 block scales = 4.500; with double-quantized 8-bit scales =
4.127, saving 0.373 bits, at +0.008% extra relative error. Qwen3-0.6B: 440M of 596M parameters (74%)
are quantizable linear weights; the tied embedding stays bf16.

**Discussion:**
- Per parameter: full ≈ 16 bytes (fp32 weight, gradient, two AdamW moments); LoRA ≈ 2 (bf16 frozen
  base); QLoRA ≈ 0.5 on the quantized part. That's where the 100 / 13 / 4.6 pattern comes from.
- These figures exclude activations, which Predict 9 showed can dominate for a small model with a
  large vocabulary.

### Learner request: a hands-on workbook (paused before Predict 14)

**Request:** a more fine-grained Day 12 notebook where the learner *writes* small, key lines of
code (not large blocks), with tests that check the result. Either a new notebook or an
improvement of notebook 03.

**Decision:** a new companion, `notebooks/day-12/04_instruction_tuning_workbook.ipynb`. Notebook 03
stays as the executed, reviewed reference ("answer key"). The workbook follows the same Parts A–D:
each exercise asks for 1–3 key lines, a check cell reports ✅/❌ with a hint, and a reference
solution follows. Check functions live in a tested module under `src/dongxi_llms/`.

Status: **built 2026-10-22**. Fourteen exercises (1–4 key lines each) across Parts A–D, checked by
`src/dongxi_llms/sft_workbook.py` (8 unit tests). Verified on the Spark GPU: blank (14 ⏳, all
references pass), with solutions filled (all 92 checks ✅) and from a personal copy. Personal copies
for live study live in the git-ignored `learner_work/notebooks/day-12/`, so the reference notebooks
stay output-free. Predict 14 in notebook 03 is not yet answered.

### Workbook session · Exercise 1 question: is the think block always empty in SFT?

**Learner's question:** the template shows an empty thinking block. What happens in the real
world? Is the think block always empty in instruction tuning / SFT?

**Answer: no.** It's empty here only because our data has no reasoning. Verified with the Qwen3
template on the Spark (2026-10-22): an assistant message with a `reasoning_content` field renders as

```
<|im_start|>assistant
<think>
17 + 25: 17 + 20 = 37, then 37 + 5 = 42.
</think>

42<|im_end|>
```

**Real-world practice:**
- **Reasoning SFT** (e.g. DeepSeek-R1's "cold start" data, Qwen3's long chain-of-thought stage)
  fills the think block with worked reasoning, and **grades it**: the model must learn to
  *generate* the reasoning, not just the answer.
- **Non-reasoning SFT** (chat, short answers) has no reasoning, so a hybrid model like Qwen3 shows
  an empty block. Qwen3's post-training mixes both kinds of data so one model can answer with or
  without thinking (its report calls this "thinking mode fusion").
- **History is stripped:** the template removes reasoning from *earlier* assistant turns (verified:
  turn 1's reasoning disappears in a two-turn render) and keeps it only in the final turn. Old
  reasoning isn't fed back as context.

**Whether to grade an empty block depends on how inference prompts are built** (same principle:
grade what the model must generate):
- `enable_thinking=False`: the template itself writes the empty block into the *prompt*, so the
  model never generates it. Mask it (the workbook's choice).
- Soft switch (e.g. "/no_think" in the user message, thinking enabled): the model must *produce*
  `<think>\n\n</think>` itself, so the empty block is graded.

**Learner's follow-up:** so SFT also affects thinking? And it's mainly RL that improves thinking?

**Answer: both shape thinking, in different ways.**

*What SFT does to thinking:*
- **Format and style:** where reasoning goes (`<think>…</think>`), how long it is, what it looks
  like (step by step, self-checking).
- **A starting behavior ("cold start"):** a few thousand good traces teach the model to begin
  reasoning at all, readably. DeepSeek-R1 added this because RL alone (R1-Zero) reasoned well but
  produced hard-to-read, language-mixing traces.
- **Distillation can transfer real ability:** DeepSeek's R1-Distill models are small models trained
  with SFT only, on about 800k traces from R1, and they reason strongly. For small models DeepSeek
  reported this beat running RL directly. But SFT only imitates: the student copies the teacher's
  reasoning, mistakes included.

*What RL (RLVR) does:* the model generates its own reasoning and is rewarded when the final answer
checks out (Chapter 13). Reasoning that leads to correct answers becomes more likely. That improves
accuracy and can bring out behaviors such as re-checking and longer reasoning on hard problems. No
human-written trace is needed, only a verifier.

*An open debate:* does RL create new reasoning ability, or sharpen what the base model could
already do? Some 2025 studies found RLVR raises single-attempt accuracy (pass@1), while the
untrained base model, given many attempts (large pass@k), solves a similar or larger set of
problems. That suggests RL mainly makes good paths more likely rather than inventing new ones,
whereas distillation from a stronger teacher can bring in new patterns. Treat this as current
evidence, not settled.

*Typical pipeline (DeepSeek-R1, Qwen3, broadly):* cold-start reasoning SFT → reasoning RL →
rejection-sampling SFT (keep the RL model's best traces, retrain) → general RL. So SFT and RL
alternate: **SFT teaches the form and seeds the behavior; RL optimizes it against correctness.**

Course links: Chapter 13 (GRPO/RLVR), Chapter 15 (distillation), and the learner's three-family map
(instruction tuning, preference alignment, RLVR).

### Workbook · Exercise 2 question: what does `ids.index(IM_END, start)` mean?

**Explanation given:** `list.index(value, start)` returns the position of the **first** occurrence
of `value` at or after position `start` (and raises `ValueError` if there's none). Here it finds the
`<|im_end|>` that closes *this* answer. The `start` argument matters: the conversation contains
several earlier `<|im_end|>` tokens (after the system and user turns, at positions 9, 24, …), and
searching from 0 would return the wrong one. In the example, answer 1 starts at 29, and
`ids.index(IM_END, 29)` returns 32, so the span is 29–32 (`El`, `ated`, `.`, `<|im_end|>`).

**Learner's lines for Exercise 2:** `start = i + len(HEADER)` and `start = start + len(EMPTY_THINK)`,
both correct and worked out by the learner. Line 3 (`end = ids.index(IM_END, start)`) follows from
the explanation above.

**Exercise 3:** the learner wrote `labels[start:end + 1] = ids[start:end + 1]`, correct, including the
`+ 1` that keeps `<|im_end|>` graded (a slice stops before its end index).

### Workbook · Exercise 3 · build the labels

**Learner's line:** `labels[start:end + 1] = ids[start:end + 1]`, correct, with the `+ 1` worked
out from Python's exclusive slice end so that `<|im_end|>` is graded.

**Learner's question:** do the labels contain everything from `<|im_start|>assistant` to `<|im_end|>`?

**Answer: no, the header is excluded.** Two clarifications:
1. `labels` always has one entry per input token (54 here). The question is which entries hold a
   real token ID instead of `-100`.
2. Graded entries run from the **first answer token** (after `<|im_start|>assistant\n`, and
   after the empty think block if present) through `<|im_end|>`. For answer 1: positions 26–28
   (`<|im_start|>`, `assistant`, `\n`) stay `-100`, and 29–32 (`El`, `ated`, `.`, `<|im_end|>`) are
   graded.

Why the header isn't graded: at inference the prompt already ends with
`<|im_start|>assistant\n` (`add_generation_prompt=True` writes it). The model never generates the
header, so it isn't trained to. Same rule as for the think block and the trailing newline: grade
exactly what the model must produce.

### Workbook · Exercise 4 question: what is a logit again?

**Explanation given:** a **logit** is the model's raw score for one candidate next token, before
being turned into a probability. At every position the model outputs one score per vocabulary
entry: shape `[T, V]`, with V = 151,936 for Qwen3. Softmax turns a row of scores into
probabilities: `p_k = exp(z_k) / Σ_j exp(z_j)`. A higher logit means more likely; only differences
between logits matter.

Toy example (vocabulary of 4): logits `[2.0, 1.0, 0.1, -1.0]` for `[El, Hi, ., <|im_end|>]` give
probabilities ≈ `[0.64, 0.23, 0.10, 0.03]`.

`F.cross_entropy(logits, targets)` takes the raw logits and applies log-softmax internally, so you
pass logits directly, not probabilities. Per graded position, the loss is −log(probability of the
correct token). Course: Chapter 3 (logits, softmax, NLL).

**Learner's follow-up:** why does `logits[:-1]` drop the last prediction and `labels[1:]` drop the
first label?

**Explanation given (4-token example `A B C D`):** the logit at position t predicts the token at
t + 1, so the pairs are logit 0 → B, logit 1 → C, logit 2 → D.
- **Logit 3** predicts "what comes after D", which isn't in the sequence, so there's no target.
  Drop it: `logits[:-1]` keeps logits 0–2.
- **Label 0 (A)** is the first token, and no position before it exists to predict it. Drop it:
  `labels[1:]` keeps labels 1–3.
- Both now have length T − 1 and line up one-to-one: `logits[:-1][k]` is graded against
  `labels[1:][k]`.

**Learner's line for Exercise 4:** `F.cross_entropy(logits[:-1], labels[1:], ignore_index=-100)`,
correct, written from Rule 2.
