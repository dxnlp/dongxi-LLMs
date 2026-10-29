# The Birth of a Token: What Happens Between Your Question and the Answer

Ask a chat model *Is OpenAI open?* and the answer appears piece by piece: *Partly. Some weights are open.*

Behind those five words is a procedure that runs once for every piece. Take a small open model, Qwen3-0.6B. For it, this reply is 9 pieces: 8 for the text and 1 that ends the turn. To write it, the model made 9 complete passes through its 596 million weights. Each pass produced 151,936 scores, one for every entry in its vocabulary, and kept one. That is 1,367,424 scores for nine pieces.

The pieces are called **tokens**. This article follows one request from the moment you press Enter to the moment the answer appears. It uses the real tokens and token IDs of Qwen3. Today's chat models all generate text with this same loop. They differ in size, vocabulary and the template that wraps your question.

[IMAGE 01]

> A language model does not write text directly. Each pass computes scores for the next token, and a short loop around the model turns those scores into text.

## Six steps

1. **Chat template:** your question is wrapped in a fixed format the model was trained on.
2. **Tokenize:** the wrapped text becomes a list of numbers.
3. **Prefill:** the model reads all of those numbers in one pass and scores the first token of the answer.
4. **Pick:** the scores become one token.
5. **Decode:** that token is fed back in, and the model scores the next one. Steps 4 and 5 repeat until a stop token appears.
6. **Detokenize:** the numbers become text again.

Steps 1 to 3 happen once per request. Steps 4 to 6 happen once per token of the answer.

## Step 1: your question gets a wrapper

The model never sees your bare question. The app wraps it in a **chat template**: a fixed text format the model saw during its training as an assistant. For Qwen, the wrapped question is:

```
<|im_start|>user
Is OpenAI open?<|im_end|>
<|im_start|>assistant
```

`<|im_start|>` and `<|im_end|>` are **special tokens**: single vocabulary entries that mark where a turn begins and ends. The last line opens the assistant's turn and stops. The answer continues from the empty line that follows. (Qwen3 also inserts an empty thinking block here when thinking is turned off; the figures leave it out.)

[IMAGE 02]

Each model family has its own template. A model given the wrong one sees text in a format it was never trained on, and its answers degrade. That is a common reason a self-hosted model behaves worse than its official demo.

## Step 2: text becomes numbers

A model computes with numbers, so the text must become numbers first. The **tokenizer** does this with a **vocabulary**: a fixed list of text pieces, each with a row number called its **token ID**. Qwen3's vocabulary has 151,669 entries.

The tokenizer cuts the wrapped question into pieces that are entries in that list, then replaces each piece with its ID. Our question becomes 13 tokens:

```
151644 872 198 3872 5264 15469 1787 30 151645 198 151644 77091 198
```

[IMAGE 03]

A few things stand out:

- A token can be a whole word, part of a word, a punctuation mark or a special marker. *OpenAI* is two tokens, ` Open` and `AI`.
- The space before a word belongs to that word's token. ` open` (1787) and `open` are different entries.
- Common Chinese words are often one entry: 开放 is token 100713.
- The newline is a token too (198), and the template uses it three times.

The vocabulary is fixed before training starts. It does not change while the model learns, and it does not change while the model answers.

## Step 3: the model reads the whole prompt at once

The **weights** are the model: 596 million numbers set during training and fixed while the model answers. The 13 IDs pass through them in three stages.

1. **Embedding.** Each ID selects one row of a table with 151,936 rows of 1,024 numbers. The tokenizer uses 151,669 of those rows; the rest are unused padding. This table alone holds 155.6 million weights, 26% of the model.
2. **Layers.** The 13 vectors pass through 28 layers. In each layer, every position looks at the positions before it (this is called attention) and updates its own vector. All 13 positions go through together, in one pass. This phase is called **prefill**.
3. **LM head.** Only the vector at the last position goes on. The LM head compares it with every vocabulary entry and gives one score per entry: 151,936 scores, called **logits**. In Qwen3-0.6B, the LM head reuses the embedding table.

Along the way, every layer stores two vectors for each position, its **key** and its **value**. Together they form the **KV cache**. Later tokens need them, and storing them means the prompt is never processed twice.

[IMAGE 04]

> One pass gives one score for every token in the vocabulary. The highest score is a candidate, nothing more.

## Step 4: 151,936 scores become one token

Logits are raw scores. Turning them into one token takes a few lines of arithmetic, with no weights involved:

- **Softmax** turns the logits into probabilities: all positive, summing to 1.
- **Temperature** divides the logits before softmax. Below 1 it sharpens the distribution toward the top token; above 1 it flattens it.
- **Top-p** keeps the smallest set of tokens whose probabilities add up to p (0.9 here) and drops the rest.
- **Greedy decoding** then takes the most probable token. **Sampling** draws one at random, in proportion to its probability.

In our illustration, three tokens survive top-p: `Part` at 57%, ` Open` at 29% and `No` at 14%. Greedy decoding writes `Part` every time. Sampling writes `Part` most of the time, ` Open` sometimes and `No` occasionally.

[IMAGE 05]

These settings belong to the app or the API call, not the model. The same weights give a fixed answer under greedy decoding and varied answers under sampling.

## Step 5: the loop

The picked token is appended to the sequence, and the model runs again. This phase is called **decode**.

Decode does not re-read the whole prompt. Only the newest token passes through the 28 layers. In each layer it reads the keys and values already in the KV cache, one per earlier position, and adds its own. The LM head scores the next token, one is picked, and it is fed back in.

The loop stops when the model picks the end-of-turn token `<|im_end|>`, or when the app's length limit is reached. In our example, prefill produced `Part`, and eight decode passes produced `ly`, `.`, ` Some`, ` weights`, ` are`, ` open`, `.` and finally `<|im_end|>`. That makes nine passes for nine tokens.

[IMAGE 06]

> Prefill handles the whole prompt in one pass. Decode handles one token per pass, and every pass reads the model's weights again.

## Step 6: numbers become text again

The output IDs go back through the same vocabulary, in reverse:

- each ID becomes its text piece;
- the stop token is dropped;
- leading spaces stay with their words;
- the pieces join into the reply.

Apps do this as each token arrives, which is why the answer streams in. Some characters, including many Chinese characters, span more than one token. The app holds back the partial bytes until the character is complete.

[IMAGE 07]

## What this explains

This loop accounts for several things people notice when using chat models.

- **Answers stream in.** The model produces one token per pass, and the app shows each one as it arrives.
- **The same question gives different answers.** Chat apps usually sample. A lower temperature, or greedy decoding, makes answers more repeatable.
- **There are two kinds of waiting.** A long prompt delays the first word, because prefill must process all of it. A long answer takes longer to finish, because decode needs one pass per token. Speed is reported the same way: time to first token, and tokens per second.
- **Counting letters is hard.** In *How many r are in strawberry?*, Qwen3 sees ` strawberry` as a single token, ID 72600. The letters are not in its input. The model can only answer from what it learned in training about the spelling of token 72600. Without the leading space, `strawberry` is three tokens: `str`, `aw` and `berry`.
- **Limits and prices are counted in tokens.** Context windows, API prices and speed limits all count tokens, not words. The ratio of tokens to words depends on the language and the vocabulary.
- **The model does not learn while you chat.** The weights stay fixed. Your conversation exists only as the token sequence the app sends with each message, and the KV cache built from it.

> Every token in every answer comes from the same loop: score the whole vocabulary, pick one, append it, repeat.

## Summary

- The app wraps your question in a chat template, and the tokenizer turns it into token IDs from a fixed vocabulary.
- Prefill runs the whole prompt through the model in one pass. It fills the KV cache and produces logits: one score for every vocabulary entry.
- Softmax, temperature, top-p and greedy decoding or sampling turn the logits into one token. No weights are involved in that step.
- Decode feeds each new token back in, one pass per token, until a stop token appears.
- Detokenizing turns the IDs back into text, as the tokens arrive.
