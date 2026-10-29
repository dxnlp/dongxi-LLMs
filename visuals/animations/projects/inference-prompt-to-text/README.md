# Inference: from a prompt to text

`CAND-ANIM-040`, seven GIFs that follow one chat request through inference. The learner requested
them on 2026-10-29, after the discussion of vocabulary, tokens, logits and weights: "create
animations to clearly illustrate the steps … visually strong, and less text load, we add
explanation as separate text". The figures carry only one-word labels. The explanation for each
one is below.

```sh
uv run --project visuals/animations python visuals/animations/projects/inference-prompt-to-text/render.py --preview
uv run --project visuals/animations python visuals/animations/projects/inference-prompt-to-text/render.py
```

Each `rendered/<stem>/` folder holds a 1280×720 GIF at 20 fps, a 1080p MP4 (ignored by Git), a
poster and a contact sheet. The review page is `gifs.html`; serve this folder and open it.

## Evidence

- **Real:** the token strings and IDs come from the Qwen3 tokenizer (`Qwen/Qwen3-0.6B`, local
  cache):
  - 151,669 tokenizer entries;
  - 151,936 embedding rows;
  - 28 layers;
  - tied embeddings.
- **Chosen, not generated:** the reply "Partly. Some weights are open."
- **Simplified template:** the template is the ChatML core that Qwen uses. Qwen3 with thinking
  disabled also inserts an empty `<think></think>` block, which the figures leave out.
- **Illustrative:** the logits, probabilities and vector shades. Softmax, temperature and
  top-p are computed exactly on those numbers.
- **Schematic:** three layers stand for 28, and frames show the order of operations, not
  measured time.

## Cast

| colour | meaning |
|---|---|
| amber | text the chat template adds: special tokens, role names, newlines |
| slate | the user's own text |
| blue | tokens the model writes |
| violet | weights: embedding table, layers, LM head |
| green | the KV cache |

## Captions (the explanation text)

**01 · From prompt to text.** The model never outputs text directly.
1. The prompt is tokenized into a sequence of IDs.
2. One forward pass of the model reads the whole sequence and gives logits for the next token
   only.
3. One token is picked and appended to the sequence, and its text joins the reply.
4. The loop runs again.

The reply "Partly. Some weights are open." takes nine passes, one per token. The ninth produces
`<|im_end|>`, and generation stops.

**02 · The chat template.** The model does not see the bare question. The serving code wraps it in
the template the model was trained on: `<|im_start|>user`, the question, `<|im_end|>`, then
`<|im_start|>assistant` and a newline. The model writes its answer from the empty line that follows.
A wrong template is a common reason for a chat model to behave badly.

**03 · Tokenize.** The tokenizer cuts the templated text into entries of a fixed vocabulary of
151,669 strings and replaces each piece with its row number. A leading space belongs to the
following token: ` Open` (5264) and ` open` (1787) are different entries from `Open` and `open`.
Special tokens such as `<|im_start|>` are single entries (151644). The model receives only the 13
numbers.

**04 · Prefill.** Each ID selects one row of the embedding table, giving one vector per position.
All 13 positions pass through the layers together, which is why prompt processing is fast.
Every layer stores each position's keys and values: this is the KV cache. Only the last
position's output goes on to the LM head, which turns it into 151,936 logits, one score per
vocabulary row.

**05 · Pick the next token.** The steps from logits to one token:
- **Softmax** turns logits into probabilities.
- **Temperature** divides the logits before softmax. T below 1 sharpens the distribution and T
  above 1 flattens it.
- **Top-p** (0.9 here) keeps the smallest set of tokens whose probabilities reach 90%, then
  renormalizes.
- **Greedy decoding** always takes the largest, so it writes "Part" every time.
- **Sampling** draws in proportion to probability, so the same logits give "Part" most often,
  "Open" sometimes and "No" occasionally.

None of these steps uses weights. The numbers are illustrative.

**06 · Decode.** After prefill, each pass feeds only the newest token:
1. Its vector rises through the layers.
2. In each layer it reads the keys and values already cached (the green lines) and adds its own.
3. The LM head gives the next logits, a token is picked, and it drops back in as the next input.

The cache grows by one position per pass. Eight passes after prefill produce the rest of the
reply and the stop token. A long answer is slow because every pass handles one token and still
reads all the weights.

**07 · Detokenize.** The output IDs are looked up in the same vocabulary, in reverse:
- each ID becomes its string;
- the stop token is dropped;
- each leading-space marker becomes a real space;
- the pieces join into the reply.

The vocabulary is used twice in one request: once to encode the prompt and once to decode the
answer.

## Status

- 2026-10-29: storyboard by the agent from the learner's request (the learner asked for the
  animations, which counts as approval to produce). Rendered on the Mac Studio.
- 2026-10-29: the learner called the set "generally very good" and asked the agent to review every
  GIF itself. The agent checked frames at 0.9-second intervals and fixed:
  - 01: the picked token now holds at the pick point for the first three passes; a curved "append"
    arrow closes the loop back to the sequence; the reply grows by fading in new glyphs instead of
    re-morphing the whole line.
  - 03: wider gaps between chips, so neighbouring IDs no longer touch.
  - 05: the caption cross-fades from logits to probabilities instead of morphing letters; the
    strip's token names stay above the moving copies.
  - 06: the attention reads were straight lines along the slab and merged into one line. They are
    now arcs above each layer, one per cached position. A light band marks the active position, the
    new token's vector is larger, and the reply grows cleanly.
  - 07: the reply pieces turn blue before leaving their chips. Before, they were white on white
    and vanished on the way to the bubble.
- 2026-10-29: the seven GIFs illustrate the X article `X-TOKEN-001`, whose Chinese version the
  learner published on X.
