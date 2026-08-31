# Solutions — Chapter 2: Text, Tokens, and Embeddings

These are model answers rather than scripts to memorize. A strong answer names
the mechanism, the relevant evidence, and the boundary of the claim.

## Exercise 1 — Interface corruption

The tensor can be structurally valid because both tokenizers emit integers in the
expected shape and numeric range. The model does not inspect the original text;
it uses each integer as an address into its embedding table. If tokenizer A's ID
42 means `cat` while the model's tokenizer assigned ID 42 to another piece, the
model retrieves the latter row without raising an error.

This is semantic interface corruption:

```text
intended text → IDs under tokenizer A → rows defined under tokenizer B
```

Record at least:

- tokenizer repository and immutable revision;
- tokenizer files or their hashes;
- normalization and pre-tokenization configuration;
- added-special-token behavior and chat template where relevant;
- model repository and immutable revision;
- confirmation that the paired tokenizer and embedding interface were loaded
  together.

Matching shapes and vocabulary bounds are necessary but not sufficient checks.

## Exercise 2 — Coverage versus understanding

UTF-8 encodes `数` as bytes `E6 95 B0`. A byte-level tokenizer has a base symbol
for every byte value, so the conservative fallback is:

```text
数 → [E6] [95] [B0] → three byte-token IDs
```

The decoder can reverse the ID-to-byte mapping and reconstruct the original
character. That exact round trip establishes coverage and reversibility for the
input. It does not show efficiency: a tokenizer with learned Chinese merges may
represent the same character or a larger phrase with fewer tokens.

It also does not establish understanding. Language capability depends on how the
model parameters were trained on those IDs in meaningful contexts. A model can
receive a perfectly valid byte sequence that was rare or absent during model
training.

The answer must be qualified as byte-level BPE or another tokenizer with complete
byte fallback; not every tokenizer guarantees this behavior.

## Exercise 3 — A failed prediction

The original prediction used genuine linguistic facts: Swedish forms compounds,
and Chinese has no spaces between written words. It became unreliable when those
facts were treated as tokenizer rules. BPE compression depends on the patterns
that won merge capacity under a particular corpus, weighting, normalization,
pre-tokenization scheme, and vocabulary budget.

The pinned Qwen3 example contradicted the prediction because it encoded several
Chinese multi-character spans, including `下一个`, while splitting the single
Swedish word `språkmodellen` into five tokens. The observation is:

```text
Chinese 9 < English 11 < Swedish 20
```

The bounded interpretation is that this tokenizer compressed these fixed strings
in that order.

A language-wide comparison would need:

- a frozen, representative sample across domains and sentence lengths;
- comparable or carefully matched content across languages;
- documented normalization and special-token policy;
- code-point, UTF-8-byte, word or grapheme, and token denominators as appropriate;
- distributional results rather than only means, such as quantiles and per-domain
  variation;
- ideally multiple tokenizers and uncertainty from sampling.

Token count still would not establish comprehension or generation quality.

## Exercise 4 — Shape and repetition

Lookup preserves the batch and sequence axes and adds the embedding width:

```text
input_ids.shape = [3, 7]
E.shape         = [151936, 1024]
output.shape    = [3, 7, 1024]
```

If ID $i$ appears at four positions with position-level lookup gradients
$g_1,g_2,g_3,g_4 \in \mathbb{R}^{1024}$, its shared row receives:

\[
\left.\frac{\partial L}{\partial E[i]}\right|_{lookup}
= g_1+g_2+g_3+g_4.
\]

The contributions need not be equal. Lookup returned the same initial row, but
position, causal context, attention, and later nonlinear computation can send a
different gradient back from each occurrence. The equation expresses parameter
reuse, not equal contextual meaning.

If the model ties input and output weights, this lookup sum is only one part of
the total row gradient; output-classifier contributions must also be added.

## Exercise 5 — Where meaning emerges

At the `bank` position in a causal decoder, the available prefix is approximately
`The bank`. The future tokens `approved the loan` cannot influence that position's
hidden state because the causal mask blocks future-to-past information. The model
may preserve an ambiguous representation rather than choosing a fully financial
sense there.

At `loan`, the available prefix contains `The bank approved the loan`. Its hidden
state can attend to and combine all relevant earlier positions, so financial-bank
information can be represented at this later position and in later predictions.

The decoder does not need to rewrite the earlier `bank` state. Sequence meaning
can live in evolving later hidden states. A bidirectional encoder would have a
different information boundary and could use words on both sides of `bank`.

## Exercise 6 — Same loss, different future

Equal tensor values do not imply equal parameter identity.

In the untied model:

```text
input table E_in ── lookup path
output table W_out ── classifier path
```

Even if `E_in` and `W_out` initially contain equal numbers, they are independent
parameters. The input table receives direct lookup gradients only from selected
input rows; the output matrix receives dense classifier gradients.

In the tied model:

```text
shared E ── lookup path
        └── classifier path
```

The shared table receives the sum of both paths. It can therefore have nonzero
gradients on rows that were absent from the input. The lab observed equal forward
loss `0.699771`, but nonzero input-embedding rows `[1,3]` when untied and all six
rows when tied.

After the optimizer applies those different gradients, the matrices no longer
contain the same values. Subsequent logits, losses, and training trajectories can
therefore diverge even though the pre-update forward pass was identical.

## Exercise 7 — Tokenizer-model boundary

The verified observations are:

- tokenizer IDs end at 151,668;
- the model has rows and logits through 151,935;
- input and output matrices are the same runtime parameter;
- the last 267 IDs have no tokenizer piece in this snapshot.

Ordinary tokenizer output cannot select those final rows, so they receive no
ordinary direct lookup-path gradient. The model still emits a logit for every
row. Under dense cross-entropy, each candidate generally receives an output-side
contribution:

\[
\left.\frac{\partial L}{\partial E[i]}\right|_{output}
= \left(p_i-\mathbf{1}[i=y]\right)h.
\]

Because an unassigned ID is not a valid target under the paired tokenizer, its
indicator term is zero in ordinary data, but its probability term can still
produce a gradient.

The model vocabulary's divisibility by 128 is observed. “The rows exist for
hardware alignment” is a plausible hypothesis, not a conclusion supported by
the configuration or checkpoint alone. Establishing that rationale would require
design documentation or a controlled implementation/performance comparison.

## Exercise 8 — Visibility versus supervision

A response hidden state can attend to prompt keys and values. Its logits and loss
therefore depend on prompt representations. Backpropagation follows that
dependency in reverse, reaching attention parameters, prompt hidden states, and
prompt-token embedding rows.

The prompt has no **local** loss term, but it is still an ancestor of the response
loss in the computation graph.

The mechanisms differ:

- **Loss masking:** removes selected prediction errors from the scalar objective.
- **Detaching:** cuts a tensor's earlier computation from gradient propagation.
- **Freezing:** prevents optimizer updates to selected parameters, even if their
  values participate in forward computation and gradients may be computed
  elsewhere.

The Day 2 lab used loss mask `[0,0,1]`; both prompt rows still received nonzero
gradients because the supervised response attended to them.

## Exercise 9 — PAD versus EOS

If PAD remains an ordinary label, the model is rewarded for predicting artificial
batch formatting:

\[
L_{pad}=-\log P(\text{PAD}\mid\text{prefix}).
\]

Frequent padded positions can waste optimization capacity and distort the learned
termination distribution.

If every EOS label is masked, the model loses direct supervision for when a
sequence ends. It may learn content tokens while becoming worse at emitting the
semantic stopping event used by generation.

Before trusting a trainer, inspect:

- whether padding is left or right and which ID represents it;
- the attention mask actually passed to the model;
- the shifted input/label alignment used for causal prediction;
- the exact ignore index and which labels are replaced by it;
- whether prompt, response, EOS, and packed-document boundaries are treated as
  intended;
- the denominator used to reduce token-level losses.

An attention mask should not be assumed to modify the loss automatically.

## Exercise 10 — Design an honest tokenizer study

One defensible design would freeze a corpus of several thousand aligned or
carefully matched passages per language across conversation, news, technical
writing, code-adjacent text, and informal text. Deduplicate it, record immutable
revisions, and preserve the exact normalization pipeline.

For each pinned tokenizer:

1. encode every passage without chat-template or special-token overhead, unless
   overhead is itself a declared study question;
2. require exact decode round trips where the tokenizer promises reversibility;
3. record tokens per UTF-8 byte, Unicode code point, grapheme cluster, written
   word where meaningful, and matched passage;
4. report medians, means, quantiles, long-tail examples, and per-domain results;
5. use paired passage-level comparisons where translations are aligned;
6. quantify sampling uncertainty, for example with bootstrap intervals over
   documents;
7. inspect representative segmentations instead of relying only on aggregate
   counts.

Controls include tokenizer revision, special-token behavior, normalization,
pre-tokenization, and the exact same frozen inputs for each tokenizer.

The study could support bounded claims about compression and context-window
occupancy on the sampled distribution. It still would not establish model
understanding, language quality, fairness, capability, or the causal effect of
the unknown tokenizer-training mixture.
