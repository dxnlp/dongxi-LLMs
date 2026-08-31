# Tokenizer Identity and Multilingual Efficiency

- Day: 02
- Date opened: 2026-08-30
- Status: demonstrated; Qwen3 worked example verified
- Book destination: Chapter 2 sections on tokenizer identity and multilingual efficiency
- Related evidence: `experiments/reports/2026-08-30-qwen3-multilingual-tokenization.md`
- Related production tasks: `X-BPE-001`

## Questions that drove the discussion

- Should Chinese use approximately one token per character?
- Should Swedish compounds make Swedish more token-efficient than English?
- Why do Qwen3, gpt-oss, and a Swedish-focused tokenizer segment the same text
  differently?
- What happens if token IDs from one tokenizer are passed to another model?

## Learner's initial model

Before inspection, the learner predicted the fewest-to-most token order as
Swedish, Chinese, English. The reasoning was that Chinese characters might map
roughly one-to-one to tokens, while Swedish compounds such as `språkmodellen`
could express multiple English words in one written word.

## Refined mental model

A written word, Unicode character, and token are different units. Tokenization is
determined by the tokenizer's corpus mixture and weighting, normalization and
pre-tokenization, subword algorithm and merge priorities, vocabulary budget, and
special-token policy. Orthographic compounding does not guarantee fewer tokens:
an underrepresented compound may split into many subwords.

The tokenizer is part of the model interface contract. A numeric token ID has no
portable meaning outside its tokenizer. Feeding IDs from another tokenizer can
select unrelated embedding rows even when both ID tensors have valid shapes.

## Concrete examples and derivations

For the fixed sentences in the specification, the pinned Qwen3 lab verified and
the earlier interactive gpt-oss inspection observed:

```text
Qwen3 verified: Chinese 9, English 11, Swedish 20
gpt-oss observed: Chinese 11, English 11, Swedish 15
```

The Swedish fragment `språkmodellen` was observed as five Qwen3 pieces but four
gpt-oss pieces. Qwen3 encoded `数据库` as one token in the inspected revision,
while `数据` and `库` also had individual tokens.

Efficiency must state its denominator. Useful worked-example measures include:

```text
Unicode code points per token
UTF-8 bytes per token
```

Neither single-sentence measure establishes general efficiency for a language.

## Demonstrated understanding

- Correctly concluded that tokenizer differences arise from data, vocabulary
  size/budget, and tokenizer rules.
- Correctly explained that the same ID can represent different text under two
  tokenizers and therefore corrupt the model interface.
- Accepted the observed falsification of the original language-efficiency
  ranking rather than revising the prediction afterward.

## Evidence and limitations

The fixed inputs, prediction, tokenizer identity, token pieces, IDs, ratios, and
exact round trips are preserved in the Qwen3 specification and report. All three
Qwen3 strings round-tripped exactly. The measured fewest-to-most ranking was
Chinese, English, Swedish, reversing the initial prediction. This verifies one
worked example, not general efficiency for any language.

The gpt-oss counts remain observational context because they are not yet backed
by a repository experiment specification and report.

GPT-SW3 access was authenticated but the account lacked authorization for the
gated tokenizer repository, so no exact GPT-SW3 segmentation claim was made.

## Open edges

- Compare enough samples before making any language-level efficiency claim.

## Reuse opportunities

- Chapter 2 worked multilingual comparison.
- Exercise asking readers to predict segmentation before execution.
- X article `X-BPE-001`, using multilingual variation as supporting context.
