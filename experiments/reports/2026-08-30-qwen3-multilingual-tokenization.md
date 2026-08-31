# Experiment Report: Qwen3 Multilingual Tokenization Exploration

## Identity

- Experiment ID: `2026-08-30-qwen3-multilingual-tokenization`
- Specification:
  `experiments/specs/2026-08-30-qwen3-multilingual-tokenization.yaml`
- Execution date: 2026-08-31
- Course code commit: `a67198980239650cf4dbcc1251f4427042511456`
- Tokenizer: `Qwen/Qwen3-0.6B`
- Tokenizer revision: `c1899de289a04d12100db370d81485cdf75e47ca`
- Loaded class: `Qwen2Tokenizer` (the Transformers implementation used by this
  Qwen3 tokenizer snapshot)
- Base vocabulary size: 151,643
- Total vocabulary size including added tokens: 151,669
- Machine manifest: `manifests/2026-08-29-dgx-spark-native.json`
- Environment lock: `/home/dongxi/dgx-spark-dongxi/uv.lock`, SHA-256
  `ca1e48e9cc3a73f4a37f425181e94248a4abfb2bd0141125337bd78749cb6efa`
- Python: 3.12.14
- Transformers: 5.16.1
- Tokenizers: 0.23.1
- Exact command:
  `PYTHONPATH=src /home/dongxi/dgx-spark-dongxi/.venv/bin/python -m dongxi_llms.tokenization_lab --local-files-only`
- Exit code: 0

## Protocol

- Intended protocol: load the exact pinned tokenizer snapshot from the local Hub
  cache; encode the three precommitted strings with tokenizer-default
  normalization and no added special tokens; record code-point, UTF-8-byte, and
  token counts; preserve IDs, vocabulary pieces, offsets, and readable spans;
  then require an exact encode/decode round trip.
- Deviations from specification: none.
- The raw byte-level vocabulary pieces are retained below even though their
  reversible byte-to-Unicode display alphabet is not intended as reader-facing
  text. Human-readable spans derived from offset mappings are shown separately.

## Measurements

| Language | Code points | UTF-8 bytes | Tokens | Code points/token | Bytes/token | Exact round trip |
|---|---:|---:|---:|---:|---:|---|
| Chinese | 16 | 48 | 9 | 1.777778 | 5.333333 | yes |
| English | 58 | 58 | 11 | 5.272727 | 5.272727 | yes |
| Swedish | 58 | 63 | 20 | 2.900000 | 3.150000 | yes |

Measured ranking from fewest to most tokens:

```text
Chinese (9) < English (11) < Swedish (20)
```

### Chinese

```text
Text:   小型语言模型学习预测下一个词元。
Spans:  [小型] [语言] [模型] [学习] [预测] [下一个] [词] [元] [。]
IDs:    [105911, 102064, 104949, 100134, 104538, 108725, 99689, 23305, 1773]
Pieces: [å°ıåŀĭ] [è¯Ńè¨Ģ] [æ¨¡åŀĭ] [åŃ¦ä¹ł] [é¢Ħæµĭ]
        [ä¸ĭä¸Ģä¸ª] [è¯į] [åħĥ] [ãĢĤ]
```

### English

```text
Text:   The small language model learns to predict the next token.
Spans:  [The] [ small] [ language] [ model] [ learns] [ to] [ predict]
        [ the] [ next] [ token] [.]
IDs:    [785, 2613, 4128, 1614, 46210, 311, 7023, 279, 1790, 3950, 13]
Pieces: [The] [Ġsmall] [Ġlanguage] [Ġmodel] [Ġlearns] [Ġto] [Ġpredict]
        [Ġthe] [Ġnext] [Ġtoken] [.]
```

### Swedish

```text
Text:   Den lilla språkmodellen lär sig att förutsäga nästa token.
Spans:  [Den] [ l] [illa] [ spr] [å] [k] [mod] [ellen] [ l] [är]
        [ sig] [ att] [ för] [uts] [ä] [ga] [ nä] [sta] [ token] [.]
IDs:    [23619, 326, 6241, 8151, 3785, 74, 2593, 15671, 326, 13977,
         8366, 1619, 16641, 6128, 2305, 6743, 43117, 20491, 3950, 13]
Pieces: [Den] [Ġl] [illa] [Ġspr] [Ã¥] [k] [mod] [ellen] [Ġl] [Ã¤r]
        [Ġsig] [Ġatt] [ĠfÃ¶r] [uts] [Ã¤] [ga] [ĠnÃ¤] [sta] [Ġtoken] [.]
```

## Observations

- The process loaded the exact pinned local snapshot and exited successfully.
- Every encoded sequence decoded exactly to its original string.
- The learner's predicted order—Swedish, Chinese, English—was reversed by this
  worked example.
- Qwen3 represented most Chinese spans with multi-character tokens, including
  the three-character span `下一个`.
- `språkmodellen` occupied five tokens: ` spr`, `å`, `k`, `mod`, and `ellen`.
- Swedish characters with diacritics contributed to several byte-level pieces,
  but the experiment does not isolate diacritics as the sole causal factor.
- The tokenizer reports more total entries than base-vocabulary entries because
  added tokens are counted by `len(tokenizer)`.

## Interpretations

- Orthographic word count and compounding are poor substitutes for measuring a
  particular tokenizer. A single Swedish compound can still split many times if
  its internal patterns have weaker merge coverage.
- The strong Chinese compression in this example is consistent with the
  tokenizer having learned useful multi-character merges from its training data.
  The run does not expose the original tokenizer-training corpus or prove which
  corpus component caused each merge.
- Byte-level internal piece strings should not be mistaken for corrupted input:
  offsets and the exact round trip show the reversible mapping back to Chinese
  and Swedish text.
- Token efficiency is model-interface-specific. It is not a fixed property of a
  language or writing system.

## Claims not established

- One sentence per language does not establish average language-level token
  efficiency.
- The translations are worked examples, not proof of perfect semantic or
  stylistic equivalence.
- Token count alone does not establish comprehension or downstream model quality.
- This run does not show tokenizer optimality or explain the tokenizer-training
  data mixture.
- The earlier gpt-oss observations were not part of this pinned experiment and
  remain separate observational context.

## Failures or surprises

- No protocol or round-trip failure occurred.
- The predicted ranking failed, which is a useful negative result rather than an
  execution failure.
- The raw byte-level vocabulary display is difficult to read, motivating the
  additional preservation of offset-derived source spans.

## Decision and next experiment

- Decision: pass. The Day 2 Qwen3 worked example is now reproducible and the
  original prediction has been compared honestly with measurements.
- Exact next action: inspect the pinned Qwen3 configuration and weight metadata
  to verify embedding shape, padded vocabulary behavior, and input/output weight
  tying before integrating Chapter 2.
