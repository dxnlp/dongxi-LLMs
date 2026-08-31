# Experiment Report: Transparent Tokenizer Mechanics

## Identity

- Experiment ID: `2026-08-31-tokenizer-mechanics`
- Specification:
  `experiments/specs/2026-08-31-tokenizer-mechanics.yaml`
- Execution date: 2026-08-31
- Base code commit: `ff417960411ee495aca0bb82423bcd7f5baebe5c`
- Branch: `codex/visuals/manim-bpe`
- Code state: tokenizer-mechanics enrichment uncommitted at report time
- Tiny-BPE source SHA-256:
  `46ea87e74e4f7147736a5d2613c23dde33fc9a852da4b989404bb44740e0cefa`
- Tokenizer-mechanics source SHA-256:
  `f902150ae25fd8f99ff85a7298781988ac170f85c11150c0172ddbd2ca46c6f2`
- Specification SHA-256:
  `8c7fcf27d37e7a770b1a44f7f937ebe72ea1e4b3b967efff46494ec2d6548ab0`
- Machine: local Mac Studio, Apple silicon (`arm64`)
- Platform: macOS 15.7.9
- Python: 3.10.8
- Transformers: 5.16.1
- Tokenizers: 0.23.1
- Regex: 2026.1.15
- PyYAML: 6.0.1
- Tokenizer: `Qwen/Qwen3-0.6B`
- Tokenizer revision: `c1899de289a04d12100db370d81485cdf75e47ca`
- Raw output: `experiments/results/2026-08-31-tokenizer-mechanics.json`
- Raw-output SHA-256:
  `38922de507dc123367c5156d265370419e04298d57e4d5afdea391eefbcfb02d`
- Exact validation command:

```bash
PYTHONPATH=src uv run \
  --with 'transformers==5.16.1' \
  --with 'tokenizers==0.23.1' \
  --with 'regex==2026.1.15' --with 'pyyaml==6.0.1' \
  python -m dongxi_llms.tokenizer_mechanics_lab \
  --local-files-only \
  --output experiments/results/2026-08-31-tokenizer-mechanics.json
```

- Exit code: 0
- GPU or model weights used: no

## Protocol

The run combined four bounded checks:

1. train three deterministic BPE merges on a tiny weighted, pre-tokenized corpus;
2. measure code points, UTF-8 bytes, and extended grapheme clusters mechanically;
3. compare NFC/NFD and leading-space inputs under the exact pinned tokenizer;
4. compare raw `Hello` with the same content rendered through the tokenizer's
   one-message chat template and generation prompt.

The final recorded run used only the local tokenizer cache populated by an
earlier tokenizer-only download. No model checkpoint was loaded.

## Measurements

### Tiny BPE trace

The corpus was:

```text
hug × 5
hugs × 3
hugging × 2
```

The selection rule was highest weighted adjacent-pair count, with a documented
lexicographic tie break.

| Round | Selected pair | Count | Representative rewrite |
|---:|---|---:|---|
| 1 | `h + u → hu` | 10 | `[h,u,g,s] → [hu,g,s]` |
| 2 | `hu + g → hug` | 10 | `[hu,g,s] → [hug,s]` |
| 3 | `hug + s → hugs` | 3 | `[hug,s] → [hugs]` |

At round 1, `h+u` and `u+g` both had count 10. The tie break selected `h+u`;
frequency alone did not define a unique first merge. Frozen replay then encoded:

```text
hug     → [hug]
hugs    → [hugs]
hugging → [hug] [g] [i] [n] [g]
```

The trainer uses characters for readability. It demonstrates generic BPE
mechanics and does not claim to reproduce Qwen's byte-level training history.

### Unicode units

| Input | Code points | UTF-8 bytes | Grapheme clusters |
|---|---:|---:|---:|
| NFC `café` | 4 | 5 | 4 |
| NFD `café` | 5 | 6 | 4 |
| `👨‍👩‍👧‍👦` | 7 | 25 | 1 |

The family emoji consists of four person code points joined by three zero-width
joiners. Python string length therefore reports seven code points even though
Unicode grapheme segmentation returns one reader-perceived cluster.

### Normalization under pinned Qwen3

| Source form | Token IDs | Reader spans | Decoded text | Exact source round trip |
|---|---|---|---|---|
| NFC `café` | `[924, 58858]` | `[ca] [fé]` | NFC `café` | yes |
| NFD `café` | `[924, 58858]` | `[ca] [fe]` | NFC `café` | no |

The NFC and NFD strings are canonically equivalent but not identical sequences
of code points. The tokenizer normalized both to the same IDs and decoded the NFD
input as NFC. The NFD offset mapping `[(0,2),(2,4)]` also did not cover the final
combining mark in the original five-code-point source. Offset-derived source
spans therefore need explicit scrutiny when normalization changes the input.

This falsified the prediction and precommitted success criterion that both forms
would decode exactly to their original source strings. The failure is retained;
canonical equivalence is not silently relabeled as exact equality.

### Leading-space behavior

| Input | Token ID | Internal piece | Reader span |
|---|---:|---|---|
| `token` | 5839 | `token` | `token` |
| ` token` | 3950 | `Ġtoken` | ` token` |

Both inputs used one token, but the leading space changed its identity. The
reader-facing span is more useful than interpreting the tokenizer's reversible
internal byte-display alphabet literally.

### Raw text versus chat-template packaging

Raw `Hello` encoded as one token:

```text
[9707]
```

The pinned chat template rendered:

```text
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
```

It encoded as nine tokens:

```text
[151644, 872, 198, 9707, 151645, 198, 151644, 77091, 198]
```

The added positions came from template text and control tokens, not from BPE
learning new vocabulary at runtime. In this snapshot, `<|im_end|>` is EOS ID
151645 and `<|endoftext|>` is PAD ID 151643; no BOS or UNK token ID is assigned.

## Observations

- All local unit tests and the final tokenizer-only command completed.
- The three BPE merge counts matched the prediction.
- NFC and NFD forms had equal grapheme counts but different code-point and byte
  counts.
- The pinned tokenizer mapped the two normalization forms to the same IDs.
- Exact source round-trip equality failed for NFD because decode returned NFC.
- A leading space changed token identity without changing token count.
- Chat-template packaging expanded one content token to nine total tokens.

## Interpretations

- “Character count” is ambiguous unless the unit is named.
- Normalization is part of tokenizer behavior and can intentionally change the
  exact source code-point sequence.
- Token offsets are not automatically lossless source spans after normalization.
- Spaces can participate in token identity rather than merely separate tokens.
- Chat token counts include formatting and control policy as well as user text.
- A deterministic BPE trace requires a tie policy in addition to pair counts.

## Claims not established

- The tiny BPE trace does not reconstruct Qwen's historical merge sequence.
- The run does not compare the quality of BPE, WordPiece, and Unigram.
- One leading-space example does not describe every pre-tokenizer.
- One chat template does not generalize to other models or revisions.
- Token count does not establish understanding, generation quality, or fairness.
- The experiment does not support language-wide efficiency claims.

## Failures or surprises

- The NFD exact-round-trip hypothesis and success criterion failed.
- The normalized offset mapping omitted the combining mark from reader-facing
  source spans, exposing a limitation in naive offset presentation.
- The BPE first-round maximum was tied; deterministic training required the
  declared tie break.

## Decision and next experiment

- Decision: partial pass. Execution and measurements are valid, but the exact
  NFD source-round-trip criterion failed and is retained as a negative result.
- Use the measured normalization behavior, grapheme example, BPE trace,
  leading-space result, and chat-template overhead in Chapter 2 and the planned X
  Article.
- Keep deep chat-template masking mechanics for the later instruction-data
  chapter.
- A broader multilingual corpus is required before making language-level token
  efficiency claims.
