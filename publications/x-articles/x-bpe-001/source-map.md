# X-BPE-001 source and claim map

## Evidence map

| Article claim | Evidence | Strength and boundary |
|---|---|---|
| `下一个` is one token while `språkmodellen` is five under the tested tokenizer | `experiments/reports/2026-08-30-qwen3-multilingual-tokenization.md` | Direct observation for three fixed strings under one pinned Qwen3 tokenizer revision; not language-wide evidence |
| The measured token-count order is Chinese 9, English 11, Swedish 20 | Same multilingual report and raw output | Direct observation; translations are worked examples, not a representative corpus |
| Family emoji has 1 grapheme, 7 code points, and 25 UTF-8 bytes | `experiments/reports/2026-08-31-tokenizer-mechanics.md` | Mechanically measured with extended grapheme segmentation |
| NFC and NFD `café` map to IDs `[924, 58858]`, but the NFD source does not round-trip exactly | Same tokenizer-mechanics report | Direct negative result under the pinned tokenizer and software versions |
| `token` and leading-space ` token` each use one token but different IDs | Same tokenizer-mechanics report | Direct observation; one example does not characterize every pre-tokenizer |
| Raw `Hello` is one token; the one-message chat rendering is nine | Same tokenizer-mechanics report | Direct observation for one fixed Qwen3 chat template |
| Tiny BPE rounds select `h+u`, `hu+g`, and `hug+s` at counts 10, 10, and 3 | Same report; `src/dongxi_llms/tiny_bpe.py` | Executable teaching mechanism; not Qwen's historical training trace |
| `数` is UTF-8 bytes `E6 95 B0`; bytes provide coverage while merges add compression | `learning_artifacts/day-02-text-tokens-and-embeddings/bpe-training-and-byte-coverage.md` | UTF-8 mapping is deterministic; Chinese merge ladder is explicitly illustrative |
| BPE, WordPiece, and Unigram are different tokenization models | `book/chapters/02-text-tokens-and-embeddings.md`, section 2.3; Hugging Face Tokenizers pipeline documentation | Bounded mechanism comparison only; no quality ranking is claimed |

## Primary local sources

- `book/chapters/02-text-tokens-and-embeddings.md`
- `experiments/reports/2026-08-30-qwen3-multilingual-tokenization.md`
- `experiments/reports/2026-08-31-tokenizer-mechanics.md`
- `experiments/results/2026-08-31-tokenizer-mechanics.json`
- `src/dongxi_llms/tiny_bpe.py`
- `learning_artifacts/day-02-text-tokens-and-embeddings/bpe-training-and-byte-coverage.md`
- `learning_artifacts/day-02-text-tokens-and-embeddings/unicode-normalization-pretokenization-and-chat-packaging.md`
- `visuals/animations/rendered/bpe-byte-merges-manim-still.png`

## Primary terminology sources

- [Hugging Face Tokenizers: The tokenization pipeline](https://huggingface.co/docs/tokenizers/main/pipeline)
- [Hugging Face Transformers: Writing a chat template](https://huggingface.co/docs/transformers/main/chat_templating_writing)
- [Unicode Standard Annex #29: Unicode Text Segmentation](https://www.unicode.org/reports/tr29/)

## Editorial limits

- The article explains byte-level BPE as the main mechanism. It does not imply
  that all tokenizers use BPE or byte fallback.
- It separates observed Qwen3 behavior from transparent teaching examples.
- It does not infer tokenizer-training corpus composition from segmentation.
- It treats coverage, compression, context occupancy, and understanding as
  separate claims.
- It does not claim that lower token count means higher model quality or fairness.
- Deep chat-template masking and assistant-only loss policy are deferred to a
  later article.
