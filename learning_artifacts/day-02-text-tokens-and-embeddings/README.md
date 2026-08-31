# Day 2 — Text, Tokens, and Embeddings

- Date opened: 2026-08-30
- Day status: complete
- Book destination: `book/chapters/02-text-tokens-and-embeddings.md`
- Primary specification: `experiments/specs/2026-08-30-qwen3-multilingual-tokenization.yaml`

## Topic artifacts

1. [`tokenizer-identity-and-multilingual-efficiency.md`](tokenizer-identity-and-multilingual-efficiency.md)
2. [`bpe-training-and-byte-coverage.md`](bpe-training-and-byte-coverage.md)
3. [`embeddings-context-and-gradients.md`](embeddings-context-and-gradients.md)
4. [`padding-masks-and-special-tokens.md`](padding-masks-and-special-tokens.md)

## Current outcome

The tokenizer and BPE mental model has been demonstrated, and the pinned Qwen3
multilingual worked example now preserves exact pieces, IDs, ratios, and round
trips in a reproducible report. Embedding shapes, repeated-row accumulation, tied
and untied gradient paths, and response-only masking have also been demonstrated
and verified in a fixed PyTorch lab. The pinned Qwen3 embedding shape, tokenizer
boundary, and runtime weight tying are verified. These mechanisms are synthesized
into Chapter 2 with ten exercises and worked solutions. Open edges are explicitly
routed to later chapters or optional enrichment rather than required Day 2 work.

## Public-production links

See `LEARNING_MEMORY.md` for task packets `X-BPE-001`, `X-EMB-001`,
`ANIM-BPE-001`, `ANIM-EMB-001`, and `ANIM-CE-001`.
