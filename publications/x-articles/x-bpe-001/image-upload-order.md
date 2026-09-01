# Image Upload Order

1. `assets/01-six-units.png` — An orthographic word, grapheme cluster, code point, UTF-8 byte sequence, subword or byte token, and token ID are different units.
2. `assets/02-tiny-bpe-training.png` — Three measured BPE rounds on a tiny corpus. The first maximum is tied, so a declared tie rule is required.
3. `assets/03-byte-coverage-compression.png` — The byte vocabulary provides coverage; learned BPE merges compress repeated patterns into larger tokens.
4. `assets/04-interface-surprises.png` — Normalization can change the exact source sequence; chat template serialization can add control tokens around unchanged user content.
5. `assets/05-multilingual-measurement.png` — One pinned Qwen3 worked example: Chinese used 9 tokens, English 11, and Swedish 20; this is not a language-wide ranking.
