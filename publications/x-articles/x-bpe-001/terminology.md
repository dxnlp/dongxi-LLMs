# X-BPE-001 canonical NLP terminology

This vocabulary is shared by the English and Chinese versions. It follows the
standard tokenizer pipeline used in NLP libraries and keeps software-engineering
metaphors out of the reader-facing explanation.

| Concept | English | 中文 | Usage in this article |
|---|---|---|---|
| Complete process | tokenization pipeline | Tokenization 流程 | Raw text becomes an `Encoding` through normalization, pre-tokenization, a tokenization model, and post-processing |
| Unicode rewriting | normalization | Unicode 规范化 | May rewrite the source sequence before segmentation |
| Boundary preparation | pre-tokenization | 预切分 | Produces smaller regions that bound later tokenization |
| Learned segmentation algorithm | tokenization model | Tokenization 算法 | BPE, WordPiece, and Unigram are different tokenization models |
| Output unit | token; qualify as subword token or byte token when needed | Token；需要时写成子词 Token 或字节 Token | A discrete unit emitted by the tokenizer |
| Vocabulary index | token ID | Token ID | Integer index assigned to a vocabulary token |
| Stored inventory | vocabulary | 词表 | Mapping between tokens and integer IDs |
| Final tokenizer stage | post-processing | 后处理 | May add special tokens and produce the final `Encoding` |
| Model input | token ID sequence / input representation | Token ID 序列 / 输入表示 | The integer sequence consumed by embedding lookup |
| Matrix selection | embedding lookup | Embedding 查表 | Uses each token ID to select a row from the input embedding matrix |
| Chat conversion | chat template serialization | Chat Template 序列化 | Converts role/content messages into formatted text or token IDs, including control tokens |
| Learned BPE order | merge ranks / merge priority | Merge Rank / 合并优先级 | Determines which trained merges apply first during encoding |
| Reproducibility identity | tokenizer name, revision, and configuration | Tokenizer 名称、修订版本与配置 | Required context for any reproducible token count |

## Terms intentionally avoided

The article does not use “contract / interface / 契约 / 接口” as its main
explanatory frame. Those terms are useful in software architecture, but the NLP
mechanisms here are more precisely described through preprocessing,
segmentation, vocabulary mapping, post-processing, and model input
representation.

## Primary terminology references

- [Hugging Face Tokenizers: The tokenization pipeline](https://huggingface.co/docs/tokenizers/main/pipeline)
- [Hugging Face Tokenizer API](https://huggingface.co/docs/tokenizers/main/api/tokenizer)
- [Hugging Face Transformers: Writing a chat template](https://huggingface.co/docs/transformers/main/chat_templating_writing)
- [Unicode Standard Annex #29: Unicode Text Segmentation](https://www.unicode.org/reports/tr29/)
