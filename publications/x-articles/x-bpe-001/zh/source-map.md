# X-BPE-001 中文版主张与证据索引

中文版沿用英文版的实验证据，不增加新的经验性主张。

| 中文版主张 | 本地证据 | 适用边界 |
|---|---|---|
| 固定 Qwen3 Tokenizer 把 `下一个` 编成一个 Token，把 `språkmodellen` 编成五个子词 Token | `experiments/reports/2026-08-30-qwen3-multilingual-tokenization.md` | 三个固定字符串、一个固定 Tokenizer 修订；无法代表整种语言 |
| 三个固定句子的 Token 数为中文 9、英文 11、瑞典语 20 | 同一多语言报告与原始输出 | 直接观测；这组译文属于教学示例，不构成代表性语料 |
| 家庭 Emoji 含 1 个字素簇、7 个 Unicode 码点、25 个 UTF-8 字节 | `experiments/reports/2026-08-31-tokenizer-mechanics.md` | 使用扩展字素簇切分机械测量 |
| NFC 与 NFD `café` 得到相同 ID，但 NFD 源串无法逐码点精确往返 | 同一 Tokenizer 机制报告 | 固定 Tokenizer 与软件版本下的直接负结果 |
| `token` 与前置空格版本各占一个 Token，ID 不同 | 同一 Tokenizer 机制报告 | 单个边界案例，无法概括所有 Pre-tokenizer |
| 原始 `Hello` 占一个 Token，单消息 Chat 模板占九个位置 | 同一 Tokenizer 机制报告 | 一份固定 Qwen3 Chat 模板的直接观测 |
| 微型 BPE 三轮合并计数依次为 10、10、3 | 同一报告；`src/dongxi_llms/tiny_bpe.py` | 可执行教学机制；没有重建 Qwen 的历史训练过程 |
| `数` 的 UTF-8 为 `E6 95 B0`；字节负责覆盖，合并负责压缩 | `learning_artifacts/day-02-text-tokens-and-embeddings/bpe-training-and-byte-coverage.md` | 字节映射可确定；中文合并阶梯属于透明示意 |
| BPE、WordPiece、Unigram 是不同 Tokenization 算法 | `book/chapters/02-text-tokens-and-embeddings.md` 第 2.3 节；Hugging Face Tokenizers 流程文档 | 只做机制范围说明，不评价优劣 |

## 主要术语来源

- [Hugging Face Tokenizers：Tokenization 流程](https://huggingface.co/docs/tokenizers/main/pipeline)
- [Hugging Face Transformers：编写 Chat Template](https://huggingface.co/docs/transformers/main/chat_templating_writing)
- [Unicode 标准附录 #29：Unicode 文本切分](https://www.unicode.org/reports/tr29/)

## 编辑边界

- 主线聚焦 Byte-level BPE，并明确其他 Tokenizer 家族采用不同机制。
- Qwen3 实测结果与教学示意分别表述。
- 不从切分结果反推训练语料构成。
- 覆盖能力、压缩效率、上下文占用与模型理解力分别讨论。
- Token 数较少无法直接推出模型质量、公平性或理解力更高。
- Chat Template 的 Loss Mask 等深层机制留给后续文章。
