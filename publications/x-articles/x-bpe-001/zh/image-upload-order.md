# 图片上传顺序

1. `assets/01-six-units.png` — 正字法词、字素簇、Unicode 码点、UTF-8 字节、子词或字节 Token 与 Token ID 是六种不同单位。
2. `assets/02-tiny-bpe-training.png` — 微型语料上的三轮 BPE 实测；第一轮最高频计数并列，因此训练规则必须声明如何打破平局。
3. `assets/03-byte-coverage-compression.png` — 基础字节词表提供覆盖能力，学习到的 BPE 合并把重复模式压缩成更大的 Token。
4. `assets/04-interface-surprises.png` — 规范化可能改变精确源序列；Chat Template 序列化还会在用户内容之外加入控制 Token。
5. `assets/05-multilingual-measurement.png` — 固定 Qwen3 示例中，中文使用 9 个 Token、英文 11 个、瑞典语 20 个；这组结果无法代表语言排名。
