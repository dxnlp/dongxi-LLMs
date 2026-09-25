# Inline image upload order

01. `../assets/01-lifecycle.gif` — 输入经过 prefill，保存 prompt 的 KV，再选出第一个输出，后续输出接着出现。这里只概览两个阶段；TTFT 还可能包含排队、输入处理和传输。箭头与播放时长不代表实测耗时。
02. `../assets/02-parallel.gif` — 同一层、一个 attention head 的两位置示意。斜线标出 K₁/V₁ 对第二个位置的贡献；O₁、O₂ 是各位置的 attention 输出，O₂ 的计算不依赖 O₁。省略归一化、位置处理、多头合并等细节。
03. `../assets/03-first-token.gif` — 四个输入位置的 causal attention 与首 token 边界。x1–x4 是示意 token，y1 是第一个新 token；图中 KV 行代表各层各自保存的状态。
04. `../assets/04-decode-loop.gif` — 用 y1 指代示意 token“延期”，后续 y2、y3 同样是示意符号。动画展开两个连续步骤：新 token 送回模型，两层各自追加 KV，旧条目留在原位，再选出下一个 token。仅示意两层，其他层内操作、vocabulary head 与选择步骤被压缩；遇到停止条件时生成结束。
05. `../assets/05-workloads.gif` — 左侧是长报告换三条结论，右侧是短提示换长演讲稿。图形只对比输入与输出的多少，不代表实测 token 数、耗时或速度。
06. `../assets/06-scheduling.gif` — A 的输出继续增长，B1–B3 分块处理并保留各自的前缀状态。随后展示另一种安排：把 B 的 KV 和继续生成所需状态交给 decode 资源池。A、B 的请求状态始终独立，两种方法可以组合，动作时长不代表实际性能。
07. `../assets/07-handoff.gif` — 第三方演示中的阶段分工，以及可实现的逐层 KV 传输重叠。仅说明机制，没有本地测速，也不表示传输完全免费。
