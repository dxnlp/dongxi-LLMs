# Image Upload Order

封面（单独上传，5:2）：`../assets/cover-token-birth.png`。所有配图均为英文，中英文版共用。

1. `../assets/01-overview.gif` — 整个循环：prompt 只做一次 tokenize；每次运行给出 logits，挑出一个 token 接到序列末尾，它的文字加入回答。9 次运行写完回答，结束 token 让循环停止。
2. `../assets/02-chat-template.gif` — 问题被套进 chat template（琥珀色部分）。回答从 assistant 下面的空行开始。
3. `../assets/03-tokenize.gif` — 文字被切成词表里的条目，每个条目换成它的 token ID。token 和编号取自 Qwen3 的真实词表。
4. `../assets/04-prefill.gif` — 每个编号从 embedding 表取出一行。13 个位置一起经过各层，把 key 和 value 留在 KV cache 里。只有最后一个位置进入 LM head，得到 151,936 个 logits。图中用 3 层代表 28 层。
5. `../assets/05-pick-token.gif` — softmax、temperature、top-p 0.9；greedy decoding 每次都选 Part，sampling 依次抽到 Part、Open、Part、No、Part。数字为示意值。
6. `../assets/06-decode-loop.gif` — 每次只送入最新的 token。它在每一层读取 KV cache（绿色弧线），再存入自己的 key 和 value；选出的 token 作为下一次的输入送回。
7. `../assets/07-detokenize.gif` — 编号反向查词表；去掉结束 token，恢复空格，片段拼成回答。
