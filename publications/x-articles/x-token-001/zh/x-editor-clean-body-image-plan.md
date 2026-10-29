# X Article Inline Image Plan

Use this after pasting the clean body into X Articles.

Rule: insert images from bottom to top so earlier block positions do not shift.

- Title: `一颗 token 的诞生：从你的问题到模型的回答`
- Clean body HTML: `x-editor-clean-body.html`
- Clean body Markdown: `x-editor-clean-body.md`
- Total clean body blocks: `82`
- Images: `7`

## Reverse Insertion Order

### IMAGE 07

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-token-001/assets/07-detokenize.gif`
- Exists: `True`
- Insert after block index: `65`
- Target text: 有些字符，包括许多中文字，会跨越不止一个 token。应用会先暂存不完整的字节，等这个字完整了再显示。
- Next text: 同一张词表用了两次：进来时把文字换成编号，出去时把编号换回文字。

### IMAGE 06

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-token-001/assets/06-decode-loop.gif`
- Exists: `True`
- Insert after block index: `56`
- Target text: 模型选出结束 token <|im_end|>，或者达到应用设定的长度上限，循环就停止。在我们的例子里，prefill 给出了 Part，接下来 8 次 decode 依次给出 ly、.、␣Some、␣weights、␣are、␣open...
- Next text: prefill 一次处理整个 prompt；decode 每次只处理一个 token，每一次都要把模型的权重重新读一遍。

### IMAGE 05

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-token-001/assets/05-pick-token.gif`
- Exists: `True`
- Insert after block index: `49`
- Target text: greedy decoding 每次都写 Part。sampling 多数时候写 Part，有时写 ␣Open，偶尔写 No。
- Next text: 这些设置由应用或 API 调用决定，与模型的权重无关。

### IMAGE 04

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-token-001/assets/04-prefill.gif`
- Exists: `True`
- Insert after block index: `40`
- Target text: 一路上，每一层都为每个位置存下两个向量：key 和 value，合起来叫 KV cache。后面的 token 还要用到它们。存下来以后，prompt 就不用再算第二遍。
- Next text: 一次运行，给词表里的每个 token 各打一个分。分数最高的只是候选，还没有被选中。

### IMAGE 03

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-token-001/assets/03-tokenize.gif`
- Exists: `True`
- Insert after block index: `27`
- Target text: 151644 872 198 3872 5264 15469 1787 30 151645 198 151644 77091 198
- Next text: 从这一步开始，模型看到的只有编号，没有文字。

### IMAGE 02

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-token-001/assets/02-chat-template.gif`
- Exists: `True`
- Insert after block index: `20`
- Target text: 最后一行开启助手的发言，然后停在这里。模型的回答从下一行开始写。关闭思考模式时，Qwen3 还会在这里插入一个空的思考块，图中略去。
- Next text: 模型读到的是套好格式的整段文本，你输入的问题只是其中一行。

### IMAGE 01

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-token-001/assets/01-overview.gif`
- Exists: `True`
- Insert after block index: `4`
- Target text: 下面跟着一次真实的请求走一遍，从你按下回车，到回答出现在屏幕上。文中的 token 和编号都取自 Qwen3 的真实词表。今天的聊天模型都用同一个循环生成文字，差别在于模型大小、词表和包裹问题的模板。
- Next text: 语言模型并不直接写字。它每运行一次，只给下一个 token 打一轮分；外面一个简短的循环，把这些分数变成文字。
