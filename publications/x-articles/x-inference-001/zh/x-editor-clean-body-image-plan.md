# X Article Inline Image Plan

Use this after pasting the clean body into X Articles.

Rule: insert images from bottom to top so earlier block positions do not shift.

- Title: `LLM 推理：一条回答背后的速度、显存与调度`
- Clean body HTML: `x-editor-clean-body.html`
- Clean body Markdown: `x-editor-clean-body.md`
- Total clean body blocks: `68`
- Images: `7`

## Reverse Insertion Order

### IMAGE 07

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-inference-001/assets/07-map.png`
- Exists: `True`
- Insert after block index: `63`
- Target text: 模型放不下，或者长上下文撑满显存：分清权重与 KV 的占用，再决定量化或并行方式。
- Next text: 如果有人告诉你“推理快了两倍”，可以接着问：相同模型和精度吗？输入、输出有多长？缓存是冷的还是热的？同时有多少请求？提升的是 TTFT、输出速度，还是系统吞吐量？回答质量是否保持可接受？

### IMAGE 06

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-inference-001/assets/06-parallelism.png`
- Exists: `True`
- Insert after block index: `54`
- Target text: 复制多个实例。 模型能容纳在一个实例中，就可以部署多个副本，由服务系统分发不同请求。一个副本也可以由多张 GPU 共同承载。对普通 dense 模型，这是理解 inference data parallelism 的直接方式。
- Next text: 更多副本可以减少拥堵、承接更多用户，但一条请求通常仍由分配到的模型实例处理，其逐步生成计算不会自动按副本数量加速。

### IMAGE 05

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-inference-001/assets/05-quantization.png`
- Exists: `True`
- Insert after block index: `47`
- Target text: 做一个纯粹的账面计算：假设模型恰有 80 亿个参数，全部按 16 bit 存储，权重 payload 是 16 GB；全部按 4 bit 紧凑存储，则是 4 GB。这里使用十进制 GB，忽略 scales、元数据、未量化层和对齐开销，更没...
- Next text: 权重量化不会自动缩小另一笔 KV 占用。长文档和高并发下，还可以评估 KV cache quantization，但它有独立的精度、格式和 kernel 支持要求。

### IMAGE 04

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-inference-001/assets/04-speculation.png`
- Exists: `True`
- Insert after block index: `39`
- Target text: 假设提出了四个候选 token：a、b、c、d。目标模型按规则接受了 a、b，在 c 处拒绝。系统保留已接受的前缀，在拒绝处产生修正 token；原来基于 c 提出的 d 也不能直接沿用。
- Next text: 这里的“验证”是按目标模型的概率规则检查候选，不能理解成事实核查。正确的 exact speculative sampling 使用接受与修正规则，在理想数值条件下保持目标采样分布；这不保证每次运行得到逐字相同的答案。

### IMAGE 03

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-inference-001/assets/03-phases.png`
- Exists: `True`
- Insert after block index: `32`
- Target text: 让 prefill worker 处理输入，再把需要的 KV 状态交给 decode worker 继续生成。两个执行池可以采用不同的资源配比和并行设置，降低新输入处理对输出节奏的干扰。
- Next text: 为什么值得区分？长 prompt 的 prefill 往往能组织较大的矩阵计算；小 batch 的 decode 则常受权重、KV 的读取带宽限制。这只是常见倾向，瓶颈会随模型、上下文、batch 和硬件变化。

### IMAGE 02

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-inference-001/assets/02-batching.png`
- Exists: `True`
- Insert after block index: `20`
- Target text: 如果一批请求的成员固定，短回答结束后，新请求仍可能要等待下一批。Continuous batching 在生成迭代之间重新安排成员：完成的请求退出，等待中的请求在资源允许时加入，长回答继续生成。
- Next text: GPU 可以共同处理几个人的计算，但他们的对话不会因此混在一起。调度器还要考虑 token 预算、可用 KV 空间与公平性。让更多请求同时运行，也可能让每条请求的一步计算更重。

### IMAGE 01

- File: `/Users/yongchao/dongxi-LLMs/publications/x-articles/x-inference-001/assets/01-latency.png`
- Exists: `True`
- Insert after block index: `15`
- Target text: 全站每秒输出更多 token，你的这条回答仍可能等得更久。 请求长度、到达速度和并发量都会改变这些指标。对比吞吐量时，必须同时看负载和延迟要求。
- Next text: 还有一个容易误判的地方：reasoning 模型可能先产生未展示的中间 tokens。用户看到首个答案文字的时间，与服务记录的首 token 时间可能不同。比较性能前，先确认自己测的是哪一个时刻。
