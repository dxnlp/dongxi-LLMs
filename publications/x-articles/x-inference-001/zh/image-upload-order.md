# Image Upload Order

01. `../assets/01-latency.png` — 从请求发出到首 token，再到后续输出。时间线是教学示意，不按真实耗时比例绘制。客户端流式消息可能包含多个 token，界面刷新间隔不能直接当作模型 ITL。
02. `../assets/02-batching.png` — 同一轮中，A 已完成，B 还在继续；下一轮 C 加入。每个请求仍有自己的上下文、KV 状态和生成进度。新请求需要先处理输入；图中省略具体 prefill 调度。
03. `../assets/03-phases.png` — 上方：同一执行池内，prefill chunks 与 decode 交错安排。下方：prefill 和 decode 使用不同的执行资源，中间需要传递 KV 状态。箭头表示数据依赖，不代表零成本通信。
04. `../assets/04-speculation.png` — 四个候选进入目标模型验证。此轮示意接受 a、b，拒绝 c，并丢弃依赖它的 d，再产生修正项 x。字母是示意 token；两项被接受是设定结果，不是测得的接受率。
05. `../assets/05-quantization.png` — 同样 80 亿个权重，16 bit 与 4 bit 的理想 payload 比较。柱长只代表计算出的权重存储量，不表示推理速度、模型质量或总显存。
06. `../assets/06-parallelism.png` — 上方：一个请求由分片后的模型共同处理，需要通信。下方：两个完整模型副本分别处理请求 A、B，增加服务容量。图中展示的是两种部署思路，不是性能对比。
07. `../assets/07-map.png` — 将优化放回它作用的位置：复用减少重复 prefill；调度组织请求与阶段；speculation 改变生成推进方式；量化改变 tensor 表示；并行改变计算与请求的设备分布。它们可以组合，但需要检查兼容性与新的瓶颈。
