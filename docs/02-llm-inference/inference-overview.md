# LLM 推理系统概览

本章建立第二周最重要的心智模型：一个 LLM 请求进入服务后，系统做了什么、时间花在哪里，以及性能优化可能牺牲什么。

完成后你应能解释 Prefill 与 Decode 的差异，正确使用 TTFT、ITL、TPOT、吞吐和 Goodput，并设计控制变量明确的推理实验。

## 1. 请求生命周期

```text
HTTP / Python 调用
  → 参数校验、Chat Template 与 Tokenization
  → 排队与调度
  → Prefill
  → 分配并写入 KV Cache
  → Decode Loop
  → Sampling、Detokenization 与 Streaming
  → 请求结束并回收 KV Cache
```

先分清三个边界：

1. 模型执行只是一部分，排队、分词、网络和流式返回也贡献端到端延迟。
2. 一次请求包含一次 Prefill 和零到多次 Decode；输出越长，Decode 轮数越多。
3. 一次调度迭代可以服务多个请求，不等于只执行一个用户请求。

## 2. 自回归生成

模型根据已有 Token 计算下一个 Token 的概率分布。选出新 Token 后追加到上下文并继续预测，直到遇到停止Token、达到最大输出长度或触发其他停止条件。

如果每一步都重新计算全部历史 Token，开销会很大。系统因此保存每层注意力模块中历史 Token 的 Key 和 Value，也就是 KV Cache。Decode 时只计算新 Token 的表示，再读取历史缓存。

## 3. Prefill：处理输入上下文

Prefill 一次处理请求已有的输入 Token，并建立初始 KV Cache。

它通常具有这些特征：

- 输入 Token 能参与较大的并行矩阵运算；
- 输入越长，计算量和初始 KV Cache 越大；
- 大矩阵乘通常更容易提高 GPU 算力利用率；
- 很长的 Prompt 会占用调度预算，可能阻塞正在生成的请求。

Prefill 通常更偏计算密集，但模型结构、长度、批大小、硬件和内核都会改变实际瓶颈。

它主要影响 TTFT、长上下文排队时间、初始 KV Cache 显存，以及混合长短请求时的公平性。典型优化包括 Prefix Caching、Chunked Prefill、高效 Attention/GEMM 内核和请求准入限制。

## 4. Decode：逐 Token 生成

Prefill 后进入 Decode。每次迭代通常为每个活跃序列生成一个 Token。

- 同一序列未来 Token 之间存在依赖，普通自回归生成无法一次全部并行；
- 每轮计算规模较小，却要反复读取模型权重和增长中的 KV Cache；
- 输出长度直接决定迭代次数；
- 单请求可能无法充分利用 GPU，需要合并不同请求的 Decode Token。

Decode 常常更偏内存带宽受限。它主要影响 ITL、TPOT、长回答 E2E 和高并发生成吞吐。典型优化包括 Continuous Batching、Paged KV Cache、CUDA Graph、融合内核、量化、Speculative Decoding 和多卡并行。

## 5. Sampling 与 Streaming

模型产生 logits，Sampling 再决定实际 Token。Benchmark 时至少固定：

- `temperature`、`top_p`、`top_k`；
- `max_tokens` 与停止条件；
- 随机种子（实现支持时）；
- Chat Template 与 tokenizer 版本。

Streaming 不减少模型计算量，但能让用户更早看到结果。客户端应记录请求发出、首个有效输出 Token、后续 Token 和流结束的时间。

不要把 HTTP 响应头、SSE 心跳或空字符串算成首 Token。字符数也不等于 Token 数。

## 6. 延迟指标

设请求开始为 `t0`，首个有效 Token 到达为 `t1`，完成为 `t_end`，输出 Token 数为 `N`。

### 6.1 TTFT

`TTFT = t1 - t0`

它包含排队、服务开销、Prefill、第一次 Decode 和网络传输，描述“多久开始回答”。

### 6.2 E2E Latency

`E2E = t_end - t0`

它描述拿到完整响应需要多久。比较时必须控制输入长度、输出长度与负载。

### 6.3 ITL

`ITL_i = t_i - t_(i-1)`

这是相邻输出 Token 的到达间隔。平均值相同的系统，p99 ITL 可能完全不同。

### 6.4 TPOT

常见定义为：

`TPOT = (E2E - TTFT) / max(N - 1, 1)`

工具间定义可能略有不同，报告必须写出公式，不能只写缩写。

### 6.5 分位数

平均值会掩盖慢请求。至少报告 p50、p95、p99、最大值、失败率和超时率。请求级指标与 Token 级指标要分别聚合。

## 7. 吞吐与 Goodput

`request_throughput = 完成请求数 / 测试持续时间`

Token 吞吐应区分输入和输出：

`input_tokens_per_second = 输入 Token 总数 / 持续时间`

`output_tokens_per_second = 输出 Token 总数 / 持续时间`

当请求长度差异很大时，只报告 requests/s 会误导。

Goodput 只统计满足服务等级目标的请求，例如“TTFT 不超过 800 ms 且 p95 ITL 不超过 80 ms”。最高 tokens/s 配置不一定有最高 Goodput，因为过载会推高排队和尾延迟。

## 8. 延迟—吞吐曲线

低并发时 GPU 可能未被充分利用。提高批量或并发后吞吐通常上升；继续增加负载会出现：

1. 调度队列增长；
2. 批次增大、单轮迭代变慢；
3. KV Cache 压力上升；
4. 抢占、重算、超时或拒绝；
5. p95/p99 延迟快速恶化。

服务 Benchmark 的目标不是寻找“最大并发数字”，而是画负载曲线，找到满足 SLO 的容量边界。

## 9. 工作负载四要素

- **输入长度**：主要影响 Prefill、TTFT 与初始 KV Cache。
- **输出长度**：主要影响 Decode 轮数、E2E 与缓存占用时间。
- **并发和到达速率**：并发是同时在系统中的请求数；速率是每秒到达数。固定并发通常是闭环负载，固定 requests/s 更接近开环负载。
- **长度分布**：固定长度便于比较；真实或脱敏分布、短输入长输出、长输入短输出和长短混合更接近生产。

## 10. 优化技术地图

| 技术 | 主要问题 | 首要指标 | 常见代价 |
| --- | --- | --- | --- |
| Continuous Batching | 静态批次空洞 | 吞吐、ITL | 调度复杂度、尾延迟 |
| Paged KV Cache | 预留浪费和碎片 | 并发容量、显存 | 块管理开销 |
| Prefix Caching | 重复前缀 Prefill | 命中请求 TTFT | 容量与命中率依赖 |
| Chunked Prefill | 长 Prompt 阻塞 Decode | ITL、公平性 | Prefill 被拆分 |
| CUDA Graph | Kernel Launch 开销 | Decode ITL | Shape 与内存约束 |
| Speculative Decoding | 串行 Decode 步数 | TPOT、输出吞吐 | 草稿成本与接受率 |
| Tensor Parallel | 单卡容量/算力不足 | 容量、吞吐 | 跨卡通信 |
| Quantization | 权重/KV 带宽和显存 | 容量、吞吐 | 精度与内核支持 |

优化假设必须可证伪，例如：

> 固定模型、硬件、长度和请求速率后，Prefix Caching 会降低共享前缀请求的 p50 TTFT，但不会明显降低长输出阶段的 TPOT。

## 11. 慢请求排查顺序

1. 明确慢的是 TTFT、ITL 还是 E2E。
2. 检查输入/输出长度、并发和请求速率是否改变。
3. 尽量拆分客户端、排队和模型执行时间。
4. 检查 GPU 利用率、显存、功耗、OOM 与降频。
5. 检查 running/waiting 请求、KV Cache 使用率和抢占。
6. 排除客户端连接数、日志和事件循环瓶颈。
7. 一次只改一个配置，并比较完整分布。

## 12. 常见误区

- GPU 利用率 100% 不代表请求满足 SLO。
- tokens/s 更高可能以严重的 TTFT 和 p99 为代价。
- Streaming 改善感知延迟，但不保证 E2E 更短。
- 输出字符数不能替代 tokenizer 的 Token 数。
- 相同模型名仍需固定 revision、dtype、量化和生成配置。
- 单次运行无法排除初始化、JIT、CUDA Graph 与缓存影响。
- 并发 32 不等于 32 requests/s。

## 13. 本章实验

1. 固定输出 64 Token，把输入从 32 增加到 2048，记录 TTFT。
2. 固定输入 128 Token，把输出从 16 增加到 256，记录 E2E 与 TPOT。
3. 固定长度，把并发从 1 提高到 2、4、8、16，画吞吐与 p95 TTFT 曲线。
4. 每次先写瓶颈假设，再用结果支持或否定它。

## 14. 验收清单

- [ ] 能画出生命周期并标注 Prefill、Decode 与 KV Cache
- [ ] 能解释 TTFT、ITL、TPOT、E2E 的公式和差异
- [ ] 能区分请求吞吐、Token 吞吐与 Goodput
- [ ] 能解释输入、输出长度和并发的不同影响
- [ ] 能写出控制变量明确的优化假设
- [ ] 报告包含版本、硬件、负载、预热和分位数

## 延伸阅读

- [KV Cache 与调度](kv-cache-and-scheduling.md)
- [vLLM 推理服务实践](../07-serving/index.md)
- [LLM 服务 Benchmark](../09-benchmarking/llm-serving-benchmark.md)
