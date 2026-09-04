# KV Cache、PagedAttention 与请求调度

KV Cache 是理解 LLM 服务容量的入口，调度器则决定这些容量如何在请求间分配。本章从显存公式开始，解释 Continuous Batching、分页缓存、Prefix Caching 和 Chunked Prefill。

## 1. 为什么需要 KV Cache

Transformer 的自注意力会为每个 Token 在每一层生成 Key 和 Value。自回归 Decode 时，历史 Token 的 K/V 不再改变，可以缓存下来，只为新 Token 计算新的 K/V。

不用缓存时，第 t 步会重复计算前 t-1 个 Token；使用缓存后，历史投影可以复用。代价是缓存随序列长度和并发请求数线性增长。

```text
请求进入
  → Prefill 写入输入 Token 的 K/V
  → 每轮 Decode 追加一个 Token 的 K/V
  → 请求完成、取消或被淘汰
  → 释放缓存块
```

## 2. 显存估算

忽略对齐和运行时元数据，单序列 KV Cache 近似为：

`M_KV = 2 × layers × kv_heads × head_dim × sequence_length × bytes_per_element`

其中：

- `2`：Key 和 Value 两份缓存；
- `layers`：Transformer 层数；
- `kv_heads`：KV Head 数；
- `head_dim`：每个 Head 的维度；
- `sequence_length`：当前序列 Token 数；
- `bytes_per_element`：FP16/BF16 通常为 2。

并发 C 个同长度序列时：

`M_total ≈ C × M_KV`

### 2.1 GQA/MQA 的关键点

Grouped-Query Attention 或 Multi-Query Attention 中，Query Head 数与 KV Head 数不同。估算缓存必须使用 **KV Head 数**，不能使用 Attention/Query Head 数。

### 2.2 示例

假设模型有 32 层、8 个 KV Head、Head Dimension 为 128，KV Cache 使用 BF16，序列长度为 4096：

```text
2 × 32 × 8 × 128 × 4096 × 2 bytes
= 1,073,741,824 bytes
≈ 1 GiB / sequence
```

这只是建立量级感的理论值。真实占用还受块大小、对齐、并行切分、运行时元数据、滑动窗口和 KV Cache dtype 影响。

### 2.3 总显存不只有 KV Cache

GPU 还要容纳：

```text
模型权重
+ KV Cache
+ 临时激活和算子 Workspace
+ CUDA Context / Graph
+ 通信 Buffer
+ 分配器保留空间
```

因此不能用“总显存减权重后全部除以单请求 KV”直接承诺最大并发，容量必须用真实服务压测确认。

## 3. 连续预分配的浪费

简单实现可能按 `max_model_len` 为每个请求预留一整块连续显存。但请求长短不同，输出长度事先未知，会造成：

- **内部碎片**：预留 8192 Token，实际只使用 600；
- **外部碎片**：总空闲空间够，却缺少足够大的连续区域；
- 请求结束时间不同，连续区域难以灵活复用；
- 为最坏情况预留会降低可服务并发。

## 4. Paged KV Cache 与 PagedAttention

分页思路把序列 KV Cache 切成固定大小的逻辑块，再映射到不必物理连续的 GPU 块。

```text
Sequence A logical blocks: [A0] [A1] [A2]
                               ↓    ↓    ↓
GPU physical blocks:         [17]  [03]  [41]
```

调度器维护块表，Attention 内核根据映射读取 K/V。这样可以按需增长、快速回收复用、降低外部碎片，也为共享前缀引用同一缓存块提供基础。

PagedAttention 不是“把 GPU 显存换成磁盘分页”。核心是借鉴虚拟内存的块映射思想管理 GPU 中的 KV Cache；CPU swap、重计算或卸载取决于引擎和配置。

### 块大小权衡

- 块小：尾块浪费较少，但块表更大、管理更复杂；
- 块大：管理开销较低，但短序列和尾块内部碎片更多。

学习阶段比手调块大小更重要的是观察 KV Cache 使用率、等待请求和抢占事件。

## 5. 静态批处理与 Continuous Batching

静态批次收集一组请求后一起运行，并常常等待整批完成。输出长度不同会导致短请求完成后留下空洞，长请求仍在 Decode。

Continuous Batching（in-flight batching）会在每次迭代重新决定批次：

1. 已完成请求立即退出并释放资源；
2. 有预算时让新请求加入；
3. 活跃 Decode 可与新 Prefill 共同调度；
4. 批次形状随时间变化。

它提高利用率和吞吐，但也引入权衡：更大批次可能增加单轮延迟，长 Prefill 可能阻塞 Decode，过度接纳会让队列和 KV Cache 失控，优先级还会影响公平性。

## 6. 调度预算

一次迭代受到以下约束：

- 本轮最多处理的 Token；
- 最多同时运行的序列；
- 可用 KV Cache 块；
- 模型最大上下文；
- 并行配置和内核支持的 Shape。

可以把它看作装箱问题：在有限 Token 与显存预算中选择本轮的 Prefill 和 Decode 工作。

vLLM 常见旋钮包括：

- `max_num_seqs`：同时处理的序列上限；
- `max_num_batched_tokens`：单迭代 Token 预算；
- `max_model_len`：最大序列长度；
- `gpu_memory_utilization`：引擎可使用的 GPU 显存比例。

默认值和参数会随版本变化，应以本机 `vllm serve --help` 为准。

## 7. Prefix Caching

多个请求拥有完全相同的 Token 前缀时，可以复用前缀 KV Cache，避免重复 Prefill。

适合固定长 System Prompt、多轮共享历史、同一文档多问和多候选生成。不应期待它加速无共享前缀的请求、消除新 Token Decode，或复用“语义相似但 Token 不同”的内容。

实验至少区分冷缓存与热缓存，并记录前缀 Token 数、命中比例、缓存容量和请求顺序。

## 8. Chunked Prefill

长 Prompt 的 Prefill 可能占满一次计算预算，使 Decode 请求等待。Chunked Prefill 把它切成多个块并跨迭代完成。

典型策略：

1. 优先安排延迟敏感的 Decode Token；
2. 用剩余 Token 预算放入 Prefill；
3. Prefill 超出预算时只处理一个 Chunk；
4. 后续迭代继续剩余 Prompt。

收益可能包括降低长 Prefill 对 ITL 的阻塞、混合计算密集 Prefill 与偏带宽密集 Decode、提高公平性。代价是长 Prompt 被拆开后 TTFT 可能变化，Chunk 过小还会增加调度和 Kernel Launch 开销。

vLLM V1 在可用时默认启用 Chunked Prefill，并优先 Decode，再用剩余 `max_num_batched_tokens` 安排 Prefill。实验必须记录 vLLM 版本和完整启动参数。

## 9. 抢占与资源压力

KV Cache 不足时，系统可能让请求排队、暂停低优先级请求、释放缓存后重算、换出数据、拒绝请求或 OOM。策略随引擎版本变化。

尾延迟突增时同时检查：

- KV Cache 使用率；
- running / waiting 请求数；
- preemption 或 recomputation 计数；
- GPU 显存与 allocator 保留量；
- 输入长度是否接近 `max_model_len`。

## 10. 多卡注意事项

Tensor Parallel 会切分模型和部分 KV Cache，但每卡仍需通信 Buffer 与运行时空间。增加两张卡不保证并发翻倍，通信、批次效率和剩余显存最小的 GPU 都可能成为瓶颈。

推荐流程：先算全局理论值，再按引擎切分估算每卡份额，加上运行时空间，最后用启动日志和压力测试确认。

## 11. 推荐实验

### A. 手算 KV Cache

从模型 `config.json` 找到层数、KV Head 数、隐藏维度与 Attention Head 数，计算每 Token、1K/4K/16K 上下文，以及 1/8/32 并发的理论占用，并与服务日志比较。

### B. 输入长度与 TTFT

固定输出和并发，只改变输入长度，观察 TTFT 何时出现非线性。

### C. 共享前缀

比较冷缓存首请求、相同 Token 前缀的后续请求，以及只改前缀中间一个 Token 的请求。

### D. 混合长短请求

同时发送长 Prompt 短回答与短 Prompt 长回答，记录 p95 TTFT、p95 ITL 和完成顺序，观察 Head-of-Line Blocking。

## 12. 验收清单

- [ ] 能从模型配置手算单 Token 和单序列 KV Cache
- [ ] GQA/MQA 使用 KV Head 数而不是 Query Head 数
- [ ] 能解释连续预分配的两类碎片
- [ ] 能画出逻辑 KV Block 到物理 Block 的映射
- [ ] 能解释 Continuous Batching 为什么优于静态批处理
- [ ] 能说明 Prefix Caching 只复用相同 Token 前缀的 Prefill
- [ ] 能解释 Chunked Prefill 对 TTFT、ITL 和吞吐的权衡
- [ ] 能根据缓存压力设计单变量实验

## 延伸阅读

- [LLM 推理系统概览](inference-overview.md)
- [vLLM 推理服务实践](../07-serving/index.md)
- [LLM 服务 Benchmark](../09-benchmarking/llm-serving-benchmark.md)
