# Week 2：LLM 推理系统与 vLLM 服务

本周目标是把“会调用模型”升级为“能解释、部署、测量和定位一个推理服务”。学习顺序遵循：原理 → 容量 → 服务 → 测量 → 分析。

## 本周核心问题

1. 一个请求如何经过 Tokenization、Prefill、KV Cache、Decode 和 Streaming？
2. Prefill 为什么更偏计算密集，Decode 为什么常受内存带宽限制？
3. KV Cache 如何估算，GQA/MQA 为什么要看 KV Head？
4. Continuous Batching 和 PagedAttention 分别解决什么问题？
5. TTFT、ITL、TPOT、E2E、Throughput 与 Goodput 如何定义？
6. 如何启动 vLLM 的离线推理和 OpenAI-Compatible 服务？
7. 如何设计公平、可复现且不会被客户端瓶颈污染的 Benchmark？

## 前置条件

- 已完成 Week 1 Linux、Python 工程与 PyTorch Profiler；
- Linux + 可用的 NVIDIA GPU 环境，或可远程访问 GPU 服务器；
- 能创建 Python 虚拟环境、查看 `nvidia-smi` 和管理进程；
- 已选择一个能在硬件上稳定运行的小模型。

没有 GPU 时仍可完成 Day 1～3 的原理、计算和实验设计；服务与压测可在有 GPU 后补做。

## 本周交付物

```text
weekly/week-02-llm-inference-and-vllm.md
docs/02-llm-inference/
├── inference-overview.md
└── kv-cache-and-scheduling.md
docs/07-serving/index.md
docs/09-benchmarking/llm-serving-benchmark.md
labs/vllm-serving-benchmark/
├── README.md
├── request_client.py
└── prompts.example.jsonl
benchmarks/raw-results/<your-runs>.json
benchmarks/reports/week-02-vllm-report.md
```

最终至少产出：

- 一张请求生命周期图；
- 一张 KV Cache 手算表；
- 一个可访问的本地 vLLM API；
- 一组并发 1/2/4/8/16 的原始结果；
- 一张吞吐—p95 TTFT 曲线；
- 一份包含假设、结果、瓶颈和局限的短报告。

## Day 1：请求生命周期与指标

阅读：

- [LLM 推理系统概览](../docs/02-llm-inference/inference-overview.md)

完成：

1. 手画 Request → Prefill → Decode → Streaming 生命周期。
2. 用自己的语言写出 TTFT、ITL、TPOT、E2E 的公式与边界。
3. 为“短问短答、长文短答、短问长答”分别判断主要压力。
4. 写一个可证伪的性能假设。

当日验收：

- 不看文档解释 Prefill 和 Decode；
- 能说明 tokens/s 高但用户体验差的原因；
- 不混淆并发数和 requests/s。

## Day 2：KV Cache 与容量估算

阅读：

- [KV Cache、PagedAttention 与请求调度](../docs/02-llm-inference/kv-cache-and-scheduling.md)

完成：

1. 从目标模型 `config.json` 找到层数、KV Head、Head Dimension。
2. 计算单 Token KV、1K/4K/16K 单序列和 1/8/32 并发占用。
3. 画逻辑 KV Block 到物理 Block 的映射。
4. 列出权重之外的五类显存占用。

当日验收：

- GQA/MQA 使用 KV Head 数；
- 能解释内部碎片和外部碎片；
- 明白理论最大并发不能替代压力测试。

## Day 3：Continuous Batching 与调度

完成：

1. 用时间线比较静态批处理和 Continuous Batching。
2. 解释 `max_num_seqs`、`max_num_batched_tokens`、`max_model_len`。
3. 设计长 Prompt 与 Decode 混合实验。
4. 写出 Prefix Caching 冷/热缓存对照组。

当日验收：

- 能解释 Head-of-Line Blocking；
- 能说明 Chunked Prefill 对 TTFT、ITL 和吞吐的权衡；
- 不把 PagedAttention 误解为磁盘分页。

## Day 4：vLLM 离线推理

阅读：

- [vLLM 推理服务实践](../docs/07-serving/index.md)

完成：

1. 新建独立虚拟环境并记录版本。
2. 用 `LLM` 与 `SamplingParams` 完成离线生成。
3. 固定两个 Prompt、Sampling 参数和模型 revision。
4. 记录首次与预热后运行时间、峰值显存和实际输出长度。

当日验收：

- 代码可在新环境复现；
- 知道模型 generation config 可能改变默认采样；
- 能区分冷启动和稳态。

## Day 5：OpenAI-Compatible 服务

完成：

1. 绑定 `127.0.0.1` 启动 `vllm serve`。
2. 检查 `/v1/models`。
3. 完成非流式 Completion 和流式请求。
4. 若使用 Chat API，验证模型 Chat Template。
5. 保存完整服务命令与日志。

当日验收：

- HTTP 返回成功且输出语义正确；
- 能用 API key 访问受保护 API；
- 知道生产环境还需反向代理、TLS、全路径鉴权和限流。

## Day 6：并发 Benchmark

阅读：

- [LLM 服务 Benchmark](../docs/09-benchmarking/llm-serving-benchmark.md)
- [并发客户端实验](../labs/vllm-serving-benchmark/README.md)

完成：

1. 先用并发 1 验证 Token 流与结果格式。
2. 预热后测试并发 1/2/4/8/16。
3. 每个点使用相同 Prompt 集和生成参数，至少重复 3 次。
4. 保存原始 JSON，不手改数据。
5. 同时记录 GPU 指标和 waiting/running 请求。

当日验收：

- 结果包含成功率、TTFT、E2E、ITL 近似值和 requests/s；
- 客户端 CPU、网络和日志不是瓶颈；
- 能指出吞吐—延迟曲线的拐点。

## Day 7：分析、报告与复盘

完成报告：

1. 实验问题和预期；
2. 硬件、软件、模型与服务命令；
3. 工作负载、预热、缓存和计时口径；
4. p50/p95/p99 与吞吐结果；
5. 曲线和拐点；
6. 瓶颈证据；
7. 有效性威胁；
8. 下一步单变量实验。

建议做一次参数对照：只改变 `max_num_batched_tokens`、`max_num_seqs` 或 Prefix Caching 中的一项。

## 最终验收

### 原理

- [ ] 能解释 Prefill/Decode 的计算与内存行为
- [ ] 能手算 KV Cache 并解释误差来源
- [ ] 能说明 Continuous Batching、Paged KV Cache、Prefix Caching 与 Chunked Prefill

### 工程

- [ ] 离线推理可复现
- [ ] 在线 API 能正确处理 Streaming
- [ ] 服务只在预期地址暴露并有最小安全防护
- [ ] 版本、命令、模型 revision 和配置已保存

### Benchmark

- [ ] 明确闭环或开环负载
- [ ] 固定输入/输出、Sampling、预热和缓存状态
- [ ] 报告 p50/p95/p99、吞吐与失败率
- [ ] 原始结果未修改且能追溯
- [ ] 找到满足 SLO 的容量点，而非只报最大吞吐

## 本周完成标准

**及格**：能启动服务，跑通并发客户端并解释四个核心延迟指标。

**良好**：完成长度扫描和负载扫描，保存可复现结果并定位性能拐点。

**优秀**：完成一个调度或缓存参数的单变量对照，用 GPU/服务遥测支持瓶颈结论，并明确实验局限。
