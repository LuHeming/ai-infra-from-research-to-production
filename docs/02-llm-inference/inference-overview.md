# LLM 推理系统概览

## 请求生命周期

```text
Request
  → Tokenization
  → Scheduling
  → Prefill
  → KV Cache Allocation
  → Decode Loop
  → Sampling
  → Streaming Response
```

## Prefill

Prefill 一次处理完整输入序列，通常具有较高并行度，更接近计算密集型阶段。

## Decode

Decode 每轮通常只产生一个新 token，需要反复读取模型权重和 KV Cache，常受到显存带宽、调度开销和批处理效率影响。

## 核心指标

- **TTFT**：Time To First Token
- **TPOT**：Time Per Output Token
- **ITL**：Inter-token Latency
- **E2E Latency**：端到端请求延迟
- **Throughput**：tokens/s 或 requests/s
- **Peak Memory**：峰值显存
- **Goodput**：满足延迟 SLO 的有效吞吐

## 关键优化

- Continuous Batching
- Paged KV Cache
- Prefix Caching
- Chunked Prefill
- CUDA Graph
- Speculative Decoding
- Tensor Parallel
- Quantized Inference
