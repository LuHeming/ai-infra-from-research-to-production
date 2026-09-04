# vLLM 流式并发压测实验

本实验用 Python 标准库向 vLLM 的 OpenAI-Compatible API 发送闭环并发请求，记录用户侧 TTFT、E2E、流式数据块间隔、成功率和请求吞吐。

## 1. 启动服务

```bash
export MODEL=facebook/opt-125m
export VLLM_API_KEY=local-dev-key

vllm serve "$MODEL" \
  --host 127.0.0.1 \
  --port 8000 \
  --api-key "$VLLM_API_KEY"
```

## 2. 先跑单请求

```bash
python labs/vllm-serving-benchmark/request_client.py \
  --model facebook/opt-125m \
  --api-key "$VLLM_API_KEY" \
  --num-requests 1 \
  --concurrency 1
```

确认结果中 `success_rate=1`、`ttft_ms` 非空，再增加负载。

## 3. 并发测试

```bash
for concurrency in 1 2 4 8 16; do
  python labs/vllm-serving-benchmark/request_client.py \
    --model facebook/opt-125m \
    --api-key "$VLLM_API_KEY" \
    --prompts-file labs/vllm-serving-benchmark/prompts.example.jsonl \
    --num-requests 100 \
    --warmup-requests 5 \
    --concurrency "$concurrency" \
    --max-tokens 128 \
    --output "benchmarks/raw-results/week2-c${concurrency}.json"
done
```

这是闭环负载：同时最多有 `concurrency` 个请求；worker 完成后再处理下一个。它不等于固定 requests/s。

## 4. Chat API

基础 Completion 模型使用默认 `completions`。对于配置好 Chat Template 的 Instruct 模型：

```bash
python labs/vllm-serving-benchmark/request_client.py \
  --endpoint chat \
  --model your-instruct-model \
  --api-key "$VLLM_API_KEY" \
  --num-requests 20 \
  --concurrency 4
```

## 5. 输出字段

汇总中包含：

- `duration_seconds`：正式压测墙钟时间；
- `requests_per_second`：成功请求数除以持续时间；
- `success_rate`：成功请求占比；
- `ttft_ms`：首个非空内容块到达时间；
- `e2e_ms`：完整流结束时间；
- `stream_chunk_gap_ms`：相邻非空 SSE 内容块间隔；
- `usage`：服务返回时保存 Token 计数。

`stream_chunk_gap_ms` 是客户端可见的 SSE 数据块间隔，**不保证一个块等于一个 tokenizer Token**。严格 ITL/TPOT 优先使用 vLLM 官方 Benchmark 或服务端 Token 遥测。

## 6. 实验纪律

- 先验证正确性，再跑性能；
- 正式请求前预热，但不把预热纳入统计；
- 所有并发点使用相同 Prompt 集、输出上限和 Sampling；
- 不要在压测时逐 Token 打印；
- 同时监控客户端 CPU、网络和服务端 GPU；
- 原始 JSON 不手工修改；
- 结果文件不包含 API key，也默认不保存 Prompt/输出正文；
- 每个并发点至少重复 3 次。

## 7. 使用官方工具复核

自定义客户端用于理解计时边界和快速回归。正式结论再用当前版本的：

```bash
vllm bench serve --help
```

复核，并在报告中写明两个工具的指标口径差异。

## 8. 验收

- [ ] 单请求可正确完成
- [ ] 并发 1/2/4/8/16 均有原始 JSON
- [ ] 失败和超时没有被丢弃
- [ ] 能解释闭环负载
- [ ] 能解释 SSE chunk gap 为什么不是严格 ITL
- [ ] 能找到吞吐与 p95 TTFT 的拐点
