# LLM 服务 Benchmark 方法

LLM 服务压测不是“多发几个请求看 tokens/s”。可信结果必须同时定义工作负载、到达模型、指标口径、预热、缓存状态、失败处理和环境信息。

## 1. 先写实验问题

好的问题应能被数据证伪：

- 在固定模型与长度下，vLLM 相比 Transformers 单请求和高并发分别快多少？
- 在 TTFT p95 小于 1 秒的条件下，服务最大可承受多少 requests/s？
- Prefix Caching 在 80% 请求共享 1024 Token 前缀时能降低多少 TTFT？
- `max_num_batched_tokens` 增大后，输出吞吐、ITL 和 TTFT 如何变化？

“测试一下 vLLM 性能”没有控制变量，不能作为实验问题。

## 2. 两类负载模型

### 2.1 闭环：固定并发

每个 worker 完成一个请求后才发送下一个。并发数固定，但实际 requests/s 会随服务变慢而下降。

适合：

- 找不同并发下的吞吐—延迟曲线；
- 模拟有限用户持续交互；
- 快速发现容量拐点。

局限：过载时客户端会自动降速，可能掩盖队列增长。

### 2.2 开环：固定到达速率

请求按照预定速率到达，不等待前一个请求完成。可使用均匀、Poisson 或回放真实时间戳。

适合：

- 验证某个生产到达率；
- 观察排队、过载和尾延迟；
- 测试突发流量。

报告必须写明 requests/s、到达分布、突发规则和是否允许积压。并发与请求速率不能互换。

## 3. 请求规格

至少控制：

- 模型、revision、tokenizer 与 Chat Template；
- 输入 Token 长度与分布；
- 最大/实际输出 Token 长度；
- `temperature`、`top_p`、停止条件；
- 是否 Streaming；
- System Prompt、共享前缀比例；
- 超时、重试与取消策略。

### 固定长度与真实分布

固定长度便于复现和解释；真实分布代表外部有效性。建议两者都做：

| 场景 | 输入 | 输出 | 目的 |
| --- | ---: | ---: | --- |
| 短问短答 | 128 | 64 | 交互基线 |
| 长文短答 | 2048 | 64 | Prefill/TTFT |
| 短问长答 | 128 | 512 | Decode/TPOT |
| 长文长答 | 2048 | 512 | 显存与容量 |
| 脱敏分布 | p50/p95 | p50/p95 | 生产代表性 |

若服务允许模型提前输出 EOS，`max_tokens=128` 不代表实际输出 128 Token。报告实际 Token 数，或使用官方 Benchmark 提供的输出长度控制选项并说明行为。

## 4. 指标定义

### 请求级延迟

- `TTFT = first_token_time - request_start`
- `E2E = request_end - request_start`
- `TPOT = (E2E - TTFT) / max(output_tokens - 1, 1)`

### Token 级延迟

- `ITL_i = token_i_time - token_(i-1)_time`

### 吞吐

- 成功 requests/s；
- 输入 tokens/s；
- 输出 tokens/s；
- 总 tokens/s；
- 满足 SLO 的 Goodput；
- 失败率、超时率与取消率。

每个指标至少报告样本数、mean、p50、p95、p99。明确分位数插值方法，避免不同工具产生小幅口径差异。

## 5. 计时边界

客户端 E2E 包含连接、网络、网关和服务端处理。服务端指标通常拆出内部排队与执行。

Streaming 客户端要注意：

- 首个非空内容才算首 Token；
- 一个 SSE data chunk 不一定等于一个 tokenizer Token；
- 网络缓冲可能把多个 Token 合并；
- usage 常在最后一个 chunk 才出现；
- 客户端解析和磁盘日志也可能成为瓶颈。

因此自定义客户端适合近似用户体验；严格 Token 级指标优先使用引擎官方工具或服务端遥测。

## 6. 预热与缓存状态

至少区分：

1. **冷启动**：进程启动到首请求，包括模型加载和初始化；
2. **冷缓存稳态**：服务已启动，但 Prefix Cache 未命中；
3. **热缓存稳态**：Kernel、CUDA Graph 和前缀缓存均可能命中。

稳态压测前运行预热请求，但不要把预热结果混入正式统计。记录预热数量、Prompt 和是否清空缓存。

Prefix Caching 实验必须给出冷/热两组，且请求顺序可复现。

## 7. 客户端不能成为瓶颈

压测前检查：

- 客户端 CPU 是否满载；
- HTTP 连接池与文件描述符上限；
- 是否每个 Token 同步写盘或打印；
- 客户端与服务是否争用同一 GPU/CPU；
- 单机网络是否达到上限；
- 超时和重试是否制造额外流量；
- 客户端时钟是否单调。

先对一个轻量 mock 服务压测客户端，或同时观察客户端资源。

## 8. vLLM 官方 Benchmark

安装对应 extra 后查看命令：

```bash
python -m pip install "vllm[bench]"
vllm bench --help
vllm bench serve --help
```

一个固定长度示例：

```bash
vllm bench serve \
  --backend vllm \
  --base-url http://127.0.0.1:8000 \
  --model facebook/opt-125m \
  --dataset-name random \
  --input-len 128 \
  --output-len 128 \
  --num-prompts 200 \
  --request-rate inf \
  --save-result \
  --save-detailed
```

不同版本支持的 backend、dataset 和参数可能变化，以当前 `--help` 为准。将实际执行命令、版本和原始 JSON 一起保存。

可用命令按版本通常包括：

- `vllm bench latency`：单进程/延迟型测试；
- `vllm bench throughput`：离线吞吐；
- `vllm bench serve`：在线服务压测。

不要把三类结果直接横向比较，它们的执行和排队模型不同。

## 9. Transformers 与 vLLM 对比矩阵

| 变量 | 基线 |
| --- | --- |
| 模型与 revision | 完全相同 |
| tokenizer/template | 完全相同 |
| dtype/量化 | 完全相同 |
| Sampling | temperature=0，固定停止条件 |
| 输入 | 同一 Token 序列 |
| 输出 | 同一上限并记录实际长度 |
| 预热 | 明确且一致 |
| 并发 | 1、2、4、8、16 |
| 重复 | 每点至少 3 次 |
| 指标 | TTFT/TPOT/E2E/吞吐/显存/失败率 |

Transformers 简单 `generate` 与 vLLM 在线服务不是完全相同的系统边界。公平报告应分别给出：

1. 离线 Transformers vs 离线 vLLM；
2. 在线 vLLM 的服务指标；
3. 若有 Transformers 服务基线，说明其批处理和队列实现。

## 10. 推荐实验矩阵

### 长度扫描

固定并发 1，输入长度为 32/128/512/2048，输出固定 64，观察 TTFT。

### 输出扫描

输入固定 128，输出为 16/64/256/512，观察 TPOT 与 E2E。

### 负载扫描

固定 128/128，闭环并发为 1/2/4/8/16/32，观察吞吐与尾延迟拐点。

### 混合请求

80% 短请求 + 20% 长请求，观察长请求是否造成 Head-of-Line Blocking。

### Prefix Cache

固定共享前缀长度与命中比例，对比冷缓存、热缓存和无共享前缀。

## 11. 结果目录

```text
benchmarks/
├── configs/
│   └── week-02-vllm.yaml
├── raw-results/
│   ├── run-001.json
│   └── run-002.json
└── reports/
    └── week-02-vllm-report.md
```

原始结果不可手工修改。聚合脚本读取 raw-results，报告链接配置和原始文件。

每次运行记录：

```yaml
run_id: week2-c8-001
timestamp: 2026-08-24T20:00:00+08:00
git_commit: "<commit>"
model: facebook/opt-125m
model_revision: "<revision>"
vllm_version: "<version>"
hardware:
  gpu: "<gpu>"
  count: 1
server_command: "<full command>"
workload:
  mode: closed_loop
  concurrency: 8
  input_tokens: 128
  output_tokens: 128
  requests: 200
warmup_requests: 10
timeout_seconds: 120
```

## 12. 报告结构

1. 实验问题与假设；
2. 硬件、软件和模型版本；
3. 工作负载与计时口径；
4. 预热、缓存与失败处理；
5. 结果表和延迟—吞吐曲线；
6. 瓶颈解释与证据；
7. 威胁有效性的因素；
8. 可复现实验命令；
9. 下一轮实验。

## 13. 常见错误

- 只报平均延迟，不报 p95/p99；
- 只报最大 tokens/s，不报失败和 SLO；
- 把 SSE chunk 数当 Token 数；
- 比较时输入/输出长度不同；
- 测试过程让模型提前 EOS，却假设输出固定；
- 把冷启动和稳态混在一起；
- 客户端与服务端争抢同一资源；
- 在同一次测试中改变多个参数；
- 不保存启动命令、版本和原始结果；
- 用一次短跑得出生产容量结论。

## 14. 验收清单

- [ ] 能区分开环与闭环负载
- [ ] 能解释并发、request rate 与突发性的区别
- [ ] 报告包含 TTFT、ITL/TPOT、E2E、吞吐和失败率
- [ ] 固定并记录模型、长度、Sampling 与 Chat Template
- [ ] 区分冷启动、冷缓存和热缓存
- [ ] 验证客户端不是瓶颈
- [ ] 保存原始结果、配置与完整命令
- [ ] 能从曲线找出满足 SLO 的容量边界

## 延伸阅读

- [通用 Benchmark 方法论](benchmark-methodology.md)
- [并发客户端实验（GitHub）](https://github.com/LuHeming/ai-infra-from-research-to-production/tree/main/labs/vllm-serving-benchmark)
- [vLLM 推理服务实践](../07-serving/index.md)
