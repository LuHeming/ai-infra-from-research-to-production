# vLLM 推理服务实践

本章把模型推理变成可测量的 OpenAI-Compatible HTTP 服务。重点不是记住所有参数，而是建立“环境确认 → 离线验证 → 在线服务 → 正确性检查 → 负载测试 → 生产防护”的流程。

> vLLM 的参数和默认策略演进较快。实验必须记录 `vllm --version`，并以当前版本的 `vllm serve --help` 为准。

## 1. 环境与边界

vLLM 主要面向 Linux 与受支持的加速器环境。开始前记录：

```bash
python --version
python -m pip --version
nvidia-smi
vllm --version
```

建议单独创建虚拟环境并固定依赖：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install vllm
python -m pip freeze > environment.lock.txt
```

正式项目应根据硬件、CUDA/PyTorch 兼容矩阵选择安装方式，而不是直接复制其他机器的 wheel。Benchmark 辅助命令若未安装，可按官方说明使用 `vllm[bench]` extra。

## 2. 先做容量估算

启动前确认：

- 模型权重大小和 dtype；
- 单卡显存与 GPU 数量；
- 计划的 `max_model_len`；
- 典型输入/输出长度；
- 并发目标与 KV Cache 量级；
- 是否使用量化或 Tensor Parallel。

权重能加载不代表服务能承受目标并发。模型加载后剩余显存还要容纳 KV Cache、Workspace、CUDA Graph 和通信 Buffer。

## 3. 离线推理最小闭环

离线模式适合验证模型可加载、生成参数生效、输出格式正确，以及单进程批量推理。

```python
from vllm import LLM, SamplingParams

prompts = [
    "Explain prefill and decode in two sentences.",
    "Why does KV cache grow with sequence length?",
]

sampling = SamplingParams(
    temperature=0.0,
    max_tokens=64,
)

llm = LLM(
    model="facebook/opt-125m",
    generation_config="vllm",
)

outputs = llm.generate(prompts, sampling)

for item in outputs:
    print("prompt:", item.prompt)
    print("output:", item.outputs[0].text)
```

注意：vLLM 默认可能读取模型仓库中的 Hugging Face `generation_config.json`，覆盖你以为的默认生成行为。需要使用 vLLM 默认生成配置时可显式设置 `generation_config="vllm"`；无论如何，Benchmark 都应显式给出关键 Sampling 参数。

离线实验记录：

- 模型标识和 revision；
- vLLM、PyTorch、CUDA、驱动版本；
- dtype、量化与并行配置；
- Sampling 参数；
- 输入/输出 Token 数；
- wall time、峰值显存和预热方式。

## 4. 启动 OpenAI-Compatible 服务

先用一个小模型打通流程，再替换目标模型。

```bash
export MODEL=facebook/opt-125m
export VLLM_API_KEY=local-dev-key

vllm serve "$MODEL" \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype auto \
  --api-key "$VLLM_API_KEY"
```

学习阶段绑定 `127.0.0.1`，避免无意暴露到局域网或公网。启动日志应保存到实验目录，并记录完整命令。

检查模型列表：

```bash
curl http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer $VLLM_API_KEY"
```

发送 Completion 请求：

```bash
curl http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -d '{
    "model": "facebook/opt-125m",
    "prompt": "KV cache is useful because",
    "max_tokens": 64,
    "temperature": 0,
    "stream": false
  }'
```

## 5. Chat Completion 与 Chat Template

`/v1/chat/completions` 接收 role/content 消息，但服务端需要知道如何把消息渲染为模型期望的 Token 序列。不是所有基础模型都自带 Chat Template。

使用聊天接口前确认：

1. 选择 Instruct/Chat 模型；
2. tokenizer 配置包含正确 Chat Template，或显式提供；
3. 使用模型官方推荐的 system/user 格式；
4. 先用确定性小样本检查输出语义。

请求示例：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -d '{
    "model": "your-instruct-model",
    "messages": [
      {"role": "user", "content": "Explain continuous batching briefly."}
    ],
    "max_tokens": 64,
    "temperature": 0,
    "stream": true
  }'
```

Chat Template 不一致会让“服务成功返回 200”但回答质量异常，因此正确性检查必须早于性能测试。

## 6. 关键配置怎么理解

### `--gpu-memory-utilization`

控制引擎计划使用的 GPU 显存比例。调高可能增加 KV Cache 容量，但会压缩运行时余量并提高 OOM 风险。不要只看能否启动，要跑目标长度与并发。

### `--max-model-len`

限制最大上下文。设置过大可能占用更多容量，过小则拒绝长请求。根据真实输入与输出分布设置，而不是机械使用模型理论上限。

### `--max-num-seqs`

限制同时处理的序列数量。增加它可能提高高并发吞吐，也可能增加单轮时延和缓存压力。

### `--max-num-batched-tokens`

限制单次迭代的 Token 预算，影响 Prefill/Decode 混合、吞吐和 ITL。vLLM V1 的 Chunked Prefill 会在可用时优先安排 Decode，再把剩余预算用于 Prefill。

### `--tensor-parallel-size`

把模型切到多张 GPU。它能解决单卡容量或算力问题，但引入跨卡通信。必须与单卡结果在相同工作负载下比较。

### Prefix Caching

共享长前缀明显时再开启并测冷/热缓存。没有命中率说明的“开启后更快”结论不可复现。

## 7. 正确的调参顺序

1. 小并发下验证输出正确、停止条件和 Token 计数。
2. 固定工作负载，建立默认配置基线。
3. 确认显存余量和是否发生抢占/OOM。
4. 一次只调整一个容量或调度参数。
5. 同时观察吞吐、p50/p95/p99 TTFT、ITL、E2E 和失败率。
6. 找到满足 SLO 的容量点，而不是只追求最大 tokens/s。
7. 在固定长度与真实分布上都复测。

## 8. 可观测性

至少收集：

- 请求数、成功/失败/超时；
- running、waiting 请求；
- TTFT、ITL、E2E 分布；
- 输入/输出 Token 总数；
- KV Cache 使用率与抢占；
- GPU 利用率、显存、功耗和温度；
- 服务启动参数、版本和模型 revision。

客户端时间包含网络，服务端指标更接近内部排队与执行。两者都保留，才能定位慢点。

## 9. 安全与生产边界

`--api-key` 并不等于完整的生产安全方案。vLLM 官方文档特别说明，API key 校验主要保护 `/v1`、`/v2` 和 `/inference` 路径，其他端点可能不受同一机制保护。

生产部署应：

- 在反向代理或 API Gateway 后暴露服务；
- 默认不直接公开 vLLM 进程；
- 对所有路径实施鉴权、TLS、限流和请求体上限；
- 限制 `max_tokens`、上下文长度和并发；
- 隔离租户并避免把 Prompt 写入非必要日志；
- 设置健康检查、启动探针和优雅终止；
- 固定镜像、依赖、模型 revision 与配置；
- 对取消、超时、重试和过载做显式策略。

## 10. 常见问题

### 服务启动即 OOM

降低 `gpu_memory_utilization`、`max_model_len` 或并行规模，检查是否有其他进程占显存，并区分权重加载 OOM 与 KV Cache 初始化 OOM。

### 请求提示模型不存在

请求体的 `model` 必须与服务暴露的模型名一致。先查询 `/v1/models`，需要稳定别名时检查当前版本提供的 served-model-name 参数。

### Chat 请求报 Template 错误

确认使用 Chat/Instruct 模型及其 tokenizer 配置，或按模型要求提供 Chat Template。不要用任意基础模型直接验证聊天接口。

### 吞吐上升但 TTFT 恶化

系统可能已接近过载点。降低到达速率或并发，查看 waiting 队列，并重新评估 Token 预算与最大序列数。

### 第一次请求特别慢

可能包含模型初始化、Kernel 编译、CUDA Graph 捕获或缓存冷启动。将冷启动单独报告，稳态实验先预热。

## 11. 本章实验

1. 用离线 API 对两个固定 Prompt 生成结果，保存配置。
2. 启动本地服务，完成 models、completion 与 streaming 请求。
3. 使用仓库的并发客户端跑并发 1/2/4/8/16。
4. 画出吞吐与 p95 TTFT，标出满足 SLO 的最大负载。
5. 修改一个调度参数复测，并解释因果假设。

## 12. 验收清单

- [ ] 能完成离线与在线推理
- [ ] 能解释 Chat Template 为什么影响正确性
- [ ] 能保存版本、模型 revision 与完整启动参数
- [ ] 能解释四个常用容量/调度参数
- [ ] 能区分冷启动与稳态结果
- [ ] 能从吞吐和尾延迟共同判断过载
- [ ] 知道 API key 的保护边界并使用反向代理
- [ ] 能运行并解释一份并发 Benchmark

## 延伸阅读

- [LLM 推理系统概览](../02-llm-inference/inference-overview.md)
- [KV Cache 与调度](../02-llm-inference/kv-cache-and-scheduling.md)
- [LLM 服务 Benchmark](../09-benchmarking/llm-serving-benchmark.md)
- [Week 2 学习计划（GitHub）](https://github.com/LuHeming/ai-infra-from-research-to-production/blob/main/weekly/week-02-llm-inference-and-vllm.md)
