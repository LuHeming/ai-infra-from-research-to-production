# AI Infra：从研究到工程落地

本项目记录一条以大模型压缩为起点的 AI Infra 学习路线。

## 为什么建立这个仓库

只完成 PTQ 或剪枝算法并不等于获得真实推理收益。企业工程通常还需要回答：

1. 压缩后的权重格式能否被推理引擎使用？
2. Kernel 是否真正减少显存访问或计算量？
3. TTFT、TPOT、吞吐和显存是否改善？
4. 多 GPU 环境下，通信是否抵消了计算收益？
5. 实验能否在固定环境中复现？
6. 服务在并发请求下是否稳定？

本仓库将围绕这些问题组织知识、代码和实验。

## 推荐阅读顺序

1. [Linux 工程基础](00-foundations/linux-fundamentals.md)
2. [Python 工程基础](00-foundations/python-engineering.md)
3. [开发环境与实验规范](00-foundations/development-environment.md)
4. [PyTorch 性能分析](01-pytorch/profiler.md)
5. [LLM Prefill、Decode 与 KV Cache](02-llm-inference/inference-overview.md)
6. PTQ/PTP 与部署鸿沟
7. CUDA/Triton Kernel
8. [vLLM 推理服务](07-serving/index.md)
9. 分布式与 NCCL
10. Docker、Slurm、Kubernetes
11. [LLM 服务 Benchmark](09-benchmarking/llm-serving-benchmark.md)
