# AI Infra 学习路线图

## 总体目标

用 4 周建立可求职的 AI Infra 基础能力，用 8 周完成一个能够公开展示的压缩部署项目。

## Week 1：工程基础与 PyTorch 性能分析

### 学习内容

- Linux、SSH、tmux、Bash
- Git 分支、PR、rebase 和冲突处理
- Python 包结构、logging、pytest、类型标注
- PyTorch Tensor 存储、stride、显存管理
- 正确的 GPU 计时、warmup 和同步
- PyTorch Profiler 与 trace 分析
- Docker 基础

### 交付物

- 一篇 PyTorch Profiler 正式文档
- 一个 Hugging Face 推理 Benchmark
- 一份规范的环境记录
- 一篇故障排查记录

## Week 2：LLM 推理系统与 vLLM

### 学习内容

- Prefill 与 Decode
- KV Cache 结构和显存估算
- Continuous Batching
- Paged KV Cache
- TTFT、TPOT、吞吐、p50/p95/p99
- Prefix Caching、Chunked Prefill
- vLLM 离线推理和在线服务
- 并发压测

### 交付物

- Transformers 与 vLLM 对比报告
- OpenAI-compatible API 服务
- 并发请求客户端
- 输入/输出长度与性能关系分析

## Week 3：CUDA 与 Triton

### 学习内容

- Grid、Block、Thread、Warp
- Global、Shared、Register Memory
- Memory Coalescing
- Kernel Launch Overhead
- Compute-bound 与 Memory-bound
- CUDA Event、Stream 和异步执行
- Triton Vector Add、Softmax、MatMul
- RMSNorm 与量化/反量化 Kernel

### 交付物

- 一个 Triton RMSNorm
- 一个块级量化或反量化 Kernel
- 多种 shape 下的误差和性能报告

## Week 4：分布式与综合项目

### 学习内容

- DDP、FSDP、ZeRO
- Tensor Parallel、Pipeline Parallel
- NCCL Collectives
- `torchrun`、rank、world size
- GPU 拓扑与通信量分析
- 压缩模型端到端服务 Benchmark

### 交付物

- DDP 最小实验
- TP 矩阵切分示例
- LLM Compression Serving Benchmark v0.1

## Week 5–8：工程深化

- TensorRT-LLM
- Nsight Systems / Nsight Compute
- Docker GPU 镜像
- Slurm
- Kubernetes 基础
- GitHub Actions
- 可观测性和性能回归
- 项目 README、架构图与面试材料

## 每周验收规则

- [ ] 至少 5 篇 daily note
- [ ] 1 篇 weekly review
- [ ] 1–2 篇正式 docs
- [ ] 1 个可运行 lab
- [ ] 1 组可复现 Benchmark
- [ ] 1 条排错记录
- [ ] 所有新增代码通过测试或基本语法检查
