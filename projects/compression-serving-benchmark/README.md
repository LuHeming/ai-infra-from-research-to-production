# LLM Compression Serving Benchmark

## 项目目标

建立从压缩算法到真实推理服务的统一评估链路。

## 计划流程

```text
Base Model
  → PTQ / PTP
  → Accuracy Evaluation
  → Transformers Baseline
  → vLLM / TensorRT-LLM
  → Profiler
  → Concurrent Benchmark
  → Docker
  → Reproducible Report
```

## 计划对比

- FP16 / BF16
- W8A16 / W4A16
- 非结构化剪枝
- 2:4 半结构化剪枝
- Transformers
- vLLM
- TensorRT-LLM

## 核心指标

- WikiText-2 / C4 PPL
- Zero-shot Accuracy
- Model Size
- TTFT
- TPOT
- p50 / p95 / p99
- Tokens/s
- Requests/s
- Peak GPU Memory
- 实际执行 Kernel

## 里程碑

- [ ] v0.1：OPT-125M Transformers 基线
- [ ] v0.2：压缩精度对比
- [ ] v0.3：vLLM 服务 Benchmark
- [ ] v0.4：Triton 量化 Kernel
- [ ] v0.5：多 GPU 与 TensorRT-LLM
