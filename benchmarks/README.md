# Benchmarks

## 目录约定

```text
benchmarks/
├── configs/              # 固定实验配置
├── raw-results/          # 原始 JSON/CSV，不手工修改
├── processed-results/    # 汇总后的表格
├── reports/              # 结果分析
└── schema/               # 结果格式约束
```

## 文件命名

```text
<model>_<method>_<date>_<run-id>.json
```

例如：

```text
opt-125m_fp16_2026-08-05_run01.json
```

## 最低报告要求

- 环境
- 模型
- 精度与量化/剪枝配置
- batch size / concurrency
- input length / output length
- warmup / repeat
- latency mean / p50 / p95
- throughput
- peak memory
## Week 3：Model Compression

- [实验配置模板](configs/week-03-compression.yaml)
- [报告模板](reports/week-03-compression-report.md)
- 原始结果统一写入 raw-results/week-03/，不要手工修改。
- 报告必须同时包含正确性、表示格式、Kernel 证据和系统性能。
