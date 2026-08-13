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
