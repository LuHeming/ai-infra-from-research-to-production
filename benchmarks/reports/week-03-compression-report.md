# Week 3 Model Compression Report

> 本文件是待填写模板，不包含虚构实验结果。所有数字必须能追溯到 raw-results、环境快照和 Git commit。

## 1. 问题与假设

- 目标瓶颈：
- 候选压缩方法：
- 精度预算：
- 性能目标：
- 可证伪假设：

## 2. 环境

| 项目 | 值 |
| --- | --- |
| Git commit |  |
| Model / revision |  |
| Tokenizer revision |  |
| GPU / CPU |  |
| PyTorch / CUDA / Driver |  |
| Compression library |  |
| Inference backend |  |

## 3. 数据

| 用途 | Dataset / revision | Samples | Length | Seed | 备注 |
| --- | --- | ---: | ---: | ---: | --- |
| Calibration |  |  |  |  |  |
| Validation |  |  |  |  |  |
| Test |  |  |  |  |  |

说明 Calibration、Validation、Test 是否有重叠，以及预处理、截断、Padding 和 Label Mask。

## 4. 配置

| ID | Method | Bits/Dtype | Granularity | Group | Sparsity/Pattern | Rank | Excluded Layers |
| --- | --- | --- | --- | ---: | --- | ---: | --- |
| baseline | BF16 |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |

## 5. 表示检查

| ID | Weight Type | Dtype | Layout | Data Bytes | Metadata Bytes | Reload Passed |
| --- | --- | --- | --- | ---: | ---: | --- |
| baseline |  |  |  |  |  |  |
|  |  |  |  |  |  |  |

回答：

- 是否为 Fake Quant？
- 是否真实 Packed？
- 是否同时保存原始权重？
- 加载时是否 Repack 或 Dequantize？
- Scale、Zero-point、Mask、Index 占多少？

## 6. 正确性

| ID | Weight Rel-MSE | Output Rel-MSE | Output Cosine | PPL | Task Accuracy | Finite |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 0 | 0 | 1 |  |  | yes |
|  |  |  |  |  |  |  |

记录 Golden Input、Prompt 和失败样例的位置。

## 7. Microbenchmark

| ID | Shape M/N/K | Batch | Median ms | P95 ms | Speedup | Operator/Kernel |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| baseline |  |  |  |  | 1× |  |
|  |  |  |  |  |  |  |

说明 Warmup、同步、迭代次数、首次/稳态和 Compile 状态。

## 8. 模型与服务性能

| ID | File MB | Peak GB | TTFT P50/P95 | TPOT P50/P95 | Output tok/s | Success |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| baseline |  |  |  |  |  |  |
|  |  |  |  |  |  |  |

## 9. Profiler 证据

- Baseline Trace：
- Compressed Trace：
- 主要 Operator：
- 实际 Kernel：
- Dequant/Cast/Repack：
- Memcpy：
- GPU Idle：
- 瓶颈从什么变成什么：

## 10. 失败或无加速反例

- 现象：
- 初始假设：
- 排除项：
- Profiler 证据：
- 根因：
- 下一实验：

## 11. 有效性威胁

- 硬件限制：
- Shape/Batch 限制：
- 数据代表性：
- 测量抖动：
- 库版本/API：
- 尚未评估的任务：
- 是否可推广到更大模型：

## 12. 部署决定

- [ ] 精度在预算内
- [ ] Checkpoint 可独立重载
- [ ] 后端支持格式
- [ ] Kernel 已确认
- [ ] 收益超过噪声
- [ ] 尾延迟和失败率可接受
- [ ] 有高精度回滚

结论：采用 / 暂缓 / 拒绝

适用范围：

回滚方案：

## 13. 原始材料

- Config：
- Raw Results：
- Environment Snapshot：
- Profiler Trace：
- Reproduction Command：
