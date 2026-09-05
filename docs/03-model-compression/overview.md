# 大模型压缩概览

模型压缩的工程目标，是在可接受的质量损失下减少存储、显存、计算、带宽或服务成本。不同方法优化的资源不同，因此不能只用“压缩率”评价。

## 1. 先定义优化目标

LLM 推理的主要成本包括：

```text
Checkpoint 存储
+ 模型权重显存
+ KV Cache
+ Activation 与 Workspace
+ HBM 带宽
+ GEMM / Attention 计算
+ 多卡通信
+ 服务排队与调度
```

不同方法直接影响的对象不同：

| 方法 | 直接改变 | 可能改善 | 不会自动改善 |
| --- | --- | --- | --- |
| Weight-only quantization | 权重 bit-width | 文件、权重显存、权重带宽 | KV Cache、激活 |
| W8A8 / FP8 | 权重和激活 | GEMM 与带宽 | 未量化算子 |
| KV Cache quantization | K/V 表示 | 长上下文容量 | 权重大小 |
| 非结构化剪枝 | 非零权重数 | 理论计算量 | Dense Kernel 延迟 |
| 结构化剪枝 | 层、通道、Head | 参数、Shape、FLOPs | 精度保持 |
| 低秩分解 | 矩阵秩 | 参数、理论 FLOPs | Kernel Launch 数 |
| 蒸馏 | 模型架构 | 长期推理成本 | 训练与数据成本 |

目标应写成可验证条件，例如：“在 PPL 增幅不超过 0.2 时，把 TPOT p95 降低 20%”。

## 2. 四层评价框架

### 算法层

检查压缩后模型是否正确：

- Weight/Output MSE；
- Cosine similarity；
- Logit KL divergence；
- PPL；
- 下游 Accuracy、F1、Exact Match；
- 生成质量、安全和领域能力。

### 表示层

检查压缩结果如何保存：

- 真实 dtype 与 logical dtype；
- Packed layout；
- Scale、zero-point、codebook；
- Group size、block size；
- Sparse indices 或 mask；
- 是否同时保留原始权重；
- Checkpoint 能否独立重新加载。

### Kernel 层

检查实际执行：

- PyTorch operator；
- CUDA/Triton Kernel 名；
- 输入、权重、输出 dtype；
- GEMM Shape；
- Dequant、cast、repack；
- CPU fallback；
- Dynamic Shape 与 graph break。

### 系统层

检查最终收益：

- Checkpoint bytes；
- 加载时间和峰值显存；
- Prefill、TTFT；
- Decode TPOT/ITL；
- Tokens/s、并发和 Goodput；
- 功耗；
- 转换、校准与维护成本。

四层必须同时记录。算法误差小但没有对应 Kernel，不能得出部署加速结论。

## 3. 量化术语

- W8A16：8-bit 权重，16-bit 激活；
- W4A16：4-bit 权重，16-bit 激活；
- W8A8：8-bit 权重与激活；
- WOQ：Weight-Only Quantization；
- PTQ：Post-Training Quantization；
- QAT：Quantization-Aware Training。

“16”可能是 FP16 或 BF16；“4”可能是 INT4、NF4、FP4 等格式。报告必须写完整。

## 4. Affine Quantization

浮点到整数：

`q = clamp(round(x / scale) + zero_point, q_min, q_max)`

反量化：

`x_hat = scale × (q - zero_point)`

### 对称量化

通常令 zero_point 为 0：

`scale = max(abs(x)) / q_max`

实现简单，但分布不对称时可能浪费量化范围。

### 非对称量化

`scale = (x_max - x_min) / (q_max - q_min)`

`zero_point = round(q_min - x_min / scale)`

它能覆盖非对称分布，但 zero-point 可能增加执行复杂度。

### 误差来源

- 舍入与饱和；
- Clipping 范围；
- Scale/zero-point 估计；
- Accumulator 精度；
- 重复 quant/dequant；
- 实现使用的整数范围不同。

实验要记录 rounding mode、qmin/qmax、clipping 和 accumulator dtype。

## 5. 量化粒度

| 粒度 | 共享范围 | 优点 | 成本 |
| --- | --- | --- | --- |
| Per-tensor | 整个 Tensor | 元数据最少 | 容易受离群值影响 |
| Per-channel | 每输出通道 | 精度较好 | 每通道 Scale |
| Per-group | 每通道的固定组 | 适应局部分布 | 更多 Scale |
| Per-token | 每个 Token | 适应动态激活 | 运行时量化 |
| Per-block | 二维 Block | 灵活 | Layout 与 Kernel 约束 |

Group 越小通常误差越低，但 Scale 更多，打包和 Kernel 限制也更强。

假设 N 个权重、每个 b bit、每 G 个权重一个 s-byte Scale：

`bytes ≈ N × b / 8 + N / G × s + zero-point 与其他元数据`

因此 INT4 相对 FP16 不一定正好缩小 4 倍。

## 6. Weight-only 与激活量化

### Weight-only

权重离线量化，激活保持 FP16/BF16。它常用于 Decode 权重带宽占主导的场景。

理想执行：

```text
BF16 Activation × Packed INT4 Weight
  → Fused Dequant/Matmul
  → BF16 Output
```

若先完整反量化回 BF16 再调用普通 GEMM，收益可能消失。

### 动态激活量化

运行时为当前 Token/Batch 计算 Scale，适应分布但增加运行时开销。

### 静态激活量化

用校准集提前确定范围。运行时便宜，但对校准域和离群值敏感。

## 7. PTQ 与 QAT

PTQ 在训练结束后压缩，成本较低。典型流程：

1. 加载高精度模型；
2. 选择目标层；
3. 收集权重/激活统计；
4. 选择 Scale、zero-point 和 clipping；
5. 量化与打包；
6. 层级误差检查；
7. PPL/任务评估；
8. Kernel 与系统 Benchmark。

QAT 在训练时插入 Fake Quant，让模型适应舍入和饱和。它通常精度更高，但需要训练数据、算力和稳定配方。训练结束仍要 Convert/Export 到部署表示。

## 8. 常见 LLM PTQ 方法

| 方法 | 目标 | 校准 | 核心思想 | 风险 |
| --- | --- | --- | --- | --- |
| RTN | 快速 Weight-only 基线 | 可选 | Round-to-nearest | 低 bit 误差 |
| GPTQ | W4/W3 | 是 | 近似二阶补偿 | 校准和格式 |
| AWQ | W4 | 是 | 激活识别重要通道并缩放 | 域偏移 |
| SmoothQuant | W8A8 | 是 | 把激活量化困难迁移到权重 | 平滑系数 |
| QAT | 极低 bit | 训练数据 | 训练适应量化噪声 | 成本高 |

算法名不是统一格式。同为 GPTQ 或 AWQ，不同工具的 packing、group 和 Kernel 可能不兼容。

## 9. 离群值与校准

一个极端值可能决定整组 Scale，使普通值只能使用少数量化级别。

常见处理：

- 更小 Group；
- Clipping；
- Percentile 或 Min-MSE Scale；
- 重要通道保护；
- 等价缩放；
- 混合精度；
- 敏感层保持高精度。

校准集要覆盖目标领域、长度和 Prompt 类型。Calibration、Validation 与 Test 必须分开。

## 10. Fake Quant、Packed Storage 与 Kernel

### Fake Quantization

```text
FP Weight → 映射到整数级别 → Dequantize → FP Tensor
```

它模拟误差，但存储和执行可能仍是浮点。

### Packed Storage

多个低 bit 值打包到 Byte/Word，并保存 Scale 和 zero-point。它能降低文件和权重显存。

### Quantized Kernel

Kernel 直接读取 Packed Weight，融合解包、反量化与矩阵乘，或使用硬件低精度指令。

只有第三层能直接支持“低 bit 执行更快”的结论。

## 11. 剪枝与稀疏

### 非结构化剪枝

删除任意权重，通常较容易保持精度，但索引不规则。Magnitude Pruning 删除绝对值最小的权重。

普通 Dense GEMM 不会因为 Tensor 中出现零值而自动跳过计算。

### 结构化剪枝

删除完整通道、Head、Neuron、Block 或 Layer。它能改变 Dense Shape，部署更直接，但精度风险更高。

Transformer 结构剪枝还要保持：

- Hidden Size 与 Head 数整除；
- GQA/MQA 和 RoPE 配置；
- Residual Shape；
- Checkpoint 与 Config 一致；
- Tensor Parallel 可切分。

### N:M 半结构化稀疏

每 M 个元素保留 N 个，如 2:4。要产生加速，必须同时满足：

- Pattern 正确；
- Shape、dtype 和硬件支持；
- 转成 Compressed Sparse Layout；
- 调用 Sparse Kernel；
- 计入 Metadata 成本。

Dense Tensor 每 4 个有 2 个零，不等于 2:4 加速。

## 12. PyTorch 剪枝重参数化

教学剪枝 API 可能保存 weight_orig 与 weight_mask，在 Forward 时计算乘积。这会增加逐元素操作，不是压缩存储。

即使固化零值，也要检查 Tensor Layout 和实际字节数。

## 13. 精度恢复

可用方法：

- 逐层 Reconstruction；
- 小学习率 Fine-tuning；
- QAT；
- 渐进提高压缩率；
- 敏感层保留；
- 蒸馏；
- 联合搜索。

报告恢复训练的数据、步数、学习率、优化器、GPU-hour 和能耗。

## 14. 低秩分解

对 W 进行近似：

`W ≈ A B`

若 W 形状为 m×n，Rank 为 r：

- 原参数量：m×n；
- 低秩参数量：r×(m+n)；
- 原理论乘加：约 2mn；
- 低秩理论乘加：约 2r(m+n)。

只有 r(m+n) 小于 mn 才有理论压缩。执行从一次 GEMM 变成两次 GEMM，可能增加 Launch 和中间 Tensor 成本。

SVD 最小化矩阵重构误差，但不保证最小化模型任务损失。

## 15. 低秩压缩与 LoRA

- 低秩压缩用因子替代原权重；
- LoRA 保留原权重并添加低秩增量；
- 未合并 LoRA 可能增加执行分支；
- 合并后通常恢复原 Dense Shape；
- 只有真实替换或移除原权重才能减少基础模型成本。

## 16. 知识蒸馏

Logit Distillation：

`L_KD = T² × KL(softmax(z_teacher/T), softmax(z_student/T))`

组合任务损失：

`L = alpha × L_task + (1-alpha) × L_KD`

还可以匹配 Hidden State、Attention Map 或 Teacher 生成序列。

必须有同架构 Student 的无蒸馏基线。Teacher 数据生成和训练成本也要记录。

## 17. 组合压缩与消融

常见组合：

- Pruning → Fine-tune → Quantization；
- Distillation → Quantization；
- Structured Pruning → Low-rank；
- SmoothQuant → W8A8；
- SparseGPT → Weight Quantization。

至少保留 Baseline、A、B、A+B 四组，并记录转换顺序。

## 18. 实验工作流

```text
定义精度预算和系统目标
  → 冻结模型、数据、Seed、环境
  → 高精度 Baseline
  → 单层/小模型原型
  → 压缩与表示检查
  → 数值和任务评估
  → 保存/重载一致性
  → Kernel 验证
  → 端到端与并发 Benchmark
  → 部署决策与回滚
```

任一步输出错误就停止，不要继续做无意义的性能测试。

## 19. 评估顺序

正确性：

1. NaN/Inf；
2. Shape、dtype、device；
3. 单层误差；
4. 固定输入 Logits；
5. PPL；
6. 下游任务；
7. 生成和安全样例；
8. 长上下文。

性能：

1. Checkpoint Bytes；
2. 加载后的权重对象；
3. GPU Memory；
4. 单算子 Latency；
5. Operator/Kernel；
6. Prefill/Decode；
7. 完整 Generate；
8. 在线并发；
9. 长时间稳定性。

## 20. 场景选择

- 权重显存主导：优先 Weight-only INT8/INT4；
- KV Cache 主导：权重量化帮助有限，关注 KV dtype 和调度；
- Prefill 计算主导：关注 W8A8/FP8 或结构化 Shape；
- Decode 权重带宽主导：关注 Packed Weight-only Kernel；
- 无稀疏 Kernel：非结构化剪枝不会立即加速；
- 可以训练：考虑 QAT、结构化剪枝与蒸馏。

## 21. 最小报告表

| 配置 | 表示 | 文件 MB | PPL | Peak GB | TTFT p95 | TPOT p95 | tok/s | Kernel |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| BF16 | Dense |  |  |  |  |  |  |  |
| INT8 WO | Packed |  |  |  |  |  |  |  |
| INT4 g128 | Packed + Scale |  |  |  |  |  |  |  |
| 50% Unstructured | Dense/Sparse? |  |  |  |  |  |  |  |
| 2:4 | Compressed |  |  |  |  |  |  |  |
| Low-rank | A+B |  |  |  |  |  |  |  |

## 22. 常见误区

- INT4 级别不等于实际每参数 4 bit；
- 零值不等于 Sparse Kernel；
- Checkpoint 变小不等于运行时显存同比下降；
- FLOPs 减少不等于 Latency 同比下降；
- PPL 稳定不代表所有任务能力不变；
- 只测一个 Batch/Shape 就泛化结论；
- 忽略 Scale、Index、Workspace；
- 把首次编译混入稳态；
- 用 Test Set 调压缩参数；
- 没有高精度回滚版本。

## 23. 验收清单

- [ ] 能从目标资源选择压缩方法
- [ ] 能推导 Affine Quantization
- [ ] 能解释粒度、离群值和校准
- [ ] 能区分 PTQ、QAT、GPTQ、AWQ、SmoothQuant
- [ ] 能区分 Fake Quant、Packed Storage 和低比特 Kernel
- [ ] 能区分三类剪枝和 N:M
- [ ] 能推导低秩参数量
- [ ] 能设计有 Student Baseline 的蒸馏实验
- [ ] 能检查表示、Kernel 与系统收益
- [ ] 能提交可复现且有限定条件的报告

## 继续学习

- [从压缩率到真实加速](compression-deployment-gap.md)
- [PyTorch Profiler](../01-pytorch/profiler.md)
- [Benchmark 方法论](../09-benchmarking/benchmark-methodology.md)
- [Week 3 实践计划（GitHub）](https://github.com/LuHeming/ai-infra-from-research-to-production/blob/main/weekly/week-03-model-compression.md)
