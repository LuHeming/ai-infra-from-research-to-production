# 从压缩率到真实加速：部署鸿沟

参数量下降、零值增多或理论 FLOPs 降低，并不自动产生真实加速。压缩算法只有被表示格式、推理后端、Kernel、硬件和真实工作负载共同支持，才会转化为系统收益。

## 1. 五种不同的“变小”

| 说法 | 含义 | 证据 |
| --- | --- | --- |
| 参数更少 | 数学自由度下降 | 参数计数 |
| 逻辑 Bit 更低 | 量化级别更少 | 整数范围和 Scale |
| 文件更小 | 序列化字节减少 | Checkpoint Bytes |
| 显存更少 | 运行时常驻或峰值降低 | Allocated、Reserved、进程显存 |
| 推理更快 | 目标负载改善 | TTFT、TPOT、Tokens/s |

这五者可能同时发生，也可能只发生其中一部分。

## 2. 真实加速链

算法先产生可接受误差，然后导出后端支持的表示；表示需要匹配硬件与 Kernel；Kernel 还必须在目标 Shape 下有优势；最后调度、网络和通信不能抵消收益。

链上的任何一环断开，都会出现“压缩率很好但部署没加速”。

## 3. Roofline 视角

性能上界可粗略理解为：

Performance ≤ min(Peak Compute, Memory Bandwidth × Arithmetic Intensity)

### Compute-bound

计算单元接近饱和。减少有效乘加或使用更高吞吐低精度指令可能有效。

### Memory-bound

权重或 KV Cache 读取占主导。Weight-only Quantization 可以通过减少权重字节获益。

### Overhead-bound

矩阵太小、Kernel 太碎或 Launch/Python/调度占主导。理论 FLOPs 和字节下降可能看不到收益。

Prefill、Decode、不同 Batch 和长度的瓶颈可能完全不同。

## 4. 量化为什么可能不加速

### Fake Quant 仍是浮点

Fake Quant 只模拟舍入误差。若权重最后恢复为 FP16/BF16 并执行 Dense GEMM，存储和执行路径没有真正低比特化。

### INT4 使用 INT8 容器

如果每个 4-bit 值占一个 Int8 元素，物理存储仍是一字节。接近理论大小需要 Nibble Packing，并计入 Scale/Zero-point。

### 在线完整反量化

每次 Forward 都把 Packed Weight 完整恢复为 BF16，会增加读取、写入和 Workspace。高效路径通常融合解包、反量化和 Matmul。

### Group Size 与 Shape 不匹配

小 Group 精度可能更好，但 Scale 更多、Padding 增加、Tile 不兼容或 Repack 成本更高。

### 其他内存主导

Weight-only 不压 Activation 和 KV Cache。长上下文或高并发时，权重变小不代表峰值显存同比下降。

## 5. 剪枝为什么可能不加速

### Dense Zero 不会自动跳过

普通 Dense GEMM 仍按完整 Shape 读取和计算。50% 元素为零时，文件和延迟都可能不变。

### 稀疏格式有额外元数据

CSR、CSC、Block Sparse 和 N:M 需要 Index 或 Mask。低稀疏率、小矩阵或低 Bit Value 时，元数据与不规则访存会抵消收益。

### Pattern 与硬件不匹配

非结构化稀疏灵活但难加速；2:4 等模式受限，却更易映射专用指令。

必须共同确认 Pattern、Layout、Dtype、Shape、硬件、Sparse Operator 和 Kernel。

### Reparameterization 增加操作

若 Forward 仍计算 WeightOrig × Mask，剪枝会多一个逐元素操作。固化零值仍不等于压缩布局。

## 6. 低秩为什么可能不加速

低秩把一次 GEMM 变成两次 GEMM，可能增加：

- 两次 Kernel Launch；
- 中间 Tensor 读写；
- 不适合 Tensor Core 的 Rank；
- 小 Batch 下的低利用率；
- Graph 中无法融合的节点。

必须联合扫描 Rank、Batch 和 Shape。

## 7. 蒸馏的隐性成本

Student 往往更容易获得真实推理收益，但仍要检查：

- Tokenizer 与词表；
- 架构是否适合后端；
- Teacher 数据生成成本；
- 蒸馏训练 GPU-hour；
- Student 是否需要更长输出；
- 真实任务质量是否保持。

蒸馏降低长期推理成本，却会增加一次性训练和数据成本。

## 8. 多卡与 Amdahl 定律

压缩计算后，通信占比可能升高。检查 Tensor Parallel All-Reduce、Pipeline Bubble、Scale/Metadata 复制、通信 Dtype、拓扑和最慢 Rank。

若目标 GEMM 原本占总时间 60%，即使它加速 2 倍，端到端加速上限约为：

1 / (0.4 + 0.6 / 2) = 1.43 倍

不能把 Kernel 加速倍数直接当端到端倍数。

## 9. 服务层抵消因素

即使单算子更快，在线服务仍可能不变或变差：

- Tokenization 或网络主导 TTFT；
- 排队主导尾延迟；
- 量化格式限制 Batch；
- Dequant Workspace 挤压 KV Cache；
- Kernel 对动态 Shape 不稳定；
- 首次请求包含 JIT/Autotune；
- 错误重试增加负载；
- 多租户产生缓存抖动。

所以要分别测 Microbenchmark、Offline Generate 和 Online Serving。

## 10. 必答问题

1. 压缩后的文件格式是什么？
2. 权重对象的 Dtype、Layout、Type 是什么？
3. Scale、Zero-point、Index 占多少？
4. 加载时是否反量化或 Repack？
5. Forward 调用了哪个 Operator 和 Kernel？
6. 是否减少 HBM 读取或有效计算？
7. 是否改变 GEMM Shape或增加 Kernel？
8. 优化覆盖 Prefill、Decode 还是二者？
9. 端到端延迟、吞吐、显存和容量是否改善？
10. 结论适用于哪些硬件、Shape、后端和版本？

## 11. 三层 Benchmark

### Microbenchmark

隔离 Linear、Matmul 或 Sparse Operator，固定 M/N/K、Dtype、Layout、Warmup、同步、重复和 Compile 状态。

输出 Median/P95 Latency、带宽、FLOPS/TOPS、Kernel 和 Peak Memory。

### Model Benchmark

测完整 Forward/Generate，分开 Prefill 与 Decode，扫描 Batch、输入/输出长度，并区分首次与稳态。

### Serving Benchmark

测 TTFT、ITL、TPOT、E2E、Requests/s、Tokens/s、P50/P95/P99、Goodput、KV Cache、失败和抢占。

三层结果不能混成一个“速度”。

## 12. 最关键的对照

| 组 | 表示 | 执行 | 目的 |
| --- | --- | --- | --- |
| A | BF16 Dense | BF16 Kernel | 基线 |
| B | Fake INT4 | BF16 Dense | 只看数值误差 |
| C | Packed INT4 | Quantized Kernel | 真实量化 |
| D | 50% Dense Zero | Dense Kernel | 稀疏反例 |
| E | 2:4 Compressed | Sparse Kernel | 真实稀疏 |
| F | Low-rank A+B | Two GEMMs | 低秩路径 |

B 对 C、D 对 E 是理解部署鸿沟的核心。

## 13. 无加速排查顺序

1. 确认输出正确、无 NaN。
2. 确认不是 Fake Quant 或 Dense Zero。
3. 确认加载时没有恢复高精度。
4. 确认目标 Operator/Kernel 执行。
5. 确认 Shape、Tile、Alignment、Group。
6. 确认 Warmup、同步和统计方法。
7. 判断 Compute、Bandwidth 或 Overhead 瓶颈。
8. 扩大工作量，排除小 Shape 噪声。
9. 用 Amdahl 估算端到端上限。
10. 检查队列、KV Cache、通信和客户端。

## 14. Profiler 证据

压缩前后分别保存：

- Top CUDA Operators；
- Kernel Timeline；
- Matmul Shape；
- Dequant/Cast；
- Memcpy；
- Kernel Calls；
- GPU Idle Gap；
- Allocated/Reserved Peak。

结论要写清设备、Shape、Batch、Kernel、额外开销和端到端变化。

## 15. 结果表

| 方法 | 理论压缩 | 实际文件 | PPL | TTFT | TPOT | Tokens/s | Peak Memory | Kernel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| BF16 | 1× |  |  |  |  |  |  |  |
| Fake W4 | 4× Logical |  |  |  |  |  |  | Dense? |
| Packed W4 |  |  |  |  |  |  |  | INT4? |
| 50% Zero | 2× Logical |  |  |  |  |  |  | Dense? |
| 2:4 |  |  |  |  |  |  |  | Sparse? |
| Low-rank |  |  |  |  |  |  |  | Two GEMMs |

## 16. 部署决策门

- [ ] 精度在预算内
- [ ] Checkpoint 可保存和独立重载
- [ ] 推理引擎支持格式
- [ ] 目标硬件有对应 Kernel
- [ ] 相同工作负载下收益超过噪声
- [ ] 尾延迟和失败率可接受
- [ ] 校准数据合规且无 Test 泄漏
- [ ] 有高精度回滚模型
- [ ] 转换流程和版本可复现
- [ ] 监控能区分模型配置

## 17. 推荐实验

1. Fake INT4 与 Packed INT4 的大小和执行路径；
2. Dense Zero 与 2:4 Compressed 的 Kernel；
3. Weight-only 下短输入长输出；
4. 长上下文下 KV Cache 是否主导；
5. Low-rank 在 Batch 1/8/32/128；
6. 单卡与 Tensor Parallel 通信占比；
7. 一个“压缩成功但没加速”的完整反例。

## 18. 验收清单

- [ ] 能解释五种“变小”
- [ ] 能用 Roofline 选择方向
- [ ] 能用 Amdahl 估计端到端上限
- [ ] 能识别 Fake Quant 和 Dense Zero
- [ ] 能确认 Packed/Sparse Layout 与 Kernel
- [ ] 能设计三层 Benchmark
- [ ] 能按顺序排查无加速
- [ ] 能给出有限定条件的部署结论

## 延伸阅读

- [模型压缩概览](overview.md)
- [PyTorch Profiler](../01-pytorch/profiler.md)
- [LLM 推理系统概览](../02-llm-inference/inference-overview.md)
- [LLM 服务 Benchmark](../09-benchmarking/llm-serving-benchmark.md)
