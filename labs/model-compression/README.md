# Model Compression Lab

这个实验用同一组矩阵和输入比较 Baseline、分组对称量化、非结构化剪枝、2:4 剪枝和 SVD 低秩分解。

目标是同时观察：

- 数值误差；
- 逻辑压缩率和实际存储字节；
- 零值比例或保留谱能量；
- Dense 执行路径的延迟；
- 为什么 Fake Quant 或 Dense Zero 不等于生产加速。

实验不下载模型，CPU 可运行。使用 CUDA 时加上 --device cuda 和合适的 dtype。

## 1. 环境检查

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python labs/model-compression/compression_lab.py --help
```

## 2. 建立 Baseline

```bash
python labs/model-compression/compression_lab.py   --method baseline   --rows 512   --cols 512   --batch-size 32   --warmup 5   --iterations 20   --seed 42   --output benchmarks/raw-results/week-03/baseline.json
```

后续实验保持 rows、cols、batch-size、seed、warmup 和 iterations 一致。

## 3. Group-wise Quantization

INT8：

```bash
python labs/model-compression/compression_lab.py   --method quantize   --bits 8   --group-size 128   --seed 42   --output benchmarks/raw-results/week-03/quant-int8.json
```

INT4：

```bash
python labs/model-compression/compression_lab.py   --method quantize   --bits 4   --group-size 128   --seed 42   --output benchmarks/raw-results/week-03/quant-int4.json
```

Group Size 扫描：

```bash
for group in 32 64 128 256; do
  python labs/model-compression/compression_lab.py     --method quantize     --bits 4     --group-size "$group"     --seed 42     --output "benchmarks/raw-results/week-03/quant-g$group.json"
done
```

脚本的 INT4 会把两个 signed nibble 打包进一个 Byte，用于计算物理存储量；执行前会反量化为 Dense 浮点权重。因此：

- Artifact Bytes 能展示打包后的理论表示；
- Candidate Latency 不能当作 INT4 Kernel 性能；
- 这个差异正是部署鸿沟实验的一部分。

## 4. 离群值实验

```bash
python labs/model-compression/compression_lab.py   --method quantize   --bits 4   --group-size 128   --outlier-scale 20   --seed 42   --output benchmarks/raw-results/week-03/quant-outliers.json
```

与无离群值版本比较 relative_weight_mse 和 relative_output_mse。

## 5. 非结构化剪枝

```bash
python labs/model-compression/compression_lab.py   --method unstructured   --sparsity 0.5   --seed 42   --output benchmarks/raw-results/week-03/prune-50.json
```

该路径把小幅值权重置零，但仍以 Dense Tensor 运行。重点验证：

- actual_zero_fraction 接近目标值；
- materialized_execution_weight_bytes 没有下降；
- 普通 Dense Linear 不保证加速；
- 精度随 Sparsity 变化。

## 6. 2:4 半结构化剪枝

```bash
python labs/model-compression/compression_lab.py   --method 2to4   --cols 512   --seed 42   --output benchmarks/raw-results/week-03/prune-2to4.json
```

脚本保证每连续 4 个值只保留 2 个，但仍使用 Dense Tensor。若要测硬件加速，需要进一步转成后端支持的 2:4 Compressed Layout 并确认 Sparse Kernel。

## 7. Low-rank SVD

```bash
for rank in 32 64 128 256; do
  python labs/model-compression/compression_lab.py     --method low-rank     --rank "$rank"     --seed 42     --output "benchmarks/raw-results/week-03/low-rank-$rank.json"
done
```

Low-rank 路径实际执行两个 Linear。比较：

- retained_spectral_energy；
- 因子 Artifact Bytes；
- Relative Output MSE；
- 两次 GEMM 延迟；
- Rank 提高后的收益拐点。

## 8. CUDA 路线

```bash
python labs/model-compression/compression_lab.py   --method quantize   --bits 4   --group-size 128   --rows 2048   --cols 2048   --batch-size 32   --device cuda   --dtype float16   --warmup 20   --iterations 100   --output benchmarks/raw-results/week-03/cuda-quant-int4.json
```

注意该命令仍不是 INT4 Kernel。它用于观察相同 Quantize-Dequantize 权重在 GPU Dense 路径上的误差和延迟。

真实低比特实验应使用 torchao 或目标推理引擎，并用 Profiler 验证 Kernel。

## 9. JSON 字段

- environment：Python、PyTorch、Device、dtype；
- config：方法、Shape、Batch、Seed、Warmup；
- representation：Dense/Artifact Bytes、零值、Scale、Padding、执行路径；
- correctness：Weight/Output Relative MSE、Cosine、Max Error、Finite；
- performance：Baseline/Candidate 的 mean、median、p95、min/max；
- warning：对当前执行路径的限制说明。

## 10. 如何写结论

不合格：

> INT4 压缩 4 倍且速度更快。

合格：

> 在 512×512 Weight、Batch 32、CPU FP32 下，Group-wise INT4 g128 的 Packed Weight 与 Scale 理论占用为 __ Bytes，Relative Output MSE 为 __。本 Lab 执行前将权重恢复为 FP32 Dense Tensor，Profiler/代码路径不包含 INT4 Matmul，因此 Candidate Latency 不能证明真实 INT4 加速。

## 11. 验收清单

- [ ] 相同 Shape/Seed 下完成 Baseline
- [ ] 完成 INT8 与 INT4
- [ ] 扫描至少 3 个 Group Size
- [ ] 完成离群值对照
- [ ] 完成至少 3 个 Sparsity
- [ ] 验证 2:4 Pattern
- [ ] 扫描至少 3 个 Rank
- [ ] 能解释 Artifact 与 Execution Weight Bytes
- [ ] 不把 Dense 路径延迟当作低比特/稀疏 Kernel
- [ ] 原始 JSON 可追溯且未手工修改
