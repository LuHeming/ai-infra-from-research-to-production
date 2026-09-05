# Week 3：Model Compression 从算法到真实部署

本周目标不是“把权重变小”这么简单，而是建立一条完整、可验证的压缩链路：

```text
高精度基线
  → 明确压缩目标
  → 选择量化 / 剪枝 / 低秩 / 蒸馏方法
  → 生成真实压缩表示
  → 检查数值误差与任务精度
  → 确认推理后端和 Kernel
  → 测量模型大小、显存、延迟与吞吐
  → 判断压缩是否产生真实系统收益
```

完成本周后，你不仅要能解释 W4A16、group-wise quantization 和 2:4 sparsity，还要能回答：

- 压缩后的参数到底以什么 dtype 和布局保存？
- Scale、zero-point、稀疏 mask 和低秩因子占多少额外空间？
- 推理时执行的是低比特/稀疏 Kernel，还是先反量化回浮点？
- 精度损失来自离群值、校准集、量化粒度，还是压缩率过高？
- 模型文件变小后，峰值显存、TTFT、TPOT 和 tokens/s 是否真的改善？
- 如果没有加速，瓶颈在哪一层？

## 0. 本周范围与时间预算

建议每天投入 2～3 小时：

| 环节 | 每日建议时间 | 产出 |
| --- | ---: | --- |
| 阅读与推导 | 30～45 分钟 | 公式、概念图、问题清单 |
| 编码与实验 | 60～90 分钟 | 命令、原始 JSON、Profiler 证据 |
| 分析与复盘 | 30～45 分钟 | 结论、失败原因、下一步 |

本周主线：

1. Weight-only PTQ：INT8、INT4、per-channel、group-wise；
2. 剪枝：非结构化、结构化、N:M 半结构化；
3. 低秩分解：SVD 与秩—误差—计算量关系；
4. 知识蒸馏：Teacher/Student、logit 与 hidden-state distillation；
5. 算法精度、表示格式、Kernel 和系统指标的四层验证。

QAT、GPTQ、AWQ、SmoothQuant、SparseGPT 等方法本周以“理解假设、识别适用场景、设计复现实验”为目标，不要求从零实现完整论文。

## 1. 前置条件

- 已完成 Week 1 的 Python 工程、GPU 正确计时和 PyTorch Profiler；
- 已完成 Week 2 的 Prefill/Decode、KV Cache 和服务 Benchmark；
- Python 3.10+；
- 已安装 PyTorch；
- GPU 可选：基础实验支持 CPU，真实低比特 Kernel 与服务 Benchmark 需要兼容 GPU；
- 至少预留 5 GB 磁盘；使用真实 LLM 时按模型大小额外准备空间。

确认仓库状态：

```bash
git status --short
python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

进入仓库后创建结果目录：

```bash
mkdir -p benchmarks/raw-results/week-03
mkdir -p benchmarks/reports
mkdir -p notes/week-03
```

## 2. 两条实践路线

### 路线 A：无 GPU / 小显存

使用仓库自带的矩阵实验，理解表示、误差和性能边界：

```bash
python labs/model-compression/compression_lab.py --help
```

它不会下载模型，可以完成：

- group-wise INT8/INT4 quantize-dequantize；
- magnitude unstructured pruning；
- 2:4 semi-structured pruning；
- SVD low-rank approximation；
- 重构误差、余弦相似度、理论存储量与 dense 执行延迟对比。

### 路线 B：有兼容 GPU

在路线 A 基础上增加：

- torchao 真实 weight-only 量化；
- Hugging Face 小模型 PPL/生成质量测试；
- PyTorch Profiler 验证实际 Kernel；
- Transformers 或 vLLM 端到端推理 Benchmark。

先完成 A 再做 B。直接调用第三方一键量化而不了解表示和 Kernel，容易得到无法解释的数字。

## 3. 模型与数据选择

### 3.1 第一次实验

优先选小模型，目的是跑通方法和记录链路：

| 用途 | 建议 | 原因 |
| --- | --- | --- |
| Completion 基线 | `facebook/opt-125m` | 小、容易加载、与项目里程碑一致 |
| Chat 扩展 | 可在显存允许时选小型 Instruct 模型 | 可测试 Chat Template 和生成质量 |
| 无下载实验 | 仓库矩阵 Lab | CPU 可运行、变量完全可控 |

模型名称只是示例。正式实验必须记录 repository id、commit/revision、tokenizer revision、dtype、remote code 设置、许可证和访问条件。

### 3.2 校准数据与评估数据必须分开

```text
Calibration set：估计 scale、zero-point、激活范围或重要性
Validation set：调参和选择配置
Test set：最终一次无偏评估
```

不能在最终测试集上选择 group size、clip ratio 或 sparsity，否则会产生数据泄漏。

建议记录：

| 字段 | 示例 |
| --- | --- |
| calibration dataset | WikiText-2 train 子集 |
| sample count | 128 |
| sequence length | 512 |
| shuffle seed | 42 |
| preprocessing hash | Git commit / 脚本版本 |
| evaluation dataset | WikiText-2 test |
| overlap check | 无重复样本 |

## 4. 本周交付物

```text
weekly/week-03-model-compression.md
docs/03-model-compression/
├── overview.md
└── compression-deployment-gap.md
labs/model-compression/
├── README.md
└── compression_lab.py
tests/test_compression_lab.py
benchmarks/raw-results/week-03/
├── quant-int8.json
├── quant-int4-g128.json
├── prune-50.json
├── prune-2to4.json
└── low-rank.json
benchmarks/reports/week-03-compression-report.md
```

最低交付标准：

- 一张压缩方法选择表；
- 一份 FP32/FP16 或 BF16 基线；
- 至少两种量化配置；
- 一种非结构化剪枝和一种 2:4 剪枝；
- 一种低秩配置；
- 一张“压缩率—误差—延迟”对比表；
- 一段 Profiler 或算子证据，说明是否调用目标 Kernel；
- 一份明确写出失败案例和部署限制的周报。

## 5. 全周实验纪律

任何对比都固定：

- 相同模型与 revision；
- 相同输入 Tensor / Token 序列；
- 相同 batch、sequence length 与输出长度；
- 相同设备、功耗模式和后台负载；
- 相同 warmup、重复次数和同步方式；
- 相同精度评估数据；
- 相同推理边界：纯 Linear、单步 forward、完整 generate 或在线服务；
- 每次只改变一个压缩变量。

每组实验保存：

```yaml
run_id: week3-int4-g128-seed42
git_commit: "<git rev-parse HEAD>"
timestamp: "<ISO-8601>"
method: groupwise_weight_only
weight_bits: 4
activation_dtype: bf16
group_size: 128
calibration:
  dataset: "<name>"
  samples: 128
  sequence_length: 512
  seed: 42
execution:
  backend: "<eager/torch.compile/vllm/...>"
  kernel: "<observed operator/kernel>"
  batch_size: 1
  input_length: 512
  output_length: 128
metrics:
  model_bytes: null
  peak_memory_bytes: null
  ppl: null
  ttft_ms_p95: null
  tpot_ms_p95: null
  output_tokens_per_second: null
```

---

# Day 1：建立基线与压缩心智模型

## 学习目标

- 区分参数量、模型文件大小、运行时显存、FLOPs 与延迟；
- 建立算法层、表示层、Kernel 层、系统层四层评价；
- 得到一份不可被后续实验悄悄改变的高精度基线。

## 1.1 必读内容

- [模型压缩概览](../docs/03-model-compression/overview.md)
- [从压缩率到真实加速](../docs/03-model-compression/compression-deployment-gap.md)
- [通用 Benchmark 方法论](../docs/09-benchmarking/benchmark-methodology.md)

## 1.2 环境快照

```bash
{
  date --iso-8601=seconds
  uname -a
  python --version
  python -m pip freeze
  git rev-parse HEAD
  nvidia-smi || true
} > notes/week-03/environment.txt
```

不要只保存 `requirements.txt`。CUDA Driver、GPU 型号、PyTorch 构建版本和模型 revision 都可能改变结果。

## 1.3 四层评价表

在 `notes/week-03/day-01.md` 填写：

| 层级 | 必答问题 | 可观测证据 |
| --- | --- | --- |
| 算法 | 输出误差和任务精度变化多少？ | MSE、cosine、PPL、Accuracy |
| 表示 | 权重实际如何保存？ | dtype、shape、scale、zero-point、索引 |
| Kernel | 执行了什么算子？ | Profiler operator、CUDA Kernel |
| 系统 | 用户体验和容量是否改善？ | 文件大小、显存、TTFT、TPOT、吞吐 |

## 1.4 建立矩阵基线

```bash
python labs/model-compression/compression_lab.py   --method baseline   --rows 1024   --cols 1024   --batch-size 32   --warmup 10   --iterations 50   --seed 42   --output benchmarks/raw-results/week-03/baseline.json
```

预期看到原始权重元素数、dense 存储字节数、baseline latency、device/dtype/shape/seed，且误差为 0、cosine 接近 1。

如果 CPU 运行太慢，把 rows/cols 降到 512；后续所有配置必须使用相同 shape。

## 1.5 当日思考题

1. 参数量减半是否必然使延迟减半？
2. 权重从 FP16 变 INT4，理论大小为什么不一定正好缩小 4 倍？
3. 为什么 Decode 更可能从 weight-only quantization 获益？
4. 为什么小矩阵上低比特 Kernel 可能比 FP16 更慢？

## Day 1 验收

- [ ] 已保存环境快照和 Git commit
- [ ] 已生成 baseline JSON
- [ ] 能解释四层评价
- [ ] 能区分逻辑压缩率与物理文件大小
- [ ] 写出本周一个可证伪假设

示例假设：

> 在同一 GPU、同一模型和 batch=1 Decode 场景中，真实 packed INT4 weight-only Kernel 会降低权重读取量和 TPOT；若仅做 fake quantize-dequantize，则误差会变化但模型大小与延迟不会按 4 bit 理论值改善。

---

# Day 2：量化基础与手工 Group-wise PTQ

## 学习目标

- 理解 affine quantization；
- 区分 symmetric/asymmetric、per-tensor/per-channel/per-group；
- 明白 fake quantization、packed storage 和低比特 Kernel 是三件事；
- 实际比较 bits 与 group size。

## 2.1 核心公式

浮点值 `x` 映射到整数 `q`：

`q = clamp(round(x / scale) + zero_point, q_min, q_max)`

反量化：

`x_hat = scale × (q - zero_point)`

对称量化通常令 `zero_point = 0`：

`scale = max(abs(x)) / q_max`

INT4 对称量化常用 `[-7, 7]` 或实现定义的其他范围。报告必须写清范围，不能只写“INT4”。

## 2.2 粒度权衡

| 粒度 | Scale 数量 | 精度 | 元数据 | 常见用途 |
| --- | ---: | --- | --- | --- |
| per-tensor | 1 | 较低 | 最少 | 教学、简单激活 |
| per-channel | 每输出通道 1 个 | 较高 | 较少 | INT8 weight-only |
| per-group | 每通道每组 1 个 | 通常更高 | 更多 | INT4 weight-only |
| per-token | 每 Token 动态计算 | 适应激活变化 | 运行时开销 | 动态激活量化 |

Group 越小通常越能适应局部分布，但 scale 更多，打包与 Kernel 约束也更强。

## 2.3 运行 INT8 与 INT4

```bash
python labs/model-compression/compression_lab.py   --method quantize   --bits 8   --group-size 128   --rows 1024 --cols 1024   --batch-size 32   --seed 42   --output benchmarks/raw-results/week-03/quant-int8-g128.json

python labs/model-compression/compression_lab.py   --method quantize   --bits 4   --group-size 128   --rows 1024 --cols 1024   --batch-size 32   --seed 42   --output benchmarks/raw-results/week-03/quant-int4-g128.json
```

再做 group size 扫描：

```bash
for group in 32 64 128 256; do
  python labs/model-compression/compression_lab.py     --method quantize     --bits 4     --group-size "$group"     --rows 1024 --cols 1024     --batch-size 32     --seed 42     --output "benchmarks/raw-results/week-03/quant-int4-g$group.json"
done
```

## 2.4 结果解释

重点比较：

- relative weight MSE；
- output MSE 和 cosine similarity；
- packed weight bytes；
- scale metadata bytes；
- materialized execution weight bytes；
- baseline 与 compressed-path latency。

仓库 Lab 使用 quantize-dequantize 来教学数值误差，不声称提供生产低比特 Kernel。如果执行前恢复为浮点 dense weight，延迟不应被解释成 INT4 Kernel 性能。

## 2.5 离群值实验

使用 `--outlier-scale 20` 人为制造少量大值：

```bash
python labs/model-compression/compression_lab.py   --method quantize   --bits 4   --group-size 128   --outlier-scale 20   --seed 42   --output benchmarks/raw-results/week-03/quant-int4-outliers.json
```

解释为什么一个极端值会扩大 scale，使同组普通值落到更少的有效量化级别。

## Day 2 验收

- [ ] 能推导 quantize/dequantize 公式
- [ ] 完成 INT8、INT4 和至少 3 个 group size
- [ ] 能解释 scale 元数据为何影响实际压缩率
- [ ] 能区分 fake quant、packed weight 与量化 Kernel
- [ ] 能用离群值解释误差变化

---

# Day 3：真实 Weight-only PTQ 与方法谱系

## 学习目标

- 理解 RTN、GPTQ、AWQ、SmoothQuant 的核心差异；
- 用 torchao 尝试真实量化表示；
- 检查硬件、shape 和 Kernel 是否支持目标配置。

## 3.1 方法选择表

| 方法 | 主要目标 | 是否需要校准 | 核心思想 | 常见风险 |
| --- | --- | --- | --- | --- |
| RTN | 快速基线 | 可不需要 | 直接 round-to-nearest | 对离群值敏感 |
| GPTQ | 低比特 weight-only | 需要 | 近似二阶信息逐层补偿误差 | 校准成本、后端格式 |
| AWQ | 低比特 weight-only | 需要 | 用激活识别重要通道并缩放保护 | 校准域偏移 |
| SmoothQuant | W8A8 | 需要 | 把激活离群困难迁移到权重 | 平滑系数与后端支持 |
| QAT | 极低比特/高精度 | 训练数据与训练 | 训练中模拟量化误差 | 成本高、训练不稳定 |

不要把算法名当作格式名。同为 INT4，GPTQ/AWQ/torchao 的 scale、zero-point、group、packing 和 Kernel 可能不同。

## 3.2 安装 torchao

在独立环境中安装，并根据 PyTorch/CUDA 兼容矩阵选择 wheel：

```bash
python -m pip install torchao
python -c "import torchao; print(torchao.__version__)"
```

不要在已经稳定运行 vLLM 的环境里直接升级 PyTorch。建议创建单独的 `.venv-compression`。

## 3.3 最小真实量化示例

当前 torchao 使用配置对象和 `quantize_` 修改匹配的 Linear：

```python
import torch
from torch import nn
from torchao.quantization import Int4WeightOnlyConfig, quantize_

device = "cuda"
model = nn.Sequential(
    nn.Linear(1024, 4096, bias=False),
    nn.GELU(),
    nn.Linear(4096, 1024, bias=False),
).to(device=device, dtype=torch.bfloat16).eval()

quantize_(
    model,
    Int4WeightOnlyConfig(group_size=128),
    device=device,
)

print(model)
for name, parameter in model.named_parameters():
    print(name, type(parameter), parameter.dtype, parameter.shape)
```

API 和受支持 Kernel 会随版本变化。运行前保存：

```bash
python -c "import torch, torchao; print(torch.__version__, torchao.__version__)"
```

## 3.4 必做检查

1. `nn.Linear` 是否真的被替换/转换？
2. 权重 Tensor 是普通 BF16，还是量化 Tensor subclass？
3. 保存后的 checkpoint 是否使用真实 packed 表示？
4. group size 是否与 `in_features` 整除关系兼容？
5. Profiler 中是低比特 matmul，还是 dequant + BF16 matmul？
6. 第一次运行是否包含 compile/Kernel autotune？
7. 对比是否固定了输入和 warmup？

## 3.5 Profiler 证据

```python
import torch
from torch.profiler import ProfilerActivity, profile

x = torch.randn(32, 1024, device="cuda", dtype=torch.bfloat16)

for _ in range(10):
    model(x)
torch.cuda.synchronize()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(20):
        model(x)
torch.cuda.synchronize()

print(prof.key_averages().table(
    sort_by="self_cuda_time_total",
    row_limit=20,
))
```

结论中写算子/Kernel 名和证据，不要只写“使用 INT4”。

## Day 3 验收

- [ ] 能用一句话区分 RTN、GPTQ、AWQ、SmoothQuant
- [ ] 在隔离环境安装或记录无法安装的兼容性原因
- [ ] 检查真实权重对象、保存格式与 Kernel
- [ ] 分开报告首次运行和稳态
- [ ] 不把论文理论收益直接当作当前硬件实测收益

---

# Day 4：剪枝、稀疏模式与硬件支持

## 学习目标

- 区分 unstructured、structured 与 N:M sparsity；
- 实现 magnitude pruning 和 2:4 mask；
- 验证“零更多”为什么不自动加速。

## 4.1 三类稀疏

| 类型 | 删除单位 | 灵活性 | 精度 | 加速难度 |
| --- | --- | --- | --- | --- |
| 非结构化 | 单个权重 | 高 | 通常较好 | 高，需要索引与稀疏 Kernel |
| N:M 半结构化 | 每 M 个保留 N 个 | 中 | 中 | 依赖硬件和 shape |
| 结构化 | Head、Channel、Layer | 低 | 风险更高 | 较容易映射 dense Kernel |

PyTorch 参数中 50% 元素为零，Tensor 仍可能以 dense FP16 保存，普通 GEMM 仍会读取和计算这些位置。

## 4.2 非结构化剪枝

```bash
python labs/model-compression/compression_lab.py   --method unstructured   --sparsity 0.50   --rows 1024 --cols 1024   --batch-size 32   --seed 42   --output benchmarks/raw-results/week-03/prune-unstructured-50.json
```

扫描稀疏率：

```bash
for sparsity in 0.25 0.50 0.70 0.90; do
  python labs/model-compression/compression_lab.py     --method unstructured     --sparsity "$sparsity"     --rows 1024 --cols 1024     --seed 42     --output "benchmarks/raw-results/week-03/prune-$sparsity.json"
done
```

观察输出误差是否在高稀疏率附近急剧恶化。

## 4.3 2:4 半结构化剪枝

每连续 4 个权重保留绝对值最大的 2 个：

```bash
python labs/model-compression/compression_lab.py   --method 2to4   --rows 1024 --cols 1024   --batch-size 32   --seed 42   --output benchmarks/raw-results/week-03/prune-2to4.json
```

检查：

- 每组 4 个是否恰好 2 个非零；
- 实际 dense Tensor 大小是否改变；
- 当前设备是否有 2:4 compressed layout 与 sparse matmul；
- baseline 和普通 dense-zero 路径延迟是否有可靠差异。

## 4.4 与 PyTorch pruning reparameterization 的关系

`torch.nn.utils.prune` 教学 API 常通过 `weight_orig` 和 `weight_mask` 重参数化。调用 `prune.remove` 会固化零值，但不自动把 dense Tensor 变成压缩稀疏格式。

必须分别检查：

```python
print(module._parameters.keys())
print(module._buffers.keys())
print((module.weight == 0).float().mean())
print(module.weight.layout)
```

## 4.5 恢复策略

剪枝后的精度恢复可能包括小学习率 fine-tuning、逐层重构、保留敏感层、渐进稀疏、校准激活/近似二阶信息和蒸馏。

任何恢复训练都要报告数据、步数、学习率、优化器和额外成本。

## Day 4 验收

- [ ] 完成非结构化稀疏率扫描
- [ ] 验证 2:4 每组的非零数量
- [ ] 能解释 dense zeros 不等于 compressed sparse format
- [ ] 能说明 mask/indices 的存储开销
- [ ] 能从 Profiler 判断是否执行稀疏 Kernel

---

# Day 5：低秩分解与知识蒸馏

## 学习目标

- 用 SVD 理解矩阵有效秩；
- 计算低秩因子的参数量和计算量；
- 区分低秩压缩与 LoRA 微调；
- 设计 Teacher/Student 蒸馏实验。

## 5.1 低秩分解

对权重 `W ∈ R^(m×n)`：

`W ≈ U_r Σ_r V_r^T = A B`

原参数量：`m × n`

低秩参数量：`r × (m + n)`

只有当 `r × (m + n) < m × n` 时，参数量才下降。执行从一次大 GEMM 变成两次 GEMM，理论 FLOPs 减少不保证端到端更快。

运行秩扫描：

```bash
for rank in 32 64 128 256; do
  python labs/model-compression/compression_lab.py     --method low-rank     --rank "$rank"     --rows 1024 --cols 1024     --batch-size 32     --seed 42     --output "benchmarks/raw-results/week-03/low-rank-$rank.json"
done
```

比较 retained spectral energy、weight/output MSE、因子存储量、两次 GEMM 延迟和精度崩溃点。

### 低秩压缩不等于 LoRA

- 低秩压缩试图用低秩因子替代原权重；
- LoRA 通常冻结原权重并添加低秩增量；
- 未合并 LoRA 时总推理路径可能多一次分支；
- 合并后权重仍可能是原始 dense shape，不自动减少模型大小。

## 5.2 知识蒸馏

Teacher 给 Student 提供比硬标签更丰富的分布信息。

Logit distillation 常用：

`L_KD = T^2 × KL(softmax(z_teacher / T), softmax(z_student / T))`

组合目标：

`L = α × L_task + (1 - α) × L_KD`

可扩展为 hidden-state matching、attention map matching、sequence-level distillation 和 feature projection。

## 5.3 蒸馏实验设计

本周不要求训练大模型，先写可执行方案：

| 项目 | 选择 |
| --- | --- |
| Teacher | 较大、冻结、eval 模式 |
| Student | 层数/宽度更小 |
| 数据 | 训练/验证/测试严格分开 |
| 温度 T | 1、2、4 |
| α | 0.2、0.5、0.8 |
| 对齐 | logits；可选 hidden projection |
| 指标 | task accuracy/PPL、训练成本、推理成本 |
| 基线 | Student 仅硬标签训练 |

教师生成数据的成本不能从系统报告中消失。蒸馏节省长期推理成本，但会增加训练和数据处理成本。

## Day 5 验收

- [ ] 完成至少 4 个 rank
- [ ] 能计算低秩前后参数与 FLOPs
- [ ] 能解释两次 GEMM 可能抵消理论收益
- [ ] 能区分低秩压缩与 LoRA
- [ ] 写出包含无蒸馏 Student 基线的实验设计

---

# Day 6：精度、性能与部署鸿沟联合评估

## 学习目标

把算法指标和系统指标放到同一张表中，不再用单一压缩率下结论。

## 6.1 正确性指标

由便宜到昂贵：

1. 权重 MSE / relative MSE；
2. Layer output MSE / cosine similarity；
3. Logit KL divergence；
4. 固定语料 PPL；
5. 下游任务 Accuracy / F1 / exact match；
6. 生成质量、安全与领域测试。

局部 MSE 小不代表 PPL 一定稳定；PPL 稳定也不代表所有下游能力不退化。

PPL 评估固定 tokenizer、truncation/stride、sequence length、padding/label mask、dataset revision、sample 数与 seed，以及 BOS/EOS 处理。

## 6.2 系统指标

至少报告：

- checkpoint 总字节数；
- 加载后 allocated/reserved/进程显存；
- 首次与稳态加载时间；
- Prefill latency、Decode TPOT；
- 输出 tokens/s；
- 并发下 TTFT p50/p95/p99；
- 能否增大 batch、上下文或并发；
- 实际 operator/Kernel；
- 压缩转换和校准耗时。

## 6.3 对比矩阵

| 配置 | 理论 bits | 实际文件 MB | Peak GB | PPL | TTFT p95 | TPOT p95 | tok/s | Kernel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| BF16 | 16 |  |  |  |  |  |  |  |
| INT8 WO | 8 |  |  |  |  |  |  |  |
| INT4 g128 | 4 + scale |  |  |  |  |  |  |  |
| 50% dense-zero | 16 |  |  |  |  |  |  | dense GEMM? |
| 2:4 compressed | 约 50% + metadata |  |  |  |  |  |  | sparse GEMM? |
| Low-rank r=128 | 因子化 |  |  |  |  |  |  | two GEMMs |

## 6.4 必须做的反例

至少保留一个“压缩成功但没加速”的案例：

- INT4 fake quant 后仍以 BF16 计算；
- 50% 非结构化置零仍走 dense GEMM；
- 低秩 rank 对小 batch 增加 launch overhead；
- 模型文件变小，但 KV Cache 主导峰值显存；
- 单请求变快，但高并发尾延迟变差。

反例不是失败，它是最能展示系统理解的证据。

## 6.5 Profiler 闭环

对基线和一个压缩配置分别采集 operator table、CUDA timeline、shape/dtype、self CUDA time、memcpy/dequant/cast、Kernel 数量和 GPU 空洞。

结论模板：

> 配置 X 的权重文件减少了 __%，但在 shape __、batch __ 下仍执行 __ Kernel，并新增 __ dequant/cast 开销。因此纯算子延迟变化 __%，端到端 TPOT 变化 __%。当前证据支持/不支持“该配置在本硬件上加速”的假设。

## Day 6 验收

- [ ] 算法和系统指标进入同一张表
- [ ] 精度评估固定数据预处理
- [ ] 保存至少一组 Profiler 证据
- [ ] 包含至少一个无加速反例
- [ ] 结论限定在实际硬件、shape 和后端

---

# Day 7：报告、复现与部署决策

## 7.1 周报模板

创建 `benchmarks/reports/week-03-compression-report.md`：

```markdown
# Week 3 Model Compression Report

## 1. 问题与假设
- 目标瓶颈：
- 候选方法：
- 可证伪假设：

## 2. 环境
- Git commit：
- 模型与 revision：
- GPU / CPU：
- PyTorch / CUDA / Driver：
- 压缩库版本：

## 3. 数据
- Calibration：
- Validation：
- Test：
- 预处理和 seed：

## 4. 配置
- bits / dtype：
- granularity / group size：
- clipping / zero-point：
- sparsity / pattern：
- rank：
- backend / kernel：

## 5. 正确性
| 配置 | Weight MSE | Output Cosine | PPL | Task Accuracy |
| --- | ---: | ---: | ---: | ---: |

## 6. 系统性能
| 配置 | File MB | Peak GB | TTFT p95 | TPOT p95 | tok/s | Kernel |
| --- | ---: | ---: | ---: | ---: | ---: | --- |

## 7. Profiler 证据
- 基线：
- 压缩配置：
- 新增 dequant/cast：
- 瓶颈变化：

## 8. 失败案例
- 现象：
- 根因假设：
- 证据：
- 下一实验：

## 9. 结论与部署建议
- 是否采用：
- 适用硬件/负载：
- 精度风险：
- 回滚方案：
```

## 7.2 部署决策门

只有同时满足以下条件才进入服务测试：

- 任务精度/PPL 在预算内；
- checkpoint 能稳定保存和重新加载；
- 推理引擎明确支持该格式；
- 目标硬件存在对应 Kernel；
- baseline 与压缩版使用相同工作负载；
- 性能收益超过测量抖动；
- 没有新增不可接受的失败模式；
- 有高精度回滚路径。

## 7.3 推荐结论格式

不要写：

> INT4 将模型压缩 4 倍并加速 4 倍。

应写：

> 在 GPU __、模型 __、batch __、输入/输出长度 __ 和后端 __ 下，INT4 group-size __ 的 checkpoint 比 BF16 小 __%，PPL 变化 __，峰值显存变化 __%，TPOT p95 变化 __%。Profiler 显示执行 __ Kernel；因此该结论仅适用于 __ 场景。TTFT 未改善的主要原因可能是 __。

## Day 7 验收

- [ ] 报告可从原始 JSON 追溯
- [ ] 所有表格写明单位和聚合方式
- [ ] 结论包含硬件、shape 和后端边界
- [ ] 明确压缩转换成本
- [ ] 明确回滚方案和未解决风险
- [ ] 提出下一周 CUDA/Triton Kernel 验证问题

---

# 常见故障与排查

## 量化后误差突然很大

1. scale 是否为 0 或出现 NaN/Inf；
2. qmin/qmax 是否和 signed/unsigned 格式一致；
3. reshape/group 维度是否正确；
4. 最后一组是否 padding 并在反量化后裁剪；
5. 模型是否处于 `eval()`；
6. 校准输入和真实输入域是否一致；
7. Embedding、LM Head、Norm 等敏感层是否被误量化。

## 模型大小没有按 bit 数下降

检查 fake quant 是否仍保存浮点权重、INT4 是否用 int8 一元素一字节保存、scale/zero-point 元数据、是否同时保存原始权重、tied weights、序列化支持和 sparse layout。

## 权重变小但显存没有明显下降

可能是加载时反量化为 BF16、activation/KV Cache 主导、allocator reserved、量化 Kernel workspace、对比负载不同或 CUDA context 占比较高。

## 理论 FLOPs 降低但延迟更高

检查 dequant/cast/packing 是否在热路径、低秩两次 GEMM launch overhead、稀疏索引与不规则访存、Kernel tile/shape、batch 太小、CPU fallback、graph break 和首次编译成本。

## 第三方量化模型无法由 vLLM 加载

检查 vLLM 版本与量化方法兼容性、`quantization_config`、packing/group size、architecture、compute capability、tokenizer/Chat Template、完整 shard/index 和官方兼容矩阵。

# 最终能力清单

## 原理

- [ ] 能推导 affine quantization
- [ ] 能比较 symmetric/asymmetric 和四种 granularity
- [ ] 能解释 outlier 对量化的影响
- [ ] 能区分 PTQ、QAT、GPTQ、AWQ、SmoothQuant
- [ ] 能区分 unstructured、structured、N:M sparsity
- [ ] 能推导低秩参数量与计算量
- [ ] 能解释知识蒸馏温度和组合损失

## 表示与 Kernel

- [ ] 检查真实 dtype、layout、scale、zero-point、mask/indices
- [ ] 区分 fake quant 与 packed low-bit
- [ ] 区分 dense zeros 与 compressed sparse
- [ ] 用 Profiler 确认 Kernel
- [ ] 检查 dequant/cast/packing 是否进入热路径
- [ ] 理解 shape、group size 和硬件约束

## 实验

- [ ] 保存 baseline 和全部原始 JSON
- [ ] 固定 seed、shape、输入和计时方法
- [ ] 报告误差、PPL/Accuracy、大小、显存、延迟和吞吐
- [ ] 区分首次运行与稳态
- [ ] 至少重复 3 次关键配置
- [ ] 包含一个压缩但不加速的反例
- [ ] 结论能追溯到环境、代码和原始数据

# 完成等级

**及格**

- 跑通矩阵 Lab；
- 完成 INT8/INT4、50% pruning 和 low-rank；
- 能解释 fake quant 不等于真实加速。

**良好**

- 完成 group size、sparsity 和 rank 三组扫描；
- 建立精度—存储—延迟表；
- 用 Profiler 对比一个真实压缩配置。

**优秀**

- 在真实小型 LLM 上完成量化前后 PPL 与生成测试；
- 使用推理引擎支持的真实 packed 格式；
- 测量 TTFT/TPOT/吞吐和并发容量；
- 用 Kernel 证据解释收益或无收益；
- 给出可执行部署建议和回滚策略。

# 官方与论文参考

- [torchao Quantized Inference](https://docs.pytorch.org/ao/stable/workflows/inference.html)
- [torchao Quantization API](https://docs.pytorch.org/ao/stable/api_reference/api_ref_quantization.html)
- [torchao Sparsity Overview](https://docs.pytorch.org/ao/stable/contributing/sparsity.html)
- [PyTorch Pruning Tutorial](https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html)
- [GPTQ](https://arxiv.org/abs/2210.17323)
- [AWQ](https://arxiv.org/abs/2306.00978)
- [SmoothQuant](https://arxiv.org/abs/2211.10438)
- [SparseGPT](https://arxiv.org/abs/2301.00774)
