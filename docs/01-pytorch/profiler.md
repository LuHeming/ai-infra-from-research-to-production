# PyTorch 性能分析：从正确计时到 Profiler Trace

模型参数量、理论 FLOPs 或文件大小下降，不代表端到端推理一定加速。
性能分析的目标不是生成一张漂亮的 trace，而是建立一条证据链：

```text
现象与指标
  → 可复现 Benchmark
  → CPU/GPU 时间线
  → 高开销算子、内存或同步点
  → 可验证的优化假设
  → 相同条件下重新测量
```

## 1. 先定义要回答的问题

Profiler 适合回答：

- CPU 时间花在哪些 PyTorch 算子和 Python 代码段？
- GPU 上执行了哪些 Kernel，它们何时开始、持续多久？
- CPU 是否及时向 GPU 提交工作，GPU 时间线上是否有空洞？
- 是否存在意外的 Host-to-Device 拷贝、dtype 转换或同步？
- 哪些 shape 调用了昂贵算子，是否出现大量小 Kernel？
- Tensor 内存分配发生在哪里，峰值前有哪些事件？

Profiler 不直接回答：

- 线上 p99 延迟是否达标；
- 系统最大稳定并发是多少；
- 优化后是否在所有输入分布上更快；
- Kernel 为什么没有达到硬件峰值。

端到端负载测试、系统监控、Nsight Systems/Compute 和硬件计数器需要与它配合。

## 2. Tensor 内存模型是分析基础

### 2.1 Tensor 不只是数值

一个普通 Tensor 至少要关注：

- `shape`：每个维度的长度；
- `stride`：沿各维移动一个元素时跨过的存储步长；
- `dtype`：单元素类型和字节数；
- `device`：CPU、CUDA 等设备；
- storage、offset 和内存布局；
- 是否跟踪 autograd。

```python
import torch

x = torch.arange(12).reshape(3, 4)
print(x.shape, x.stride(), x.dtype, x.device)

y = x.transpose(0, 1)
print(y.shape, y.stride(), y.is_contiguous())
```

`transpose` 通常只改变视图元数据，不复制 storage，因此结果可能不连续。
某些算子能处理任意 stride，某些算子内部会触发连续化或选择不同 Kernel。

### 2.2 view、reshape 与 contiguous

- `view()` 要求当前 shape/stride 与目标形状兼容，否则报错。
- `reshape()` 在可能时返回 view，否则可能复制。
- `contiguous()` 在布局不满足要求时创建连续副本。

不要无条件调用 `contiguous()`“修复”问题。它可能隐藏一次大拷贝，
应该在 Profiler 中确认拷贝是否存在，以及下游算子是否真的需要它。

```python
y = x.transpose(0, 1)
z = y.contiguous()
print(y.data_ptr() == z.data_ptr())
```

### 2.3 dtype 与显存估算

Tensor 数据的理论字节数：

```text
numel × element_size
```

```python
bytes_used = tensor.numel() * tensor.element_size()
```

但进程显存还包含缓存分配器保留空间、临时 workspace、上下文、
Kernel 库、激活、KV Cache 和碎片。`memory_allocated()` 与
`memory_reserved()` 的含义不同，不能把 `nvidia-smi` 数值直接当成模型权重大小。

## 3. 推理模式与测量边界

```python
model.eval()

with torch.inference_mode():
    output = model(**inputs)
```

- `model.eval()` 改变 Dropout、BatchNorm 等模块行为，不关闭 autograd。
- `torch.no_grad()` 关闭梯度记录。
- `torch.inference_mode()` 进一步减少部分 autograd 相关开销，但限制更强。

推理 Benchmark 通常使用 `eval()` 和 `inference_mode()`，并在报告中明确记录。
训练性能分析则必须保留 forward、backward、optimizer 和梯度清零的真实边界。

首先决定测量哪一层：

| 边界 | 包含内容 | 适合回答 |
|---|---|---|
| 单算子 | 一个 op 或 Kernel | 算子实现比较 |
| 模型 forward | 已准备好的 Tensor 输入 | 模型计算基线 |
| `generate()` | tokenization 后到生成完成 | 自回归推理行为 |
| 请求端到端 | 排队、网络、tokenize、调度、生成 | 服务体验 |
| 冷启动 | 下载、加载、编译、初始化 | 部署启动成本 |

不同边界的数字不能直接比较。

## 4. CUDA 异步执行与正确计时

### 4.1 为什么普通计时会错

CUDA Kernel 启动通常是异步的：CPU 把工作提交到 stream 后就可继续执行。
如果直接用 `time.perf_counter()` 包围模型调用，测到的可能主要是 CPU 提交时间，
而不是 GPU 完成时间。

端到端同步计时：

```python
import time

torch.cuda.synchronize()
start = time.perf_counter()
model(**inputs)
torch.cuda.synchronize()
elapsed_ms = (time.perf_counter() - start) * 1000
```

同步必须在起点和终点都考虑：起点前清空之前遗留的 GPU 工作，
终点等待本轮工作完成。

### 4.2 CUDA Event 设备侧计时

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
model(**inputs)
end.record()
end.synchronize()
elapsed_ms = start.elapsed_time(end)
```

Event 在 CUDA stream 时间线上记录，适合测量 GPU 工作区间。
如果工作跨多个 stream，应明确依赖和完成条件；否则单一 stream 上的 Event
未必覆盖所有异步工作。

### 4.3 warmup 与冷启动

首次运行可能包含：

- CUDA context 初始化；
- 库加载、句柄和内存池初始化；
- cuDNN/cuBLAS 算法选择；
- lazy module 或缓存创建；
- `torch.compile` 图捕获、编译和 autotune；
- 数据下载、磁盘缓存和页面缓存。

稳定态 Benchmark 应先 warmup；冷启动成本则单独作为另一项指标报告，
不能把首次运行静默丢弃后仍称为“启动延迟”。

## 5. Benchmark 的最小规范

一个可信的基线至少固定并记录：

- 模型名称与 revision；
- PyTorch、CUDA、驱动、GPU 和操作系统；
- dtype、device、batch、输入/输出长度和 shape；
- eager、compile、量化和 attention backend；
- warmup 次数、重复次数、同步方式；
- 是否包含 tokenize、数据传输、模型加载和生成；
- mean、median、p95、min/max 和原始样本；
- GPU 峰值 allocated/reserved memory。

不要只报告“tokens/s”。至少说明：

- 输入 token 还是输出 token；
- 单请求还是所有并发请求总和；
- prefill、decode 还是完整请求；
- 是否排除 padding；
- 时间窗口和并发策略。

## 6. PyTorch Profiler 基本用法

```python
from torch.profiler import ProfilerActivity, profile, record_function

activities = [ProfilerActivity.CPU]
if torch.cuda.is_available():
    activities.append(ProfilerActivity.CUDA)

with profile(
    activities=activities,
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    with record_function("model_forward"):
        model(**inputs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

sort_key = "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
print(prof.key_averages().table(sort_by=sort_key, row_limit=20))
prof.export_chrome_trace("profile-trace.json")
```

常用参数：

| 参数 | 用途 | 成本与注意 |
|---|---|---|
| `activities` | 选择 CPU、CUDA 等活动 | 不需要的活动不要记录 |
| `record_shapes` | 记录算子输入 shape | 增加数据量和开销 |
| `profile_memory` | 关联 Tensor 内存事件 | 不是完整进程显存解释器 |
| `with_stack` | 记录调用栈 | 开销和 trace 体积更大 |
| `with_modules` | 尝试记录模块层级 | 支持范围依执行模式而异 |
| `record_function` | 标记自定义阶段 | 便于关联业务阶段与算子 |

Profiler 本身会改变运行时间。它用于定位，不应把 Profiler 打开时的耗时
当作正式 Benchmark 数字。

## 7. 长任务使用 schedule

训练或多轮推理不应记录整个任务，否则 trace 可能巨大：

```python
from pathlib import Path
from torch.profiler import ProfilerActivity, profile, schedule

trace_dir = Path("results/traces")
trace_dir.mkdir(parents=True, exist_ok=True)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=2, warmup=2, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_dir)),
    record_shapes=True,
    profile_memory=True,
) as prof:
    for batch in dataloader:
        train_step(batch)
        prof.step()
```

阶段含义：

- `wait`：跳过，不记录；
- `warmup`：让 Profiler 自身进入稳定状态，但不保存；
- `active`：真正记录；
- `repeat`：重复多少个周期。

每次迭代必须调用 `prof.step()` 推进状态机。采样窗口要覆盖有代表性的稳定阶段，
不能只挑最快的一段。

## 8. 如何读 key averages 表

```python
table = prof.key_averages(group_by_input_shape=True).table(
    sort_by="cuda_time_total",
    row_limit=30,
)
print(table)
```

重点字段：

- **Self CPU**：事件自身在 CPU 上的时间，不含子事件；
- **CPU total**：包含子事件的 CPU 总时间；
- **Self CUDA**：直接归属于事件的 CUDA 时间；
- **CUDA total**：包含子事件关联的 CUDA 时间；
- **# of Calls**：调用次数；
- **CPU/CUDA time avg**：平均每次调用时间；
- **Input Shapes**：同名算子的 shape 分组；
- **CPU Mem / CUDA Mem**：与事件关联的内存变化。

分析顺序：

1. 按 total 找影响端到端的大类。
2. 按 self 找真正执行工作的叶子算子或 Kernel。
3. 同时看调用次数和平均时间，区分“一次很慢”与“大量很小”。
4. 按 shape 分组，避免同名算子的不同尺寸被平均掉。
5. 回到 trace 检查并发、空洞、拷贝和同步，而不是只看聚合表。

父算子和子算子的 total 时间会重叠，不能把整列简单相加。

## 9. 如何读 CPU/GPU 时间线

在 Chrome Trace、Perfetto 或兼容查看器中，先定位一个完整迭代：

```text
Python / user annotation
  → ATen operator
  → CUDA Runtime launch
  → GPU stream kernel or memcpy
```

### 模式 A：GPU 时间线有大空洞

可能原因：

- DataLoader 或预处理慢；
- Python 调度和大量小算子；
- CPU-GPU 同步；
- 网络或磁盘等待；
- 分布式 rank 在 barrier/collective 等待。

先看空洞前 CPU 正在执行什么，不要仅凭低 GPU 利用率猜测。

### 模式 B：大量很短的 Kernel

可能说明 launch overhead 高、算子过碎、batch/shape 太小，
或存在本可融合的逐元素操作。验证方向包括向量化、批处理、算子融合和 compile，
但必须重新测量数值正确性、编译成本和动态 shape 行为。

### 模式 C：频繁 memcpy 或 dtype conversion

检查输入是否每轮 `.to(device)`、是否混入 CPU Tensor、是否发生隐式 cast、
布局转换或 `contiguous()` 拷贝。优化后要确认数据生命周期和 stream 依赖仍正确。

### 模式 D：CPU 调用后立刻等待

常见同步点：

- `.item()`；
- `.cpu()` / `.numpy()`；
- 打印 CUDA Tensor；
- 显式 `torch.cuda.synchronize()`；
- 某些内存查询、异常检查或跨设备操作。

Benchmark 终点的同步是必要的，但热路径中的意外同步会破坏流水。

## 10. 内存分析

### 10.1 先区分 allocated 与 reserved

```python
torch.cuda.reset_peak_memory_stats()
run_once()
torch.cuda.synchronize()

allocated = torch.cuda.max_memory_allocated()
reserved = torch.cuda.max_memory_reserved()
```

- allocated：活跃 Tensor 由缓存分配器管理的内存；
- reserved：缓存分配器向 CUDA 申请并保留的内存；
- `nvidia-smi`：进程/上下文层面的设备内存观察，口径更宽。

### 10.2 Profiler memory 字段怎么用

`profile_memory=True` 可把部分 Tensor 分配/释放关联到算子，适合寻找：

- 大型临时 Tensor；
- 重复分配；
- dtype/布局转换副本；
- 某阶段净增长。

但内存峰值是时间序列问题，聚合后的净变化可能把“先分配再释放”抵消。
需要结合时间线、峰值统计和必要时的 memory snapshot 工具分析。

### 10.3 OOM 排查顺序

1. 保存失败配置和错误信息。
2. 确认权重、激活、KV Cache、workspace 和其他进程的占用。
3. 比较 allocated、reserved 与设备总占用。
4. 检查引用是否跨迭代保留，如把带计算图的 Tensor 放进列表。
5. 用更小 shape 构造最小复现。
6. 修改 batch、序列长度、dtype 或算法后重新验证正确性与性能。

`torch.cuda.empty_cache()` 只释放缓存分配器中未使用的缓存给其他应用，
不会释放仍被 Tensor 引用的内存，也不是修复内存泄漏的通用办法。

## 11. 用 record_function 标记业务阶段

ATen 算子名不足以表达“tokenize、prefill、decode、postprocess”等业务语义：

```python
from torch.profiler import record_function

with record_function("prepare_inputs"):
    inputs = prepare(batch)

with record_function("prefill"):
    outputs = model(**inputs, use_cache=True)

with record_function("postprocess"):
    result = decode(outputs)
```

命名要稳定、层级清晰，不要把 request id 等高基数字段放进事件名，
否则聚合困难且 trace 体积膨胀。

## 12. 编译、量化与模型压缩分析

### 12.1 torch.compile

分开报告：

- eager 冷启动；
- compile 首次编译成本；
- 编译后稳定态性能；
- 新 shape 导致的重编译或 graph break。

Profiler 中看到的算子和 Kernel 可能因融合而改变。优化目标应是端到端指标，
不是“算子数量越少越好”。

### 12.2 fake quant 不等于低比特 Kernel

fake quant 常用浮点 Tensor 模拟量化误差，可能不会减少模型文件、显存访问或计算。
必须确认：

- 权重实际存储 dtype 和打包格式；
- 推理引擎是否识别该格式；
- trace 中是否出现目标量化 GEMM/反量化 Kernel；
- 反量化、格式转换和 padding 是否抵消收益；
- 精度与端到端延迟是否在同一配置下测量。

### 12.3 稀疏和剪枝

权重中出现零值并不自动加速。需要结构化稀疏格式、硬件支持和对应 Kernel。
Profiler 应验证 Kernel 是否变化，Benchmark 应验证真实 shape 与 batch 下的收益。

## 13. 多 GPU 与分布式场景

单进程 Profiler 只能看到局部视角。分布式分析还要记录：

- rank、local rank、world size 和节点；
- compute stream 与 NCCL collective 的重叠；
- 不同 rank 的 straggler；
- barrier、all-reduce、all-gather、reduce-scatter 的时间；
- 输入不均衡、网络拓扑和通信量。

先在单卡建立计算基线，再分析通信；否则容易把单卡效率问题误判为 NCCL 问题。

## 14. Profiler 开销与数据安全

- 只记录代表性窗口，不记录整个长任务。
- `with_stack`、shape 和 memory 信息按问题需要开启。
- trace 可能非常大，不能直接提交 Git。
- stack、路径、算子参数和自定义标记可能暴露用户名、模型路径或业务信息。
- 分享前脱敏，并记录 PyTorch 版本，因为字段和查看器支持会变化。

生产环境启用 Profiler 前要评估开销、存储和隐私；优先在可复现的预生产负载中定位。

## 15. 常见误区

1. CUDA 未同步就报告延迟。
2. 没有区分冷启动与稳定态。
3. 只测试一个 shape、batch 或输入长度。
4. 把 Profiler 打开后的时间当正式性能数字。
5. 把 CPU total 和 CUDA total 简单相加。
6. 只看排序表，不看时间线上的空洞和重叠。
7. 把 `nvidia-smi` 瞬时利用率当作算子分析。
8. 把 reserved memory 当成活跃 Tensor 大小。
9. 把 fake quant、零权重或模型文件变小当成真实加速。
10. 优化后只看速度，不验证输出正确性和精度。
11. 只报告最优一次，不保存原始样本和波动。
12. trace 未脱敏就上传到公开仓库。

## 16. 第一周 Profiler 实战

仓库提供
[Profiler Demo](https://github.com/LuHeming/ai-infra-from-research-to-production/tree/main/labs/pytorch/profiler-demo)。

### 练习 A：Tensor 与布局

1. 构造连续 Tensor，并通过 transpose 得到非连续 view。
2. 记录 shape、stride、storage 共享关系和 `is_contiguous()`。
3. 对比 `view()`、`reshape()` 和 `contiguous()`。
4. 用 Profiler 确认何时发生真实拷贝。

### 练习 B：错误计时与正确计时

对同一模型分别使用：

1. 不同步的 `perf_counter()`；
2. 前后同步的 `perf_counter()`；
3. CUDA Event；
4. 含首次运行和 warmup 后稳定态。

保存所有原始样本，解释不同结果的原因。

### 练习 C：算子与时间线

1. 导出 Chrome Trace。
2. 找到一个完整 forward 的 CPU 调用链和 GPU Kernel。
3. 列出 CUDA total 最高的 10 个算子。
4. 用 shape 分组，并找出调用次数最多的小算子。
5. 标记至少一个同步点、memcpy 或 GPU 空洞；若没有，也要记录证据。

### 练习 D：压缩前后对比

选择 FP32/FP16/BF16、真实量化或剪枝中的一种对比：

- 固定模型、revision、输入和输出；
- 比较精度/输出、延迟分布、吞吐、allocated/reserved memory；
- 比较算子与 Kernel 是否变化；
- 解释理论压缩收益与实测收益的差异。

## 17. 第一周完成检查表

- [ ] 能解释 shape、stride、view、copy 和 contiguous 的关系。
- [ ] 能区分 `eval()`、`no_grad()` 和 `inference_mode()`。
- [ ] 能明确单算子、forward、generate、请求和冷启动测量边界。
- [ ] 能用同步计时和 CUDA Event 获得可信延迟。
- [ ] 能设计 warmup、repeat、shape 和统计方法。
- [ ] 能配置 Profiler 并使用 schedule 控制采样窗口。
- [ ] 能解释 self/total、CPU/CUDA、调用次数和 shape 分组。
- [ ] 能从时间线定位 GPU 空洞、小 Kernel、拷贝和同步。
- [ ] 能区分 allocated、reserved、峰值和进程总显存。
- [ ] 能验证量化或剪枝是否真正改变 Kernel 和端到端指标。
- [ ] 能说明 Profiler 的开销、局限和 trace 隐私风险。
- [ ] 能提交一份包含证据、假设、修改与复测的性能报告。

## 18. 官方参考

- [PyTorch Profiler API](https://docs.pytorch.org/docs/stable/profiler.html)
- [PyTorch Profiler Recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [CUDA Event](https://docs.pytorch.org/docs/stable/generated/torch.cuda.Event.html)
- [Tensor view](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.view.html)
- [Tensor storage](https://docs.pytorch.org/docs/stable/storage.html)
