# PyTorch Profiler Demo

## 目标

对 Hugging Face CausalLM 的前向传播进行正确计时，保存原始延迟样本，
并导出 PyTorch Profiler Chrome Trace。

本实验用于回答：

- warmup 后的 forward 延迟分布是什么？
- CUDA 时间最高的算子和输入 shape 是什么？
- allocated 与 reserved 峰值显存相差多少？
- CPU 提交、CUDA Runtime 与 GPU Kernel 在时间线上如何关联？

## 安装

```bash
pip install torch transformers
```

## 运行

```bash
python labs/pytorch/profiler-demo/benchmark.py \
  --model facebook/opt-125m \
  --model-revision main \
  --device cuda \
  --dtype float16 \
  --prompt "AI infrastructure is"
```

不导出 trace：

```bash
python labs/pytorch/profiler-demo/benchmark.py \
  --model facebook/opt-125m \
  --no-profile
```

## 输出

- 原始延迟样本、平均延迟与 p50/p95
- tokens/s
- CUDA allocated/reserved 峰值显存
- 按输入 shape 聚合的算子表
- `results/profile-trace.json`
- `results/benchmark.json`

## 查看 trace

可将 `profile-trace.json` 导入 Chrome Trace 或 Perfetto。按以下顺序观察：

1. 找到 `model_forward` 自定义区间。
2. 沿 CPU operator 找到 CUDA Runtime launch。
3. 在 GPU stream 上定位对应 Kernel。
4. 检查 GPU 空洞、memcpy、dtype/layout 转换与同步。
5. 将观察写入报告，不要只附一张截图。

trace 可能包含本地路径和调用栈，公开前必须检查并脱敏；trace 文件本身已被
`.gitignore` 排除。

## 建议对比矩阵

| 变量 | 建议值 |
|---|---|
| dtype | `float32`、`float16`、`bfloat16` |
| 输入长度 | 短、中、长三组固定 token 长度 |
| warmup | `0` 与 `5`，观察冷启动差异 |
| device | CPU 基线与 CUDA 基线 |

每轮只改变一个变量，并保存独立输出目录。

## 注意

- 该实验测试模型前向传播，不包含自回归服务调度。
- 第一次运行可能包含下载、CUDA 初始化和缓存开销。
- 小模型上的结果不代表大模型或高并发服务性能。
