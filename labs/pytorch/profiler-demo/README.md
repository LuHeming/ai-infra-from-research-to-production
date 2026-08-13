# PyTorch Profiler Demo

## 目标

对 Hugging Face CausalLM 的一次前向传播进行正确计时，并导出 PyTorch Profiler Chrome Trace。

## 安装

```bash
pip install torch transformers
```

## 运行

```bash
python labs/pytorch/profiler-demo/benchmark.py \
  --model facebook/opt-125m \
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

- 平均延迟与 p50/p95
- tokens/s
- CUDA 峰值显存
- `results/profile-trace.json`
- `results/benchmark.json`

## 注意

- 该实验测试模型前向传播，不包含自回归服务调度。
- 第一次运行可能包含下载、CUDA 初始化和缓存开销。
- 小模型上的结果不代表大模型或高并发服务性能。
