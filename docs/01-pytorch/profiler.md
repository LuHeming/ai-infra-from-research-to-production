# PyTorch 性能分析

## 为什么需要 Profiler

大模型压缩的参数量、理论 FLOPs 或模型文件大小下降，不代表端到端推理一定加速。Profiler 用于判断时间真正花在：

- CPU 调度
- Host-to-Device 数据传输
- Kernel 启动
- GEMM 或 Attention
- 量化/反量化
- 显存分配
- CPU/GPU 同步
- GPU 间通信

## 正确计时

CUDA 默认异步执行。使用普通 `time.perf_counter()` 时，应在计时区间前后同步：

```python
torch.cuda.synchronize()
start = time.perf_counter()
output = model(**inputs)
torch.cuda.synchronize()
elapsed = time.perf_counter() - start
```

正式测试应包含 warmup，并重复多次报告稳定统计量。

## Profiler 基本流程

```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    model(**inputs)

print(prof.key_averages().table(sort_by="cuda_time_total"))
prof.export_chrome_trace("trace.json")
```

## 分析清单

- CUDA 时间最高的算子是什么？
- 是否存在大量小 Kernel？
- CPU 是否频繁阻塞 GPU？
- 是否有意外的 dtype 转换？
- 量化前后 Kernel 是否发生变化？
- 显存峰值来自权重、激活还是 KV Cache？
- 测试是否包含模型加载或编译时间？

## 常见误区

1. 不同步 CUDA 就报告延迟。
2. 没有 warmup。
3. 只测试一个 shape。
4. 比较不同输入长度或输出长度。
5. 把 fake quant 当作真实低比特推理。
6. 只报告最优值，不报告波动。
