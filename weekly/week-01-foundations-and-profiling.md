# Week 01：工程基础与 PyTorch Profiling

## 本周目标

- [ ] 配置主开发环境
- [ ] 完成 Linux 文件、权限、进程、资源和网络基础练习
- [ ] 使用 SSH、tmux 和 rsync 完成一次远程工作流
- [ ] 掌握正确 GPU 计时
- [ ] 使用 PyTorch Profiler 分析一个小型 LLM
- [ ] 建立第一份性能基线
- [ ] 完成一篇排错记录

## 核心问题

- 如何从 PID 出发定位一个任务的命令、资源、文件与网络连接？
- 退出码、标准流、管道和信号如何组成可观察的命令行工作流？
- CPU、内存、磁盘或网络如何让 GPU 任务表现为“GPU 利用率低”？
- CUDA 异步执行为什么会导致错误计时？
- 首次推理和稳定态推理为什么不同？
- 模型时间主要花在哪些算子上？
- FP16、BF16 和 fake quant 是否改变实际 Kernel？
- 模型大小下降是否带来显存和吞吐收益？

## 每日安排

| 天数 | 主题 | 实践与验收 |
|---|---|---|
| Day 1 | Linux 文件系统、Shell、标准流与权限 | 完成 Linux 练习 A；能解释路径、管道、退出码和 `rwx` |
| Day 2 | 进程、信号、CPU、内存、磁盘、GPU 与日志 | 完成练习 B；能从 PID 建立进程资源画像 |
| Day 3 | SSH、tmux、rsync、网络与环境变量 | 完成练习 C；SSH 断开后能恢复任务 |
| Day 4 | Git 与 Python 工程基础 | 运行测试和 lint，记录依赖与环境 |
| Day 5 | PyTorch Tensor、CUDA 异步与正确计时 | 对比首次运行、warmup 后和同步计时结果 |
| Day 6 | PyTorch Profiler | 导出 trace，找出耗时和显存占用最高的算子 |
| Day 7 | Benchmark 与复盘 | 固化原始结果，完成 troubleshooting 和 weekly review |

Linux 的详细知识、命令和练习见 [Linux 工程基础](../docs/00-foundations/linux-fundamentals.md)。

## 本周交付物

- Linux 基础练习记录
- `docs/01-pytorch/profiler.md`
- `labs/pytorch/profiler-demo/`
- 一份环境 JSON
- 一份 Benchmark JSON
- 一篇 troubleshooting 文档
