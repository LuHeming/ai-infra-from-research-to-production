# Benchmark 方法论

## 实验变量

每次对比只改变一个核心变量，其余条件保持一致：

- 模型与 commit
- 输入 prompt 集
- 输入长度与输出长度
- batch size 或并发数
- dtype
- 采样参数
- GPU 和软件环境
- warmup 与重复次数

## 延迟统计

至少报告：

- mean
- p50
- p95
- p99
- 标准差或置信区间

## GPU 测试规则

- 模型加载和编译时间单独记录。
- warmup 后再开始统计。
- CUDA 异步执行必须正确同步。
- 记录峰值显存和稳定态显存。
- 说明是否启用 CUDA Graph、编译和缓存。
- 并发服务使用固定请求集合和到达模式。

## 精度测试

- PPL 使用固定数据预处理。
- Zero-shot 使用固定任务版本和 few-shot 设置。
- 报告各任务结果，不只报告平均值。
- 量化与剪枝共用相同校准数据时，需要明确说明。

## 结果目录

```text
benchmarks/
├── configs/
├── raw-results/
├── processed-results/
└── reports/
```
