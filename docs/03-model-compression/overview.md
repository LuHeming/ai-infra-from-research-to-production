# 大模型压缩概览

## 主要方向

- Post-Training Quantization
- Quantization-Aware Training
- Post-Training Pruning
- Structured / Semi-structured Sparsity
- Low-rank Approximation
- Knowledge Distillation

## 评估维度

压缩方法至少需要从四层评价：

1. **算法层**：PPL、Accuracy、重构误差。
2. **表示层**：权重格式、scale/zero-point、稀疏索引。
3. **Kernel 层**：是否存在对应低比特或稀疏 Kernel。
4. **系统层**：端到端吞吐、延迟、显存和并发能力。

## 研究记录建议

每个方法都应记录：

- 问题定义
- 核心假设
- 优化目标
- 校准数据
- 复杂度
- 实际权重格式
- 推理框架支持
- 精度和性能结果
- 失败场景与局限
