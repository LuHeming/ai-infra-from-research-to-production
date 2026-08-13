# Contributing

欢迎提交 Issue、修正文档、补充实验或增加可复现 Benchmark。

## 分支命名

```text
docs/<topic>
lab/<experiment>
bench/<benchmark>
fix/<problem>
```

## Commit 规范

```text
docs: add KV cache notes
lab: add PyTorch profiler demo
bench: add OPT-125M latency results
paper: add GPTQ paper notes
fix: correct CUDA event timing
chore: update dependencies
```

## Pull Request 要求

PR 描述应说明：

- 修改了什么
- 为什么修改
- 如何验证
- 是否影响已有实验结果
- 是否引入新的依赖或硬件要求

## 实验贡献要求

每个实验必须记录：

- 硬件与软件环境
- 模型和数据
- 执行命令
- warmup 与重复次数
- 评价指标
- 原始结果
- 结论与局限

请勿提交模型权重、访问密钥、私有数据和无法追溯来源的内容。
