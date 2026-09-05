# AI Infra: From Research to Production

> 从大模型压缩研究到 AI Infra 工程实践

这是一个面向 **大模型压缩研究生与 AI Infra 初学者** 的开源学习仓库。项目以 PTQ、后训练剪枝和模型评测为起点，逐步补齐 GPU 性能分析、LLM 推理系统、CUDA/Triton、分布式计算、服务部署与工程交付能力。

## 项目目标

本仓库不是零散课程笔记，而是一套持续演进的：

- AI Infra 学习手册
- 实验代码库
- 性能 Benchmark
- 故障排查知识库
- 求职作品集

核心主线是：

```text
压缩算法
  → 精度评估
  → GPU 性能分析
  → 推理框架接入
  → Kernel 优化
  → 分布式扩展
  → 服务部署
  → 可复现实验报告
```

## 当前学习路线

| 阶段 | 主题 | 目标状态 |
|---|---|---|
| Week 1 | Linux、Python 工程、PyTorch Profiler | ✅ 已完成 |
| Week 2 | LLM 推理、KV Cache、vLLM | ✅ 已完成 |
| Week 3 | Model Compression：量化、剪枝、低秩、蒸馏 | 🚧 进行中 |
| Week 4 | CUDA、Triton、量化 Kernel | ⏳ 待开始 |
| Week 5 | 分布式、NCCL、综合项目 | ⏳ 待开始 |
| Week 6–9 | TensorRT-LLM、Docker、Slurm、Kubernetes、CI | ⏳ 待开始 |

详细计划见 [ROADMAP.md](ROADMAP.md)。

## 仓库导航

| 目录 | 内容 |
|---|---|
| `docs/` | 已理解、验证并整理后的正式知识章节 |
| `daily/` | 每日学习记录和原始思考 |
| `weekly/` | 每周复盘与知识压缩 |
| `labs/` | 可运行的小型实验 |
| `benchmarks/` | 配置、原始结果、处理结果和报告 |
| `projects/` | 可以用于作品集的完整项目 |
| `paper-notes/` | 面向工程落地的论文笔记 |
| `troubleshooting/` | 环境、性能和框架问题排查记录 |
| `references/` | 术语表、阅读清单和资源索引 |
| `templates/` | 每日笔记、实验、论文和排错模板 |
| `scripts/` | 自动创建笔记、采集环境和汇总结果的工具 |

## 每日工作流

```text
学习资料
   ↓
daily/YYYY/MM/YYYY-MM-DD.md
   ↓
实验代码进入 labs/
   ↓
原始数据进入 benchmarks/raw-results/
   ↓
踩坑进入 troubleshooting/
   ↓
每周整理到 weekly/
   ↓
成熟内容沉淀到 docs/
```

创建当天笔记：

```bash
python scripts/create_daily_note.py
```

采集当前实验环境：

```bash
python scripts/collect_environment.py \
  --output benchmarks/raw-results/environment.json
```

汇总 Benchmark JSON：

```bash
python scripts/build_result_table.py \
  --input-dir benchmarks/raw-results \
  --output benchmarks/processed-results/summary.csv
```

## 文档站点

安装依赖并本地预览：

```bash
pip install -r requirements-docs.txt
mkdocs serve
```

构建静态站点：

```bash
mkdocs build --strict
```

## 首个综合项目

仓库会围绕 [LLM Compression Serving Benchmark](projects/compression-serving-benchmark/README.md) 持续演进，对比：

- FP16 / BF16
- Weight-only PTQ
- 非结构化与半结构化剪枝
- Transformers / vLLM / TensorRT-LLM
- PPL、Zero-shot Accuracy
- TTFT、TPOT、吞吐、显存与模型大小
- 压缩率与真实加速之间的差异

## 内容沉淀规则

1. `daily/` 允许不完整，但必须记录来源、环境和待解决问题。
2. `docs/` 只保留已经验证过的结论。
3. 每个实验必须包含目标、环境、命令、指标、结果、结论和局限。
4. 原始结果不可手工修改；图表和表格从原始 JSON/CSV 自动生成。
5. 每周至少将一项 daily 内容沉淀为正式章节。
6. 不提交模型权重、数据集、密钥、Token 和大型二进制结果。

## 开源协议

代码与文档采用 MIT License。引用论文、图片或第三方代码时，请保留原始出处和许可证说明。
