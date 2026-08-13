# 开发环境与可复现性

## 推荐环境

AI Infra 主环境优先使用 Linux。Windows 用户可以使用 WSL2 Ubuntu 或远程 Linux GPU 服务器。

## 每次实验应记录

```text
操作系统
GPU 型号
驱动版本
CUDA Runtime
Python
PyTorch
Transformers
推理框架
模型版本或 commit
数据集版本
随机种子
执行命令
```

## 常用检查命令

```bash
uname -a
python --version
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda)"
pip freeze
git rev-parse HEAD
```

## 可复现原则

- 区分首次运行和 warmup 后性能。
- GPU 计时前后执行正确同步。
- 固定输入长度、输出长度和请求集合。
- 报告重复次数、均值和分位数。
- 原始结果以 JSON/CSV 保存，不手工覆盖。
