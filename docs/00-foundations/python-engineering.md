# Python 工程基础：把实验脚本变成可复现项目

AI Infra 中的 Python 既负责模型实验，也负责配置管理、数据处理、
Benchmark、服务入口和自动化工具。一个脚本“能运行”只是起点；工程化的目标是让它：

- 换一台机器仍能安装和运行；
- 输入、配置、环境和输出都有记录；
- 出错时能从日志和异常中定位原因；
- 核心逻辑可以测试、复用和审查；
- 性能实验不会被隐式状态或错误统计污染。

## 1. 从脚本到项目的边界

一次性验证可以从单文件开始，但出现以下任一情况时就应该拆分：

- 同一段逻辑被两个入口复用；
- 参数超过三到五个，或需要多组实验配置；
- 需要测试、日志、缓存、结果文件或多个模型后端；
- 需要其他人安装和执行；
- 需要区分“业务逻辑”和“命令行/文件系统副作用”。

推荐的最小布局：

```text
my-benchmark/
├── pyproject.toml
├── README.md
├── src/
│   └── my_benchmark/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── runner.py
│       └── metrics.py
├── tests/
│   ├── test_config.py
│   └── test_metrics.py
└── results/
```

职责划分：

| 文件 | 职责 |
|---|---|
| `cli.py` | 解析参数、调用核心逻辑、设置退出码 |
| `config.py` | 配置结构、默认值和合法性校验 |
| `runner.py` | 模型加载、warmup、执行和资源清理 |
| `metrics.py` | 纯统计逻辑，便于单元测试 |
| `tests/` | 验证边界条件和回归行为 |
| `results/` | 生成物，不应与源代码混在一起 |

`src/` 布局可避免测试时意外从仓库根目录导入未安装的包，
更接近用户实际安装后的行为。小型学习仓库也可以不使用 `src/`，
但必须明确包根目录和运行入口。

## 2. 解释器、虚拟环境与依赖

### 2.1 先确认你运行的是哪个 Python

```bash
python --version
python -c "import sys; print(sys.executable)"
python -m pip --version
python -m pip list
```

优先写 `python -m pip`，因为它明确使用当前 Python 对应的 pip。
`pip install` 和 `python` 可能来自不同环境，这是常见的“已经安装但导入失败”根因。

### 2.2 使用可重建的虚拟环境

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

Windows PowerShell 激活命令为：

```powershell
.venv\Scripts\Activate.ps1
```

虚拟环境是可丢弃、可重建的运行环境，不应提交到 Git，也不应复制到另一台机器。
依赖文件描述“如何重建”，环境采集描述“这次实际用了什么”。

### 2.3 区分依赖声明与环境快照

- `pyproject.toml`：项目元数据、直接依赖和工具配置的主要入口。
- `requirements*.txt`：简单环境或 CI 中常用的安装清单。
- lock 文件：固定完整依赖解析结果，适合需要严格复现的应用。
- `pip freeze`：当前环境快照，不等于经过设计的直接依赖列表。

AI 项目还要记录 PyTorch 构建版本、CUDA Runtime、驱动和 GPU 型号。
同一个 Python 包版本在不同 CUDA 构建下并不等价。

## 3. 模块、包与导入

### 3.1 基本概念

- 模块通常是一个 `.py` 文件。
- 包是可被导入的模块集合，常包含 `__init__.py`。
- 导入时 Python 按 `sys.path` 查找模块，并在首次导入后缓存到 `sys.modules`。

```bash
python -c "import sys; print(*sys.path, sep='\n')"
python -c "import torch; print(torch.__file__)"
```

不要用下面这些办法长期修复导入问题：

- 在源码里随意 `sys.path.append(...)`；
- 设置全局 `PYTHONPATH` 指向个人目录；
- 把脚本命名为 `torch.py`、`json.py` 或 `logging.py`，遮蔽真实模块；
- 依赖“必须从某个目录执行”的隐式前提。

开发本地包时可使用 editable install：

```bash
python -m pip install -e .
python -m my_benchmark.cli --help
```

### 3.2 让导入没有重副作用

模块导入时不应该自动下载模型、解析全部命令行、占用 GPU 或启动服务。
入口代码放入 `main()`：

```python
def main() -> int:
    config = parse_args()
    run(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

这样核心模块可以安全地被测试和复用，退出码也由入口统一管理。

## 4. 配置设计：显式、可校验、可记录

配置来源通常有四层，优先级应写清楚：

```text
代码默认值 < 配置文件 < 环境变量 < 命令行参数
```

敏感信息适合环境变量或密钥系统；实验超参数适合可版本控制的配置文件；
临时覆盖适合命令行。不要把 token 写进 YAML、日志或结果 JSON。

用 dataclass 表达经过校验的运行配置：

```python
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkConfig:
    model: str
    device: str = "cuda"
    warmup: int = 5
    repeat: int = 20
    output_dir: Path = Path("results")

    def validate(self) -> None:
        if self.warmup < 0:
            raise ValueError("warmup must be non-negative")
        if self.repeat <= 0:
            raise ValueError("repeat must be positive")

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        return data
```

配置保存到结果中时，应保存解析后的最终值，而不是只保存用户显式输入的部分。

## 5. 命令行接口

一个合格的 CLI 至少应该：

- 提供 `--help`；
- 明确类型、默认值和合法选项；
- 非法参数返回非零退出码；
- 不通过交互式输入阻塞自动化任务；
- 将输出目录、随机种子和运行模式设为显式参数。

```python
import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a reproducible inference benchmark."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()
```

脚本被自动化调用时，stdout 适合机器可消费的最终结果，stderr 适合诊断信息。
如果面向人和机器两类消费者，最好把结构化结果写入 JSON/CSV，日志单独输出。

## 6. 路径、文件与结构化输出

使用 `pathlib.Path`，避免手工拼接 `/` 和 `\`：

```python
import json
from pathlib import Path


def write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)
```

这里先写临时文件再替换，能降低进程中断后留下半个 JSON 的概率。
并发写同一路径时还需要唯一运行目录、文件锁或外部存储事务。

实验输出建议按运行隔离：

```text
results/
└── 2026-08-20T120000Z-opt125m-fp16/
    ├── config.json
    ├── environment.json
    ├── benchmark.json
    ├── run.log
    └── profile-trace.json
```

原始结果一旦生成就不手工修改；派生表格和图从原始 JSON/CSV 自动生成。

## 7. 日志：记录事件，不是到处 print

库代码使用模块级 logger，入口统一配置格式、级别和输出位置：

```python
import logging

logger = logging.getLogger(__name__)


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def load_model(name: str) -> None:
    logger.info("loading model name=%s", name)
```

建议在日志中包含：

- 时间、级别、模块名和事件；
- 模型、设备、dtype、shape、batch size 和 run id；
- 阶段边界，如 load、warmup、measure、profile、save；
- 异常堆栈和可操作的上下文。

避免记录 token、密码、私钥、完整请求正文和未脱敏的数据路径。
高频循环内的日志会改变性能，Benchmark 测量区间应关闭不必要日志。

## 8. 异常、退出码与资源清理

异常应在能够补充上下文或恢复的位置捕获，不要用空 `except` 吞掉根因：

```python
try:
    result = run_benchmark(config)
except FileNotFoundError as exc:
    logger.error("input file is missing: %s", exc.filename)
    return 2
except RuntimeError:
    logger.exception("benchmark failed model=%s", config.model)
    return 1
```

规则：

- 参数和配置错误尽早失败；
- 异常链使用 `raise NewError(...) from exc` 保留原始原因；
- 可预期错误给出修复建议；
- 不可恢复错误返回非零退出码；
- 文件、锁和临时资源用上下文管理器释放。

不要在捕获 CUDA OOM 后假装本轮结果有效。记录失败配置，释放引用，
必要时终止当前进程，重新启动一个干净进程继续实验矩阵。

## 9. 类型标注与数据模型

类型标注的价值是让模块边界、可空值和数据形状更清晰，而不是消灭所有运行时错误。

```python
from collections.abc import Sequence


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    if not 0 <= q <= 1:
        raise ValueError("q must be between 0 and 1")
    ...
```

建议优先标注：

- 公共函数参数和返回值；
- 配置、结果和外部 I/O 边界；
- `None`、异常和多种返回类型；
- 回调函数和可迭代对象。

Tensor 的 dtype、device 和 shape 通常不能只靠普通 Python 类型完整表达，
仍需运行时断言、测试和结果元数据。

## 10. 测试：先测纯逻辑，再测昂贵集成

测试金字塔可以分为：

| 层次 | 示例 | 特点 |
|---|---|---|
| 单元测试 | 分位数、配置校验、路径生成 | 快、确定、无需 GPU |
| 集成测试 | 小模型一次 CPU 前向 | 检查模块协作 |
| GPU smoke test | CUDA 可用时运行小 shape | 数量少、条件跳过 |
| Benchmark 回归 | 固定环境比较性能分布 | 不能当普通单测运行 |

pytest 示例：

```python
import pytest

from my_benchmark.metrics import percentile


def test_percentile_rejects_empty_values() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        percentile([], 0.5)


@pytest.mark.parametrize(
    ("values", "q", "expected"),
    [([1.0], 0.5, 1.0), ([1.0, 3.0], 0.5, 2.0)],
)
def test_percentile(values: list[float], q: float, expected: float) -> None:
    assert percentile(values, q) == pytest.approx(expected)
```

GPU、网络和大模型测试不应成为所有开发者运行单元测试的前提。
通过 marker、环境检测和小型 fixture 把它们隔离。

## 11. 可复现性与随机性

固定随机种子只是可复现的一部分：

```python
import random

import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

还要固定或记录：

- 代码 commit、依赖版本和模型 revision；
- 数据集版本、样本顺序和预处理；
- dtype、device、shape、batch、线程数和环境变量；
- warmup、repeat、同步点和统计方法；
- 确定性算法设置及其性能影响。

完全确定性可能降低性能或不被某些算子支持。正确做法是明确目标：
功能回归需要尽量确定；真实性能测试需要记录实际生产设置，不能为了复现而偷偷改变算法。

## 12. Python 与性能的边界

Python 代码不一定是 GPU 工作负载的瓶颈，但以下模式很常见：

- 循环启动大量小 Kernel，CPU launch 开销占比高；
- 在热路径中调用 `.item()`、`.cpu()` 或打印 CUDA Tensor，引入同步；
- 每轮重新创建 Tensor、Tokenizer 或模型；
- 用 Python 列表逐元素处理本可向量化的数据；
- 频繁 JSON 序列化、日志或磁盘写入污染测量区间；
- 数据加载进程、线程或 GIL 限制让 GPU 等待输入。

优化顺序应是：先定义端到端指标，再测量，再用 Profiler 找证据，最后修改。
不要因为“Python 慢”就提前改写成 C++ 或增加多进程。

## 13. 代码质量与自动化

本仓库的最小检查：

```bash
python -m ruff check .
python -m pytest
python -m mkdocs build --strict
```

提交前还应检查：

- `git diff --check` 没有空白错误；
- 没有模型权重、数据集、trace 和缓存；
- 没有密钥、`.env` 或私人路径；
- README 中的命令可以从干净环境执行；
- 生成物与源代码分离。

CI 是团队共享的最低门槛，不替代本地测试，也不能证明性能结论正确。

## 14. 第一周 Python 工程实战

### 练习 A：解释器与项目结构

1. 创建 `.venv` 并确认 `python`、`pip` 指向同一环境。
2. 将一个单文件脚本拆为 CLI、配置、核心逻辑和指标模块。
3. 使用 `python -m package.cli --help` 运行入口。
4. 解释为什么导入核心模块不会加载模型或占用 GPU。

### 练习 B：配置、日志与结果

1. 用 dataclass 表达模型、设备、dtype、warmup 和 repeat。
2. 对非法配置尽早抛错。
3. 为 load、warmup、measure 和 save 阶段增加结构化日志字段。
4. 将最终配置、环境和结果写入独立运行目录。

### 练习 C：测试与失败路径

1. 为分位数函数编写正常值、边界值和异常测试。
2. 用临时目录测试 JSON 输出，不污染仓库。
3. 模拟 CUDA 不可用、输出目录不可写和非法 dtype。
4. 运行 Ruff 与 pytest，修复所有错误。

### 练习 D：可复现检查

从空环境重新安装并运行一次：

```bash
python -m venv .venv-rebuild
source .venv-rebuild/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest
```

对比两次环境记录，说明哪些信息相同、哪些仍可能导致不同结果。

## 15. 第一周完成检查表

- [ ] 能解释模块、包、入口、`sys.path` 和虚拟环境的关系。
- [ ] 能将 CLI、副作用和可测试核心逻辑分离。
- [ ] 能设计显式、可校验、可保存的运行配置。
- [ ] 能用 logging 记录阶段和上下文，而不污染测量区间。
- [ ] 能安全地写入结构化结果并隔离每次运行。
- [ ] 能正确传播异常、退出码并清理资源。
- [ ] 能为纯逻辑、集成路径和 GPU 路径设计分层测试。
- [ ] 能记录代码、依赖、模型、数据、硬件和 Benchmark 方法。
- [ ] 能识别 Python 热路径中的同步、I/O 和小 Kernel 问题。
- [ ] 能在干净环境中重建项目并通过全部检查。

## 16. 官方参考

- [Python Modules](https://docs.python.org/3/tutorial/modules.html)
- [Python venv](https://docs.python.org/3/library/venv.html)
- [Python argparse](https://docs.python.org/3/library/argparse.html)
- [Python logging HOWTO](https://docs.python.org/3/howto/logging.html)
- [Python pathlib](https://docs.python.org/3/library/pathlib.html)
