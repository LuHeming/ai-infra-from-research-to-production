# Linux 工程基础：AI Infra 必备知识与实战

Linux 是 AI Infra 的主要工作环境。本章不追求背完所有命令，而是建立一套能够回答以下问题的工作方法：

1. 文件、进程、设备和网络连接分别在哪里、由谁拥有？
2. 一个训练或推理任务正在做什么，为什么慢、为什么退出？
3. 如何安全地在本机、远程服务器和容器之间移动代码与结果？
4. 如何留下足够的环境、命令和日志，让实验可以复现？

## 1. 先建立 Linux 心智模型

Linux 可以先理解为四层：

| 层次 | 作用 | 常见观察入口 |
|---|---|---|
| 硬件 | CPU、内存、磁盘、网卡、GPU | `lscpu`、`free`、`lsblk`、`nvidia-smi` |
| 内核 | 管理进程、内存、设备、文件系统和网络 | `/proc`、`/sys`、`dmesg` |
| 用户空间 | Shell、systemd、SSH、Python、CUDA 工具 | `bash`、`systemctl`、`ssh` |
| 应用 | 训练脚本、推理服务、Benchmark | Python 进程、日志、指标 |

三个重要原则：

- **一切皆可通过文件或文件描述符访问**：普通文件、终端、设备、管道和 socket 都可被进程读写。
- **进程是资源和权限的基本边界**：排障时先找到 PID，再看它的用户、命令、资源、文件和网络连接。
- **退出码和标准流是命令协作协议**：退出码 `0` 通常表示成功，非 `0` 表示失败；标准输入、输出和错误让命令可以组合。

## 2. Shell、命令与帮助系统

### 2.1 命令是怎样被执行的

Shell 会从左到右解析命令名、参数、重定向和管道，然后在当前目录或 `PATH` 中寻找可执行文件。

```bash
pwd                          # 当前目录
echo "$SHELL"                # 当前登录 Shell
echo "$PATH" | tr ':' '\n'   # 按行查看命令搜索路径
type -a python               # python 是别名、内建命令还是可执行文件
which python                 # PATH 中首先匹配的可执行文件
command -v python            # 脚本中更适合使用的查询方式
```

Shell 会展开变量、通配符和命令替换。无法确定内容时始终给变量加双引号：

```bash
project_dir="$PWD"
printf '%s\n' "$project_dir"
files=$(find docs -type f -name '*.md')
```

双引号允许 `$VAR` 展开，单引号保留字面内容。不要把不可信字符串拼成命令交给 `eval`。

### 2.2 获取帮助

```bash
man rsync
rsync --help
apropos "copy files"
help cd                      # Bash 内建命令用 help
```

阅读命令说明时重点找：参数格式、默认行为、退出码、是否递归、是否覆盖以及示例。

### 2.3 命令历史与快捷键

```bash
history
history | grep ssh
```

常用快捷键：

- `Ctrl+C`：向前台进程发送 `SIGINT`。
- `Ctrl+Z`：暂停前台进程，配合 `bg`、`fg` 使用。
- `Ctrl+D`：发送输入结束；空 Shell 中通常会退出会话。
- `Ctrl+R`：反向搜索历史命令。
- `Ctrl+A` / `Ctrl+E`：移动到行首 / 行尾。
- `Tab`：补全命令、路径和部分参数。

## 3. 文件系统与常用目录

### 3.1 目录结构

| 路径 | 典型用途 |
|---|---|
| `/home/<user>` | 用户代码、配置和个人文件 |
| `/root` | root 用户的家目录，不是文件系统根目录 |
| `/etc` | 系统级配置 |
| `/var/log` | 持久化日志 |
| `/tmp` | 临时文件，可能在重启后清理 |
| `/proc` | 内核提供的进程和系统信息 |
| `/sys` | 设备与内核对象信息 |
| `/dev` | 设备文件，如磁盘和终端 |
| `/mnt`、`/data` | 常见数据盘或共享存储挂载点，具体以服务器为准 |
| `/usr/local`、`/opt` | 本地或第三方软件 |

Linux 路径区分大小写。绝对路径从 `/` 开始，相对路径以当前目录为基准；`.` 表示当前目录，`..` 表示父目录。

### 3.2 查看、创建、复制与移动

```bash
ls -lah
find docs -type f -name '*.md'
mkdir -p benchmarks/raw-results
cp -a source_dir backup_dir
mv old_name new_name
touch run.log
file artifact.bin
stat artifact.bin
```

使用删除命令前，先用 `pwd`、`realpath` 和不带删除动作的查询命令确认范围：

```bash
pwd
realpath benchmarks/raw-results
find benchmarks/raw-results -maxdepth 1 -type f
```

`rm -rf` 不经过回收站。尤其不要对空变量、通配符展开结果、根目录或家目录执行递归删除。

### 3.3 查看与搜索文本

```bash
less run.log
head -n 20 run.log
tail -n 50 run.log
tail -f service.log
rg -n "CUDA|out of memory" logs/
find . -type f -size +1G
```

`less` 中可用 `/keyword` 搜索、`n` 跳到下一个结果、`q` 退出。代码仓库内搜索优先用 `rg`，按文件属性查找使用 `find`。

### 3.4 链接、inode 与磁盘空间

- 硬链接指向同一个 inode，通常不能跨文件系统。
- 符号链接保存另一个路径，可以跨文件系统，但目标删除后会失效。
- `df` 看文件系统剩余空间，`du` 看目录实际占用，两者回答的问题不同。

```bash
ln -s /data/models models
readlink -f models
df -hT
du -sh ./* | sort -h
df -i                         # inode 耗尽也会表现为无法创建文件
lsblk -f
mount | column -t
```

AI 项目中，模型权重、数据集、编译缓存和 trace 很容易占满磁盘。将大文件放到合适的数据盘，并通过 `.gitignore` 排除。

## 4. 标准流、重定向与管道

每个进程通常有三个标准文件描述符：

| 编号 | 名称 | 默认位置 |
|---|---|---|
| `0` | stdin，标准输入 | 键盘或上一个管道 |
| `1` | stdout，标准输出 | 终端或下一个管道 |
| `2` | stderr，标准错误 | 终端 |

```bash
python benchmark.py > run.log              # 覆盖标准输出
python benchmark.py >> run.log             # 追加标准输出
python benchmark.py > run.log 2> error.log # 分开记录
python benchmark.py > run.log 2>&1          # 合并到同一文件
python benchmark.py 2>&1 | tee run.log      # 屏幕和文件各一份
```

管道把左侧命令的 stdout 交给右侧命令的 stdin：

```bash
ps aux | rg 'python|vllm'
find . -type f -name '*.json' -print0 | xargs -0 -r du -h
```

在 Bash 脚本中推荐启用严格模式，并让管道中任一命令失败时整个管道失败：

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
```

使用 `$?` 查看上一条命令的退出码：

```bash
python -m pytest
echo "$?"
```

## 5. 用户、用户组与权限

### 5.1 读懂权限

```bash
id
whoami
ls -l scripts/create_daily_note.py
```

`-rwxr-x---` 可拆成：文件类型、所有者权限、所属组权限、其他用户权限。`r`、`w`、`x` 分别是读、写、执行；对目录而言，`x` 表示能够进入和访问目录项。

```bash
chmod u+x script.sh
chmod 640 config.yaml
chown user:group file        # 通常需要管理员权限
umask
```

数字权限由 `r=4`、`w=2`、`x=1` 相加。`640` 表示所有者读写、组只读、其他用户无权限。

### 5.2 sudo 与最小权限

`sudo` 只应用在确实需要管理员权限的单条命令上。不要为了省事用 root 运行 Python、Jupyter 或训练任务，否则生成的缓存和结果可能变成普通用户无法修改的 root 文件。

密钥和配置的常见权限：

```bash
chmod 700 ~/.ssh
chmod 600 ~/.ssh/id_ed25519
chmod 600 .env
```

不要把 token、私钥、`.env`、云凭据或带签名的下载链接提交到 Git。

## 6. 进程、作业、信号与会话

### 6.1 找到目标进程

```bash
ps aux
ps -ef --forest
pgrep -af 'python|vllm'
top
```

每个进程有 PID 和父进程 PPID。排障时记录启动用户、完整命令、启动时间、CPU/内存占用和父子关系。

```bash
ps -o pid,ppid,user,stat,%cpu,%mem,etime,cmd -p <PID>
tr '\0' ' ' < /proc/<PID>/cmdline
ls -l /proc/<PID>/fd
cat /proc/<PID>/status
```

### 6.2 信号与退出

```bash
kill -TERM <PID>             # 请求进程优雅退出
kill -INT <PID>              # 类似 Ctrl+C
kill -KILL <PID>             # 最后手段，进程无法清理资源
```

先发送 `SIGTERM` 并等待；只有进程无响应且确认 PID 正确时才使用 `SIGKILL`。进程被信号终止时，Shell 中常见退出码为 `128 + 信号编号`，例如 `137` 往往表示收到 `SIGKILL`，但还需结合内核日志判断是否为 OOM Killer。

### 6.3 前后台作业

```bash
python train.py > train.log 2>&1 &
jobs -l
fg %1
bg %1
wait <PID>
```

单纯在命令末尾加 `&` 不保证 SSH 断开后进程继续运行。长任务优先使用任务调度系统；个人调试可使用 `tmux`。`nohup` 适合简单场景，但不提供资源调度、重试和完整状态管理。

## 7. CPU、内存、磁盘与 GPU 观察

### 7.1 CPU 与负载

```bash
lscpu
uptime
top
pidstat -p <PID> 1           # 通常由 sysstat 提供
```

Load Average 表示可运行或不可中断等待的任务数量，不等于 CPU 使用率。解释负载时要同时考虑逻辑 CPU 数量、I/O 等待以及任务是否被 CPU 配额限制。

### 7.2 内存与 OOM

```bash
free -h
vmstat 1
ps -eo pid,user,rss,vsz,cmd --sort=-rss | head
dmesg -T | rg -i 'out of memory|killed process'
```

- RSS 是进程当前驻留在物理内存中的页。
- VSZ 是虚拟地址空间，不等于实际物理内存占用。
- Linux 会利用空闲内存做缓存，判断是否紧张应重点看 `available`，不能只看 `free`。
- 容器或作业系统中的 OOM 可能由 cgroup 限额触发，即使宿主机仍有空闲内存。

### 7.3 磁盘与 I/O

```bash
df -hT
du -sh ~/.cache/* | sort -h
iostat -xz 1                 # 通常由 sysstat 提供
lsof -p <PID>                # 进程打开的文件
lsof +L1                     # 已删除但仍被进程占用的文件
```

如果 `df` 显示空间不足但 `du` 找不到大文件，检查被删除但仍打开的文件、挂载点差异、文件系统配额和 inode。

### 7.4 NVIDIA GPU

```bash
nvidia-smi
nvidia-smi pmon -s um -d 1
watch -n 1 nvidia-smi
```

重点看 GPU 利用率、显存占用、功耗、温度、时钟、进程 PID 和驱动版本。注意：

- `nvidia-smi` 的瞬时利用率只是采样值，不能代替 Profiler。
- 有显存占用不等于 GPU 正在计算。
- GPU 利用率低可能是 CPU 数据准备、磁盘、网络、同步或小 Kernel 启动开销导致。
- 进程退出后显存仍占用时，先核对 PID、容器命名空间和残留子进程，不要直接重启服务器。

## 8. 日志、systemd 与内核消息

```bash
journalctl -b                           # 本次启动日志
journalctl -u ssh --since today
journalctl -f                           # 持续跟踪
systemctl status <service>
systemctl list-units --type=service
dmesg -T | tail -n 100
```

诊断服务时区分三类信息：

1. 应用日志：请求、异常和业务状态。
2. 服务管理器日志：启动命令、退出码、重启行为和环境。
3. 内核日志：OOM、磁盘、驱动、网络和硬件问题。

部分日志需要管理员权限。不要在公开 issue 中粘贴含用户名、内网地址、token、完整环境变量或数据路径的日志。

## 9. 网络基础与端口排障

### 9.1 基本对象

- IP 标识网络接口地址。
- 端口标识主机上的服务入口。
- DNS 把域名解析为地址。
- TCP 提供可靠字节流；UDP 不建立连接、开销更低。
- `127.0.0.1` 只允许本机访问；`0.0.0.0` 表示监听所有 IPv4 接口，不代表一个可访问的目标地址。

```bash
ip addr
ip route
getent hosts github.com
ss -lntp
ss -ntp
curl -v http://127.0.0.1:8000/health
```

服务无法访问时按顺序检查：进程是否存在、端口是否监听、绑定地址是否正确、本机请求是否成功、防火墙或安全组是否放行、DNS/代理是否正确。

### 9.2 下载与接口调试

```bash
curl -fL -o artifact.tar.gz 'https://example.com/artifact.tar.gz'
curl -fsS http://127.0.0.1:8000/health
```

下载大模型或数据时记录 URL、版本或 commit、校验和与许可协议。对重要文件使用 `sha256sum` 验证完整性。

## 10. SSH、tmux 与 rsync

### 10.1 SSH

```bash
ssh-keygen -t ed25519 -C "your-name@device"
ssh user@gpu-server
ssh -J user@jump-host user@gpu-server
ssh -L 8888:127.0.0.1:8888 user@gpu-server
```

`~/.ssh/config` 示例：

```sshconfig
Host gpu-dev
    HostName gpu.example.com
    User alice
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60
```

不要上传私钥。首次连接要核对服务器主机指纹；主机指纹意外变化时先联系管理员确认，不要直接删除 `known_hosts` 记录绕过检查。

### 10.2 tmux

```bash
tmux new -s profiling
tmux ls
tmux attach -t profiling
```

默认前缀是 `Ctrl+B`：

- `Ctrl+B` 后按 `d`：退出当前会话但保持任务运行。
- `Ctrl+B` 后按 `c`：新建窗口。
- `Ctrl+B` 后按 `n` / `p`：切换下一个 / 上一个窗口。
- `Ctrl+B` 后按 `[`：进入复制和滚动模式。

### 10.3 rsync

```bash
rsync -avh --progress ./ user@gpu-dev:~/project/
rsync -avh --dry-run --delete ./docs/ user@gpu-dev:~/project/docs/
```

`--delete` 会删除目标端不存在于源端的文件，必须先使用 `--dry-run` 核对。同步代码优先使用 Git；`rsync` 更适合实验产物、数据和无法进入版本控制的文件。

## 11. 环境变量、软件包与动态库

```bash
env | sort
printenv PATH
export CUDA_VISIBLE_DEVICES=0
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip freeze
```

优先使用 `python -m pip`，避免 `pip` 与 `python` 指向不同环境。定位包和动态库问题：

```bash
python -c 'import sys; print(sys.executable); print(*sys.path, sep="\n")'
python -c 'import torch; print(torch.__file__)'
ldd /path/to/library.so
echo "$LD_LIBRARY_PATH" | tr ':' '\n'
```

`PATH` 决定可执行文件搜索顺序，`LD_LIBRARY_PATH` 会影响动态库加载。不要把未经理解的大段环境修改永久写入 `.bashrc`；先在当前 Shell 验证，并记录修改原因。

在 Ubuntu/Debian 中：

```bash
sudo apt update
apt search <package>
apt show <package>
sudo apt install <package>
```

区分 NVIDIA 驱动、系统 CUDA Toolkit、Python 环境中的 CUDA Runtime 和 PyTorch 构建版本。它们相关但不是同一个组件，不要仅凭 `nvcc --version` 判断 PyTorch 实际使用的 CUDA 版本。

## 12. Bash 脚本最小规范

```bash
#!/usr/bin/env bash
set -Eeuo pipefail

readonly project_dir="${1:-$PWD}"

if [[ ! -d "$project_dir" ]]; then
    printf 'directory not found: %s\n' "$project_dir" >&2
    exit 1
fi

printf 'project=%s\n' "$(realpath "$project_dir")"
```

建议：

- 变量引用使用双引号：`"$value"`。
- 路径和参数通过命令行传入，不硬编码个人目录。
- 失败时输出到 stderr 并返回非零退出码。
- 用 `mktemp -d` 创建临时目录，用 `trap` 清理。
- 用 ShellCheck 做静态检查。
- 复杂的数据处理和业务逻辑使用 Python，而不是无限扩张 Bash 脚本。

## 13. 通用故障排查流程

遇到“任务慢、服务不通、进程消失、磁盘写不进去”时，遵循以下流程：

1. **定义现象**：期望、实际、首次发生时间、是否稳定复现。
2. **保留现场**：记录完整命令、退出码、日志、PID、环境和版本。
3. **确认作用域**：单个进程、用户、容器、节点，还是整个集群。
4. **从外到内检查资源**：磁盘、内存、CPU、GPU、网络、权限。
5. **建立最小复现**：固定输入，移除无关组件，只改变一个变量。
6. **验证假设**：每次操作前写下预期观察，避免无目标地尝试命令。
7. **记录结论与局限**：将稳定结论沉淀到 `docs/`，个案进入 `troubleshooting/`。

一个最小现场采集清单：

```bash
date --iso-8601=seconds
hostname
whoami
pwd
git rev-parse HEAD
uname -a
uptime
free -h
df -hT
nvidia-smi
python --version
python -c 'import torch; print(torch.__version__, torch.version.cuda)'
```

公开前需删除主机名、用户名、内网 IP、凭据和不应公开的数据路径。

## 14. 第一周 Linux 实战

### 练习 A：文件、管道与权限

1. 创建 `~/ai-infra-lab/logs` 和 `results`。
2. 用一条管道找出项目内最大的 10 个文件。
3. 运行一个同时输出 stdout 和 stderr 的 Python 命令，并分别保存两类输出。
4. 创建一个只允许所有者读写的配置文件，解释其数字权限。

验收：能够说明绝对/相对路径、`df`/`du`、stdout/stderr 和目录 `x` 权限的区别。

### 练习 B：进程与资源

1. 启动一个持续运行的 Python 进程并记录 PID。
2. 使用 `ps`、`top` 和 `/proc/<PID>` 查看命令、资源与文件描述符。
3. 先用 `SIGTERM` 结束，记录退出行为；再解释为什么不应默认使用 `SIGKILL`。
4. 查看系统内存、磁盘和 GPU 状态，保存为带时间戳的文本记录。

验收：给定一个 PID，能在 5 分钟内回答“谁启动、运行多久、占用什么资源、在读写什么”。

### 练习 C：远程开发

1. 配置一个 SSH Host 别名并用密钥登录测试机。
2. 创建 `tmux` 会话，在其中运行任务，断开 SSH 后重新连接。
3. 用 `rsync --dry-run` 预览一次同步，再执行同步。
4. 使用本地端口转发访问远程 Jupyter 或测试服务。

验收：断网或关闭终端后任务仍可恢复查看，且没有在仓库中提交私钥或 token。

### 练习 D：故障定位演练

从以下场景选择一个，填写 `templates/troubleshooting-template.md`：

- 端口已被占用。
- 磁盘空间或 inode 耗尽。
- Python 与 pip 指向不同环境。
- 进程收到 `SIGKILL` 或发生内存不足。
- GPU 有显存占用但利用率很低。

验收：记录复现步骤、证据、根因、修复方法、验证方式和预防措施。

## 15. 第一周完成检查表

- [ ] 能解释 Linux 用户空间、内核、进程和文件描述符的关系。
- [ ] 能安全地查找、复制、移动、同步和清理文件。
- [ ] 能组合管道、重定向并检查退出码。
- [ ] 能读懂并正确设置文件与目录权限。
- [ ] 能定位进程并使用合适的信号结束它。
- [ ] 能判断 CPU、内存、磁盘、GPU 或网络是否为当前瓶颈。
- [ ] 能从应用日志、systemd 日志和内核日志收集证据。
- [ ] 能通过 SSH、tmux 和 rsync 完成远程实验工作流。
- [ ] 能解释 Python 环境、`PATH`、动态库和 CUDA 组件的边界。
- [ ] 能提交一份已脱敏、可复现的排错记录。

完成这些项目后，再进入 PyTorch 正确计时和 Profiler。Linux 观察工具回答“系统发生了什么”，Profiler 回答“框架和算子内部发生了什么”，两类证据应互相验证。
