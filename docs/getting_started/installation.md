# 安装与主机准备

普通用户从 PyPI 安装带版本 Tag 的发行包，统一运行 `acprof <command>`，无需获取源码。
本文只说明安装、主机条件、权限和模型认证；采集示例见[实验指南](../usage/experiments.md)。
开发环境与发行包构建见[开发者安装](../development/distribution.md#本地开发环境)。

[文档导航](../README.md) · [CLI 参数](../usage/cli.md) · [运行排障](../usage/troubleshooting.md)

## 安装

### uv tool install（推荐）

准备 [uv](https://docs.astral.sh/uv/getting-started/installation/) 后执行：

```bash
uv tool install acprof
acprof --version
acprof --help
acprof tui
```

`acprof` 已在 [PyPI](https://pypi.org/project/acprof/) 发布。如遇 `No solution found` / `not found`，请先检查 Python 3.12+、包索引镜像是否同步以及网络连通性；下方的 GitHub Release Wheel 可作为备用发行来源，但第三方依赖仍可能需要从包索引下载。

工具安装使用独立环境，包含 TUI 和分析依赖；uv 可按需准备 Python 3.12+。
普通用户无需 clone 仓库、创建或激活 `.venv`，也无需运行开发用的 `uv sync`。
源码开发环境与 `uv tool` 的安装环境彼此独立；切换或删除源码工作树的 `.venv` 不会卸载已安装的工具。
若找不到命令，执行 `uv tool update-shell`，按提示刷新 PATH 或重新打开终端。
`acprof` 不带子命令时只显示 CLI 帮助；启动 TUI 请运行 `acprof tui`，查看版本和帮助不会启动采集。

### GitHub Release fallback

如需直接安装 GitHub 发布的 Wheel，可访问[最新正式版 GitHub Release](https://github.com/kainam15/AC-Prof/releases/latest)。在 Bash 终端执行以下命令，自动识别最新正式版 Tag，下载 Wheel 和同版本的 `SHA256SUMS`，校验成功后再安装：

```bash
(
  set -euo pipefail
  release_url="$(curl -fsSL -o /dev/null -w '%{url_effective}' https://github.com/kainam15/AC-Prof/releases/latest)"
  release_tag="${release_url##*/}"
  wheel="acprof-${release_tag#v}-py3-none-any.whl"
  base="https://github.com/kainam15/AC-Prof/releases/download/${release_tag}"
  curl -fLO "${base}/${wheel}"
  curl -fLO "${base}/SHA256SUMS"
  grep -F " ${wheel}" SHA256SUMS | sha256sum --check -
  uv tool install "./${wheel}"
  acprof --version
)
```

若 Release 的 Wheel 文件名或平台标签发生变化，请以[发布页](https://github.com/kainam15/AC-Prof/releases/latest)列出的实际资产为准。下载与哈希校验使用同一 Release 的文件，不要混用不同版本。

同一 Release 也提供 `acprof-linux-x86_64` standalone，可按需下载、对照 `SHA256SUMS` 核验并赋予执行权限。它面向 Linux x86_64，仍需符合 Docker 和 GPU / profiling 的主机要求。Wheel 的依赖仍由 uv 解析下载；从 GitHub 下载 Wheel **不等于**全部依赖离线可用。

### uv pip install / pip install

已有虚拟环境时激活后直接安装；新环境可使用：

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install acprof
acprof --version
```

使用 pip 的环境中执行 `pip install acprof`，随后同样运行 `acprof tui`。
安装包可以脱离仓库运行，Docker、驱动及采集工具仍由主机提供。

### 升级与固定版本

```bash
uv tool upgrade acprof
# 当前虚拟环境安装：
uv pip install -U acprof
# 使用 pip 时：
pip install -U acprof
acprof --version
```

复现实验应将安装命令中的 `acprof` 改为 `acprof==<已发布版本>`，并保留实验产物中的
runtime、模型 revision 和环境身份。发布版本与 Git Tag 的对应规则见[版本策略](../development/distribution.md#版本策略与失败恢复)。

## 快速开始

### 1. 检查主机环境

AC-Prof 的 FULL 采集要求 Native Linux、本机 Docker Engine 和统一 cgroup v2。
无 Docker 的 Linux VM 或 GPU 容器可选择 `--runtime process --profiling-mode basic`，
使用 uv 管理的隔离 Python 环境和进程级指标；前提、TUI 操作及验证边界见 [Linux Runtime](../platforms/linux-runtimes.md)。
WSL2 支持开发及 basic/PARTIAL 采集；Docker 路径需发行版内的 Docker Engine 和 cgroup v2，详见 [WSL2](../platforms/wsl2.md)。
Docker Desktop 与远程 Docker daemon 不能用于 Docker 采集；Windows 和 macOS 不能作为采集主机。
当前推荐并验证的是 Ubuntu 24.04。

Docker 路径必需条件：

- Python 3.12+；uv 可按需准备，standalone 已内置。
- 当前用户可以直接访问 `unix:///var/run/docker.sock`，无需使用 `sudo docker`。
- Host 使用统一 cgroup v2；`/sys/fs/cgroup/cgroup.controllers` 必须存在。
- Hugging Face Hub 可访问；私有或 gated 模型还需要 `HF_TOKEN`。
- `full` 模式要求 Linux RAPL powercap 可读。
- `full` 模式要求 Linux `perf` 可以访问硬件 `instructions` 事件。
- `full` 模式要求 `tcpdump`、`tshark` 和本机 Docker bridge 可用。
- 运行 `--gpus on` 时，还需要 NVIDIA driver 和 NVIDIA Container Toolkit。

安装 AC-Prof 后可先运行只读检查，查看当前缺项及处理建议：

```bash
acprof doctor --profiling-mode basic
acprof doctor --runtime process --profiling-mode basic
acprof doctor --profiling-mode full --gpus on
```

standalone 不要求目标机安装 Python。`doctor` 不自动安装软件、更改权限或启动容器；
通过仅表示所选模式的前置检查通过，仍需运行最小实验验证。

先确认 Docker 指向本机 daemon。下面的命令会清除当前 shell 的 Docker 覆盖变量，
并切换默认 context；确认你要使用本机 Docker 后再执行：

```bash
unset DOCKER_HOST DOCKER_CONTEXT
docker context use default
docker context inspect default --format '{{(index .Endpoints "docker").Host}}'
docker info --format 'OperatingSystem={{.OperatingSystem}}'
test -f /sys/fs/cgroup/cgroup.controllers
cat /proc/self/cgroup
```

`docker context inspect` 应输出 `unix:///var/run/docker.sock`。`full` 模式还需检查采集工具：

```bash
command -v perf tcpdump tshark
getcap "$(command -v tcpdump)"
ip link show docker0
find /sys/class/powercap -name energy_uj -readable -print -quit
perf stat -e instructions -- true
```

权限安装见下一节；`acprof run` 会在下载模型之前执行完整 preflight，权限不足时报告错误。

### 最小权限安装

在 TUI 中按 `F2` → **连接与权限 → 采集权限**，先“检查环境”，再选择需要授权的
perf / tcpdump 并点击“配置所选权限”。界面展示真实可执行文件与变更命令，确认后临时
切换到系统终端，由 `sudo` 请求管理员授权，结束后返回 TUI 并重新检查。管理员密码不保存。
该入口要求已安装 `perf`、`tcpdump`、`libcap2-bin` 和 `acl`；缺失工具会在执行前报告。
TUI 把工具限制到 `acprof-perf` / `acprof-capture` 组，并给当前用户增加执行 ACL，
因此当前登录会话立即可用。原 owner/group/mode、ACL 与 capability 保存在 TUI 设置目录的
`permissions-before-*.json`，失败时保留备份并停止后续步骤。内核升级后需重新配置新版本 perf。

以下是 Ubuntu 管理员的 CLI 安装方案。`setup.sh`、`doctor`、环境检查和采集进程
均不自动执行 sudo 或授予 capability；只有上述显式配置入口会请求授权。basic 模式无需 perf/tcpdump 权限。
已退役的 `ACPROF_SUDO_PASSWORD` 应从 shell、`.env` 和 `.env.local` 删除；发现该键时明确报错。
无需降低整台机器的 `perf_event_paranoid`，也无需 sudoers 免密规则。

```bash
sudo apt-get install -y "linux-tools-$(uname -r)" linux-tools-common tcpdump tshark libcap2-bin acl

# Ubuntu 的 /usr/bin/perf 通常是脚本；给当前内核对应的真实 ELF 文件授权。
acprof_perf_bin="$(readlink -f "/usr/lib/linux-tools/$(uname -r)/perf")"
acprof_tcpdump_bin="$(readlink -f "$(command -v tcpdump)")"
file "$acprof_perf_bin" "$acprof_tcpdump_bin"
stat -c '%n %U:%G %a' "$acprof_perf_bin" "$acprof_tcpdump_bin"
getcap "$acprof_perf_bin" "$acprof_tcpdump_bin"
```

确认两个路径均指向系统包安装的 ELF 可执行文件后，保存原权限输出，再执行：

```bash
sudo groupadd -f acprof-perf
sudo groupadd -f acprof-capture
sudo usermod -aG acprof-perf,acprof-capture "$USER"
sudo chown root:acprof-perf "$acprof_perf_bin"
sudo chmod 0750 "$acprof_perf_bin"
sudo setcap cap_perfmon=ep "$acprof_perf_bin"
sudo chown root:acprof-capture "$acprof_tcpdump_bin"
sudo chmod 0750 "$acprof_tcpdump_bin"
sudo setcap cap_net_raw=ep "$acprof_tcpdump_bin"
getcap "$acprof_perf_bin" "$acprof_tcpdump_bin"
```

从普通用户登录会话执行这些 sudo 命令；`$USER` 是待授权用户。两组应仅包含可信采集用户：
`CAP_PERFMON` 允许性能观测，`CAP_NET_RAW` 允许抓包，并不只限于 AC-Prof 的进程或端口。
AC-Prof 使用 `tcpdump -p` 关闭 promiscuous mode，不要求 `CAP_NET_ADMIN`；perf stat 只授予
`CAP_PERFMON`，不授予 `CAP_SYS_ADMIN`。Linux 5.9+ 支持以 `CAP_PERFMON` 附加其他用户的进程，
实际容器 PID 仍需在采集前探测；LSM、capability bounding set 或 `nosuid` 挂载可能继续限制访问。

重新登录以取得组权限，然后用普通用户执行 `perf stat -e instructions -- sleep 0.01` 和
`acprof doctor --profiling-mode full --gpus off`。抓包检查只证明工具、网卡及文件 capability 存在，
不代替真实抓包。内核或工具包升级后需重新检查真实文件路径和 capability。
撤销时由管理员对这两个真实文件执行 `setcap -r`，移除用户的采集组成员身份，并按保存的原值恢复 owner/group/mode。

方案参考 [Linux perf 安全文档](https://github.com/torvalds/linux/blob/master/Documentation/admin-guide/perf-security.rst)
和 [libpcap Linux 权限说明](https://github.com/the-tcpdump-group/libpcap/blob/master/pcap.3pcap.in)，
仅使用系统已有 capability 机制，不新增采集依赖或测量窗口内的授权操作。

### 2. 检查安装

完成上方的[安装](#安装)后，在要保存实验结果的工作目录执行：

```bash
acprof --version
acprof doctor --profiling-mode basic --gpus off
acprof tui
```

采集前先核对 TUI 中的模型、模式及输出目录；认证配置和输出路径相对于当前工作目录。

容器运行依赖由独立的平台和完整制品锁管理：7 个任务族的逻辑 profile 共享依赖环境，
当前数量和版本统一见[当前配置](../models/environment.md#当前配置)，其中 `onnxruntime-cpu` 完全不安装 Torch。
镜像按需构建和复用，分层及锁更新命令见[运行兼容](../models/environment.md#当前配置)。

### Hugging Face 认证

私有或 gated 模型可在 TUI 按 `F2` → **连接与权限 → 连接配置**填写 Token，保存后立即用于
新启动的任务；也可以手动创建当前工作目录的 `.env.local`（源码根目录启动时就是项目根目录）：

```env
HF_TOKEN=hf_xxx
# 默认自动尝试国内入口、可信存储重定向和官方 Hub
HF_DOWNLOAD_MODE=auto
HF_ENDPOINT=https://hf-mirror.com
# 可选：批量 payload 预算，未知或超限时在下载前停止
# ACPROF_MAX_DOWNLOAD=5GB
```

`.env.local` 已被 Git 忽略，可用 `chmod 600 .env.local` 限制读取权限。程序自动读取
`.env` 和 `.env.local`；令牌只用于主机检测和 Model Store 下载，正式推理容器
从只读挂载的固定 snapshot 离线加载模型，不接收令牌或在运行中下载权重。
地址、令牌的优先级与空白值处理见 [主机环境与 Hugging Face 认证](../usage/configuration.md#主机环境与-hugging-face-认证)。
