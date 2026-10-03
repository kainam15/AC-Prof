# 安装包、standalone 与发布

AC-Prof 支持源码开发、`uv tool install` 隔离安装和 Linux x86_64 standalone。
三种安装方式通过 `acprof <command>` 执行相同的主机采集代码；Docker Engine、cgroup v2、GPU driver 和采集工具仍由主机提供。
安装与首次运行见[安装指南](Getting_Started.md)，环境检查参数见 [doctor](CLI_Reference.md#acprof-doctor)。

分发包的 `License-Expression` 为 `Apache-2.0 AND CC-BY-4.0`：项目代码采用 Apache-2.0，
内置 LibriSpeech 音频采用 CC-BY-4.0。`LICENSE`、`NOTICE` 和 `licenses/CC-BY-4.0.txt`
通过 `project.license-files` 随 sdist/wheel 分发；`NOTICE` 保留原始项目贡献者和音频变换来源。
standalone 从 wheel 收集同一份 dist-info 元数据。wheel 的 Docker 构建资源也携带许可文件，
最终服务镜像在 `/usr/share/licenses/acprof/` 保存它们；许可文件变化参与服务层指纹。
外部依赖和下载的模型分别保留原许可。

## Host dependency split 评估

默认安装保持完整可用：`pip/uv install` 与 `setup.sh` 安装相同的 host 依赖，打开 TUI
和执行分析不要求理解 extras。当前不拆分发行依赖；`requirements/host.lock` 继续包含完整环境及 hashes，
runtime profile 的严格版本锁独立维护。

| 候选边界 | 当前依赖与约束 |
| --- | --- |
| core | requests、Hugging Face Hub/socksio；numpy 与 Pillow 也被确定性 workload/input preparation 使用，不能简单归入 analysis |
| TUI | Textual 可在命令路由处惰性加载；完整默认安装仍必须带上它 |
| analysis | pandas、matplotlib、Plotly；只读分析与绘图已有模块边界，但 core-only 发行还需要完整的缺依赖提示与安装测试 |
| GPU | nvidia-ml-py；CPU 路径已有可选加载，但 GPU capability 与错误提示仍须单独验收 |

新增 extras 在保持默认完整依赖时不会减少默认安装成本；改为精简默认又不符合当前开箱即用约定。
因此本轮仅保留上述职责边界，未新增组合安装模式或重写主机锁。若将来有明确的 headless/minimal
部署需求，应为每种组合增加隔离安装与实际 CLI smoke，同时继续生成完整 host hashed lock。
代码路径拆分不改变依赖许可证，也不把容器模型框架移入 host 默认安装。

## Clone 后初始化

准备原生 Linux x86_64、本机 Docker Engine/Buildx、cgroup v2 和 Git 后：

```bash
git clone https://github.com/kainam15/AC-Prof.git
cd AC-Prof
./setup.sh
```

`setup.sh` 先检查主机和 Docker，再使用已有 uv；找不到时通过官方安装脚本准备 uv 0.12.13，
默认安装到 `~/.local/bin`（可用 `UV_INSTALL_DIR` 指定）。随后用 Python 3.10 和
`requirements/host.lock` 的版本约束隔离安装当前 checkout；缺少 Python 3.10 时由 uv 下载。
重复执行会更新当前源码包、复用依赖缓存，保留项目 `.venv`、认证文件、TUI 设置和旧结果。
脚本不自动安装系统包、修改 Docker 权限或 GPU 驱动；错误修复后重试原命令即可。

安装后使用现有 `doctor --profiling-mode basic --gpus off` 检查，并打印 GPU/full 的检查命令。
只有 basic 前置检查通过才进入 TUI；检查通过不代表真实推理已成功。
交互终端自动预填 BERT 和基础 CPU smoke，每次分配独立的
`results/first-run-<时间>-<随机后缀>/`，用户点击“开始采集”后再准备镜像、下载模型和运行。
runtime 由所选模型、backend 和设备配置决定，沿用已有按需拉取/核验/构建流程。

| 参数 | 行为 |
| --- | --- |
| `--no-tui` | 完成安装和检查后退出，打印启动命令；非交互终端也自动采用此行为 |
| `--no-modify-path` | 不修改 shell 启动文件；使用打印的完整路径命令 |
| `--help` | 显示说明，不安装或检查系统 |

默认调用 `uv tool update-shell` 配置后续终端的命令路径，当前终端和自动启动均使用完整路径，
不要求重新登录。初始化及其结果目录以源码根目录为工作目录；手动启动的 `acprof` 则以调用时的目录为准。
自动化隔离安装可使用 uv 原生的 `UV_TOOL_DIR`、`UV_TOOL_BIN_DIR`、`UV_CACHE_DIR` 和
`UV_PYTHON_INSTALL_DIR`，配合 `--no-tui --no-modify-path`。

安装完成后，在新终端统一使用：

```bash
acprof tui
```

需要入门预设时运行 `acprof tui --preset smoke`。`setup.sh` 通过 `uv tool dir --bin`
定位已安装的 `acprof`，直接调用其 `tui` 子命令，支持自定义 `UV_TOOL_BIN_DIR`。
当前终端尚未刷新 PATH 时，使用脚本输出的完整路径命令；隔离安装无需创建项目 `.venv`。

## Python 工具安装

在包含 `pyproject.toml` 的源码目录中：

```bash
uv tool install .
uv tool update-shell
```

重新打开终端或按 uv 提示刷新 `PATH` 后运行：

```bash
acprof --version
acprof doctor --profiling-mode basic
acprof tui
```

也可以安装 Release 的 wheel：`uv tool install ./acprof-0.2.0-py3-none-any.whl`。
远端源码包含本版本后，可直接运行
`uv tool install git+https://github.com/kainam15/AC-Prof.git`；复现实验应固定 Git tag 或 commit。
这里只使用源码和 Release 制品，不假设 PyPI 已有同名官方发行包。

唯一公开入口是 `acprof <command>`；子命令见 [CLI 参数](CLI_Reference.md#cli-参数)。
源码开发先安装 editable 包，再使用同一入口。根目录不包含 Python 文件或额外启动器。
`run --help` 等命令沿用各自的参数定义；顶层帮助和版本查询不会加载 Textual、绘图库或推理框架。
Python 依赖声明位于 `pyproject.toml`；开发和 Release 构建采用 `requirements/host.lock` 中已验证的制品。

## 工作目录与资源

输出目录、用户 workload 相对路径及 `.env` / `.env.local` 相对于**启动时的当前工作目录**。
TUI 设置仍写入 XDG 用户配置目录，以工作目录的摘要隔离不同实验工作区。
在源码仓库根目录启动时，路径行为与原来的操作示例相同。

wheel 内置 Dockerfile、平台/环境锁、扩展声明、音频素材及构建所需的 Python 源码。
`installation.resource_root()` 定位这些只读资源；它不是输出目录。
唯一 custom build hook 位于 `packaging/hatch_build.py`，由 wheel target 的 `hooks.custom.path` 指定，
并随 `packaging/` 进入 sdist。它使用临时目录复制 `acprof/`、`dockerfiles/`、`assets/`、`examples/`，
按原有文件后缀白名单筛选，排除 `.env`、`AGENTS.md`、`__pycache__` 与嵌套 `_bundle`。
`.dockerignore`、`LICENSE`、`NOTICE`、`licenses/CC-BY-4.0.txt` 一并复制，通过
`build_data["force_include"]` 写入 wheel 的 `acprof/_bundle`；构建结束清理临时目录。
Docker 模型层仍由本机按固定 revision 下载，令牌经 BuildKit secret 传入。

离线 report 的 HTML/CSS/JavaScript 和 Plotly.js MIT 许可随 `acprof.plotting` 打包；
standalone 同时收集 Plotly 的 bundle 数据。报告生成时内嵌资源，不从 CDN 下载。
`scripts/check_distribution.py` 在空工作目录实际生成 HTML，核对模板与 bundle 可用性。

editable 安装（`uv pip install -e .`）直接从当前 checkout 读取这些资源，build hook
不生成 `acprof/_bundle` 副本，避免保留过期源码副本，以及 IDE 同时索引两份同名 Python 符号。
普通 wheel 仍携带完整资源，standalone 继续从 wheel 收集资源。

### 发行包验证

使用独立的构建环境安装 `build` 后执行 `python -m build`，生成 sdist 并从该 sdist 构建 wheel；
另用 `python -m build --wheel --outdir <direct-wheel-dir>` 验证直接从 checkout 构建的 wheel。
两份 wheel 分别安装到全新 venv，依赖使用 `requirements/host.lock`，从仓库外的空目录执行：

```bash
<venv>/bin/python -I <checkout>/scripts/check_distribution.py \
  --wheel <wheel-file> --sdist <sdist-file> --report <evidence-file>
```

此检查通过真实 console script 运行所有公共命令的帮助，检查 wheel 资源与 sdist 的 hook、
根目录约束，确认安装包来源位于 venv 内，并实际暂存 Docker service context、核对源文件指纹。
同时验证缺少 Docker 时的 doctor JSON、离线 HTML 与 packet worker。context 暂存不代表镜像构建或推理成功。
checkout 与 editable 从同一源码树取资源；两种 wheel 从 `_bundle` 取资源，相同输入应生成相同的
service context 指纹。editable 的安装路径应回到 checkout，且不生成 `_bundle`。

## Linux standalone

[`release.yml`](../.github/workflows/release.yml) 在 Ubuntu 22.04、Python 3.10 上构建
`acprof-linux-x86_64`，目标为 glibc 2.35+ 的原生 Linux x86_64。
它包含 Python 解释器、主机依赖和构建资源；目标机无需先安装 Python 或 uv。
Docker、RAPL、perf、抓包及 NVIDIA 的要求仍按所选模式检查。

Release 发布后，从 [GitHub Releases](https://github.com/kainam15/AC-Prof/releases)
下载 `acprof-linux-x86_64` 和 `SHA256SUMS`，在下载目录核对并运行：

```bash
sha256sum --check --ignore-missing SHA256SUMS
chmod +x acprof-linux-x86_64
./acprof-linux-x86_64 doctor --profiling-mode basic
./acprof-linux-x86_64 tui
```

PyInstaller 单文件模式启动时会解压到临时目录，该目录需要支持可执行文件和符号链接。
需要时通过 `TMPDIR` 选择合适的临时目录。采集子进程使用同一可执行文件的内部 worker 入口，
不会把它误当成系统 Python；worker 只允许 client 和两个 packet 模块。

本地重现构建（已有 `.venv/bin/uv` 时可替换下列 `uv`）：

```bash
uv build --out-dir internal-testing/distribution/dist
uv venv internal-testing/distribution/venv --python 3.10
uv pip install --python internal-testing/distribution/venv/bin/python -r requirements/host.lock
uv pip install --python internal-testing/distribution/venv/bin/python --no-deps internal-testing/distribution/dist/*.whl
uv pip install --python internal-testing/distribution/venv/bin/python 'pyinstaller==6.22.3'
internal-testing/distribution/venv/bin/python scripts/check_distribution.py
internal-testing/distribution/venv/bin/python -m PyInstaller --noconfirm packaging/acprof.spec \
  --distpath internal-testing/distribution/standalone --workpath internal-testing/distribution/build
internal-testing/distribution/venv/bin/python scripts/check_distribution.py \
  --binary internal-testing/distribution/standalone/acprof
```

本机生成的 binary 使用本机构建环境的 glibc 下限；只有相应 runner 上构建并验证的资产才能按
Release 的平台范围分发。wheel/standalone 的 smoke 验证不能替代真实 Docker/GPU 采集。

## 发布入口与范围

维护者更新 `acprof.__version__`，验证后推送对应 `v<version>` tag：

- `release.yml` 构建 sdist、wheel、standalone 与 SHA256 清单；tag 发布还必须等待同一提交调用的
  `ci.yml` 完成 lint、全部主机分片、ONNX CPU 和 runtime 容器测试，以及隔离安装/worker 验证。
- `runtime-images.yml` 先构建/核验 4 个平台，再让 24 个环境 job 拉取已发布平台，构建、核验并发布环境。
- 手动运行 Release workflow 只构建并保存 Actions artifacts；手动运行 GHCR workflow 会发布镜像。

Release 附带 `verification.json`，记录源码 SHA、CI run 和硬件证据范围。硬件报告仅关联同一 SHA
上成功的 `hardware.yml` run，且 `hardware` artifact 尚未过期；否则明确标为 `not_verified`。
这不是硬件强制门禁，也不代表所有设备或 profiler 均已验收，具体范围以对应 artifact 为准。
复用工作流遵循 [GitHub reusable workflows](https://docs.github.com/en/actions/reference/workflows-and-actions/reusing-workflow-configurations)，
本地相对路径保证执行同一提交；普通 CI 处理分支 push，tag 的 CI 由 release 调用，避免重复运行。

工作流文件存在不代表远端资产已发布。实际发布需要仓库中的 Actions 正常完成，以及 GitHub 的
`contents: write` / `packages: write` 权限。GHCR package 首次发布后，维护者需在 package 设置中
确认 public 可见性，匿名用户才能直接拉取；私有 package 需要先 `docker login ghcr.io`。

GHCR 只预构建平台和依赖环境，不发布模型权重、用户数据或包含令牌的层。
拉取策略、内容身份和失败回退见[预构建依赖镜像](Runtime_Compatibility.md#ghcr-预构建依赖镜像)。
发布脚本的 `--report` 保存 image ID、内容 tag 和清单核验范围；依赖核验不代表 GPU 推理已经通过。

## 参考实现与取舍

- [uv tools](https://github.com/astral-sh/uv/blob/main/docs/guides/tools.md)（MIT / Apache-2.0）：
  采用标准 console script 与隔离工具环境。`setup.sh` 只串联安装和已有诊断；uv 引导使用
  [官方 installer](https://docs.astral.sh/uv/reference/installer/) 的 `UV_INSTALL_DIR` / `UV_NO_MODIFY_PATH`，
  不复制包管理逻辑。uv 引导版本与开发锁、Release 工具链一同维护。
  `setup.sh` 复用 [uv 的目录查询接口](https://github.com/astral-sh/uv/blob/main/crates/uv/src/commands/tool/dir.rs)
  定位已安装的 `acprof`，不推测内部虚拟环境布局；查询只发生在安装后，不增加测量期开销。
- [Hatch build hooks](https://github.com/pypa/hatch/tree/master/backend/src/hatchling/builders/hooks)
  （MIT）：用一个小型 build hook 打包既有资源；按官方
  [wheel 构建版本](https://github.com/pypa/hatch/blob/master/docs/plugins/builder/wheel.md)
  区分 `standard` 与 `editable`，只在发行 wheel 中复制资源，不改变运行时依赖和镜像配方。
  [custom hook 源码](https://github.com/pypa/hatch/blob/master/backend/src/hatchling/builders/hooks/custom.py)
  支持项目内的显式 `path`；迁移路径时保留 `BuildHookInterface` 和 `force_include` 逻辑。
  [Issue #1627](https://github.com/pypa/hatch/issues/1627) 记录了 editable 中 force-include 同名包遮蔽源码的风险，
  因此保留 editable 跳过 bundle 的分支。复用既有 Hatchling 1.x 构建接口，无新增运行依赖或测量期开销。
- [PyInstaller](https://github.com/pyinstaller/pyinstaller)（GPL 与分发例外）：使用官方冻结工具，
  按其[资源与子进程说明](https://pyinstaller.org/en/stable/runtime-information.html)处理真实源码、动态模块和系统库路径。
  构建工具不进入主机运行依赖，不将 Torch/CUDA 安装到主机包。
- [Docker Actions](https://github.com/docker/build-push-action)（Apache-2.0）：借鉴 GHCR 登录与分阶段发布方式，
  继续使用项目自己的依赖构建器，保留完整包清单、父镜像 ID 和内容指纹验证。

这些工具均有官方维护仓库；这里只复用公开接口和发布方式。安装、doctor、下载和镜像核验都位于测量窗口之外。
