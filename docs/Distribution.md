# 安装包、PyPI、standalone 与发布

普通用户推荐 `uv tool install acprof`，也支持虚拟环境内的 `uv pip install acprof` / `pip install acprof`，
以及 Linux x86_64 standalone；源码安装用于开发。所有方式通过 `acprof <command>` 执行相同的主机代码；
Docker Engine、cgroup v2、GPU driver 和采集工具仍由主机提供。
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

## Python 工具安装

安装、升级与固定版本命令集中在[安装指南](Getting_Started.md#安装)。distribution name 为 `acprof`，
console script 为 `acprof = "acprof.cli.main:main"`。`acprof` 默认打开 TUI，也可明确运行
`acprof doctor`、`acprof tui` 或 `acprof run ...`；全部子命令见 [CLI 参数](CLI_Reference.md#cli-参数)。
安装后的运行不依赖 Git、源码 checkout、`tests/` 或 `docs/`。

开发者可以安装本地 wheel：`uv tool install ./dist/acprof-<version>-py3-none-any.whl`，
或在源码目录执行 `uv tool install .`。源码开发使用 editable 包，入口仍为 `acprof <command>`。
根目录不包含 Python 文件或额外启动器；顶层帮助和版本查询不会加载 Textual、绘图库或推理框架。
Python 依赖范围由 `pyproject.toml` 声明；CI 同时验证已锁定环境和按发行依赖范围解析的干净环境。

## 从源码安装（开发者）

日常开发按[开发安装](Getting_Started.md#开发安装)创建 `.venv` 并 editable 安装。
下方脚本是把当前 checkout 安装成隔离工具的可选辅助入口。

### Clone 后初始化

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

## 工作目录与资源

输出目录、用户 workload 相对路径及 `.env` / `.env.local` 相对于**启动时的当前工作目录**。
TUI 设置仍写入 XDG 用户配置目录，以工作目录的摘要隔离不同实验工作区。
在源码仓库根目录启动时，路径行为与原来的操作示例相同。

wheel 内置 Dockerfile、平台/环境锁、扩展声明、音频素材及构建所需的 Python 源码。
`installation.resource_root()` 定位这些只读资源；它不是输出目录。
唯一 custom build hook 位于 `packaging/hatch_build.py`，由 wheel target 的 `hooks.custom.path` 指定，
并随 `packaging/` 进入 sdist。它使用临时目录复制 `acprof/`、`dockerfiles/`、`assets/`、`examples/`，
按文件后缀白名单筛选，排除 `.env`、`AGENTS.md`、`__pycache__`、嵌套 `_bundle`，以及
任意层级的 `tests/`、`docs/`、`.git/`、`.github/`、`.codex/`；wheel 主包也排除这些开发文件。
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

`uv build` 在隔离构建环境中先生成 sdist，再从 sdist 构建待发布 wheel。
Release workflow 将它们与 `SHA256SUMS` 保存为唯一的 `python-dist` artifact；后续 job 下载并核对这份制品。
`verify-dist` 使用未附加仓库 lock 的依赖解析，分别验证 `uv pip` 安装、`uv tool` 安装，
以及从仓库外重新构建 sdist 后用 pip 安装。验证用的重建 wheel 留在 runner 临时目录，绝不替换待发布制品。
已有 CI 与 standalone 环境仍安装 `requirements/host.lock`，保留已验证版本的回归范围。

本地构建后，在独立 venv 安装 wheel，再从仓库外执行：

```bash
<venv>/bin/python -I <checkout>/scripts/check_distribution.py \
  --wheel <wheel-file> --sdist <sdist-file> \
  --expected-version <version> --report <evidence-file>
```

此检查核对 wheel/sdist metadata、安装包与 CLI 的版本，通过真实 console script 运行所有公共命令的帮助，检查 wheel 资源与 sdist 的 hook、
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

普通分支 push（包括 `main`）只执行 CI。只有 `push` 事件的 `v*` Tag 可以发布；
Release 的 PR 检查和 `workflow_dispatch` 仅构建、验证和保存 artifacts，不上传 PyPI 或创建 GitHub Release。

`release.yml` 在构建前要求 Tag 精确等于 `v` + `acprof.__version__`，不一致直接失败。
发布流程如下：

```text
同一提交的 ci.yml ─────────────────────────────────────┐
uv build → python-dist → verify-dist → standalone ────┤
                                                     ├→ GitHub Release
                                                     └→ PyPI → 安装与哈希回查
```

GitHub Release 与 PyPI job 下载同一份 `python-dist`，发布前各自核对 SHA256，不重新构建。
PyPI 目录仅包含 wheel/sdist，standalone、校验和与 `verification.json` 只作为 GitHub Release 资产。
PyPI 发布成功后，`pypi-smoke` 最多尝试 5 次（间隔 15 秒），核对 PyPI JSON 中的制品 SHA256，
再执行 `uv tool install acprof==<version>`、版本及帮助检查。失败显示告警，不能触发重新发布。

### 首次配置 Trusted Publishing

先在 [PyPI 项目页](https://pypi.org/project/acprof/) 和 [JSON API](https://pypi.org/pypi/acprof/json)
检查 distribution name。404 仅表示未查询到公开项目；保留名或已删除名称等限制仍以 PyPI 创建结果为准。
若名称被第三方占用，先确定新的 distribution name，再同步包 metadata 和安装文档；CLI 保持 `acprof`。

在 GitHub 仓库 Settings → Environments 创建 `pypi`，将部署来源限制为所用版本 Tags。
在 PyPI 账户的 [Publishing 设置](https://pypi.org/manage/account/publishing/) 添加 pending publisher；
项目已由自己持有时，在项目 Publishing 设置中添加同样的 publisher：

| 字段 | 值 |
| --- | --- |
| PyPI project name | `acprof` |
| Owner | `kainam15` |
| Repository | `AC-Prof` |
| Workflow filename | `release.yml` |
| Environment | `pypi` |

pending publisher 在首次成功发布时创建项目，不提前占用名称。
仓库、workflow 文件名与 environment 必须精确匹配，不使用本地 checkout 名称或旧 remote URL。
`pypi` job 使用 `id-token: write` 与官方 PyPA action，通过 GitHub OIDC 获取短期授权；
不配置 `PYPI_TOKEN` / `PYPI_PASSWORD`。它不 checkout 源码、不构建，也不执行已安装包。
`id-token` 权限只授予发布 job。账户侧配置完成与远端同一 Tag 的成功运行才构成 Trusted Publishing 验收。

### 版本策略与失败恢复

版本继续显式维护在 `acprof/__init__.py`，不引入自动递增工具。开发阶段新一批功能递增 minor
（如 `0.3.0` → `0.4.0`），修复递增 patch（如 `0.4.0` → `0.4.1`）；稳定后再进入 `1.0.0`。
每个发布版本只能对应一个源码 Tag。修改版本、提交并验证后，维护者再创建并推送新的 `v<version>` Tag。
已有 Tag 不移动、不覆盖；即使某个旧 Tag 未发布到 PyPI，也不能把当前不同代码按该版本重新发布。

PyPI 版本发布后不可用另一份代码覆盖。上传失败时先检查 PyPI 已接受的文件及 SHA256；
如果已经部分或全部发布，不删除版本后重传，也不启用 `skip-existing` 掩盖冲突。
代码或制品需要变化时使用新版本、新 Tag。GitHub Release 单独失败时，可以使用保留的原始 artifact
恢复 GitHub 资产；不得因恢复 Release 而重跑已经成功的 PyPI 上传。

`v0.4.0` Tag、GitHub Release `v0.4.0` 和 PyPI `acprof 0.4.0` 一一对应。
PyPI 项目描述来自同一制品的 README，metadata 的 Changelog 链接指向 GitHub Releases，发布说明在该版本 Release 维护。
实验 `static_meta.json` 中的 `platform_runtime.acprof_version` 保存执行包版本，源码开发态另外保留
`git_commit`，模型/runtime 身份按原协议记录；历史缺失版本保持未知，见[环境身份与能力支持](Profiling_Protocol.md#环境身份与能力支持)。

Release 附带 `verification.json`，记录源码 SHA、CI run 和硬件证据范围。硬件报告仅关联同一 SHA
上成功的 `hardware.yml` run，且 `hardware` artifact 尚未过期；否则明确标为 `not_verified`。
这不是硬件强制门禁，也不代表所有设备或 profiler 均已验收，具体范围以对应 artifact 为准。
复用工作流遵循 [GitHub reusable workflows](https://docs.github.com/en/actions/reference/workflows-and-actions/reusing-workflow-configurations)，
本地相对路径保证执行同一提交；普通 CI 处理分支 push，tag 的 CI 由 release 调用，避免重复运行。

工作流文件存在不代表远端资产已发布。GitHub Release 需要 `contents: write`，PyPI 需要上述
Trusted Publisher 配置。GHCR 使用独立的 `runtime-images.yml` 和 `packages: write`：版本 Tag 自动发布，
手动运行该 GHCR workflow 也会发布镜像。它先构建/核验 4 个平台，再让 24 个环境 job 发布依赖环境。
GHCR package 首次发布后，维护者需在 package 设置中
确认 public 可见性，匿名用户才能直接拉取；私有 package 需要先 `docker login ghcr.io`。

GHCR 只预构建平台和依赖环境，不发布模型权重、用户数据或包含令牌的层。
拉取策略、内容身份和失败回退见[预构建依赖镜像](Runtime_Compatibility.md#ghcr-预构建依赖镜像)。
发布脚本的 `--report` 保存 image ID、内容 tag 和清单核验范围；依赖核验不代表 GPU 推理已经通过。

## 参考实现与取舍

- [PyPA publish action](https://github.com/pypa/gh-action-pypi-publish)（BSD-3-Clause）：
  采用其构建/发布分 job、OIDC 与同份 artifact 的官方模式，固定 `v1.14.2` 对应的完整 commit SHA。
  [Issue #283](https://github.com/pypa/gh-action-pypi-publish/issues/283) 记录了 reusable workflow 的身份/attestation 限制，
  因此 PyPI job 直接留在 `release.yml`，仅测试调用 reusable CI。由 PyPA 持续维护，无需自行实现上传或保存长期凭据；
  action 及其依赖仅用于 CI，不进入 host 安装依赖或测量窗口。
- [uv build 源码与文档](https://github.com/astral-sh/uv/blob/main/docs/concepts/projects/build.md)
  明确默认先构建 sdist 再构建 wheel；沿用已有 uv 工具链，并在独立 job 重建验证，避免维护第二套构建配置。
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
