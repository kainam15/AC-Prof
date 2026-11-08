# 镜像构建与管理

[← 返回专题目录](runtime.md)

## 构建、复用和验证

`prepare_image` 将模型分支解析为完整 commit，再通过 profile 引用准备依赖环境。
构建链路为 `platform.Dockerfile` → `runtime.Dockerfile` → `runtime-model.Dockerfile` →
`runtime-final.Dockerfile`，分别缓存平台、完整环境、模型计划清单和服务代码。
模型权重由主机 [Model Store](model-store.md#下载网络与-model-store) 管理并只读挂载，不写入新镜像。
Docker build context 忽略 `tests/`、`docs/`、pytest/mypy/Ruff 缓存、覆盖率和本地运行证据；
服务镜像构建仍由 `source_identity.service_context_files` 按白名单独立暂存源文件，
不会将开发凭据、实验结果或测试目录复制进推理镜像。
最终服务镜像先以 root 完成固定的源码清单生成，再创建 UID/GID 65532 的 `acprof`
账户并切换为非 root 运行，`HOME` 和 Torch 缓存指向容器临时目录；主机 Model Store 始终只读。
可缓存的 Python 镜像仍在 root 构建阶段安装已锁定依赖，不需要运行时安装权限。
非 root 运行须分别验收 CPU 模型 `/ready`、`/predict`、CUDA 设备和 GPU profiler；
这些基础 smoke **不能替代** 当前源码指纹对应的 CPU/GPU 完整采集与正式测量窗口验收。
Docker bridge 保留本机回环地址端口映射。`--network none` 无法承载现有 HTTP 测量链路；
internal bridge 仍可能允许访问宿主网关上的服务，且不能假设它保留现有端口发布行为。
模型离线加载不等于 Docker 层级的出站网络封锁。
阻断非必要主动外联的候选策略、独立验收与接入边界见[测量容器出站隔离](../profiling/egress.md)。
标准推理容器还使用 `--cap-drop=ALL` 和 `no-new-privileges`，不赋予额外 Linux capabilities；
CPU/GPU 正式推理所需的具体能力由运行时验证核对。独立的 profiler 工具容器
（如 NCU/nsys）有自己的权限与能力限制，不能将普通推理成功当作 GPU profiler 的验证通过。

`environment_id` 是规范化平台声明、系统锁及全部 Python 包版本、来源 URL、制品 SHA256 的摘要。
锁的文件名、注释、顺序、profile、adapter、模型和业务代码不参与该身份。平台镜像的指纹另计
Dockerfile 和安装脚本；环境镜像再计对应配方及不可变平台 image ID。依赖相同而构建配方不同，
可以产生新的镜像缓存。

服务标签使用 `request_fingerprint` 前 20 位查找候选，覆盖逻辑 profile、环境/构建声明、
模型 commit、下载策略、后端、构建参数和 AC-Prof 代码；完整 `build_fingerprint` 再绑定实际模型
父镜像 ID。模型层指纹绑定实际环境 image ID。标签是查找入口，执行与补采始终使用不可变 ID。
模型来源参与模型身份；Hub 主地址及备用列表仅控制下载入口，不独立改变模型层或服务层的请求指纹。切换 endpoint 不因地址本身使已有模型失效。
默认自动选择下载入口；模型来源与传输记录见[下载网络与 Model Store](model-store.md#下载网络与-model-store)。
实际成功地址随 `model_download.endpoint` 保存，历史缺失字段不推算。

服务源码由 `acprof/source_identity.py` 统一选择，供 Docker 服务层、隔离接口检查与 Process
环境暂存使用，并参与各自的源码身份。除推理包外，清单还包含推理实际依赖的共享约束、
指标定义、runtime 设置与依赖声明模块及其包初始化文件。移动这些模块时应同步维护清单，
并运行 `tests/runtime/containers/test_service_source_imports.py`：它从暂存目录在禁用
checkout、`PYTHONPATH` 和 editable 安装路径的独立进程中导入服务入口。

分层构建与文件选择细节见[模型文件选择规则](adaptation.md#模型文件选择规则)。

采集和最大输入探测默认核验并复用指纹、内部清单和 Model Store 均通过校验的镜像，执行引用固定为 Docker image ID。
查不到目标指纹则自动构建；已有标签内容不符或模型文件校验失败时明确报错，不自动绕过校验。
`--rebuild-image` 显式重建模型镜像，仍允许 Docker 构建缓存和已验证依赖层复用；不等于无缓存构建或重新下载模型。
旧 `--skip-build` 参数已删除，原命令移除该参数即可沿用自动复用行为。
TUI 的“高级参数 → 镜像高级操作 → 重新构建镜像（仅本次）”作用于下一次采集或探测，启动后清除，不保存为默认配置。
镜像准备或运行验证失败时，也可在失败详情中选择“重新准备环境”。恢复实验继续使用原始镜像，不能与重建同用。
专用硬件验收例外：设置 `ACPROF_REQUIRE_PREPARED_IMAGE=1` 后，只有精确指纹复用路径被允许，缺镜像或要求重建均直接失败，不触发隐式下载。构建期间代码变动会使构建失败，避免
用旧指纹标记新代码。旧 `:latest` 镜像可留存供历史实验使用，但不直接用于新环境的采集。

模型解析身份将未设置的 task/backend 统一为空值；CLI、TUI 和 `probe` 传入的空字符串
与 `None` 不会造成不同的镜像指纹。显式指定的非空选项仍参与身份计算。已有实验保存的
provenance 在 resume 和补采时保持冻结，不用新的默认值重新计算或覆盖历史证据。

主机构建统一经 `build_image` → `build_runtime_image` → `prepare_environment_image`；环境验证脚本
和 CI 复用最后一个入口。所有 profile 必须引用已锁定环境。构建前拒绝父层包缺失、版本或制品
冲突；环境安装仅安装平台之外的差量。安装后执行 `pip check` 并严格核对完整 Python 和系统包
集合，额外包同样报错。缓存命中时核对完整身份标签和内部清单；构建使用独立输入目录及
`--iidfile`，核验输入未变、父镜像引用未变、成品清单正确后才发布标签。

正式 server 和 profiler 使用只读 Model Store 中固定的本地 snapshot；历史 baked image 保留原有解释。显式 `MODEL_LOCAL_PATH` 不存在时
立即报错，不回退到 Hub/cache 加载；未知 backend 不会自动选择同任务族的其他 handler。

在正式资源矩阵之前，使用输入计划的最小尺度、最大已选 CPU／内存，为每个请求的设备模式
启动独立验证容器，执行完整的加载、预处理、推理和输出序列化。模型加载与推理禁用 Hub 在线访问，
但为保留本机 HTTP 测量协议，默认 Docker bridge 并不封锁所有出站流量；退出后清理。
验证错误和超时保留日志并退出；Docker 明确报告的 cgroup OOM 记为资源限制，允许矩阵继续。
验证不会写测量 CSV、计算能耗或充当正式 warmup。
报告按 execution、load、preprocess、predict、completion、postprocess、validate_output、metadata
分别保存阶段结果，失败保留原异常类型及 `failed_stage`。已声明的自定义分类 pipeline 还需返回
非空 label/score，不能仅凭一个可序列化的 object 判定兼容；其余任务继续使用相应 validator。

验证可能预热宿主机文件缓存；正式容器初始化、首次请求和测量窗口的区别见[采集生命周期](../profiling/lifecycle.md#采集生命周期)。

`static_meta.json` v7 保存 `image_id`、`image_name`、`runtime_environment` 和成功返回的
`runtime_validation`；单独的 `runtime_validation.json` 与设备日志也保留失败信息。
只读取当前 schema 的 CSV／静态元数据。补采要求记录不可变 `image_id`，且原镜像存在并匹配构建指纹，
不自动升级依赖，也不将工作区代码覆盖进该镜像。历史清单缺失的新平台/环境字段不推算、
不回填；新构建必须具备并核验这些身份字段，schema 版本保持不变。Torch、NCU、Massif、Nsys 的工具版本、
可用性、输出与错误继续由各自计划记录；普通推理成功不代表所有工具已验证成功。

## Docker 内容完整性与恢复

镜像标签、`docker image inspect` 和容器可运行，只能证明相应元数据或解包文件存在。
依赖缓存、服务镜像以及构建父镜像还会独立检查 OCI 内容可用性；检查仅发生在准备阶段。

| 状态 | 处理 |
| --- | --- |
| `complete` | 所需内容对象已存在；继续原有指纹、标签、依赖 Manifest 和 Model Store 校验 |
| `incomplete` | 缺少部分内容对象，停止复用及后续构建；显示已存在／所需数量 |
| `unavailable` | 连所需内容集合也无法完整解析，停止复用及后续构建 |
| `unverified` | 没有可靠的检查能力，明确显示未验证；继续原有身份校验，不宣称内容完整 |

目前支持 Linux x86_64 上、标准本机 Docker Unix socket 对应的 containerd image store。
检查使用当前用户可执行的 `ctr`，固定 Docker 管理的 socket 与 `moby` namespace，
并将检查返回的 target digest 与 Docker 的 OCI descriptor 对齐；检查的是 linux/amd64 所需内容。
无 `ctr`、权限不足、传统 graphdriver、Rootless、Docker Desktop、远程 context、其他架构、
旧 Engine 不暴露 descriptor、结果不一致或超时均为 `unverified`。不会调用 sudo、导出整个镜像、
写入 containerd 数据库或删除内容。`complete` 表示内容可用性检查通过，不等于重新哈希全部大层、
磁盘健康检查或模型推理验收。

准备失败使用稳定的 `reason_code` 区分内容缺失、仓库鉴权、网络、磁盘及一般构建错误。
保留失败步骤、Docker 退出码、脱敏输出和准备日志路径；日志沿用
`logs/preparation/<attempt>.log` 的有界记录，结构化错误进入准备摘要的 `last_failure`。
该日志包括本次 Docker CLI／Engine 输出，不保证保留无限日志；超限会显式记录截断。

TUI 提供以下动作：

- **重新检查**：保留缓存、重试失败阶段。已确认缺层后，必须复查为完整才能继续；
  检查器随后不可用不能将已知缺层转成成功。
- **修复缓存**：仅在缺层镜像具有匹配的不可变 registry digest 时启用。先复查，再 pull 同一 digest，
  最后独立验证；即使 Docker 返回 `Image is up to date`，仍缺层也会失败。已有有效层由 Docker 复用。
  仓库网络／权限失败不会触发删除。`ACPROF_REQUIRE_PREPARED_IMAGE` 和下载预算模式禁止此路径绕过独立下载准备。
- **重新准备环境**：环境阶段也可使用。对识别出的本地失败依赖目标重新构建并校验，
  其构建使用 `--no-cache`；其他完整依赖、Model Store 和实验结果保留。普通 `--rebuild-image`
  仍保留原来的 Docker cache 语义。重新构建不能修复缺失的基础镜像 blob，基础内容仍需先恢复。

无法检查的 Docker 主机可先执行 `docker context show`、`docker version` 和
`docker image inspect <image>`，由 Docker 主机管理员核对实际 store 与权限。
仅在确认是本机 Docker 管理的 containerd 后，使用
`ctr --address /var/run/docker/containerd/containerd.sock --namespace moby images check 'target.digest==sha256:…'`
检查目标；不要将系统默认 containerd 当作 Docker store。镜像来自 registry 时，可手动
`docker pull <repository@sha256:…>`，随后重新启动准备以复查。没有准确 digest 或检查能力时不提供自动修复承诺。

[Moby PR #53615](https://github.com/moby/moby/pull/53615) 修复了已有解包 snapshot 时跳过必要 blob 的 pull 行为，
已进入 [Docker Engine 29.9.0](https://docs.docker.com/engine/release-notes/29/#2990)。旧版本的相关错误会提示评估升级，
不会自动升级 Engine。升级与 GPU Runtime、现有镜像的真实兼容性仍需用户另行验证。

准备进度复用 TUI 下载面板：Platform、Environment、Model、Service 分别显示检查、下载、构建和验证状态，
模型权重仍使用原来的 Hugging Face／ModelScope 进度协议。BuildKit 使用 `rawjson`，本机匿名 Unix socket pull
使用 Engine JSON 流；配置了 Docker 凭据／credential helper、SSH／TLS context 时保留 Docker CLI 的连接和鉴权行为，
只显示阶段及日志。百分比只描述有同一范围分母的当前下载、解压或内容就绪量，不代表整个镜像的下载百分比。
准备计时与正式 `measurement_started`／`measurement_stopped` 窗口累计时间分开；测量期间该行仅在边界更新，
不新增轮询或周期刷新。均为 UI 耗时，不能当作结果延迟指标。

实现参考 [containerd `images check`](https://github.com/containerd/containerd/blob/main/cmd/ctr/commands/images/images.go)
的 required／present／missing 语义，以及 [BuildKit SolveStatus](https://github.com/moby/buildkit/blob/master/client/graph.go)
和 [Buildx rawjson](https://github.com/docker/buildx/blob/master/docs/reference/buildx_build.md#progress)。这些项目采用 Apache-2.0，
本项目借鉴协议与检查边界，未复制其实现；复用已有 HTTPX 和 TUI 组件，没有增加 runtime 依赖。

### 下载总量与传输统计

预检合并当前选择路径的 OCI Manifest、Python／Debian 锁定制品和模型文件；预构建镜像与其
条件性本地构建不重复相加。`expected_download_bytes` 仍是完整已知时的预计待下载量，任何未知项
都会让它保持 `null`，有限预算仍拒绝继续。新增的 `known_download_bytes` 保留已知待下载部分，
`known_resource_bytes` 加上已确认缓存，`unknown_items` 与 `by_category` 列出未知数量和来源类别。
模型按缺失文件、可解析 Manifest 按缺失大小的 descriptor 统计；整个 Manifest 不可达时计一个未解析来源，
不能将它解释为只缺一层。缓存未知与缓存命中独立报告。OCI 仍使用逐来源的完整压缩上界，
不同镜像的共享层和未验证 BuildKit 缓存可能使预计量高于真正下载量。

| 展示 | 统计范围 |
| --- | --- |
| 预计下载 | 预检估算；不完整时显示已知字节 + 未知项，不作为整体进度分母 |
| 当前传输／解压／内容就绪量 | 最近一个有标识的对象及活动；大小未知只显示当前字节，不显示百分比 |
| Docker 累计下载 | 本轮准备各 Docker 操作已报告的下载 payload，按 operation ID 去重；重试重新下载的字节仍计入 |
| 已识别资源／未知大小 | 当前 Docker 操作中已发现对象的体积和未知项；新对象、后续 total 会动态补充，不保证已枚举全部资源 |
| 缓存字节／项、复用构建步骤 | 已确认复用的文件或层、独立的 BuildKit cached vertex 数；未知缓存大小不会伪造为已节省字节 |

Engine `Downloading` 按对象和显式重试累计；Moby 续传的下一次计数表示本次新读 payload，分母可能
缩为剩余大小。重复事件不重复相加；没有重试标记的计数回退保留原有证据并标为部分统计。
BuildKit 无名称的 digest 状态表示内容存储可用量，可能来自已有缓存或并发复用，因此仅更新资源体积，
不记作网络下载。解压、本地 context 上传和普通构建工作量也不计入下载。

构建依赖下载器通过现有 BuildKit 日志发送 `ACPROF_DEPENDENCY_PROGRESS`，按实际读取累计 payload，
包括失败片段与重新下载；HTTP Range 的历史缓存不计入本次传输。事件最多每 250 ms 更新一次，
完成、缓存命中和失败会强制刷新。并行 vertex 的日志片段独立拼接。Docker EOF、异常和取消保留
最后的有效计数；外层验证／失败事件不会清空这些证据，新一轮运行才重置累计值。

最终显示的是操作结束时已确认的计数，不是抓包流量：HTTP/TLS 头、认证／Manifest 元数据、
未报告的 Docker 内部重试或续传基线、普通 `RUN` 程序和无字节事件的下载仍无法完整计量。
缓存构建只报告 `CACHED` 时，仅确认步骤复用，不从步骤数推算下载量或百分比。
模型下载继续使用现有 SDK payload／文件重建统计，不与 Docker 字节合成一个整体百分比。
依赖 helper 与下载策略已经属于构建指纹，本次更新会改变依赖镜像身份；不会替换历史实验记录的 image ID。

实现参考 [Moby 下载／重试](https://github.com/moby/moby/blob/402d0a16f131d93ee89f1b06cb5fc8c98e766dfc/daemon/internal/distribution/xfer/download.go)、
[Moby Range 续传读数](https://github.com/moby/moby/blob/402d0a16f131d93ee89f1b06cb5fc8c98e766dfc/daemon/internal/distribution/pull_v2.go)
及 [BuildKit 内容存储进度](https://github.com/moby/buildkit/blob/cedaeca8850a2bb536bbd1db72419f3cf491bdb5/util/pull/pullprogress/progress.go)。
沿用受维护的 Apache-2.0 上游协议思路与现有日志、HTTPX、Textual 能力，不复制源码或增加依赖；
所有计数发生在准备阶段，正式测量窗口仍拒绝下载事件和 UI 刷新。

## 镜像管理与清理

### GHCR 预构建依赖镜像

依赖准备按“本地核验缓存 → GHCR 预构建镜像 → 本机锁定构建”执行。
默认 registry 为 `ghcr.io/kainam15/ac-prof/runtime`，可用
`ACPROF_RUNTIME_REGISTRY` 指向镜像仓库。镜像必须已发布且对当前 Docker 用户可访问。

`ACPROF_RUNTIME_IMAGE_SOURCE` 可选择：

| 值 | 行为 |
| --- | --- |
| `auto`（默认） | 优先本地缓存，再拉取；缺标签、缺少 AMD64 或网络故障时警告并回退本地构建；有限下载预算禁止未预检的构建回退 |
| `pull` | 优先本地缓存；拉取失败直接报错，不自动构建依赖 |
| `build` | 优先本地缓存；缺失时本机构建，跳过预构建 Registry；仍可能下载前端、base 和依赖 |

标签为 `platform-<完整平台构建指纹>` 或 `environment-<完整环境构建指纹>`，
后者绑定实际平台 image ID。拉取后先核对标签和 image config 的 `linux/amd64`，再读取内部系统/Python 依赖清单，
通过后才写本地缓存标签；内容不符直接报错，不静默回退。
鉴权、TLS 证书、digest 校验和已知内容缺失也不会触发 `auto` 构建回退。
国内制品镜像设置不会隐式改变 `auto|pull|build` 或 GHCR 仓库。希望避免 GHCR 请求时明确选择
`build`；已有可信预构建镜像时用 `ACPROF_RUNTIME_REGISTRY` 指定其完整 repository。
正式构建与运行仍使用不可变 image ID。此策略不更改模型下载、文件选择、服务验证和采集协议。

### GHCR 不可用时恢复

缺少当前指纹的 Tag 与只有 ARM64 的 Manifest 是不同原因；TUI 分别显示“标签不存在”与
“不支持 linux/amd64”。`auto` 在不设置下载预算时自动转入本地锁定构建，显示警告、原因和
构建进度；只有本地构建也失败，环境准备才最终失败。原始拉取错误与后续构建错误都保留在
准备日志中，失败对话框默认仅显示摘要，可展开详细信息。认证、TLS、内容校验失败、磁盘满
或无法识别的 Docker 错误直接停止，不通过构建掩盖。

恢复步骤：

1. 在“设置 → 连接与权限 → Docker 下载源”检查 `auto|pull|build`。
   `pull` 遇到缺标签或架构不匹配会停止，需要本机准备时选择 `auto` 或 `build`。
2. 有下载预算时选择 `build`，重新预检并确认 base、BuildKit frontend 与锁定依赖的下载量；
   不清除预算来绕过未知下载量。已经核验的本地缓存在三种模式下都优先复用。
3. 按下表分别配置 Docker Hub、Debian、PyPI/PyTorch 来源，再重新准备 Runtime。
   “回退到本地构建”是准备策略变化；`endpoint fallback` 只表示同一制品下载入口变化，
   不表示 GHCR 镜像可以从 Debian 或 PyPI 下载。
4. 维护者可在对应版本 checkout 执行只读核验：

   ```bash
   uv run --locked python scripts/publish_runtime_images.py --platform cpu --verify-only
   uv run --locked python scripts/publish_runtime_images.py --environment onnxruntime-cpu --verify-only
   ```

   命令仅读取远端 Manifest/config；缺少当前指纹、架构或身份不符时非零退出，不拉取镜像层。
   需要发布缺失镜像时，在目标版本上手动运行 **Runtime Images** workflow。
   预览版不自动发布镜像矩阵，不能凭 Python 包已发布推断 GHCR 有对应 Tag。

### 下载来源分流（限额 VPN 场景）

| 阶段 | 当前锁定来源 | 建议 |
| --- | --- | --- |
| Python PyPI wheel（含 NVIDIA `nvidia-*` CUDA 库） | `files.pythonhosted.org` | `ACPROF_PYPI_ARTIFACT_MIRROR=tencent`；腾讯云镜像直连、保留官方 SHA256 |
| PyTorch wheel | `download-r2.pytorch.org` | `ACPROF_PYTORCH_ARTIFACT_MIRROR=aliyun`；只改传输地址，不改 PyTorch 锁及 SHA256 |
| Debian `.deb` | `snapshot.debian.org` | `ACPROF_DEBIAN_ARTIFACT_MIRROR=tencent`；固定包名版本由腾讯云 Debian 镜像提供，仍须原锁 SHA256 校验 |
| BuildKit Dockerfile 前端 / Python base | Docker Hub | `ACPROF_DOCKER_HUB_MIRROR=<registry/path>` 显式映射，保留原 digest；留空沿用官方引用及 Docker 配置 |
| 预构建依赖镜像 | GHCR | `ACPROF_RUNTIME_REGISTRY=<registry/repository>` 指定自有或可信同步仓库；按完整指纹和内部清单验证 |
| 模型元数据与权重 | HF endpoint / Model Store、可选 ModelScope | 缓存优先；HF 镜像或 ModelScope 按各自来源规则处理 |
| 安装 AC-Prof 的宿主机依赖（uv / pip） | 进程包管理器 | 可以单独配置 uv / pip 国内 index，不受 runtime 镜像配置控制 |
| Process Runtime 环境 | uv 的原始锁定 URL、必要时解释器下载源 | 不受 Docker 制品镜像开关控制；缺少解释器时来源/大小可能未验证，有限预算会拒绝未知量 |

```bash
# 在项目根目录执行，或写入本地 .env.local。
export ACPROF_PYPI_ARTIFACT_MIRROR=tencent
export ACPROF_PYTORCH_ARTIFACT_MIRROR=aliyun
export ACPROF_DEBIAN_ARTIFACT_MIRROR=tencent
# 可选示例；先确认此服务能提供所需 digest，也可以换成自有 OCI 镜像服务：
export ACPROF_DOCKER_HUB_MIRROR=m.daocloud.io/docker.io
# 需要本机构建且希望避免先拉取 GHCR 时：
export ACPROF_RUNTIME_IMAGE_SOURCE=build
```

三个制品镜像开关均为**独立 opt-in**：PyPI 仅从 `files.pythonhosted.org/packages/` 替换为腾讯云，
PyTorch 仅从 `download-r2.pytorch.org/whl/` 替换为阿里云 `pytorch-wheels/`，Debian 仅从固定
`snapshot.debian.org/archive/debian[-security]/<timestamp>/pool/` 替换为腾讯云 `debian/pool/`。
默认均关闭；启用后的制品下载和 HEAD 预检绕过进程 HTTP(S) 代理，只允许同 HTTPS 镜像 host 重定向。
镜像故障、历史文件缺失或哈希不匹配会明确失败，不自动回退官方源。锁中的 URL、版本、SHA256
及缓存键保持原值，最终必须校验完整制品。HEAD 成功不能证明完整 wheel 或系统包下载成功。
未启用 Debian 镜像时，官方 snapshot 的既有瞬态失败备用地址为 `deb.debian.org` 同名 pool 文件，
沿用三次总尝试、原 SHA256 和显式来源提示，预检会列出该条件性来源。

TUI 在 **应用设置 → 管理连接与权限 → Docker 下载源**提供相同设置。
“选用国内依赖镜像”只填写三个制品选项，保存后供新任务使用；OCI 地址和 Runtime 策略分别选择。
可逐项改回“官方源”，清空 OCI 地址恢复官方引用。CLI 读取相同 `.env.local`，显式导出的环境变量
优先；保存机制见[本地设置](../usage/configuration.md#tui-本地设置)。不按语言、时区或 IP 自动切源：
这些信号无法证明镜像覆盖、网络出口或用户授权。国内镜像作为明确推荐，实际可用性由每次预检与传输确认。

### BuildKit 前端、缓存与网络边界

两个依赖 Dockerfile 的 `# syntax` 固定官方 `docker/dockerfile:1@sha256:...`，使用现有 Dockerfile v1
语法与 cache/bind mount。主机通过官方 `--build-arg BUILDKIT_SYNTAX=...` 机制传入选定 OCI 路径。
`ACPROF_DOCKER_HUB_MIRROR` 只接受无 scheme、凭据、tag 或 digest 的 registry/path；映射
`docker/dockerfile` 与 `library/python` 时保留原 SHA256，不能用该设置执行任意替换前端。
地址映射不会改变锁或构建身份；修改 Dockerfile 中的前端 digest 会随配方进入平台和环境构建指纹。
本次从浮动前端固定到 digest 会产生新的构建指纹，已有实验仍按记录的不可变 image ID 恢复。

| 网络环节 | 使用的配置与证据边界 |
| --- | --- |
| 主机 metadata / `docker buildx imagetools inspect --raw` | 主机网络、Buildx 认证与代理；不是 daemon 或 BuildKit 的传输证明 |
| `docker pull` 预构建镜像 | Docker daemon 的网络、代理和认证；与 shell 的代理配置可能不同 |
| BuildKit 前端与 `FROM` | builder 的 Registry resolver；`docker` driver 使用 Engine 配置，独立 `docker-container` / Kubernetes builder 使用自己的 BuildKit 配置 |
| Dockerfile `RUN` 下载依赖 | 构建容器网络和传入的代理 ARG；`ACPROF_DEPENDENCY_BUILD_NETWORK=host` 只影响这一环节 |

Engine 的 `registry-mirrors` 及独立 builder 的 `[registry."docker.io"].mirrors` 是可继续使用的兼容路径。
BuildKit resolver 会将官方源放在 mirrors 之后，因此仅配置 mirror **不保证禁止官方直连**。
显式 OCI 路径让 AC-Prof 直接请求选定 Registry；失败时不在应用中改回 Docker Hub。
Registry 自身的认证服务、blob 重定向和管理员配置仍由 Docker 管理，应用不能承诺其公网出口或
隐藏上游全部在国内。需要严格出口限制时，在相应 Docker/BuildKit 网络配置中控制。

前端缓存属于 builder 的内容存储；`docker image ls` 中有前端镜像不等于当前 builder 命中缓存。
OCI 预检复用发布脚本已采用的 Buildx `imagetools --raw`，只读取 index 与目标 linux/amd64 manifest，
校验固定 digest，两个查询共享 20 秒总时限；不拉取层、不调用可能阻塞的旧 `docker manifest inspect` 备用路径。
预检列出前端、base、依赖与 `auto` 的条件性构建来源；前端缓存标为未知，按完整压缩大小计入预算，
两个配方使用同一 digest 时只计一次。镜像 metadata 或大小未知时保留 unknown，有限预算拒绝继续；
不会把未知量当作零或为确认缓存而提前拉取层。实际 `CACHED`、解析目标和失败步骤由构建日志确认。
完整平台及依赖环境命中时无需前端下载预检；原有内容、标签、清单核验仍在复用前执行。

实现参考 [Docker 前端配置](https://docs.docker.com/build/buildkit/frontend/)、
[BuildKit Registry 配置](https://docs.docker.com/build/buildkit/configure/)、
[BuildKit resolver 源码](https://github.com/moby/buildkit/blob/master/util/resolver/resolver.go)
和[上游前端解析延迟 Issue #4205](https://github.com/moby/buildkit/issues/4205)。
国内 OCI 映射示例参考 [DaoCloud 的前缀与 digest 说明](https://github.com/DaoCloud/public-image-mirror)。
复用已有 Docker/BuildKit（Apache-2.0）的受维护机制和标准 OCI digest 校验，不复制上游实现、
不增加 Registry 客户端依赖、守护进程或正式测量开销。第三方镜像服务没有被设为不可替换的默认值。

### 依赖缓存与代理

本机构建的系统与 Python 依赖制品使用 BuildKit cache mount 按 SHA256 缓存，成功制品不随单次构建失败而删除。
构建中断或网络超时后，若 `.part` 与来源侧车同时匹配锁定的 URL、SHA256，可通过 HTTP Range 继续下载；
响应必须给出正确的 `Content-Range`，服务器不支持续传时从头覆盖，最终仍以完整 SHA256 校验后才发布正式缓存。
传输过程会输出 `fetching`、每 16 MiB 的 `progress` 与失败重试诊断；并行连接与下载镜像源不自动改变。
构建依赖镜像时，若宿主机已配置标准 `HTTP_PROXY`、`HTTPS_PROXY`、`NO_PROXY` 或 `ALL_PROXY`，
会仅向 `docker build --build-arg` 传递变量名称，由 Docker 从环境中获取值；不将代理地址写入命令行、
锁定制品 URL 或镜像指纹。没有配置代理时继续正常直连，不会替用户发明网络出口。
部分 Linux 网关/NAT 环境中，Docker 默认 bridge 在依赖下载时可能显著慢于 host 网络。
可仅对锁定的依赖镜像构建设置 `ACPROF_DEPENDENCY_BUILD_NETWORK=host`，
让 `platform.Dockerfile` 和 `runtime.Dockerfile` 使用 `docker build --network=host`；
默认仍为 `default`（Docker 原网络隔离）。此开关不会更改正式推理容器的网络模式、下载来源、
依赖 SHA256、镜像身份或实验测量窗口。host 模式放宽了构建过程的网络隔离，
建议仅在受信任的单机 Linux 环境中、经过网络速度对照验证后启用，
不支持任意自定义 Docker network 值。
再次启动同一实验须使用新的空输出目录，不能把镜像构建中断误报为 Native profiling 已通过。

发布 workflow 先发布平台，再发布环境。`scripts/publish_runtime_images.py --push` 使用 Docker Buildx 轻量读取 GHCR 的 OCI manifest 与镜像配置（不下载文件系统层），核对完整内容指纹标签、`linux/amd64`、manifest SHA256、不可变 config digest（Docker image ID）以及环境对平台层链的继承关系。**匹配时跳过构建和推送**，缺少精确指纹 Tag 时才调用现有锁定构建与完整包清单验证；新构建推送后再次确认远端 image ID。无权限、网络超时、元数据损坏或标签不匹配必须失败，不把这些情况当作缺失而重建/覆盖。环境只在缺失需要构建时拉取已发布的平台镜像并核对不可变 ID。
这属于远端 OCI 身份核验，不是重新执行容器内安装包审计；首次发布时的完整依赖清单验证仍由既有本地构建实现。内容指纹不包含单纯的项目版本 Tag，依赖、平台、配方不变时后续版本可复用同一 GHCR 内容镜像，`evidence/*.json` 记录 `reused`、`pushed`、manifest digest、`linux/amd64`、内容指纹、AC-Prof 版本、发布 Tag 和源码 SHA。
CI 在生成矩阵前通过 `--release-tag` 拒绝 Tag/package version 不一致；发布后用 `--verify-only` 再独立读取远端精确内容标签。
发布脚本复用现有构建和清单验证，不将“已发布依赖”视为 GPU/模型推理通过。
远端轻量查询参考 [Docker Buildx imagetools inspect](https://docs.docker.com/reference/cli/docker/buildx/imagetools/inspect/) 的 `.Image` / `.Manifest` 接口，以及 [go-containerregistry/crane config](https://github.com/google/go-containerregistry/blob/main/cmd/crane/doc/crane_config.md) 的分层元数据读取思路；此处沿用已安装的 Docker Buildx，不增加新的发行依赖。
回退状态借鉴 [Docker Compose 的拉取与构建警告](https://github.com/docker/compose/blob/main/pkg/compose/pull.go)，
架构检查参考 [Moby 平台选择 Issue #48197](https://github.com/moby/moby/issues/48197) 及其关联 PR #43434。
Compose 与 Buildx 均为持续维护的 Apache-2.0 项目；仅借鉴控制流程，沿用现有 Python/Docker API，
不复制实现或增加依赖，所有检查位于正式测量窗口之前。
发布权限、资产和验证方式见[发行包说明](../development/distribution.md#发布入口与范围)。

### 镜像分类与复用

TUI 根据镜像标签和 AC-Prof 元数据判定类型。下表列出常见名称与用途；名称中的 `*` 表示不同模型、环境或标签。

| 界面类型 | 名称前缀或示例 | 内容与用途 |
| --- | --- | --- |
| 公共基础 | `acprof-platform-<platform_id>:<指纹前20位>`；历史 `acprof-base:*` | 平台镜像包含固定 Python 和系统包；`cpu`/CUDA 平台另含 Torch 必需闭包，`python-cpu` 不预装 Torch，供 ONNX Runtime 等独立运行时使用。不含任务族 Python 依赖或模型。 |
| 运行依赖 | `acprof-runtime-env:<指纹前20位>`；历史 `acprof-runtime-*` | 完整依赖环境，供多个 profile 或模型共用；不含模型权重或 AC-Prof 业务代码。 |
| 模型清单 | `acprof-model-plan-*`；历史 `acprof-weights-*` | 新镜像只保存固定 Model Store 计划，不保存权重。历史 `weights` 镜像仍含原模型文件，按旧身份保留或显式清理。 |
| 推理服务 | `acprof-nlp-*`、`acprof-cv-*` 等任务族前缀 | 在运行环境与模型文件上加入 AC-Prof 服务代码和环境清单，实际运行模型推理。 |
| 调试镜像 | `acprof-blip-reuse-base:*`、`acprof-massif-*`、`acprof-nsys-*`、`acprof-ncu-*`；名称含 `dependency-check` 或 `reuse-base` | 调试、修复或旧 profiler 兼容流程留下的镜像；可能继承某个模型的权重。 |
| 其它镜像 | 例如 `acprof-validation-host:*` | 未匹配上述分类的已标记镜像。此例用于开发时验证主机依赖、Python 兼容性和回归测试，常规采集不会自动创建或使用它。 |
| 无标签 | Docker 中显示为 `<none>:<none>` | 没有名称标签的镜像，仍需按 image ID 核对内容和引用；无标签不等于可以释放其全部空间。 |

镜像、容器和单配置结果文件名中的模型标识统一转小写，将 `/` 替换为 `--`，保留点号 `.` 和原有下划线 `_`。
例如 `Qwen/Qwen2.5-0.5B` 生成 `qwen--qwen2.5-0.5b`，对应服务镜像
`acprof-nlp-qwen--qwen2.5-0.5b:<request_fingerprint前20位>`、结果文件 `result_case_qwen--qwen2.5-0.5b_1c_4g_off.csv`。
点号符合 [Docker 镜像名称规则](https://github.com/distribution/reference/blob/main/regexp.go)；下载、加载与元数据仍保留原始模型 ID。
镜像搜索和“选择同模型”区分点号与下划线。已有镜像标签与结果文件不会自动改名。

`org.acprof.image-kind` 标签分别标记 `platform/environment/model-plan/model`；原有 `weights` 镜像仍可通过
历史标签和名称识别。常规模型的继承关系是 **平台 → 运行依赖 → 模型清单 → 推理服务**，权重从主机 Model Store 只读挂载。
`acprof-build-source:<image ID>` 是构建时给已有镜像添加的别名，
不另存一份镜像内容；TUI 按 image ID 合并这些标签。分类与构建实现分别见
[`image_management.py`](../../acprof/host/images/inventory.py) 和 [`runtime_images.py`](../../acprof/host/images/runtime.py)。

| 构建情况 | 镜像复用与新增 |
| --- | --- |
| 新模型，已有匹配的运行依赖 | 通常只新增模型文件和推理服务镜像。 |
| 新模型，需要尚未构建的依赖组合 | 先新增对应运行依赖，再构建模型文件和推理服务镜像。 |
| 同一模型，版本、环境、代码和构建配置均未变化 | 指纹与内部清单匹配时，可以复用已有镜像。 |
| 仅容器执行代码或共享声明变化，运行依赖与模型文件指纹不变 | 复用运行依赖和模型文件，只重建推理服务镜像。 |
| 仅主机编排、TUI 或绘图变化 | 复用服务镜像；主机采集代码变化仍会影响续跑身份。 |

服务代码上下文由 `source_identity.service_context_files()` 明确选择：`acprof/` 顶层共享
Python/JSON 文件、`container/`、`workloads/`、`extensions/`，以及 final Dockerfile 和许可证。
先复制到临时目录，再核对内容摘要，Docker 只接收该目录；`host/`、TUI、测试、`.env` 和
仓库历史不进入服务代码层。request fingerprint 使用相同范围，另绑定模型构建配方与依赖身份。
新指纹域会使旧服务标签首次重新构建，平台、依赖和模型文件层仍按各自身份复用。

正式采集前的独立推理验证使用本次模型的推理服务镜像，具体契约见[构建、复用和验证](#构建复用和验证)。

### 查询与删除

镜像页右上角“存储空间”弹窗按需执行 `docker info` 与 `docker system df --format '{{json .}}'`，
沿用当前清单的 Docker 连接并核对 daemon ID；弹窗打开期间暂停镜像自动扫描。
磁盘统计只对能核验为本机 Unix socket、主机名一致且非 Docker Desktop 的 daemon 读取 `DockerRootDir` 的 `statvfs`；
读取失败或远程环境显示未知，不回退到客户端根目录。使用率为已用／总容量，可用量采用非特权用户可用块数，不包含预留块。
Docker 各分类使用 CLI 汇总并将十进制单位换算为 IEC 单位，缺失项不会当作零或计入不完整合计。
“Docker 合计”是分类统计之和，镜像与 Build Cache 可能共享数据，不等于文件系统中 Docker 目录的物理独占占用。
所选镜像非空时刷新其清单、标签及容器引用，复用现有层去重算法估算释放上限；身份或引用变化后估算显示未知。
“删除后可用”是当前可用空间加上该估算，实际释放量取决于共享层、缓存和存储驱动。弹窗只有刷新与关闭，不执行 prune。

TUI“镜像管理”页（`/images`）打开时自动读取当前 Docker 环境，按实际 image ID 合并全部标签。
页面空闲时，每轮读取完成后 5 秒再次更新；后台查询不重叠，清单未变化时不重建视图。
默认只显示 AC-Prof 镜像，也可筛选全部镜像、模型相关、公共基础/依赖、PyTorch CPU、PyTorch CUDA 12.4/12.8 或无标签镜像。
搜索支持模型 ID、逻辑环境名、本层依赖的包名和版本、标签、环境摘要及镜像 ID。三个视图共用筛选与勾选：

| 视图 | 用途与交互 |
| --- | --- |
| 镜像树（默认） | 显示层级、完整/新增大小和容器引用数；`←/→` 或箭头折叠/展开，点击复选框区域勾选，点击行内其他位置查看详情。搜索保留淡色祖先节点作为上下文。 |
| 镜像列表 | 显示逻辑名称与大小，点击表头按原始名称或数值排序，再点反向；右侧保留 `Repository`、`Tag` 和类型，可横向滚动。 |
| 层共享 | 每行是一个层链，显示 Diff ID、层大小、引用镜像数和 Chain ID；选行列出全部引用镜像，包含筛选外镜像。同一 image ID 的多个标签只计一次。 |

下方详情采用“摘要优先、分组详情、默认折叠”：摘要常显名称、类型、完整/继承/新增大小、删除释放估算和容器引用数。
用户日常查看的信息与排障依据分开：

| 分组（默认折叠） | 内容 |
| --- | --- |
| 依赖清单 | 当前锁匹配时显示已锁定包；历史镜像按需读取原始构建清单，缺失时尝试隔离扫描；标注来源、完整包集合或相对父镜像的新增包。 |
| 镜像信息 | 继承路径、共享/独有空间、模型、创建时间、全部标签和容器引用。 |
| 诊断信息 | 完整 image/parent ID、依赖来源、父镜像关系证据、后代数、history/df 空间来源、原始字节数与空间计算说明；有库存警告时标题提示“有警告”。 |

点击标题或用 `Tab` 聚焦后按 `Enter` 展开/折叠，在详情区滚动查看长清单。切换语言、缩放或勾选时，同一镜像的展开状态保留；
切换镜像或层时重新折叠并回到摘要，筛选无结果时隐藏详情分组。层共享视图常显层大小与引用数，引用镜像和 Diff ID/Chain ID 诊断信息分别折叠。
镜像树的名称旁显示选择状态与关系标记，依赖清单和完整 image ID 在详情中查看；同名镜像仍按各自 image ID 分别选择和管理。
当前行的祖先连接线使用主题强调色高亮，经过其它分支时只点亮竖线；切换行、折叠、搜索和缩放后随当前路径更新。
焦点移到搜索或详情区域后仍保留该路径，高亮与复选框勾选状态独立。
平台节点统一显示 `Python CPU`、`PyTorch CPU`、`PyTorch CUDA 12.4`、`PyTorch CUDA 12.8`；`Python CPU` 对应不预装 Torch 的 `python-cpu` 平台，其余平台预装相应 CPU/CUDA 版 Torch。父节点已说明同一平台时，运行依赖子节点省略与平台 ID 完全匹配的后缀，如 `nlp-cu124` 显示为 `nlp`。
列表、层引用和缺少同平台父节点的依赖镜像保留可读的平台说明，例如 `nlp · PyTorch CUDA 12.4`。已有平台筛选项使用相同名称。
已知环境别名显示为 `moss-transformers5.10.4`、`multimodal-transformers4.57.6`，其它名称中已有的小数点原样保留。
搜索同时支持完整显示名称（如 `PyTorch CUDA 12.8`）、名称片段（如 `CUDA 12.8`、`5.10.4`）与原始标签、profile 名称。
`acprof-runtime-env:<hash>` 的逻辑名称通过 `org.acprof.environment` 与当前完整依赖锁身份精确匹配，
相同环境的多个名称并列显示；旧环境无法匹配时显示环境摘要与平台，不凭标签猜任务族。原始 repository/tag 和 Docker 镜像保持原样。

依赖只在选中镜像后展开“依赖清单”时显示，可滚动查看全部包；Torch、Transformers、Diffusers 等关键包及其精确版本排在前面，核验来源单独放在“诊断信息”。
这里的“本层”指树中的逻辑构建阶段，
不是“层共享”视图中的每条物理 Diff ID：

- 公共基础：显示平台 Python 锁中的完整包集合（包含上游 Python 基础镜像已有包），以及系统锁中本层安装的 `.deb` 制品；不将完整系统包表中的上游包算成本层安装。
- 运行依赖：用完整环境锁减去其平台锁，显示本层新增 Python 包；系统包继承平台。例如 MOSS 层显示 `transformers==5.10.4`，平台已有的 `torch` 不重复列入。
- 模型文件与推理服务：最近构建步骤、父镜像关系和环境身份均可核对时显示“无新增包”，分别说明本层添加模型文件或服务代码与运行清单。
- 当前依赖锁无法匹配历史镜像时，不据当前 profile 猜测旧版本；只在展开某镜像的依赖分组后后台读取该镜像内的 `platform-manifest.json` 或 `environment-manifest.json`。原始清单保存的是构建时核验过的完整 Python / Debian 安装集合；同时校验清单中的构建身份与不可变镜像标签。
- 若镜像不存在有效构建清单，仅对 AC-Prof 管理且可识别构建步骤的基础/运行依赖镜像，在禁网、只读、去 capability、禁提权、CPU/内存/PID 限制的临时容器内查询实际 Python 发行包及 dpkg 系统包；结束即删除容器，不加载模型。构建记录不可核验或 Python/dpkg 不可用时保留“未知”及具体原因，不对任意第三方镜像执行代码。
- 历史快照记录的是**整张镜像**的安装包，不等于当前层新增包。运行环境只有取得匹配的父平台快照才计算新增差异；父镜像缺失或不匹配则仅展示完整安装集合，不能推断新增。模型文件、服务镜像只有构建步骤、身份与父级均可核对才标为“继承，无新增”。“层共享”中的物理 Diff ID 不被当作独立 Python 环境。
- 读出的快照按 Docker daemon ID、不可变 image ID 和扫描协议版本持久缓存于 `$XDG_CACHE_HOME/acprof/image-dependencies/`（未设置时 `~/.cache/acprof/image-dependencies/`）。再次展开或 TUI 定时刷新只读本地缓存；未命中才按需启动临时容器，正式测量期间禁止开始扫描。缓存读取失败可重新取得证据，不把陈旧缓存应用于另一个 Docker daemon 或新镜像 ID。

当前构建匹配仍采用本地依赖锁和已识别构建步骤；历史来源分别注明“镜像原始构建清单”或“实际扫描”。
设计参考 [Trivy 的 image/layer ID 扫描缓存](https://github.com/aquasecurity/trivy/blob/main/docs/guide/target/container_image.md#scan-cache)
和 [Syft 的已安装包目录识别](https://oss.anchore.com/docs/guides/sbom/catalogers/)，但这里仅需 Python / dpkg，无需引入重量级 SBOM 扫描依赖。
镜像枚举与空间分析仍在线程中执行；历史依赖解析只在用户展开详情时执行，不进入正式测量窗口。
界面树继续使用 [Textual Tree](https://github.com/Textualize/textual/blob/main/docs/widgets/tree.md)（MIT，已为项目依赖）。

路径高亮复用 [Textual 8.2.8 Tree](https://github.com/Textualize/textual/blob/v8.2.8/src/textual/widgets/_tree.py)
的行布局与 Rich 分段渲染，只覆盖连接线样式。Textual 采用 MIT 许可、上游持续维护，当前 `.venv` 版本已验证；
路径高亮无需额外依赖或 Docker 查询。光标切换仅重绘可见树区域，不进入正式测量窗口。

继承关系按以下证据优先级解析，并要求父镜像层链为当前镜像的前缀：构建记录中的不可变模型父镜像 ID / Docker `Parent`，
构建指纹匹配的上一级镜像，最后才是唯一的最近层前缀。最后一种只说明可能的祖先关系，详情明确标为“层前缀推断，未确认 FROM”。
父镜像缺失、多个候选或记录冲突时显示未知或明确原因，不把同名标签、新版本平台、共享某一层的镜像强行连成 `FROM`。
历史 `acprof-base` 只有在真实层链相符时才出现在祖先路径中，当前平台不保证继承它。

| 空间项 | 口径 |
| --- | --- |
| 完整大小 | Docker inspect 的 `Size`，与 `docker_image_bytes` 一致，包含继承层。 |
| 继承 / 新增 | 经核验的本地父镜像大小 / 当前完整大小减父镜像大小；父镜像未知或缺失时为 `?`，不把完整大小当作本镜像新增。 |
| 共享 / 独有 | 完整清单中被其它 image ID 引用 / 仅当前 image ID 引用的层链字节；新增层可能又被后代共享，所以新增不等于独有。 |
| 预计可释放 | 去除所选集合后不再被其它镜像引用的层只计一次；已知无可释放层或直接容器引用时为 `0 B`，正值显示 `约 X`。这是镜像层估算，实际回收受共享镜像层、构建缓存与存储驱动影响，以清理后核验为准。数据不完整时显示未知；删除确认框中的 `ℹ` 提供悬停说明。 |

层链按 [OCI ChainID 定义](https://github.com/opencontainers/image-spec/blob/main/config.md#layer-chainid)
结合有序 RootFS Diff ID 构建；相同 Diff ID 在不同父层链上分开统计，不把内容相同直接当作相同的存储快照。
层大小来自 `docker image history --no-trunc --human=false`：识别元数据指令，保留真实零字节文件层，
只有层数与完整字节数一致时才映射。不能唯一核验或大小冲突的层显示 `?`；单镜像共享/独有值可退回
`docker system df -v` 的近似值并标注来源。单镜像近似值不能直接相加得出批量释放量。

Docker 连接失败、权限不足或查询超时显示错误并清除旧选择，不把失败显示成空 Docker 环境。
附加空间查询失败不阻止基本清单与身份核验，详情提示缺失范围。

树和列表内，鼠标勾选/取消仅由 `□ / ☑ / ◩` 及左右各一格留白触发，点击区域共三个终端格；名称、数值和其余行内空白只用于聚焦与查看详情，重复点击也不切换勾选。
树的展开箭头只控制折叠/展开，勾选父镜像不会折叠子树。方向键浏览，空格或“勾选/取消”切换当前镜像分支的选择。
两个视图共用同一套选择规则：勾选上层包含自身及完整清单中的全部下层，不受折叠或筛选影响；
只选下层不向上扩大删除范围。取消下层会取消该分支及其上层的删除选择，其它已选分支保留。
`◩` 表示本镜像未选中、仅下层有选择；再次操作补齐可选分支，可选下层已经全部选中时清空该分支。
容器引用阻止选择被引用镜像及其上层，同时明确提示原因；其它下层仍可选择。
选择状态与半选显示分离的设计参考 [rc-tree](https://github.com/react-component/tree/blob/b31625a1c219c980056d473fe189dfa4334c2712/src/utils/conductUtil.ts)
（MIT），在现有 Textual 控件中实现，不引入 React 依赖或额外 Docker 查询。
“选择同模型”依据 `MODEL_ID` 或已知任务族的镜像名称，
选择当前模型可删除的服务、权重和调试镜像，排除公共基础、运行依赖、被容器引用的镜像及仍有未选下层的上层；同时显示该模型的筛选结果。
手动筛选和自动刷新保留有效勾选，并显示筛选外的选择数量；“清空选择”清掉全部勾选。
刷新时保留筛选、排序、手动列宽、焦点、折叠和浏览位置；镜像消失、标签或仓库 digest 变化、新增容器引用时取消对应勾选，
并取消仍有未选下层的上层选择。刷新中新出现的下层不会自动加入选择，需重新勾选该分支。
Docker 连接或 daemon ID 变化时清空勾选。读取失败保留上次清单并提示自动重试，恢复前暂停删除。
一次明确的删除操作结束后清空勾选；若需重试删除，应重新选择并确认。
未知命名的旧调试镜像可能无法关联模型，仍可按标签查找并手动选择。

“删除所选”先列出全部目标 ID、标签及仓库 digest、筛选外数量，并逐项注明 `≈` 层前缀推断关系；
这种关系用于镜像页的分支选择，不意味着 Docker 必须级联删除。用户确认后重新核验 Docker daemon、镜像身份和容器引用。
正在运行或已停止的容器均会阻止删除其直接引用的镜像；TUI 不代为移除这些容器。
删除按依赖深度从下层向上执行，同样处理不增加文件层的元数据镜像，移除选定镜像的全部标签，包括 `acprof-build-source` 别名。
无标签不代表无引用：多个仓库仍可能通过 `RepoDigests` 引用同一个镜像，直接删除 image ID 会被 Docker 拒绝。
删除时同时核对已确认的 digest，处理标签后重新读取剩余引用，再逐项移除；Docker 随标签自动清理的同仓库 digest 不重复删除。
确认后出现新的标签或 digest 时停止处理受影响镜像，不扩大删除范围。
每次删除后核验 image ID 已从本地清单消失；仅移除标签而仍保留镜像时不报告删除成功。
下层未选中、确认后出现新下层或下层删除失败时，保留受影响的上层；不会扩大已确认清单，其它独立分支可继续处理。
使用 `--no-prune`，不强制删除、不自动清理未选中的父镜像或构建缓存。
部分失败时保留已完成操作，并在“运行监控”日志记录 Docker 返回的详情及上层保留原因。
此页不扫描历史结果来自动判断实验是否结束；仍需续采或 profiler 补采时应保留原实验 `image_id` 对应的镜像。
删除镜像不修改 CSV、静态元数据、日志或历史指标。共享层及构建缓存仍可能占用空间，不能将镜像大小相加作为预计释放量；
对应推理服务或调试镜像仍存在时，权重层仍可能被引用。该模型的镜像和容器引用全部解除后，
权重层才可能释放；构建缓存仍可能保留它。保留不含权重的公共基础和运行依赖，不会阻止模型权重空间的释放。
实际占用可通过 `docker system df -v` 和 `docker buildx du` 另行检查；后者的 `Shared` 是与镜像等资源共享的缓存内容，
`Private` 是缓存独占内容。清理共享缓存时，仍被镜像引用的实体层会保留，不能将缓存总量当作可额外释放的空间
（见 [Docker 缓存占用说明](https://docs.docker.com/reference/cli/docker/buildx/du/)）。

查询和删除在后台执行，与本 TUI 的采集、探测、绘图、报告和补采任务互斥。
离开页面、显示确认框或运行任务时暂停自动扫描，任务结束后恢复；正式测量期间暂停刷新计时器。
自动读取期间仍可搜索、浏览和勾选；删除需要等待读取完成并核对确认清单。

自动刷新复用 [Textual Timer](https://github.com/Textualize/textual/blob/v8.2.8/src/textual/timer.py)
的暂停/重置机制和既有线程 Worker（MIT，官方维护，已核对项目使用的 8.2.8 API），不增加依赖。
Docker 查询只在上述空闲窗口执行，不增加正式测量窗口内的轮询或绘制。

交互参考 [Lazydocker 镜像面板](https://github.com/jesseduffield/lazydocker/blob/master/pkg/gui/images_panel.go)
的列表、详情与删除确认（MIT，持续维护）。其 Go/gocui 实现不直接嵌入 Python TUI；表格、后台任务和弹窗复用
[Textual](https://github.com/Textualize/textual)（MIT，官方持续维护，项目版本 8.2.8），镜像身份与删除复用现有 Docker CLI。
树和列表复用其原生 Tree/DataTable 及 Pilot 交互测试。复选框区域沿用
[Tree 的点击元数据](https://github.com/Textualize/textual/blob/v8.2.8/src/textual/widgets/_tree.py)与 DataTable 的列定位，
兼容当前 8.2.8，无需新增勾选树依赖或定时任务。层视图参考 [Dive](https://github.com/wagoodman/dive)
（MIT）的逐层呈现方式；其 Go 实现和镜像内容导出成本不适合本页的轻量清单，因此不嵌入 Dive 或导出权重层。
共享/独有值参考 [Docker CLI formatter](https://github.com/docker/cli/blob/master/cli/command/formatter/disk_usage.go)
（Apache-2.0，随 Docker 维护）的公开格式；实现只读取 JSON 和 history，兼容不可用字段。
不新增 Docker SDK 或常驻服务；AC-Prof 的分层标签识别和任务互斥由项目实现，开销只发生在用户操作时。
详情折叠复用官方 [Collapsible](https://github.com/Textualize/textual/blob/main/docs/widgets/collapsible.md)
（MIT，随 Textual 维护，兼容已有 8.2.8），保留原生点击、键盘与焦点处理。无需新增 UI 依赖；展开、折叠只使用已加载清单，不查询 Docker、扫描包或增加轮询。
