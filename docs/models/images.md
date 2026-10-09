# 镜像构建与管理

[← 返回专题目录](runtime.md)

## 构建、复用和验证

`prepare_image` 将模型分支解析为完整 commit，再通过 profile 引用准备依赖环境。
构建链路为 `platform.Dockerfile` → `runtime.Dockerfile` → `runtime-model.Dockerfile` →
`runtime-final.Dockerfile`，分别缓存平台、完整环境、模型计划清单和服务代码。
模型权重由主机 [Model Store](model-store.md#下载网络与-model-store) 管理并只读挂载，不写入新镜像。

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

分层构建与文件选择细节见[模型文件选择规则](adaptation.md#模型文件选择规则)。

`--skip-build` 仅复用指纹和内部清单均匹配的镜像，执行引用固定为 Docker image ID。
查不到目标指纹则自动构建；已有标签内容不符会报错。构建期间代码变动会使构建失败，避免
用旧指纹标记新代码。旧 `:latest` 镜像可留存供历史实验使用，但不直接用于新环境的采集。

主机构建统一经 `build_image` → `build_runtime_image` → `prepare_environment_image`；环境验证脚本
和 CI 复用最后一个入口。所有 profile 必须引用已锁定环境。构建前拒绝父层包缺失、版本或制品
冲突；环境安装仅安装平台之外的差量。安装后执行 `pip check` 并严格核对完整 Python 和系统包
集合，额外包同样报错。缓存命中时核对完整身份标签和内部清单；构建使用独立输入目录及
`--iidfile`，核验输入未变、父镜像引用未变、成品清单正确后才发布标签。

正式 server 和 profiler 使用只读 Model Store 中固定的本地 snapshot；历史 baked image 保留原有解释。显式 `MODEL_LOCAL_PATH` 不存在时
立即报错，不回退到 Hub/cache 加载；未知 backend 不会自动选择同任务族的其他 handler。

在正式资源矩阵之前，使用输入计划的最小尺度、最大已选 CPU／内存，为每个请求的设备模式
启动独立验证容器，执行完整的加载、预处理、推理和输出序列化。容器无网络，退出后清理。
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

## 镜像管理与清理

### GHCR 预构建依赖镜像

依赖准备按“本地核验缓存 → GHCR 预构建镜像 → 本机锁定构建”执行。
默认 registry 为 `ghcr.io/kainam15/ac-prof/runtime`，可用
`ACPROF_RUNTIME_REGISTRY` 指向镜像仓库。镜像必须已发布且对当前 Docker 用户可访问。

`ACPROF_RUNTIME_IMAGE_SOURCE` 可选择：

| 值 | 行为 |
| --- | --- |
| `auto`（默认） | 优先本地缓存，再拉取；拉取失败时本机构建 |
| `pull` | 优先本地缓存；拉取失败直接报错，不自动构建依赖 |
| `build` | 优先本地缓存；缺失时本机构建，不访问 GHCR |

标签为 `platform-<完整平台构建指纹>` 或 `environment-<完整环境构建指纹>`，
后者绑定实际平台 image ID。拉取后核对标签与内部系统/Python 依赖清单，
通过后才写本地缓存标签；内容不符直接报错，不静默回退。
正式构建与运行仍使用不可变 image ID。此策略不更改模型下载、文件选择、服务验证和采集协议。

发布 workflow 先发布平台，再由环境 job 拉取相同父层；避免各 job 重建平台后产生互不匹配的环境键。
发布脚本复用现有构建和清单验证，不将“已发布依赖”视为 GPU/模型推理通过。
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
[`image_management.py`](../../acprof/host/image_management.py) 和 [`runtime_images.py`](../../acprof/host/runtime_images.py)。

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
已知环境别名显示为 `moss-transformers5.6.0`、`multimodal-transformers4.57.6`，其它名称中已有的小数点原样保留。
搜索同时支持完整显示名称（如 `PyTorch CUDA 12.8`）、名称片段（如 `CUDA 12.8`、`5.6.0`）与原始标签、profile 名称。
`acprof-runtime-env:<hash>` 的逻辑名称通过 `org.acprof.environment` 与当前完整依赖锁身份精确匹配，
相同环境的多个名称并列显示；旧环境无法匹配时显示环境摘要与平台，不凭标签猜任务族。原始 repository/tag 和 Docker 镜像保持原样。

依赖只在选中镜像后展开“依赖清单”时显示，可滚动查看全部包；Torch、Transformers、Diffusers 等关键包及其精确版本排在前面，核验来源单独放在“诊断信息”。
这里的“本层”指树中的逻辑构建阶段，
不是“层共享”视图中的每条物理 Diff ID：

- 公共基础：显示平台 Python 锁中的完整包集合（包含上游 Python 基础镜像已有包），以及系统锁中本层安装的 `.deb` 制品；不将完整系统包表中的上游包算成本层安装。
- 运行依赖：用完整环境锁减去其平台锁，显示本层新增 Python 包；系统包继承平台。例如 MOSS 层显示 `transformers==5.6.0`，平台已有的 `torch` 不重复列入。
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
刷新时保留筛选、排序、手动列宽、焦点、折叠和浏览位置；镜像消失、标签变化或新增容器引用时取消对应勾选，
并取消仍有未选下层的上层选择。刷新中新出现的下层不会自动加入选择，需重新勾选该分支。
Docker 连接或 daemon ID 变化时清空勾选。读取失败保留上次清单并提示自动重试，恢复前暂停删除。
一次明确的删除操作结束后清空勾选；若需重试删除，应重新选择并确认。
未知命名的旧调试镜像可能无法关联模型，仍可按标签查找并手动选择。

“删除所选”先列出全部目标 ID 和标签、筛选外数量，并逐项注明 `≈` 层前缀推断关系；
这种关系用于镜像页的分支选择，不意味着 Docker 必须级联删除。用户确认后重新核验 Docker daemon、镜像身份和容器引用。
正在运行或已停止的容器均会阻止删除其直接引用的镜像；TUI 不代为移除这些容器。
删除按依赖深度从下层向上执行，同样处理不增加文件层的元数据镜像，移除选定镜像的全部标签，包括 `acprof-build-source` 别名。
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
