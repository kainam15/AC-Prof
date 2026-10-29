# 主机编排与测量模块

[← 返回专题目录](architecture.md)

## 主机编排与测量

| 模块 | 职责 |
| --- | --- |
| `application/task_runs` | 保存推理组集合与配置，按不同执行路径串行调度独立 `RunApplication`；组间失败隔离、恢复及 workload 物化均在测量外 |
| `model_candidates` / `task_workloads` / `inference_groups` / `host/task_selection` | 分离模型声明、语义输入模板、计算配置分组与交互选择；不以架构候选替代明确任务证据，也不为 prompt-only 变化重跑资源矩阵 |
| `preflight` | 原生 Linux、本机 Docker、cgroup 与 CPU 能耗前置检查 |
| `network_preflight` / `download_probe` | Docker/Process 下载计划与确认、来源元数据检查及共享预算的限量 GET；复用 HF transport，全部在准备阶段执行 |
| `host/command` | host 同步命令的 UTF-8、timeout、环境、退出码、耗时与脱敏 metadata；不拥有长期进程 |
| `docker_runtime` | 推理服务容器的所有权、启停、ready 检查及原有冷启动分段 |
| `runtimes/contracts` / `runtimes/service` | 统一执行契约、Runtime dispatch、客户端执行证据；编排不依赖 Docker 生命周期细节 |
| `runtimes/docker` / `runtimes/process` | 原 Docker adapter 与无 Docker 的受管理服务；启动就绪和停止清理均由 adapter 完整持有 |
| `runtimes/environment` / `runtimes/validation` | 隔离 Python 环境、锁校验、模型准备、测量前真实推理验证 |
| `runtimes/process_control` / `runtimes/supervisor` | 控制管道、subreaper、PID 身份、清理和恢复证据；不参与采样 |
| `runtimes/resources` / `runtimes/capabilities` | 继承分配与实施限制分离、逐项能力证据；全部在测量外运行 |
| `monitors/process_usage` | psutil 观测进程树 CPU/RSS、可选整卡 NVML；与 Docker cgroup 字段分开 |
| `container_state` | Docker inspect 状态证据、启动失败与运行期 OOM 解释；不拥有容器生命周期 |
| `images.identity` / `gpu_device` | 镜像、容器、payload 共用的模型名称 token；GPU mode 归一化及选定设备身份 |
| `images.environment` | 平台与依赖环境的内容缓存、安装配方指纹、完整清单和标签核验；主构建及容器 CI 共用 |
| `images.runtime` | `ImageInfo`、镜像准备、profile/平台选择、模型层与代码层构建、父镜像绑定、运行清单及不可变 image ID 核验 |
| `images.inventory` | Docker 镜像清单、标签合并、容器引用检查及按确认清单删除；不参与采集 |
| `images.graph` | 从镜像元数据、依赖锁身份与层链解析父子关系、逻辑名称、共享层和按选择集合去重的释放估算；不访问 Docker |
| `images.dependencies` | 匹配镜像身份与本地锁，核对构建阶段并生成本层依赖增量；不启动容器或扫描包 |
| `runtime_validation` | 矩阵前的独立 CPU／GPU 完整推理验证及报告，不生成测量行 |
| `input_plan/` | `types` / `scales` 定义结果与尺度，`probe` 负责 HTTP 探测，`materialize` 发布输入计划，`families` 实现任务族规划，`planner` 统一分派 |
| `startup_probe` | 独立 readiness-only 启动探测、Docker OOM 证据与保守连续前缀，不采集性能 |
| `matrix_plan` | 资源 case 与 input scale 的独立确定性排序、冻结计划、hash 与恢复校验，不采样硬件 |
| `cases` | `MatrixRequest`、单 case 四类冻结配置和 `MatrixProgress`；不持有运行时状态或执行采集 |
| `model_schema` | 从 catalog 读取任务 IO 模板与推理精度，返回独立副本供元数据补充 |
| `machine_metadata` | 准备期 GPU、内存、swap、Docker storage/image 与 CPU policy 探测；保留 unavailable/unknown 状态 |
| `static_metadata` | `StaticMeta` 组装、enrich 与写出；不直接拥有主机或 Docker 探测实现 |
| `measurement.capture` | tcpdump 前置检查及 capture/parser 命令构造 |
| `orchestrator` | case/matrix 调度、idle 稳定性、失败与超时处理、OOM pruning 和 CSV 合并 |

| `run_state` | 目录锁、实验身份、已完成 case 校验、中断备份和恢复；仅在测量窗口外运行 |
| `measurement.monitors` | `MonitorGroup` 所有权、固定启停顺序及共享 `run_matched_control_window`；不导入硬件或 workload |
| `execution_conditions` | 保存／恢复线程环境、固定 GPU UUID、CPU affinity 和请求超时；开销诊断与负载重放共用 |
| `measurement.load` | 独立非流式 HTTP 调度与连接生命周期；结果不进入正式能耗 CSV |
| `client` | 环境与 workload 初始化、短连接 HTTP 请求适配、窗口调度和完成结果发布 |
| `measurement.metrics` | 已完成采样结果的纯计算与格式化；`WindowMetrics` 保留部分计算结果、判定指标状态并组装完整结果行 |
| `measurement.execution` | `WindowExecutor.execute` 拥有单窗口 collector 构造、对照采集、请求、停止、请求证据发布和异常优先级 |
| `measurement.types` / `measurement.errors` | `WindowSpec`、`WindowObservations`、`WindowResult` 与共享的致命测量异常；不访问硬件 |
| `measurement.requests` | 串行请求进度、实际参数和有效输入尺度一致性；不拥有 collector 或文件 |
| `monitors/rapl_topology` | powercap 完整域发现、alias 去重、package/DRAM 来源选择与可用性；独立于矩阵计划 |
| `monitors/common` | Docker PID 查询与 CPU/资源采样的绝对时刻调度；保留各监控器的异常类型 |
| `host/container_lifecycle` | 按主机与进程身份确认废弃服务容器，在冷启动计时前回收 |
| `profiling.compute` / `profiling.execution` | profiler 计划编排及汇总；工具执行实现统一放在 `profilers/` 下 |
| `profilers/ncu` / `profilers/torch` / `profilers/advisor` | 各自的容器执行与产物处理；NCU 另拥有 checkpoint、export 和 resume |
| `profilers/massif` / `profilers/nsys` | 隔离执行、错误条目与产物处理；Massif 拥有 checkpoint/resume，Nsys 拥有 stats、SQLite 与 raw stream 清理 |
| `measurement.diagnostics` / `measurement.publication` | 窗口外 idle/NVIDIA 诊断，以及停止后请求 JSONL、CSV 与 sidecar 的发布 |
| `measurement.error_rows` | case 失败、timeout 和 startup-OOM 行的纯归约与发布；校验 measurement key、保留已提交成功行，并维持 sniff sidecar 先于 CSV 的提交顺序 |
| `monitors/resource_readers` / `monitors/resource_metrics` | cgroup/proc/sysfs discovery 和 raw reader；纯 sample reduction 与 counter 派生计算 |
| `profilers/compute_parsers` / `profilers/execution_parsers` | Advisor/NCU CSV、Massif snapshot 和 Nsys stats 的纯标准库解析 |
| `profilers/tool_discovery` | 可执行文件、版本目录优先级和完整工具挂载路径 |
| `profilers/execution_environment` | 原始模型镜像的 profiler 能力核验、工具版本查询 |
| `profiling.support` | 两类 profiler 的容器命令、runner 参数、输入计划与产物路径；复用 `artifacts.atomic_write` 发布原有 profiler JSON |
| `application/terminal_log` | tmux pane pipe 的启动、停止与日志发布；不覆盖已有 pipe，由本次 `RunApplication` 持有并在 finally 清理 |

`host.measurement.energy_validation` 在正式采样结束后读取 case CSV 并核验 CPU/GPU idle 功率基线，
保留既定错误状态、告警 sidecar 与阈值；`orchestrator` 仅调度该校验。
`host.result_cleanup` 只有在合并结果存在且非空时才回收逐 case 临时文件，
`cli.run` 只负责参数与调度，v2/flat 路由与请求证据保留策略由结果模块维护。

`measurement.metrics` 对跨模块使用的 CPU/GPU/资源样本转换公开 `cpu_metrics_from_result`、
`gpu_metrics_from_result` 与 `resource_usage_metrics_from_result`；其他局部计算保留模块私有，
不为降低 private API 计数机械公开所有助手函数。

`runtimes/service` 是输入规划的下层，Docker adapter 继续使用 `docker_runtime`；`static_metadata` 引用 runtime、输入计划类型和
任务 schema；这些模块均不反向引用 `orchestrator`。调用方直接引用各模块。
两类 profiler 单向依赖 `profiling.support`，TaskInfo 只在类型检查时导入；
解析与工具查找使用 `profilers/` 下的实现。`profiling.execution` 通过 Massif/Nsys 的
`profile` 与 `error_entry` 调度各工具，私有采集、恢复或解析实现只由所属模块调用。
解析器不导入 Docker、模型检测或采集编排，单独读取报告无需安装推理框架。
测试在函数实际查找依赖的位置 mock。

### 单窗口执行边界

`ClientRunner` 在准备阶段加载 profiler 计划、建立 execution profile 解析快照，
再以 `WindowSpec` 调用 `WindowExecutor.execute`。执行器返回 `WindowResult`，
包含结果行、可选 idle diagnostic 和抓包分组标识；调用者统一发布 CSV 和 sidecar。
请求适配器仍使用原有短连接 HTTP 协议，自动请求窗口仍按已完成请求的应用时延累计判断。

执行器独立拥有所有已登记 collector，即使后续构造、prepare、start 或请求失败，
也完成原定的 stop/close 收敛。请求日志在全部 collector 停止后写入，保留已完成请求、
失败请求 ID 和原有错误诊断。超时、能耗中止、PMU 中止及取消优先于清理或日志写入故障；
普通请求错误与清理失败同时出现时，清理失败阻止后续窗口。

`WindowMetrics` 在采集结束后按既定顺序归约数据。数值计算中途失败时保留此前已取得的字段，
吞吐量保持未知；请求本身失败时保留请求证据和完成数量，不将该不完整窗口重新归约为成功结果。
idle history 由执行器按 case 保存，GPU idle 来源在 workload stop 完成累计计数器回退后读取。

边界测试直接使用替代 collector 和请求端口，验证完整窗口、部分请求、初始化失败、
清理次数和组合故障的异常优先级；CLI/CSV 集成测试继续验证外部产物。
设计参考 [MLPerf LoadGen 的 IssueQueries / PerformanceResult 边界](https://github.com/mlcommons/inference/blob/master/loadgen/loadgen.cc)，
以及 [pyperf 对测量阶段状态与 teardown 上下文的讨论](https://github.com/psf/pyperf/issues/239)。
仅借鉴生命周期和结果分离，不引入依赖，也不修改现有采样策略、CSV schema 或 monitor 启停顺序。

### Host command 与 diagnostics

host 的短生命周期同步命令直接复用 `host.command.run_command`，覆盖 Docker、doctor、
preflight、perf probe、profiler、TUI diagnostics 和 packet 后处理。镜像、状态、容器与调用方
直接使用 runner；需要非零退出抛错或直通输出的调用显式设置
`check=True` / `capture_output=False`。container-side subprocess 与开发 scripts 不在此执行边界中。

runner 接收 literal argv，禁止隐式 shell；stdout/stderr 使用 UTF-8 和 `errors="replace"`。
返回值保留 `CompletedProcess` 的 args、returncode、stdout/stderr，并增加 `duration_s` 和 metadata。
默认捕获输出、`check=False`；调用方原有的 check、capture 和 timeout 选择保持原义，未设置 timeout
仍为无限等待；与 Docker daemon 交互的生产路径应显式使用有界 timeout，启动和 inspect
预算由 `host.command` 集中定义。`env` 保留完整替换语义，`env_overrides` 在它或父环境上覆盖，不修改父进程环境。
非零退出按 check 决定是否抛出 `CalledProcessError`；timeout、OSError 和取消继续传播原异常。
timeout 的部分输出保留标准库的 bytes 语义。runner 不增加 retry、进程组或 Docker 容器清理策略。
长期进程继续由 `ProcessLifecycle`、`PerfMIPSMonitor`、orchestrator/load 的 tcpdump owner 管理。

`CommandMetadata.as_dict()` 是显式提取的 evidence：command、cwd、必要环境差异、duration_s、
returncode 和 error_type。失败异常附带同一结构的 `command_metadata`。
常见 credential/token/password/key 参数、环境值及 URL 凭据脱敏；自定义位置的秘密值由调用者
通过 `redact_values` 显式标记。原始 stdout/stderr 与异常仍供业务处理，不能直接当作已脱敏 metadata。
runner 只输出已脱敏的 DEBUG logging，不输出 `[cmd]`、不记录 stdout/stderr、不写实验文件。
普通 doctor/preflight/idle diagnostic 命令不自动进入 experiment provenance；现有 profiler plan、
runtime validation log、静态元数据仍由各自 producer 按原协议发布。需要额外证据时，调用方须显式选择
metadata 并在窗口外写入独立 diagnostics/evidence，不能把全部命令自动追加到正式结果。

内部诊断使用命名 `logging` logger，当前覆盖 command duration、orchestration 取消/清理、runtime
validation 与 profiler failure 类型；默认不安装 handler、不输出 DEBUG。异常日志仅记录必要上下文和
异常类型，不自动打印可能含凭据的异常正文。CLI/TUI 的状态、警告、进度和 preparation events 继续由
presentation/event 层输出；没有增加 `--verbose` / `--quiet` 或窗口内同步日志。

输出审计保留 case/工具进度、posthoc dry-run/更新清单、input scale 规划和
`ACPROF_*` machine events。preflight 与 client 面向用户的错误使用 stderr；idle 稳定性警告、
largest-probe 错误和 tmux 警告也使用 stderr。client 启动配置、scale 内部原因和
Nsys importer 成功诊断使用 DEBUG logger。TUI 的进程层仍合并 stdout/stderr，
machine events 继续由 `RunProgressTracker` 解析。

容器职责划分参考 [Docker SDK 的 container API](https://github.com/docker/docker-py/blob/main/docker/api/container.py)
（Apache-2.0），日志边界参考 [CPython logging](https://github.com/python/cpython/blob/3.12/Lib/logging/__init__.py)
（PSF），tmux pipe 所有权核对 [tmux 源码](https://github.com/tmux/tmux/blob/master/cmd-pipe-pane.c)
（ISC）。这些项目均有持续维护的官方源码；本项目只借鉴职责与生命周期边界，
保留现有 Docker CLI、标准库 logging 与 tmux 命令，不新增运行依赖或正式测量期开销。

此边界参考 [CPython subprocess](https://github.com/python/cpython/blob/3.10/Lib/subprocess.py)
（PSF License）的执行及异常语义、[Invoke runners](https://github.com/pyinvoke/invoke/blob/main/invoke/runners.py)
（BSD-2-Clause）的显式输出/环境控制，以及 [pyperf hooks](https://github.com/psf/pyperf/blob/main/pyperf/_hooks.py)
（MIT）的准备与测量分离。上游有持续维护的源码与测试；兼容本项目 Python 3.10+ 的标准库已足够，
不引入 Invoke/pyperf 运行依赖、PTY、常驻管理进程或新的测量阶段，不复制其框架实现。

导入审计区分模块执行时、函数体与 `TYPE_CHECKING` 中的依赖；静态图中的延迟 SCC 不等于启动失败。
`config`、`metric_registry` 和 `extensions` 不为消除 late import / `E402` 注释而拆分。
只有实际依赖循环导致初始化顺序或职责问题时才调整模块边界。

`ExecutionProfileResolver` 在载入 execution profile plan 后建立一次资源索引，
常规窗口、post-hoc 完整性检查和批量回填各自复用同一解析器。
解析器只保存实际使用工具的标量结果快照，不复制 report 等大型载荷；
计划合并或变更后须重新建立解析器。输入尺度仍按原顺序使用 `isclose` 容差匹配，
首个含目标工具的资源 case、disabled/missing/invalid 和错误优先级保持不变。
单次 `find_execution_profile_entry` 仍支持直接读取计划；持久化 schema 与 provenance 不变。

`metric_registry` 统一 CSV 字段、单位、来源、窗口和 profiler 完成条件；`config.CSV_FIELDS`
保留同一列表对象。
`acprof.hardware_conditions` 仅定义历史硬件条件字段与产物路径；
`host.hardware_conditions` 负责在采集窗口外观察硬件并写入记录，
`analysis.comparison` 和 `analysis.independent_comparison` 只依赖共享只读协议，避免导入主机采集层。`analysis/audit` 和 `analysis/uncertainty` 负责只读审计与窗口统计，
`acprof.cli.audit` / `acprof.cli.stats` 仅处理参数和报告输出。生成的 `docs/results/metric_reference.md` 可在 CI 检查漂移。

指标模块不读取环境、不创建 workload 或 monitor。慢请求阈值由 client 在调用时显式传入；
冷启动状态仍由 client 管理。对照窗口、monitor 启停、正式请求和停止后的统计顺序保持一致。
界面刷新、绘图、通知与额外文件操作继续位于正式测量窗口之外。

`application.run.RunApplication.run(RunConfig)` 是准备与采集的应用入口，返回 `RunOutcome`，
不解析进程 argv、不调用 `sys.exit`，并且不反向导入 CLI 或 TUI。
每个实例独立持有 `RunState`、tmux 日志句柄和 `RunNotifications`，可注入准备工作流、
状态工厂、运行时准备、矩阵执行、环境配置与日志边界；已有测量锁仍阻止并发正式实验。
`cli.run` 只适配参数、SIGTERM 与退出码，原始参数的 `None` 保留未显式覆盖环境设置的含义。
`RunConfig.from_namespace` 和 `to_namespace` 复用既有参数声明，并保留 seed、SLO 等额外公共选项，
不另建配置 schema。准备阶段返回 `PreparedRuntime`，汇总镜像、输入计划、profiler 计划与能力报告。
`RunConfig.validate()` 汇总纯参数约束与逐字段 `ConfigIssue`，并规范化 profiler 重复次数和参考资源。`config_constraints` 提供 CLI、表单、冻结参数及 `ClientConfig` 共用的无损数值解析，拒绝布尔值、非有限值和小数截断；主机 CPU 亲和可用性、环境 SLO 与硬件能力仍在 preflight 检查。用户实验的 `repeat` 至少为 1，直接测量入口允许准备／清理用的 0 窗口；这一区别显式保留在共享规则中。底层测量还验证进程身份和证据目录等运行不变量。
`_execute()` 只串联执行选项校验、任务集合分派、主机预检、准备、矩阵执行和结果发布；
`_collect_matrix()` 消费单一 `MatrixRequest`，`_publish_matrix_results()` 负责 capability 校验、
CSV 合并、RunState 完成与清理，task child 继续使用同一 `_finalize()` 所有权和异常优先级。
`RunNotifications` 在预检前、每个 profiler/case 窗口结束后及终态完成时创建通知事件，
并负责 CSV 摘要、保密错误回显与尽力发送；不同应用实例之间不共享通知上下文。
终态通知必须在 tmux 日志关闭及 RunState 清理后执行；通知异常不得改变实验结果、污染采集窗口或泄露 webhook。
CSV 格式化、像素指标和 profiler 关联集中于 `WindowMetrics.build_row`；
`ClientConfig.from_env()` 仅在入口读取并验证配置，导入 client 不修改代理环境、
不读取实验参数、不创建 workload。每次执行持有独立状态；`WindowExecutor.execute`
复用 `MonitorGroup` 保持采样顺序，`ClientRunner.main` 负责窗口调度与结果发布。
`orchestrator._finalize_case` 负责采集后的抓包合并和结果校验，清理故障仍会执行容器回收。

`source_identity` 统一续跑身份与服务构建的文件选择。续跑包括执行声明和输入资源；服务构建
只打包共享模块与 `container/workloads/extensions`，同一文件集合同时用于上下文复制和指纹。
范围细节见[镜像分类与复用](../models/images.md#镜像分类与复用)。

清理机制参考 [CPython ExitStack](https://github.com/python/cpython/blob/3.12/Lib/contextlib.py)
的回调栈，使用现有 Python 标准库（PSF License），不增加依赖；清理错误另行聚合以保留请求证据。
实验隔离参考 [Optimum Benchmark 的隔离改进](https://github.com/huggingface/optimum-benchmark/pull/186)
（Apache-2.0），仅借鉴所有权与生命周期思路，不引入其调度框架或测量开销。

`runtime_profiles` 使用标准库将 manifest 实例化为 `RuntimeProfile`、`PlatformSpec`、`DependencyEnvironment`；
公共查询 `locked_transformers_version(environment)` 从所选环境锁读取 Transformers 版本，
供主机预检、模型契约与容器加载策略共用；保留缓存且不导入推理框架。
7 个任务族通过逻辑 profile 共享依赖环境，当前数量见[运行配置](../models/environment.md#当前配置)。`dependency_locks` 规范化和验证
制品锁，环境内容身份独立于 profile、adapter、模型及业务代码。主机检测只读元数据；handler 注册表
供 server、输入规划和 profiler 共用。`extensions/*/manifest.json` 同时提供 config 映射、任务支持、
profile 和延迟入口，读取声明不导入推理框架；声明文件参与服务镜像指纹。
`extensions/schema.py` 按类型注解校验声明，family 默认值与 task/backend 覆盖统一由 catalog 合并。
`detect` 与 `model_candidates` 共用 `CATALOG.resolve()`，规则同级使用显式 priority，冲突和未知组合明确报错；
`model_schema`、`input_plan` 和 handler 消费选中声明。`config` 的尺度/任务参数与 TUI 的任务族列表由 catalog 派生。
backend、路由和镜像身份只从当前扩展声明解析；旧 extension schema v1 直接拒绝，
完整字段与优先级见[扩展声明](../models/extensions.md#扩展声明与按需加载)。
`model_candidates` 只根据固定 commit 的仓库布局、Hub/作者声明和注册表推导候选；
`model_support` 读取 `extensions/transformers` 的已审查 Auto 注册及本地 loader 能力。
`model_resolution` 校验最终选定的执行接口，不重算候选，不加载模型权重；
profile 选择按任务／架构匹配已锁定环境，平台切换保留版本线。
标准库模块 `model_spec` 共用本地／仓库模型声明及代码引用检查；候选记录保留证据和歧义，
有效声明参与服务镜像与恢复身份，并传入容器 loader。自定义 pipeline 复用标准任务 Handler，
不在采集器增加模型分支；是否可运行由独立验证的各阶段结果判断。
元数据解析不加载模型，静态候选与独立推理、正式采集证据分别记录。timm、Chronos 三代及句向量
复用上游接口与现有 Handler，公共采集器不增加 checkpoint 分支。
`container.execution` 只加载所选声明的可选执行模块；Torch 上下文位于 `torch_execution`。
无执行模块时采用 CPU/nullcontext，复用同一个 `BaseHandler`，不增加平行适配器层次。
Workload 使用同一声明的可选 `workload_entrypoint`，按 family 延迟导入；任务参数在实现的
`from_config` 中处理。`container.execution.complete_prediction` 调用所选运行时的可选请求完成
hook，等待计入既有窗口，窗口外验证仍在独立进程。`runtime_settings` 统一可选线程/Provider
请求的读取和传递，不负责资源调度。`analysis.comparison` 只读比较已有结果条件，不参与恢复身份。
`host.images.environment` 构建固定 Python/系统平台及其依赖环境分支；旧 Torch 平台保持原锁和身份；`host.images.runtime`
绑定模型与最终服务代码的构建身份。镜像是按需缓存，不为每个 profile 强制保留一个镜像。
`scripts/compile_locks.py` 复用固定 uv 解析目标 wheel；`scripts/compile_system_lock.py` 在隔离基础容器
中解析 Debian Snapshot。普通构建仅消费锁，`--check` 只读校验锁及映射。
`container.model_files` 是标准库文件规划器，
`download_model` 负责下载与构建期完整性检查，`runtime_manifest` 与 `runtime_validate` 分别负责环境清单和独立接口验证。
分层设计将权重下载与业务代码变更解耦；加载、镜像复用与验证契约见[运行兼容](../models/images.md#构建复用和验证)，
字段与历史兼容见[采集协议](../profiling/metadata.md#static_metajson-字段)。

容器归属标签借鉴 Apache-2.0 许可的
[Testcontainers 会话标签](https://github.com/testcontainers/testcontainers-python/blob/main/src/testcontainers/core/labels.py)，
结合本机 Linux 的 boot ID 与进程启动时间判断废弃状态。继续使用现有 Docker CLI，不增加 Docker SDK
或 Ryuk 常驻容器；具体退出、恢复与旧容器处理见[排障](../usage/troubleshooting.md#中断后残留容器或端口占用)。
CPU 与资源监控共享 PID 查询和采样调度，NVML 保留自己的首采样时机；各自的 `_nan_result`、
`_result_from_samples` 保留能量积分、窗口计数和缺失值语义，不按同名强行合并。

### 单 case 请求与独立 profiler 编排

`host.cases` 定义四类冻结配置：`CaseRequest` 描述任务、镜像、资源与目录，
`CaseWorkload` 描述请求次数和超时，`CaseMeasurement` 描述采样与采集策略，
`CasePlans` 引用准备阶段已生成的输入和 profiler 计划。`run_matrix` 统一接收 `MatrixRequest`，
按冻结计划生成每个 case 的配置；`run_single_case` 接收这些配置，并将 `RunState` 作为显式执行依赖传入，
不混入配置对象。

单次 `_CaseExecution` 拥有该 case 的容器和 tcpdump 进程，并分别处理启动、客户端环境、
客户端失败、测量完成检查点、结果后处理和清理。只有客户端调用位于原有
`measurement_boundary` 内；环境构建、抓包解析、结果校验和持久化保持在窗口外。
恢复已完成测量的 case 直接进入后处理，不启动新容器或重新采样。
OOM、请求超时和 readiness 失败继续保留已完成行与原有错误证据；
抓包停止失败仍执行容器回收，清理失败保留原始执行错误并发布原有 cleanup sidecar。

`collect_execution_profile_plan` 接收 `ExecutionProfileRequest`、
`ExecutionProfileOptions` 和独立进度回调。两个工具各自使用 `ProfilerSampling` 描述
采样策略、重复次数和参考资源；执行配置不包含与工具无关的项目目录字段。
资源规范化、工具环境准备、独立工具执行、资源网格展开和 JSON 发布分别实现。
Massif/Nsys 仍按原顺序独立执行，工具不可用时保留原有诊断条目；
扩展到未直接测量的资源 case 时保留来源 CPU、内存和采样策略，持久化 schema 不变。

设计参考 [HTTPX 的 Request 与 Client.send 边界](https://github.com/encode/httpx/blob/master/httpx/_client.py)
及其异常时关闭响应和认证流程的所有权，以及
[pytest 的 setup/call/teardown 阶段编排](https://github.com/pytest-dev/pytest/blob/main/src/_pytest/runner.py)。
这里只借鉴类型化请求、阶段结果与资源所有权，不增加依赖、重试或新的测量行为。
