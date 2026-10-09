# 主机编排与测量模块

[← 返回专题目录](architecture.md)

## 主机编排与测量

| 模块 | 职责 |
| --- | --- |
| `preflight` | 原生 Linux、本机 Docker、cgroup 与 CPU 能耗前置检查 |
| `host/command` | host 同步命令的 UTF-8、timeout、环境、退出码、耗时与脱敏 metadata；不拥有长期进程 |
| `docker_runtime` | 推理服务容器的所有权、启停、ready 检查及原有冷启动分段 |
| `container_state` | Docker inspect 状态证据、启动失败与运行期 OOM 解释；不拥有容器生命周期 |
| `runtime_identity` / `gpu_device` | 镜像、容器、payload 共用的模型名称 token；GPU mode 归一化及选定设备身份 |
| `dependency_images` | 平台与依赖环境的内容缓存、安装配方指纹、完整清单和标签核验；主构建及容器 CI 共用 |
| `runtime_images` | `ImageInfo`、镜像准备、profile/平台选择、模型层与代码层构建、父镜像绑定、运行清单及不可变 image ID 核验 |
| `image_management` | Docker 镜像清单、标签合并、容器引用检查及按确认清单删除；不参与采集 |
| `image_graph` | 从镜像元数据、依赖锁身份与层链解析父子关系、逻辑名称、共享层和按选择集合去重的释放估算；不访问 Docker |
| `image_dependencies` | 匹配镜像身份与本地锁，核对构建阶段并生成本层依赖增量；不启动容器或扫描包 |
| `runtime_validation` | 矩阵前的独立 CPU／GPU 完整推理验证及报告，不生成测量行 |
| `input_plan` | 手动和自动尺度规划、规划用 probe、payload 物化与输入计划写入 |
| `startup_probe` | 独立 readiness-only 启动探测、Docker OOM 证据与保守连续前缀，不采集性能 |
| `matrix_plan` | 资源 case 与 input scale 的独立确定性排序、冻结计划、hash 与恢复校验，不采样硬件 |
| `model_schema` | 从 catalog 读取任务 IO 模板与推理精度，返回独立副本供元数据补充 |
| `static_metadata` | 主机、镜像、模型和 profiler 计划的静态元数据 |
| `packet_capture` | tcpdump 前置检查及 capture/parser 命令构造 |
| `orchestrator` | case/matrix 调度、idle 稳定性、失败与超时处理、OOM pruning 和 CSV 合并 |

| `run_state` | 目录锁、实验身份、已完成 case 校验、中断备份和恢复；仅在测量窗口外运行 |
| `measurement_window` | `MonitorGroup` 所有权、固定启停顺序及共享 `run_matched_control_window`；不导入硬件或 workload |
| `execution_conditions` | 保存／恢复线程环境、固定 GPU UUID、CPU affinity 和请求超时；开销诊断与负载重放共用 |
| `load_protocol` | 独立非流式 HTTP 调度与连接生命周期；结果不进入正式能耗 CSV |
| `client` | 环境与 workload 初始化、请求、对照窗口和正式窗口控制、结果写入 |
| `client_metrics` | 已完成采样结果到指标字段的纯计算与格式化 |

| `monitors/rapl_topology` | powercap 完整域发现、alias 去重、package/DRAM 来源选择与可用性；独立于矩阵计划 |
| `monitors/common` | Docker PID 查询与 CPU/资源采样的绝对时刻调度；保留各监控器的异常类型 |
| `host/container_lifecycle` | 按主机与进程身份确认废弃服务容器，在冷启动计时前回收 |
| `compute_profile` / `execution_profile` | profiler 计划编排及汇总；工具执行实现统一放在 `profilers/` 下 |
| `profilers/ncu` / `profilers/torch` / `profilers/advisor` | 各自的容器执行与产物处理；NCU 另拥有 checkpoint、export 和 resume |
| `profilers/massif` / `profilers/nsys` | 隔离执行、错误条目与产物处理；Massif 拥有 checkpoint/resume，Nsys 拥有 stats、SQLite 与 raw stream 清理 |
| `client_diagnostics` / `client_publication` | 窗口外 idle/NVIDIA 诊断，以及停止后请求 JSONL、CSV 与 sidecar 的发布 |
| `monitors/resource_readers` / `monitors/resource_metrics` | cgroup/proc/sysfs discovery 和 raw reader；纯 sample reduction 与 counter 派生计算 |
| `profilers/compute_parsers` / `profilers/execution_parsers` | Advisor/NCU CSV、Massif snapshot 和 Nsys stats 的纯标准库解析 |
| `profilers/tool_discovery` | 可执行文件、版本目录优先级和完整工具挂载路径 |
| `profilers/execution_environment` | 原始模型镜像的 profiler 能力核验、工具版本查询 |
| `profiler_support` | 两类 profiler 的容器命令、runner 参数、输入计划与产物路径；复用 `artifacts.atomic_write` 发布原有 profiler JSON |
| `cli/terminal_log` | tmux pane pipe 的启动、停止与日志发布；不覆盖已有 pipe，CLI 持有活动会话并负责 finally 清理 |

`host.energy_validation` 在正式采样结束后读取 case CSV 并核验 CPU/GPU idle 功率基线，
继续保留原来的错误状态、告警 sidecar 与阈值；`orchestrator` 仅调度该校验。
`host.result_cleanup` 只有在合并结果存在且非空时才回收逐 case 临时文件，
`cli.run` 不再直接拥有文件清理实现，保留既有 v2/flat 路由与请求证据保留策略。

`client_metrics` 对跨模块使用的 CPU/GPU/资源样本转换公开 `cpu_metrics_from_result`、
`gpu_metrics_from_result` 与 `resource_usage_metrics_from_result`；其他局部计算保留模块私有，
不为降低 private API 计数机械公开所有助手函数。

`docker_runtime` 是输入规划的下层；`static_metadata` 引用 runtime、输入计划类型和
任务 schema；这些模块均不反向引用 `orchestrator`。调用方直接引用各模块。
两类 profiler 单向依赖 `profiler_support`，TaskInfo 只在类型检查时导入；
解析与工具查找使用 `profilers/` 下的实现。`execution_profile` 通过 Massif/Nsys 的
`profile` 与 `error_entry` 调度各工具，不再重导出其私有采集、恢复或解析入口。
解析器不导入 Docker、模型检测或采集编排，单独读取报告无需安装推理框架。
测试在函数实际查找依赖的位置 mock。

### Host command 与 diagnostics

host 的短生命周期同步命令直接复用 `host.command.run_command`，覆盖 Docker、doctor、
preflight、perf probe、profiler、TUI diagnostics 和 packet 后处理。镜像、状态、容器与调用方
直接使用 runner，原 `docker_runtime._run` 已删除；需要非零退出抛错或直通输出的调用显式设置
`check=True` / `capture_output=False`。container-side subprocess 与开发 scripts 不在此执行边界中。

runner 接收 literal argv，禁止隐式 shell；stdout/stderr 使用 UTF-8 和 `errors="replace"`。
返回值保留 `CompletedProcess` 的 args、returncode、stdout/stderr，并增加 `duration_s` 和 metadata。
默认捕获输出、`check=False`；调用方原有的 check、capture 和 timeout 选择保持原义，未设置 timeout
仍为无限等待。`env` 保留完整替换语义，`env_overrides` 在它或父环境上覆盖，不修改父进程环境。
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

`metric_registry` 统一 CSV 字段、单位、来源、窗口和 profiler 完成条件；`config.CSV_FIELDS`
保留同一列表对象。
`acprof.hardware_conditions` 仅定义历史硬件条件字段与产物路径；
`host.hardware_conditions` 负责在采集窗口外观察硬件并写入记录，
`analysis.comparison` 和 `analysis.independent_comparison` 只依赖共享只读协议，避免导入主机采集层。`analysis/audit` 和 `analysis/uncertainty` 负责只读审计与窗口统计，
`acprof.cli.audit` / `acprof.cli.stats` 仅处理参数和报告输出。生成的 `docs/results/metric_reference.md` 可在 CI 检查漂移。

指标模块不读取环境、不创建 workload 或 monitor。慢请求阈值由 client 在调用时显式传入；
冷启动状态仍由 client 管理。对照窗口、monitor 启停、正式请求和停止后的统计顺序保持一致。
界面刷新、绘图、通知与额外文件操作继续位于正式测量窗口之外。

`cli.run` 将资源校验、运行准备和矩阵调度分开；准备阶段返回 `_PreparedRuntime`，汇总镜像、
输入计划、profiler 计划与能力报告。client 的 CSV 格式化、像素指标和 profiler 关联集中于
`ClientRunner._build_result_row`；`ClientConfig.from_env()` 仅在入口读取并验证配置，
导入 client 不修改代理环境、不读取实验参数、不创建 workload。每次执行持有独立状态；
`_execute_window` 复用 `MonitorGroup` 保持采样顺序，`main` 负责窗口调度与结果发布。
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
`detect` 与 `model_resolution` 共用 `CATALOG.resolve()`，规则同级使用显式 priority，冲突和未知组合明确报错；
`model_schema`、`input_plan` 和 handler 消费选中声明。`config` 的尺度/任务参数与 TUI 的任务族列表由 catalog 派生。
旧 `DEFAULT_BACKEND`、三个路由镜像字典、`default_backend()`、MOSS 常量及旧 architecture/profile 镜像导出均已移除，
旧 extension schema v1 直接拒绝；完整字段与优先级见[扩展声明](../models/extensions.md#扩展声明与按需加载)。
`model_resolution` 按固定 commit 的仓库布局和原生接口解析候选，`extensions/transformers` 保存
固定版本的 Auto 注册数据；profile 选择按任务／架构匹配已锁定环境，平台切换保留版本线。
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
`host.dependency_images` 构建固定 Python/系统平台及其依赖环境分支；旧 Torch 平台保持原锁和身份；`host.runtime_images`
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
