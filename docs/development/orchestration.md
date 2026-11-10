# 主机编排与测量模块

[← 返回专题目录](architecture.md)

## 主机编排与测量

| 模块 | 职责 |
| --- | --- |
| `application/task_runs` | 保存推理组集合与配置，按不同执行路径串行调度独立 `RunApplication`；组间失败隔离、恢复及 workload 物化均在测量外 |
| `application/run` / `application/outcome` | 单次运行生命周期与类型化 `RunOutcome`；应用层不解析 argv 或调用 `sys.exit` |
| `application/publication` | 合并 case CSV、校验 capability、完成 `RunState`、清理本次中间产物并输出完成回执 |
| `models/candidates` / `tasks/workloads` / `tasks/groups` / `host/preparation/task_selection` | 分离模型声明、语义输入模板、计算配置分组与交互选择；不以架构候选替代明确任务证据，也不为 prompt-only 变化重跑资源矩阵 |
| `preflight` | 原生 Linux、本机 Docker、cgroup 与 CPU 能耗前置检查 |
| `preparation/network_preflight` / `preparation/download_probe` | Docker/Process 下载计划与确认、来源元数据检查及共享预算的限量 GET；复用 HF transport，全部在准备阶段执行 |
| `system/command` | host 同步命令的 UTF-8、timeout、环境、退出码、耗时与脱敏 metadata；不拥有长期进程 |
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
| `metadata/machine` | 准备期 GPU、内存、swap、Docker storage/image 与 CPU policy 探测；保留 unavailable/unknown 状态 |
| `metadata/static` | `StaticMeta` 组装、enrich 与写出；不直接拥有主机或 Docker 探测实现 |
| `measurement.capture` | tcpdump 前置检查及 capture/parser 命令构造 |
| `orchestrator` | case/matrix 调度、idle 稳定性、失败与超时处理、OOM pruning 和 CSV 合并 |

| `run_state` | 目录锁、实验身份、状态快照与原子持久化；case 归档和清理由 `host.case_recovery` 负责，均在测量窗口外运行 |
| `measurement.monitors` | `MonitorGroup` 所有权、固定启停顺序及共享 `run_matched_control_window`；不导入硬件或 workload |
| `execution_conditions` | 保存／恢复线程环境、固定 GPU UUID、CPU affinity 和请求超时；开销诊断与负载重放共用 |
| `measurement.load` | 共享 HTTP Scheduler、配置和 `run_load` 入口；协调 `/predict` 与可选 SSE Adapter，不进入正式能耗 CSV |
| `measurement.request_inputs` | 计时前的有界 JSONL 读取、来源与有序输入摘要、凭据字段拒绝；不调度请求或下载内容 |
| `measurement.server_metrics` | 显式选择的服务端 Prometheus 快照，位于每点负载前后；保留采样错误，复用异常清理机制，不进入负载计时或正式能耗 CSV |
| `measurement.load_transport` | 单次负载的线程局部连接、HTTP 请求与活动 socket 所有权；取消时唤醒读取，线程退出后关闭连接 |
| `measurement.load_reporting` | 补齐未发送请求记录、窗口外 Token 派生与负载报告；保留原始错误及成功样本筛选，v3 输入索引由 `load` 在窗口外附加 |
| `measurement.streaming` | OpenAI-compatible SSE 解码、有效输出时间、Token 来源与流式指标；不调度负载或采集硬件 |
| `client` | 环境与 workload 初始化、短连接 HTTP 请求适配、窗口调度和完成结果发布 |
| `measurement.config` | `ClientConfig` 是子进程配置的唯一类型化对象；集中处理 `from_env()` / `to_env()`，资源限制、资源计划和输入顺序在进程内部保持结构化类型 |
| `measurement.numbers` | 请求归约与结果序列化共用的数值转换；不读取环境、collector 或文件 |
| `measurement.collector_metrics` | 已完成 collector 结果到 GPU/CPU/资源/PMU 指标组的纯投影及 monitor 告警计算 |
| `measurement.metrics` | `WindowMetrics` 保留部分归约结果、判定指标状态并组装完整结果行；idle history 由显式方法更新，行组装不修改历史 |
| `measurement.execution` | `WindowExecutor.execute` 拥有单窗口 collector 构造、对照采集、请求、停止、请求证据发布和异常优先级 |
| `measurement.types` / `measurement.errors` | `WindowSpec`、`WindowObservations`、`WindowResult` 与共享的致命测量异常；不访问硬件 |
| `measurement.requests` | 串行请求进度、HTTP body 大小、实际参数和有效输入尺度一致性；不拥有 collector 或文件 |
| `monitors/rapl_topology` | powercap 完整域发现、alias 去重、package/DRAM 来源选择与可用性；独立于矩阵计划 |
| `monitors/common` | Docker PID 查询与 CPU/资源采样的绝对时刻调度；保留各监控器的异常类型 |
| `host/container_lifecycle` | 按主机与进程身份确认废弃服务容器，在冷启动计时前回收 |
| `host.case_recovery` | case 归档、checkpoint 恢复与清理债务；`RunState` 保留锁、状态快照和持久化，`processing_checkpoint()` 只读，显式恢复入口才执行清理 |
| `profiling.compute` / `profiling.execution` | profiler 计划编排及汇总；工具执行实现统一放在 `profiling/tools/` 下 |
| `profiling.tools.ncu` / `profiling.tools.torch` / `profiling.tools.advisor` | 各自的容器执行与产物处理；NCU 另拥有 checkpoint、export 和 resume |
| `profiling.tools.massif` / `profiling.tools.nsys` | 隔离执行、错误条目与产物处理；Massif 拥有 checkpoint/resume，Nsys 拥有 stats、SQLite 与 raw stream 清理 |
| `measurement.diagnostics` / `measurement.publication` | 窗口外 idle/NVIDIA 诊断，以及停止后请求 JSONL、CSV 与 sidecar 的发布 |
| `measurement.error_rows` | 失败、timeout 和 startup-OOM 行的纯归约与发布；校验 measurement key、保留已提交成功行，并维持 sniff sidecar 先于 CSV 的提交顺序 |
| `monitors/resource_readers` / `monitors/resource_metrics` | cgroup/proc/sysfs discovery 和 raw reader；纯 sample reduction 与 counter 派生计算 |
| `profiling.tools.compute_parsers` / `profiling.tools.execution_parsers` | Advisor/NCU CSV、Massif snapshot 和 Nsys stats 的纯标准库解析 |
| `profiling.tools.tool_discovery` | 可执行文件、版本目录优先级和完整工具挂载路径 |
| `profiling.tools.execution_environment` | 原始模型镜像的 profiler 能力核验、工具版本查询 |
| `profiling.support` | 两类 profiler 的容器命令、runner 参数、输入计划与产物路径；复用 `artifacts.atomic_write` 发布原有 profiler JSON |
| `application/terminal_log` | tmux pane pipe 的启动、停止与日志发布；不覆盖已有 pipe，由本次 `RunApplication` 持有并在 finally 清理 |

`measurement.load.run_load()` 保留公共调用参数与 v1/v2/v3 报告协议。线程池和就绪屏障统一使用
请求数与配置并发数的较小值，避免短负载预启动闲置线程；报告中的配置并发上限不变。
Scheduler 在 worker 就绪后开始计时，
取消时先设置取消状态、撤销待执行任务并唤醒活动 socket，再等待线程退出；负载耗时在最终连接关闭和
Token 派生之前冻结。`load_reporting` 在此后补齐未提交请求、派生流式指标并汇总，原始异常仍携带
`load_report` 返回调用方。线程池行为沿用 [CPython ThreadPoolExecutor](https://github.com/python/cpython/blob/3.12/Lib/concurrent/futures/thread.py)，
清理阶段分工参考 [Locust Runner](https://github.com/locustio/locust/blob/42c4ac9554e6c74cbe52e67fc8f0cf2557c2dfa8/locust/runners.py)；
仅借鉴生命周期设计，不引入 gevent、新依赖或额外后台任务。三个 load 模块共同纳入 C901 ≤ 12 门禁。

`arrival-rate` 用公开的 `Future.add_done_callback` 将完成通知送入本次负载的线程安全队列，
每次到达先消费通知，再判断 `max_pending`，避免反复扫描全部 pending Future；pending 集合仍由
Scheduler 单独修改，结果检查继续传播原始失败。回调只弱引用队列，避免末批 Future 与队列循环持有。
设计参考 [Dask as_completed 的完成队列](https://github.com/dask/distributed/blob/edf67787a4e0d5c93ed8803eaa298b2ddf0f67ad/distributed/client.py#L5687)
（BSD-3-Clause），并核对 [CPython Future callback](https://github.com/python/cpython/blob/3.12/Lib/concurrent/futures/_base.py#L399)
（PSF License）的已完成调用与对象持有语义；只借鉴设计，使用标准库，不复制源码或引入 Dask 依赖。

请求集准备逐行完成模型、形态与凭据校验，只保留本阶段前缀的规范化副本；整个校验、编码及摘要
仍在启动 worker 和计时前完成。设计参考 [Hugging Face Datasets 的逐条转换](https://github.com/huggingface/datasets/blob/a1269ca9dbb6e0858a9c375e8982329a3fcc84af/src/datasets/iterable_dataset.py#L1512)，
沿用项目已有 adapter 与 CLI 的前缀保留方式，不复制源码或引入 Datasets 依赖。

回放 CLI 的抓包收尾复用 `cleanup_on_exit` 保留原始错误与取消，并独立尝试强杀后的有界回收。
设计参考 [CPython subprocess](https://github.com/python/cpython/blob/3.12/Lib/subprocess.py) 的信号与等待分工，
以及 [取消后显式回收的 PR 讨论](https://github.com/python/cpython/pull/160941)；仅借鉴设计，不复制源码或引入依赖。
抓包成功后仍执行原有 PCAP 有效性和请求覆盖检查，失败诊断见[负载报告](../results/comparisons.md#非流式负载报告)。

`host.measurement.energy_validation` 在正式采样结束后读取 case CSV 并核验 CPU/GPU idle 功率基线，
保留既定错误状态、告警 sidecar 与阈值；`orchestrator` 仅调度该校验。
`host.result_cleanup` 只有在合并结果存在且非空时才回收逐 case 临时文件，
`cli.run` 只负责参数与调度，v2/flat 路由与请求证据保留策略由结果模块维护。

`measurement.collector_metrics` 对跨模块使用的 CPU/GPU/资源样本转换公开 `cpu_metrics_from_result`、
`gpu_metrics_from_result` 与 `resource_usage_metrics_from_result`；`measurement.numbers` 提供共享数值转换，
其他局部计算保留模块私有，不为降低 private API 计数机械公开所有助手函数。

`runtimes/service` 是输入规划的下层，Docker adapter 继续使用 `docker_runtime`；`configure_client()` 将 Runtime session
直接写入类型化 `ClientConfig`，orchestrator 不再维护另一套环境变量字段映射。`metadata/static` 引用 runtime、输入计划类型和
任务 schema；这些模块均不反向引用 `orchestrator`。调用方直接引用各模块。
两类 profiler 单向依赖 `profiling.support`，TaskInfo 只在类型检查时导入；
解析与工具查找使用 `profiling/tools/` 下的实现。`profiling.execution` 通过 Massif/Nsys 的
`profile` 与 `error_entry` 调度各工具，私有采集、恢复或解析实现只由所属模块调用。
解析器不导入 Docker、模型检测或采集编排，单独读取报告无需安装推理框架。
测试在函数实际查找依赖的位置 mock。

Process 启动失败时先尝试读取 OOM 证据，再执行既有的进程终止与资源释放。
普通诊断错误不得绕过清理或替代原始启动故障／取消；错误作为异常备注保留，
普通启动失败的 stderr 与错误 CSV 同时保留这些备注。只有成功读取的 OOM 证据才能改变故障分类。
进程清理未确认时仍保留 Model Store 租约与资源范围，传播 `RuntimeCleanupError` 阻止后续执行，
其 `run_error` 保留原始启动异常。轻量真实服务加故障注入可验证所有权与传播；真实 cgroup OOM 仍需硬件验收。

`ManagedProcess` 构造期间即持有控制管道，父端关闭或所有权握手失败也必须尝试回收 supervisor。
构造尚未返回时发生的未确认清理统一传播 `ProcessCleanupError`，由服务与验证入口保留租约和资源范围。
确认回收后补写实际启动账本：已启动的 supervisor 保留 `started` 与 owner PID，只有未创建进程才标记
`failed`；账本仍无法发布时保留错误备注，Resume 继续拒绝缺失证据，不绕过 supervisor 状态核验。
接口与运行时验证保留清理错误类型及 `cleanup_status=incomplete`，立即停止后续设备验证；
报告或 Model Resolution 写入的普通错误仅追加诊断，不能将清理未确认改成普通兼容性失败。
该构造失败处理借鉴 [CPython multiprocessing.Pool 的部分创建回收](https://github.com/python/cpython/blob/3.12/Lib/multiprocessing/pool.py#L183-L222)，
复用项目现有异常与收尾工具，不引入依赖。轻量真实子进程、文件锁与故障注入覆盖此路径，资源范围使用 Fixture。

Process 收尾先撤销控制管道描述符的所有权，再执行一次关闭；按照
[Linux close(2) 的错误处理规则](https://man7.org/linux/man-pages/man2/close.2.html)，关闭报错后不得重试可能已被复用的描述符。
管道关闭失败仍有界等待 supervisor 并检查清理记录，多重故障保留诊断；状态读取失败或清理中断同样
传播 `ProcessCleanupError`，不能降为普通兼容性错误后继续调度。独立验证将原始失败、超时或取消写入
清理错误的 `run_error`，未确认清理时保留 Model Store 租约和资源范围。
真实子进程、文件描述符复用与父进程文件锁回归验证此边界；资源范围 Fixture 不代表真实 cgroup 验收。

Profiler 命令统一经过 `profiling.commands.run_profiler_command`：主机工具查询保留原有执行方式，
每次 `docker run` 单独分配 cidfile 和现有进程所有权标签，按不可变 ID 有界回收容器。
启动前同样复用 `container_lifecycle.recover_abandoned_containers`，只回收已确认退出的所有者
留下的容器；当前所有者的遗留容器、清单查询失败或回收失败均阻止新启动。
其他存活、身份无法确认或不在当前 host/user 范围的所有者不会被自动删除。
每次容器启动前增加一次有界 `docker ps`，对候选执行 inspect／回收；主机工具查询不走此流程。
`--rm`、客户端退出或超时不能单独证明回收完成；删除失败时由 `container_lifecycle`
检查同一 ID 是否确实不存在。清理未确认则传播 `RuntimeCleanupError`，停止后续工具、
输入尺度及计划发布，并保留 Model Store 租约；原始运行错误或取消保留在异常链和 `run_error` 中。
清理成功时继续原有工具错误语义，取消仍传播；租约关闭失败保留重试句柄和原始异常。
这些操作在独立 profiler／准备阶段执行，不进入正式测量窗口，不增加 profiler 运行时限。
参考 [Testcontainers 的容器作用域](https://github.com/testcontainers/testcontainers-python/blob/6146aa1170f415fcc825ed9bddc4eacbeb937a12/src/testcontainers/core/container.py)
（Apache-2.0）与 [Docker CLI 的 cidfile 写入](https://github.com/docker/cli/blob/master/cli/command/container/create.go)，
复用项目现有 CLI、清理和异常保留工具，不增加 SDK、后台清理服务或依赖。
所有权范围参考 [Testcontainers 的 session 标签](https://github.com/testcontainers/testcontainers-python/blob/6146aa1170f415fcc825ed9bddc4eacbeb937a12/src/testcontainers/core/labels.py)，
AC-Prof 继续使用自身 host/user/boot/PID/start-time 身份与既有恢复判定。

`RuntimeCleanupError` 在准备和多任务边界同样必须中止当前执行：`PreparationWorkflow` 记录失败后直接传播，
不提供交互 retry/rebuild；`application.preparation` 不将其包装为普通准备失败或可选 profiler 告警，
不绑定 runtime，也不启动后续 profiler 或正式矩阵。`TaskCollection` 保存当前任务失败、已有成功结果
和未执行任务的 pending 状态，将 collection 标为 interrupted 后传播原始异常；目录锁照常释放。
普通任务失败仍继续独立任务，普通可选 profiler 失败仍报告能力缺口。
这一停止调度原则参考 [CPython ProcessPoolExecutor 的 broken 状态处理](https://github.com/python/cpython/blob/7b06d6f80df2729a68f78d2a346c043d10f82fd6/Lib/concurrent/futures/process.py)，
沿用既有异常和状态格式，不引入 executor 或新依赖。应用入口与落盘状态使用离线故障注入验证。

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
执行器通过 `record_idle_history()` 显式更新 case 基线后才调用 `build_row()`；行组装可重复执行且不修改历史。
GPU idle 来源在 workload stop 完成累计计数器回退后读取。

边界测试直接使用替代 collector 和请求端口，验证完整窗口、部分请求、初始化失败、
清理次数和组合故障的异常优先级；CLI/CSV 集成测试继续验证外部产物。
设计参考 [MLPerf LoadGen 的 IssueQueries / PerformanceResult 边界](https://github.com/mlcommons/inference/blob/master/loadgen/loadgen.cc)，
以及 [pyperf 对测量阶段状态与 teardown 上下文的讨论](https://github.com/psf/pyperf/issues/239)。
仅借鉴生命周期和结果分离，不引入依赖，也不修改现有采样策略、CSV schema 或 monitor 启停顺序。

### Host command 与 diagnostics

host 的短生命周期同步命令直接复用 `system.command.run_command`，覆盖 Docker、doctor、
preflight、perf probe、profiler、TUI diagnostics 和 packet 后处理。镜像、状态、容器与调用方
直接使用 runner；需要非零退出抛错或直通输出的调用显式设置
`check=True` / `capture_output=False`。container-side subprocess 与开发 scripts 不在此执行边界中。

runner 接收 literal argv，禁止隐式 shell；stdout/stderr 使用 UTF-8 和 `errors="replace"`。
返回值保留 `CompletedProcess` 的 args、returncode、stdout/stderr，并增加 `duration_s` 和 metadata。
默认捕获输出、`check=False`；调用方原有的 check、capture 和 timeout 选择保持原义，未设置 timeout
仍为无限等待；与 Docker daemon 交互的生产路径应显式使用有界 timeout，启动和 inspect
预算由 `system.command` 集中定义。`env` 保留完整替换语义，`env_overrides` 在它或父环境上覆盖，不修改父进程环境。
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
计划合并或变更后须重新建立解析器。输入尺度按精确数值匹配，重复尺度使用首项，
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
不另建配置 schema。准备阶段返回 `PreparedRuntime`，汇总运行时制品、输入计划、profiler 计划与能力报告。
`PreparedRuntime.image_info` 和共享 case／输入规划／静态元数据使用 `RuntimeArtifact`，
明确接收 `ImageInfo | ProcessInfo`；仅用于 Docker 的执行路径继续使用 `ImageInfo`。
`ExecutionRuntime.prepare(..., gpu_device=...)` 由 Process 实现绑定 GPU 选择；
`RunState.restore_runtime()` 将实验输出根目录传给 `restore(..., output_dir=...)`，
由 Process 实现在身份校验后复制环境字典、重绑定日志目录，保留已记录的 GPU 和模型身份。
应用准备层不再直接修改 Process 制品，恢复操作也不会改写保存的快照；
Docker 实现保持既有镜像准备和不可变镜像身份校验。
`RunConfig.validate()` 汇总纯参数约束与逐字段 `ConfigIssue`，并规范化 profiler 重复次数和参考资源。`experiments.constraints` 提供 CLI、表单、冻结参数及 `ClientConfig` 共用的无损数值解析，拒绝布尔值、非有限值和小数截断；主机 CPU 亲和可用性、环境 SLO 与硬件能力仍在 preflight 检查。用户实验的 `repeat` 至少为 1，直接测量入口允许准备／清理用的 0 窗口；这一区别显式保留在共享规则中。底层测量还验证进程身份和证据目录等运行不变量。
`auto` 在环境设置、锁和模型访问之前复用该校验；只有自动准备入口允许保留尚未选择的
`profiling_mode=auto`，不提前决定 basic/full，也不放宽 Process 的显式 Profiler 或 DRAM 限制。
校验返回值通过既有 `RunConfig.to_namespace()` 转交准备与采集入口，保持路径、任务与资源值一致；
原 CLI 中未指定的选项继续为 `None`，避免规范化默认值覆盖下载环境配置。本地声明按当前工作目录解析。
参考 [Click 的参数转换、来源与命令调用分离](https://github.com/pallets/click/blob/main/src/click/core.py)
及 [CPython argparse 的参数错误处理](https://github.com/python/cpython/blob/3.12/Lib/argparse.py)，
仅借鉴设计，沿用项目共享约束与标准库，不复制代码或引入新的 CLI 依赖。
`_execute()` 只串联执行选项校验、任务集合分派、主机预检、准备、矩阵执行和结果发布；
`_collect_matrix()` 消费单一 `MatrixRequest`，`application.publication.publish_matrix_results()` 负责
capability 校验、CSV 合并、RunState 完成、清理及自动统计 JSON 发布，task child 继续使用同一 `_finalize()` 所有权和异常优先级。

**运行阶段边界。** `application.resources.resolve_runtime_options()` 接收经校验、环境配置后的
请求 Namespace 和已记录的 runtime kind，返回新的执行 Namespace；不原位改写请求的
`runtime/cpus/mems/prune_startup_oom`。这是顶层标量选项的阶段拷贝，不是任意嵌套对象的深冻结。
Docker 默认资源轴在同模块声明一次；Process 保持 auto/inherited 语义。恢复 auto 继续使用
冻结 runtime，显式选择和错误优先级不变。未知字段和环境配置附加信息原样转交后续阶段。

| 路径 | 可观察阶段与所有权 |
| --- | --- |
| 正常执行 | 校验/环境配置 → runtime 解析 → 任务解析 → 主机预检 → 获取 RunState → 日志 → 准备 → 矩阵 → 提交 → 报告 → 收尾 |
| 已完成恢复 | 校验/解析/预检和状态核验保留；直接重试报告，不重新开始日志、准备或矩阵 |
| 准备失败 / 重建 | 失败不进入矩阵；明确的 RebuildEnvironment 仅重复准备，不重做已完成的解析和预检 |
| 采集失败 / 中断 | 不发布成功结果；仍按既有规则保存失败证据、关闭日志和状态锁，再交付通知与最终事件 |
| 报告失败 | 已提交结果保持 complete；报告可通过 resume 重试，不重新测量 |
| 多任务 | 每个子任务完成日志/锁/通知收尾之后才进入下一项；普通失败记录后继续，未知清理失败保持既有阻断规则 |

`test_run_phase_contract.py` 从公开 `RunApplication.run()` 驱动真实 RunState、矩阵计划、结果
合并和 TaskCollection，只替换硬件/模型执行、报告渲染和通知/终端边界。阶段与锁断言替换了
原先仅 patch `_execute` 验证日志关闭的一项测试，其余故障组合测试继续保留。
参考 [pytest 的 wrap_session / _main](https://github.com/pytest-dev/pytest/blob/main/src/_pytest/main.py)
将阶段执行与统一收尾分离的思路，以及 [CPython ExitStack](https://docs.python.org/3.12/library/contextlib.html#contextlib.ExitStack)
的清理规则；仅借鉴结构，沿用项目的 cleanup_on_exit，不引入调度器或新依赖。

`publish_run_statistics()` 在结果提交后复用 `analysis.statistics_report`，已完成实验的 Resume
也经过同一入口；整个计算与发布期间继续持有 RunState 的测量锁和目录锁，不重复申请离线锁。
报告直接放在实验根目录的 `analysis/`，既有 Layout v2 路由清单及 HTML 报告路径保持兼容。
完成阶段发布的设计参考 [pytest-benchmark 的 session.finish / handle_saving](https://github.com/ionelmc/pytest-benchmark/blob/97c52d8e0de0ac4f307e35716f723bf0376d63d2/src/pytest_benchmark/session.py)
与 [Google Benchmark JSONReporter](https://github.com/google/benchmark/blob/42786a7876120b03e2ecff269695729337d243d9/src/json_reporter.cc)；
统计计算、快照校验、原子发布及去重均沿用项目现有实现，不复制外部源码或引入依赖。
日志关闭、RunState 关闭、终态通知与通知上下文重置按顺序尝试，任一收尾失败都不能跳过后续步骤。
应用释放实例占用后才发送 `run_finished`；事件输出失败也必须保留此前的运行与清理错误。
收尾统一复用 `system.cleanup.cleanup_on_exit`：保留原始异常及附带诊断；清理期间的取消优先于
普通运行错误，已有取消不会被后续错误替换。CLI 错误输出及 task collection 的字符串 `error`
记录保留 exception notes；任务状态、后续任务执行规则和恢复协议不变。
矩阵结束时，质量证据和运行失败证据各自独立汇总；一份汇总失败仍尝试另一份，
并保留原始采集异常或取消。成功采集后的汇总失败仍阻止完成发布；旧报告沿用原子写入保护，
修复证据后 Resume 复用已提交 case。PCAP 超时包装和 packet／energy／MIPS 的应用层转换
保留 exception notes，CLI 可以显示清理故障；超时消息不直接拼入原始命令或部分输出。
设计借鉴 [CPython unittest.doCleanups](https://github.com/python/cpython/blob/main/Lib/unittest/case.py)
的独立清理与错误保留，并核对取消传播问题
[gh-148427](https://github.com/python/cpython/issues/148427) 及其
[PR #148435](https://github.com/python/cpython/pull/148435)；只复用本项目已有机制，
不复制 PSF License 源码、不增加运行依赖，额外处理仅位于运行收尾阶段。

任务集合的中断清单与 `acprof auto` 的最终报告也复用同一收尾规则。集合锁获取成功后，
启动、执行和终态发布均在中断保存保护范围内；保存失败仍继续释放锁。自动报告在准备锁
及采集调用结束后生成，保留传入的 exception notes，并沿用现有 collection path 定位规则。
普通报告失败通过 CLI 错误输出和退出码 `2` 显式报告；取消继续传播，即使报告自身写入失败。
成功的 `SystemExit` 仍须完成最终报告，不能因报告保存失败返回成功。
此处借鉴 [Django Atomic.__exit__](https://github.com/django/django/blob/main/django/db/transaction.py)
保留原始事务异常的原则，并核对异常处理器再次失败的问题
[PR #21974](https://github.com/django/django/pull/21974)。仅参考设计，未复制 BSD-3-Clause 源码，
不增加依赖、字段或正式测量窗口内的工作。

`RunNotifications` 在预检前、每个 profiler/case 窗口结束后及终态完成时创建通知事件，
并负责 CSV 摘要、保密错误回显与尽力发送；不同应用实例之间不共享通知上下文。
终态通知必须在 tmux 日志关闭及 RunState 清理后执行；通知异常不得改变实验结果、污染采集窗口或泄露 webhook。
CSV 格式化、像素指标和 profiler 关联集中于 `WindowMetrics.build_row`；
`ClientConfig.from_env()` 仅在入口读取并验证配置，导入 client 不修改代理环境、
不读取实验参数、不创建 workload。每次执行持有独立状态；`WindowExecutor.execute`
复用 `MonitorGroup` 保持采样顺序，`ClientRunner.main` 负责窗口调度与结果发布。
`orchestrator._finalize_case` 负责采集后的抓包合并和结果校验，清理故障仍会执行容器回收。

**ClientConfig 的进程边界契约**

字段来源按下表区分，不能把运行时派生值重新变成用户配置，也不能把所有层合成一个配置对象。
每个字段的环境变量名、非默认实例和精确编码由 `test_client_config_transport.py` 独立列出；
新增字段若缺少声明示例，覆盖检查失败。配置经真实 Python 子进程解析，而不是仅在父进程内往返。
生产编解码共用只读 `CLIENT_ENV_FIELDS`：每项显式给出 wire 名、解码、编码和默认继承策略，
默认值仅从 `ClientConfig` 获取，公共数值默认复用已有 constants。没有第二份默认值表或新配置框架。
声明表保持原解析顺序；不捕获或重写底层解析错误。新增字段必须同时登记传输规则和独立测试样例。

| 来源 | ClientConfig 字段 |
| --- | --- |
| 执行 runtime / 已验证会话 | `runtime_kind`, `process_pid`, `process_start_ticks`, `resource_plan`, `process_startup_s` |
| TaskInfo / 运行制品 | `model_revision`, `task_family`, `pipeline_tag`, `runtime_backend`, `image_tag` |
| 资源 case / 设备选择 | `cpu_cores`, `mem_cap_gb`, `gpu_mode`, `device_index`, `gpu_device_uuid` |
| 服务会话 / HTTP 配置 | `base_url`, `endpoint`, `request_timeout_seconds` |
| CaseWorkload / 条件性环境覆盖 | `batch_size`, `warmup`, `repeat`, `repeat_in_window`, `repeat_window_seconds`, `auto_warmup_requests` |
| 会话冷启动证据，缺失保留 nan | `cold_start_s`, `cold_start_started_at`, `cold_start_ready_at`, `cold_start_container_launch_s`, `cold_start_server_setup_s`, `cold_start_cuda_init_s`, `cold_start_model_load_s`, `cold_start_ready_wait_s` |
| 实验、case、attempt 与产物布局 | `out_csv`, `case_name`, `container_name`, `run_id`, `case_id`, `case_attempt_id`, `window_records_dir`, `telemetry_dir`, `result_root` |
| 诊断与中间产物控制 | `sniff_groups_path`, `idle_debug`, `idle_diag_path`, `client_error_path`, `idle_debug_trace_interval_s` |
| CaseMeasurement / 采集能力选择 | `use_mips`, `profiling_mode`, `dram_energy`, `sample_hz`, `sampling_policy`, `raw_sample_limit`, `idle_seconds`, `idle_cooldown_seconds` |
| 已冻结的输入 / profiler 计划 | `input_scales_str`, `input_scale_plan_file`, `compute_profile_plan_file`, `execution_profile_plan_file`, `input_scale_order` |

大多数字段由类型化配置完整覆盖同名宿主环境变量，包括空字符串、False 与未知资源。
仅三个历史入口保留条件性继承：`AUTO_WARMUP_REQUESTS` 在值为 5 时、
`IDLE_DEBUG_TRACE_INTERVAL_S` 在值为 0.1 时、`SNIFF_GROUPS_PATH` 在空字符串时不发出键，
于是子进程可继承宿主值；非默认值始终显式覆盖。显式传入恰好等于默认值也遵循这一规则，
不能在保持行为的重构中悄悄改为“默认值总是覆盖环境”。未识别环境键不成为配置字段。

自动预热和诊断间隔由 `measurement_environment` 记录，参与既有运行条件、恢复与比较；
重放恢复来源环境后再建立子进程，退出恢复调用方环境。`SNIFF_GROUPS_PATH` 是遗留的中间
产物路径覆盖，目前不属于该身份快照；本轮仅固定此边界，不改变结果目录和恢复协议。
结构化资源计划使用 JSON object，输入顺序使用 JSON list；未知资源保持 None/nan，
布尔、路径空白、URL 尾斜线和浮点编码保留既有归一化，非法值在 client 初始化时拒绝。

orchestrator 保留代理、设备等宿主环境，显式去掉通知 webhook；这不是通用的凭据隔离沙箱。
独立 case 测试经过真实子进程确认 webhook 不下传、代理可用且受控参数不被同名环境污染。
纯 codec 测试不读取模型、容器、计划文件或硬件，也不替代真实测量验收。
优先级显式化参考 [Pydantic Settings 的来源规则](https://github.com/pydantic/pydantic-settings/blob/main/docs/index.md#field-value-priority)，
仅借鉴契约和测试思路，不引入依赖或改变 AC-Prof 现有优先级。

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
`detect` 与 `models/candidates` 共用 `CATALOG.resolve()`，规则同级使用显式 priority，冲突和未知组合明确报错；
`model_schema`、`input_plan` 和 handler 消费选中声明。`config` 的尺度/任务参数与 TUI 的任务族列表由 catalog 派生。
backend、路由和镜像身份只从当前扩展声明解析；旧 extension schema v1 直接拒绝，
完整字段与优先级见[扩展声明](../models/extensions.md#扩展声明与按需加载)。
`models/candidates` 只根据固定 commit 的仓库布局、Hub/作者声明和注册表推导候选；
`models/support` 读取 `extensions/transformers` 的已审查 Auto 注册及本地 loader 能力。
`models/resolution` 校验最终选定的执行接口，不重算候选，不加载模型权重；
profile 选择按任务／架构匹配已锁定环境，平台切换保留版本线。
标准库模块 `models/spec` 共用本地／仓库模型声明及代码引用检查；候选记录保留证据和歧义，
有效声明参与服务镜像与恢复身份，并传入容器 loader。自定义 pipeline 复用标准任务 Handler，
不在采集器增加模型分支；是否可运行由独立验证的各阶段结果判断。
元数据解析不加载模型，静态候选与独立推理、正式采集证据分别记录。timm、Chronos 三代及句向量
复用上游接口与现有 Handler，公共采集器不增加 checkpoint 分支。
`container.execution` 只加载所选声明的可选执行模块；Torch 上下文位于 `torch_execution`。
无执行模块时采用 CPU/nullcontext，复用同一个 `BaseHandler`，不增加平行适配器层次。

模型入口共用 `container.model_session`：不可变的 `ModelLoadConfig` 集中读取模型身份、
本地 snapshot、adapter、设备和线程配置；`select_handler`、`prepare_execution`、
`load_model_session` 分别选择 Handler、初始化执行上下文和加载模型，仍委派现有注册表及
`load_handler`。已加载的 `ModelSession` 只保存配置、Handler、context、execution 和 device，
不自动推理、warmup、验证或启动 profiler。
`server.create_app(session)` 将 HTTP 路由绑定到已加载会话；导入 `server` 不加载模型、
初始化设备或修改离线环境。生产入口 `create_runtime_app()` 在显式启动时设置离线策略，
保留服务准备、CUDA 初始化和模型加载的计时分段；`python -m acprof.container.server`
仍启动原有 HTTP 服务与继承 socket 路径。

`runtime_validate` 继续拥有各阶段事件、必填配置检查和输出质量门禁；
`compute_profile_runner` 继续拥有 profiler 支持检查、专用 `load_options`、warmup 与采集窗口。
三个入口只共享加载组装，正式 HTTP 请求仍按 preprocess → predict → completion → postprocess
执行。应用会话可独立注入，请求完成超时等进程级设置仍由现有 runtime settings 管理。

**模型适配器身份。** `ModelLoadConfig.from_environment()` 是服务、独立 runtime validation、
compute profiler 和 model probe 的环境配置入口；未指定 adapter 时采用 `family-default`。
直接构造 `ModelLoadConfig(adapter=...)` 使用显式值，构造时省略 adapter 也采用该默认值。
配置形成后，`select_handler`、`HandlerRegistry.get`、`prepare_execution`、`load_model_session`、
`load_handler`、`handler_declaration` 和 `registered_policy/load_policy` 都使用同一显式 adapter。
之后仅改变 `ACPROF_MODEL_ADAPTER` 不会重新路由、替换缓存项或令已捕获配置失效；
未知 adapter 明确失败，不回退默认 handler。直接调用下游 API 时同样省略即默认，不再隐式读取环境。

`BaseHandler.load` 及各任务族 loader 以 keyword-only `adapter` 接收身份；NLP 到句向量
委托、CV direct loader、Audio codec 等辅助路径也显式传递。这个参数由 AC-Prof 自身消费，
不放入 `load_options`，也不透传 Transformers、Torch、Diffusers、ONNX Runtime 等第三方接口。
interface probe 只核验导入和签名，包括 handler 的 adapter keyword，不执行模型加载或推理。

注册表保留原有默认键 `family:backend` 和专用键 `(adapter, family, backend)`；
同键复用无会话状态的 handler，不同 adapter 隔离。重复注册默认拒绝，`override=True`
替换包括已缓存实例在内的旧注册项。惰性导入和构造成功后才缓存实例，失败保留原异常链，
可在依赖修复后重试；模型、设备、权重及一次加载的上下文由各自 `ModelSession.context`
持有，不写入共享 handler 实例。

显式 adapter 不代表实际运行制品已经匹配。`load_handler` 在调用模型 loader 前执行
`registered_policy`，包括无需 Torch 的 handler；仍校验 runtime profile 的 family、adapter
及远程代码权限，Torch loader 继续检查已安装依赖版本。环境中的
`ACPROF_RUNTIME_PROFILE` 代表独立准备的运行配置，不被显式 adapter 覆盖。
构建入口 `runtime_manifest` 保留环境读取及下载计划、Model Store 和依赖身份校验；
主机镜像准备继续核对 manifest、标签、下载计划和不可变制品身份。三者分别属于构建、
服务配置捕获与模型加载边界，不重扫启动时权重，也不更改 manifest schema、指纹或恢复身份。

此设计借鉴 [Transformers Auto 工厂的显式配置与惰性映射](https://github.com/huggingface/transformers/blob/863d2620210d72fc0a9a03fdfe200bc9699154f6/src/transformers/models/auto/auto_factory.py)
（Apache-2.0）与 [pluggy 的重复注册检查及验证后发布](https://github.com/pytest-dev/pluggy/blob/9d5cfafe1a476be5376449e653970733955533f2/src/pluggy/_manager.py)
（MIT），沿用现有 registry 和四阶段协议，不引入新框架。离线 fixture 与真实 Python
子进程覆盖配置后环境漂移、交错会话、缓存覆盖、失败重试及四类入口；这些证据不代替
真实 checkpoint、Docker/GPU 或硬件性能验收，正式测量窗口不增加检查。
应用构造方式参考 [Flask application factories](https://flask.palletsprojects.com/en/stable/patterns/appfactories/)，
沿用现有 Flask 和四阶段 Handler，不增加依赖或新的适配器框架。

`NLPHandler` 在加载时消费声明中已合并的 `encoder` 选项，选定
`SentenceEncoderHandler` 处理句向量、相似度和重排；选择保存在各自的模型 context，
同一个缓存的 NLP Handler 可以交替处理不同会话。专用处理器集中拥有 SentenceTransformer／
CrossEncoder 的加载、输入准备、预测和输出摘要；通用 NLP 保留 Pipeline、生成和其他任务逻辑。
原有 `sentence_transformers`、`cross_encoder` 及 `transformers_pipeline` 的相似度／重排别名
继续使用同一扩展声明和入口，context 中的 backend、task、revision 和加载选项保持原值。
`load_handler` 仍是唯一的加载质量记录与异常包装边界。

`container.handlers.text_inputs` 提供两类 Handler 共用的模型长度、单文本和成对文本预算、
截断及 fill-mask 占位符处理；模块导入只依赖标准库。专用规则保留默认／具名 prompt、
prompt 与文本拼接后的 token 检查、完整句向量输出、逐批候选对应关系和原始重排分数。
测试使用轻量 tokenizer／SDK 替身，分别覆盖专用规则、既有路由和跨会话状态隔离，
并复用原有 fill-mask、QA、生成与服务源码打包回归。
接口边界参考 [SentenceTransformer.encode](https://github.com/huggingface/sentence-transformers/blob/main/sentence_transformers/sentence_transformer/model.py)
与 [CrossEncoder.predict](https://github.com/huggingface/sentence-transformers/blob/main/sentence_transformers/cross_encoder/model.py)；
仅借鉴职责划分，继续使用项目锁定的依赖版本和既有调用参数，未复制上游实现。

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

**包级后处理子进程协议。** 正常执行、完成测量后的 resume 和历史 capture segments
都由 `measurement.capture.packet_parser_command()` 构造可信的模块 argv；路径作为单独参数传递，
checkpoint 只提供已验证的数据，不保存或重放可执行命令。

| 边界 | 契约与失败行为 |
| --- | --- |
| parser → host | 退出码必须为 0，stdout 必须是一个有限数值的 JSON object；stderr 仅作诊断，不混入 JSON。空 stdout、畸形 JSON、原生 NaN/Infinity、错误 schema 和错误对象形态均拒绝。 |
| schema v2 | `packet.protocol` 共用版本与结构校验；`requests` 是记录对象映射，`streams` 可省略但存在时必须是记录对象映射。没有无版本裸映射/list 的回退。空 requests 是合法 parser 输出，不代表取得有效时延。 |
| 历史分段 | 使用相同解析入口及超时预算；合并后保留 capture_segment 和流级字节来源，重复请求身份仍拒绝。当前段为空但归档段有效时仍可派生，不重复计算共享流字节。 |
| merge → host | 退出码 0 还必须在本次独立临时目录写出 CSV，然后才同文件系统原子替换。此前遗留的 `.merged` 不作为本次成功证据；失败不替换输入 CSV，原始 checkpoint 不变。 |

`measurement.packet_processing` 负责有界 worker 调用和结果验收，仍使用已有进程组终止策略；
`_finalize_case` 保留空/部分测量策略、CSV 完整性和能耗诊断。CSV 缺失报错，已完成行回填包指标，
未执行或失败行继续保留错误与 `nan`。readiness 哨兵、客户端退出码及状态判定不在本轮重写。
子进程、目录分配和解析均发生在正式测量窗口之外；没有新增指标、CSV 列或 checkpoint schema。

真实 TShark 只读取可重建的小型合成 pcap，不启动 tcpdump 或网络请求。公开 `run_single_case`
测试覆盖正常执行与新 RunState 的恢复、parser/merge 失败、损坏的派生 CSV、原始字节保留及
容器/客户端只调用一次；另覆盖分段、部分错误行、缺 CSV、退出码 0 无新产物与非法 JSON。
这些测试替代一项完全 patch `_finalize_case` 的重复恢复测试，包解析算法及独立故障注入继续保留。
设计参考 [PyPA pyproject-hooks 的 _call_hook](https://github.com/pypa/pyproject-hooks/blob/main/src/pyproject_hooks/_impl.py)
的调用私有临时目录，以及其[成功退出但缺少 output.json 的问题 #173](https://github.com/pypa/pyproject-hooks/issues/173)：
退出码不能代替协议结果。仅借鉴边界思路，沿用标准库与现有 worker，不引入构建后端依赖。

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
