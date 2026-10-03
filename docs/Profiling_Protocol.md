# 采集协议与实验产物

修改请求窗口、输入计划、冷启动或产物来源时查阅。字段计算见 [指标](Metrics.md)，运行步骤见 [profiling 流程](../.agents/skills/acprof-profiling-workflow/SKILL.md)。

[文档导航](README.md)

## 采集生命周期

根 CLI 顺序为 `resolve → interface validation → prepare runtime → runtime validation → matrix measurement`。
任务解析和环境预检通过后，先检查无权重的源码接口，再准备正式镜像和输入计划，完成每个选中设备
的最小真实请求，最后运行所选 profiler 与正式矩阵。validation 不属于 measurement。
启用启动剪枝时，先执行独立 startup-OOM probe，再冻结正式矩阵计划并开始采集。
每个正式 case 创建新容器；client 控制已有预热、冷却、无请求对照与 workload 窗口，
监控停止后才计算派生值和写行。case 产物经抓包解析与校验后合并；图表及通知属于测量之外的操作。
模块顺序见[主机编排](Architecture.md#主机编排与测量)，镜像与独立验证见[运行兼容](Runtime_Compatibility.md#构建复用和验证)。

独立接口验证与 startup probe 可能预热宿主机文件缓存。冷启动描述全新容器的进程和模型初始化，
不承诺磁盘冷缓存；`cold_start_first_predict_app_s` 不计入 `/ready` 前的分段和，也不新增推理请求。

### 容器清理与失败证据

正式 case、startup probe、Interface Probe 和独立 runtime validation 使用同一组 owner 标签（主机、用户、
boot、PID 与进程 starttime），并通过 Docker 的 `--cidfile` 或成功启动返回的完整容器 ID
确定清理对象。名称只用于显示；进程被强杀后，下一次启动只回收可证明 owner 已退出的本机容器。
启动前若发现完整 owner 标签仍属于当前进程的容器（包括已退出但尚未删除的容器），
直接记录 `preflight` 清理问题并拒绝启动，不删除活 owner 的容器。同一 PID 捕获异常后显式重试
也必须先解决残留；其他活 owner 的容器继续保留，PID 相同但 boot/starttime 不同仍按原规则核验。

清理在测量窗口结束后执行：`stop --time 10` 的 CLI 上限为 15 秒，`rm -f` 为 30 秒，
状态 inspect 为 15 秒。stop 失败但 rm 成功仍算清理完成；rm 失败或超时后必须 inspect，
只有明确的该 ID 不存在响应才视为完成。容器仍存在或 Docker 状态无法确认时中止后续 case。
恢复遗留容器也使用同一移除规则，不根据删除命令中的任意 “No such” 文本推断成功。

失败 case 的 `cleanup_error.json`（flat 布局为 `<case.csv>.cleanup_error.json`）保存 schema v1：
`status=incomplete`、不可变 `container_id`、`final_state=present/unknown`、可用的 `docker_state`、
各命令的 `operations`（操作、秒数上限、返回码或异常）及原始 `run_error`（类型与详情）。
独立验证将同一结构放在 `runtime_validation.json` 的 `devices.<device>.cleanup_error`；
启动前拒绝则保存到顶层 `cleanup_error` 并设置 `cleanup_status=incomplete`，尚未产生设备验证结果。
原推理失败记录继续保留；清理不完整不会被成功验证覆盖。Docker 超时后 cidfile 缺失、损坏或不可读
同样记为 `unknown`，保留原超时阶段和预算证据，不根据容器名称猜测归属或执行删除。
历史结果缺此 sidecar 表示未记录该项证据，
不能据此推断容器清理成功。续跑将失败 case 的清理 sidecar 与原始运行证据一起归档，
新 attempt 不覆盖旧失败。诊断不改变 CSV 指标和冷启动计算。

重试已有 cleanup debt 前，调用方先核对原主机与 Docker daemon 身份，并持有测量锁；
`recover_cleanup_debt` 复用上述 abandoned-owner 恢复，再对旧错误中的完整容器 ID 逐一做有界 inspect。
另一活 owner 的容器即使被安全恢复跳过，也不能算已清理；缺失或非法 ID 时必须再次确认
同主机、同用户的 lifecycle 标签范围为空。查询超时、返回损坏或仍有容器时保留旧 debt 并拒绝后续工作。
成功证据记录 `status=complete`、`verified_container_ids`、`recovered_container_ids`、
`owner_scope`、必要时的 `scoped_inventory_empty` 与核验 `operations`；原失败证据继续归档。

### Measurement preparation 与窗口副作用

每个窗口在第一个 monitor 采样前完成 preparation。已有 CPU/RAPL、NVML 和 cgroup reader
在构造时完成初始化；`MonitorGroup.prepare()` 再调用需要显式准备的 monitor。
`PerfMIPSMonitor.prepare()` 解析容器 PID、验证 perf attach 能力并固定 executable、argv 和
错误诊断上下文；`start()` 只启动已准备的 `perf stat`，未准备直接报错。
resource 的 CPU 拓扑与频率文件路径在构造时选定，窗口中读取实时值；拓扑变化由下一组 monitor
重新发现。无请求对照和正式请求窗口分别准备，不跨窗口复用失效 PID。

所有 prepare 成功后仍按 **GPU → CPU → resource → MIPS** 启动；停止保持
**MIPS → resource → GPU → CPU**。prepare 失败不会开始采样；已登记资源仍沿原来的
`finish()` / close 路径清理。部分 start、stop、close 失败继续尝试其余清理，并保留取消与请求超时语义。

| 阶段 | 允许的操作 |
| --- | --- |
| Preparation，窗口外 | Docker PID 与环境探测、perf executable/capability probe、profiler prerequisites、模型解析、镜像准备、reader discovery、idle diagnostics |
| 测量窗口内 | 串行 HTTP inference、RAPL/NVML/cgroup/proc/sysfs 采样、perf 测量进程、必要的起止 counter snapshot、内存中的请求记录 |
| 全部 monitor finish 后 | 请求 JSONL、CSV/JSON 与诊断文件发布、统计和派生指标、CLI/TUI presentation、通知与诊断 logging |

测量窗口内禁止非必要的 Docker/preflight discovery、同步 diagnostic logging、文件发布、
界面输出、通知、模型解析及镜像准备。机器可读 progress/preparation events 仍在原来的
case/client 边界发出，不改为 logger，也不向每个请求插入事件。
成功、timeout、取消及清理失败路径均先完成 monitor finish，再写已缓冲的请求证据；
取消和 timeout 不因 publication 失败而被替换。原有 HTTP 请求次数、短连接和超时定义保持不变。

### 进程与界面边界

TUI 的 `ProcessLifecycle` 统一普通停止、回调异常和卸载清理：向独立进程组发送 SIGINT，
等待 30 秒，再发送 SIGTERM 并等待 5 秒。只有观察到子进程退出才释放引用。超时保留 PID、
错误、管道读取和后续停止能力；不自动 SIGKILL，以免跳过编排器的容器清理。
`清理未完成` 仍是忙碌状态。若终端已关闭，PID 和清理错误写入 stderr；强行结束 TUI
进程不能保证清理完成，后续仍须核查容器与测量锁。

正式矩阵向 TUI 输出 `ACPROF_EVENT ` 前缀的 JSON 控制记录，`version=1`，
`event` 为 `case_started`、`measurement_started`、`measurement_stopped` 或 `case_finished`，
携带 `case_id`；结束事件另含 `status=ok|error|cancelled`。未知版本或损坏控制记录明确报错。
收到结构化事件后，普通日志不再控制 `measurement_active`；其他 case 的迟到事件不改变当前窗口。
事件只在 case/client 边界输出，不逐请求发送、不在窗口内追加控制文件。
这里保护的是整个 client 生命周期（包含其多个采样窗口）；管道通知没有同步 ACK，
不承诺 TUI 绘制完成与第一轮采样之间存在严格的时序屏障。独立最大尺度 probe 暂沿用自己的日志进度。

### 网络与请求超时

推理端口固定发布到宿主机 `127.0.0.1`，容器内服务继续监听原地址。
本机 `/ready`、`/meta`、`/predict` 和 Docker bridge 抓包沿用原路径。
`--request-timeout-seconds` 分别作用于 Requests 的连接等待和读取无进展等待，
不是整个请求的严格 wall-clock 截止时间；持续收到数据时，总耗时可以超过该值。
请求继续串行执行，每次 `Connection: close`，不自动重试。
超时仍先停止全部 monitor，再由编排器清理容器；错误 sidecar 的 `timeout_semantics`
记录为 `connect_or_read_inactivity`。不把客户端停止等待当作模型已经停止推理。

### 独立非流式负载

```bash
acprof load results/model --gpu off --scenario concurrent --concurrency 4 \
  --requests 100 --output-dir internal-testing/load-concurrent
acprof load results/model --gpu off --scenario arrival-rate --concurrency 4 \
  --rate 10 --arrival poisson --seed 7 --requests 100 --capture \
  --output-dir internal-testing/load-arrivals
```

`load` 从已完成实验恢复固定镜像、revision、输入计划 hash、线程设置、CPU 集合、GPU UUID 和请求超时，
启动独立容器并使用主机测量锁。默认选源矩阵最大 CPU／内存及第一个输入尺度；可用 `--cpu`、
`--mem`、`--input-scale` 选择源实验已有配置。缺少 GPU UUID、设备不可解析或 CPU affinity 无法满足时
明确失败；当前 shell 的 GPU／线程变量不能覆盖源条件。硬件观测保存到独立输出目录。

`serial` 一次只有一个请求；`concurrent` 是完成后补发的闭环；`arrival-rate` 预先生成固定或泊松
到达时间，不因服务变慢重新排程，首请求在起点提交。`--concurrency` 限制工作线程，
`--max-pending`（默认 1024）限制执行中加排队请求，满额的新到达记为 dropped。
`--warmup` 默认 5 个串行请求，排除在测量窗口之外。无自动重试，超时沿用连接／读取无进展口径。

`--connections close` 为默认；`reuse` 每个工作线程复用连接，并自动要求抓包验证。
服务必须支持 HTTP keep-alive；现有 Flask development server 会主动关闭连接，对这类源镜像
`reuse` 明确失败，不把重新建连冒充复用。本次不替换源镜像的 HTTP server。
PCAP 以 HTTP `request_in` 校验逐请求响应与完整 `X-Req-Id`，并检查实际出现了共享 TCP 流。
同一流包含多个请求时，流级字节只记录一次；逐请求 wire-byte 指标标为不可分摊，避免重复归因。
关闭连接的既有正式采集仍沿用单请求独占流的字节口径。

`load.json` 保存独立 `run_id`、完整协议、源身份与新的 `identity_sha256`；连接、调度或并发不同
即属于不同实验。失败和取消也保留报告；输出目录必须为空。该产物不生成正式能耗 CSV，
不参与 `--resume`，也不接受作为正式窗口统计／跨实验比较的输入。字段定义见
[负载报告](Metrics.md#非流式负载报告)。流式 token 指标需另行实现真实服务端事件。

### 硬件条件证据

每个正式 case 在启动抓包和 client 前观测一次 CPU 型号、宿主机身份哈希、容器所有可见
进程线程的实际 CPU affinity、对应 governor/boost、所选 GPU 的型号/UUID/驱动/功耗上限/
persistence mode，以及 `/meta` 的有效线程数。证据原子写入
`metadata/hardware_conditions.json`（flat 目录为根目录同名文件），schema v1，按资源 case ID
保存到 `cases`。缺失或不可读取字段为 `null`，诊断保存在 `errors`；无 GPU case 为
`gpu=not_applicable`。这是开始边界快照，不证明整个测量期间独占 CPU、温度不变或线程策略不变。

`--cpuset-cpus 0-3,8` 是可选的正式采集/startup probe 容器条件；留空维持原有 `--cpus` 配额。
规范化后的 CPU 集合进入 run options、矩阵和 startup probe 身份，改变集合会拒绝恢复。
正式采集前必须观察到所有容器线程的 affinity 都位于所要求的 CPU 集合内；运行时可以进一步绑定子集。
无法核验或发现越界时保留证据并停止该 case。
独立 runtime validation、额外 profiler 和最大尺度诊断 probe 不继承此选项，其结果继续标识各自范围。
已有实验缺少该文件时比较为 `unknown`，不重写旧产物或补造历史硬件条件。

## 协议不变量

- CPU package、估算 vCPU、GPU device 与 container-attributed 数据分别命名和解释，整机测量不能静默替换容器归因。
- 单请求、workload window、进程生命周期、独立 profiler 使用不同窗口与分母；比较前核对请求数、尺度单位及来源资源。
- 数值 `0`、CSV `nan`、JSON `null` 和字段缺失含义不同；未知历史数据不补成零，推导或代表资源复用不冒充独立实测。
- JSON 保留原生类型；schema 版本由协议与兼容策略决定。只接受当前 schema；可选指标不可用时保持 `nan`，不转换旧字段或推测来源。
- 输入计划、`input_scale_plan_sha256`、workload 素材与模型 revision 必须一致；派生字段复用已有采样，不额外发起请求。
- 静态对象描述与采集过程来源分开保存；补采和修复记录进入 `collection_history.json`，保留原始备份和可回溯信息。

## Profiling mode 与能力证据

`profiling_mode=full` 为默认，保留原有 native Linux、本机 Docker、cgroup v2、RAPL、硬件 instructions
和抓包要求。`basic` 允许 Native Linux 或 WSL2，但仍需本机 Docker 和 cgroup v2，仅要求 application latency、throughput、容器 CPU/memory；
能耗、PMU、packet latency 不请求、不启动、数值保持 `nan`。两种模式不能混作同一画像。
Torch、NCU、Massif、Nsys 仍是显式选择的额外工具，`full` 不意味着自动开启全部工具。

`--dram-energy auto` 在 full 中尝试 DRAM，作为可选能力记录；缺失不改变原有 full 必需项。
`off` 和 basic 的 DRAM 状态为 `not_requested`。只有 `required` 将 DRAM 加入
`requested_measurements`，要求完整 package 覆盖及有效的原始、effective J/request；
权限错误、部分覆盖、采样失败不能标为 verified。`required` 与 basic 组合明确报错。

`capability_report.json` 保存统一的 execution/measurement 对象：`available`（已声明或预检可用）、
`verified`（有当前运行证据）、`unsupported`、`permission_denied`、`not_requested`、`unavailable`、`error`。
每项含 source、detail 和 evidence，运行时验证关联 environment ID。声明不能直接标 verified；
实测缺值不能标零。合法的零值仍是数值，不是能力状态。

`static_meta.json.capability_report` 保存矩阵前的冻结快照，保护恢复时的准备产物身份；
独立 `capability_report.json` 在采集结束后补充实际 CSV/profiler 证据。
它记录主采集收尾时的能力快照；后续 posthoc 补采的状态以对应 profiler plan、CSV error 和
`collection_history.json` 为准，当前不会自动改写该主采集快照。
Capability Report schema v3 将 `requested_measurements_available`（全部要求可采集）与
`requested_measurements_complete`（全部要求已经 verified）分开；预检 available 不再算完成。
`collection_complete` 保留原有“采集行完成且成功”的含义，新增 `collection_finished` 与
`collection_succeeded` 分别表示行已有终态及行执行成功，`row_counts` 区分 succeeded、failed、
not_measured、unfinished。OOM／timeout 是已结束的失败；剪枝未尝试行单列为 not_measured。
这些是已提交审计的行状态，完整计划覆盖仍由 run_state/CSV 唯一键和 audit.coverage 检查。
旧报告不能区分的完成状态为 null，不把 available 或单独的 runtime `status=ok` 冒充验证证据。
读取兼容 schema v1/v2/v3，缺版本按 v1；v1/v2 缺少环境身份时保持 unknown。未知版本及非布尔完成状态明确报错，不隐式转换字符串。
`full_profile_complete` 仅在 native_linux、full 且上述两者满足时成立。工具 permission denied 或输出全为未知数值不能标完整。
主矩阵仍可保留其它成功指标与失败计划；成功测量行缺少当前模式的必需指标时，保存诊断并以失败退出。

模式参与实验恢复身份；旧参数中缺失模式解释为历史默认 full，历史结果不会凭空补出能力证据。
新增 `profiling_mode`、`capability_report` 是 static schema v7 的可选字段，旧 v7 文件继续可读。

### 环境身份与能力支持

static schema v7 新增可选 `platform`、`collection_tier`、`comparability_class`、`environment_class`
和 `platform_runtime`，新采集必须写入。`platform` 包含 environment/native、system/kernel/kernel_version、
machine、WSL generation 与识别证据；`platform_runtime` 保存 `acprof_version`、Git commit、Docker/runtime、GPU driver、
CUDA driver/NVML 版本和探测错误，GPU 硬件身份继续保存在 `gpu_device`。版本不可读时为 null/明确错误，
不合成版本。只在 Preparation 探测，不进入请求窗口。

`platform_runtime.acprof_version` 是无单位的版本字符串，直接取当前执行包的 `acprof.__version__`，
适用于所有安装方式，不依赖 Git 或硬件；已发布版本可对应 `v<version>` Tag。
源码 checkout 额外记录既有 `git_commit`；wheel/standalone 取不到 Git commit 时保持 null 并保留错误原因。
此字段是 static schema v7 的可选来源信息，旧文件缺失表示版本未知，读取时不得用当前安装版本回填。
版本不参与数值聚合，不改变测量窗口、环境比较或 resume 身份。

Capability Report v3 增加相同的环境身份及 `metric_support`，measurement 每项增加独立 `support`。
支持状态为 supported、partial、unsupported、requires_native_validation；它们不替代原来的实测状态。
CSV 新增文本列 `environment_class`（experiment 范围、无数值单位），包括失败占位行。
旧结果缺失、冲突身份读为 unknown，不由当前运行主机反推；CSV 合并、续跑和 Native baseline 拒绝混用。
FULL/PARTIAL 是平台上限，不等于采集成功或 full/basic 模式。完整边界见 [WSL2](platforms/wsl2.md)。

## Startup probe 与冻结矩阵

startup probe 使用独立容器，只启动服务、等待 `/ready`、检查 Docker State 并记录证据。
它不发送 `/predict`，不启动 client、能耗监控、抓包或 profiler，也不写正式 CSV。
每种 GPU mode 在最低选中 CPU 下，按内存升序探测；仅 `ready=false`、
`State.OOMKilled=true`、`State.Running=false` 且未 restarting 的启动失败计入连续前缀。
第一个非 confirmed startup OOM 就停止该 GPU mode 的后续探测，不能跨过缺口扩大前缀。
错误文本、CUDA OOM、timeout、ready 后或正式采集中的运行期 OOM 均不能增加剪枝依据。
该前缀按现有资源单调性假设推广到其它选中 CPU；所有被剪枝资源配置（包括参考 CPU）
只生成 `result_origin=inferred_not_measured` 的占位行，指标为 `nan`，不冒充 probe 实测性能。

probe 证据使用 `startup_oom_pruning.json` schema v2，逐次原子保存启动结果、Docker State、
错误和时间。中断恢复可复用已记录的尝试；全部 probe 结束后才生成 `matrix_plan.json` schema v1。
禁用剪枝时直接冻结矩阵。正式 case 重新创建容器，不复用 probe 的容器或启动计时。

`matrix_plan.json` 保存完整实验身份、实际 case 顺序、每个 case 的实际 input scale 顺序、
`matrix_order`、`seed`（CLI 的 `--matrix-seed`）、`algorithm_version=sha256-sort-v1` 和 `plan_sha256`。
`seeded` 按 seed 与资源键的 SHA256 排序；input scale 由 seed 与该资源键派生独立 seed 后排序，
不共享可变 RNG 状态，也不受其它 case 数量影响。`declared` 保持参数和输入计划的声明顺序。
hash 对不含 `plan_sha256` 的规范 JSON 计算；探测时间戳不进入计划。
相同身份、probe 前缀、算法和 seed 得到相同计划；输入 payload 本身保持原物化计划不变。
独立随机源的设计参考 [MLPerf LoadGen TestSettings](https://github.com/mlcommons/inference/blob/master/loadgen/test_settings.h)，未引入 LoadGen 依赖。

resume 读取实际冻结顺序并校验身份、内容 hash、case/scale 覆盖及 probe 结论，不重新 shuffle。
`run_state.json` 同时绑定矩阵和 probe 文件 hash，已完成实验恢复时也检查这两份证据。
改变 order、seed、输入计划或剪枝选项须使用新输出目录；已有 case 却没有冻结计划、旧版
startup pruning schema v1、损坏或被改写的计划均明确拒绝恢复，不猜测历史执行顺序。

## Workload Contract

输入计划 schema v2 增加 `scenario: {"type": "serial"}`；正式能耗矩阵保持串行请求。
并发／到达率使用下方独立负载协议，不重写源输入计划的场景或正式 CSV。
资源条件仍来自矩阵 CPU/memory/GPU 字段，输入来自物化计划，场景单独声明。

每次 `/predict` 响应包含 `workload_contract` schema v1，保存 task、batch、scenario、计划尺度、实际尺度，
以及各任务可获得的实际输入／输出形状、计数和生成参数。窗口结束后 CSV 文本列 `workload_contract`
序列化本行已完成请求的摘要：`request_count` 保存总请求数，`variants` 保存各不同 contract 及其 `count`。
相同事实在窗口结束后合并，避免快速模型的重复 JSON 超过 CSV 单字段限制；不同输出数量保留分布，
不把它们简单改写成计划上限。这是工作量计数，不保存请求时间顺序。
比较时保留 variant 计数并比较归一化的联合分布，不能降为不含频率的集合；
分布不同与窗口请求总数不同分别记录，详见[跨独立实验比较](Metrics.md#跨独立实验比较)。
未知事实为 JSON `null` 并保留可用性说明，旧 CSV 缺此扩展列继续可读，不补造历史 workload。

输入计划是 planned，原请求上限是 requested，Handler 观测的张量尺寸、token 数与输出数量是
actual；三者不互相替代。新增任务通过现有 `_workload` 补充事实，保持列与 contract schema 不变。
ONNX 独立验证记录实际 Provider、线程数及制品 SHA256；制品校验在准备／验证阶段完成，
正式服务加载时不增加一次完整权重扫描。协议/任务 sanity 验证与固定参考结果检查分别报告，
通过结构检查不表示模型准确率通过。

文本的 `max_output_tokens` 是请求上限，`actual_output_tokens` 来自真实生成 token ID；
计数排除 causal prompt／已知 decoder-start，包含终止 EOS，排除 EOS 后 padding。
`actual_output_tokens_per_sequence` 和 stop reason 保留单序列事实。`output_token_count` 原有的文本重分词
口径保持不变，不能与生成步骤计数混用。无法取得 token ID 时实际计数为 null，不能用最大长度代替。
音频记录时长、采样率、声道；`processor_input_duration` 是送入处理器的波形时长，
处理器 padding/chunk 后的 `processed_duration` 无法观察时为 null，不能用原始时长替代。
图像记录原始尺寸、可观察的 processor tensor shape；
多模态分别记录 text/image/audio/video，不能用单一尺度代替全部模态。处理器内部尺寸不可观察时明确未知。

完整 raw output 的类型、shape、finite、必需字段、实际尺度和 JSON serialization 校验只在独立验证阶段执行，
不在正式测量窗口重新扫描张量。主请求继续执行既有 Handler 预处理和输出约束，附加轻量工作量摘要。
任务 sanity 不等于完整 accuracy benchmark。

## 输出文件

输出目录为 `<output-dir>/<model-dir>/`；模型 ID 中的 `/` 替换为 `--`。
下表列出可能生成的文件；probe、补采、调试与绘图产物仅在执行对应操作时出现。

### Artifact Layout v2

新主实验由 `ArtifactLayout` 创建以下布局；根目录的四个文件分别是正式结果、静态描述、
能力报告和 `result_manifest.json`。子目录按需创建，文件缺失不能据此推断实验成功或失败。

```text
<model-dir>/
├── result_all.csv
├── static_meta.json
├── capability_report.json
├── result_manifest.json
├── metadata/                # 解析、输入/矩阵/profiler 计划和 collection_history
├── raw/
│   ├── requests/<case-id>.jsonl
│   ├── compute_profiles/
│   ├── execution_profiles/
│   ├── posthoc_profiles/
│   └── probes/
├── plots/                   # <environment_class>/{cpu,gpu,gpu+cpu}/、latency_model/、analysis/
├── logs/                    # terminal.log、runtime_validation_<device>.log
├── debug/idle/<case-id>.jsonl
└── .acprof/
    ├── run_state.json
    ├── result.lock
    ├── work/cases/<case-id>/
    │   ├── result.csv
    │   ├── requests.jsonl
    │   ├── sniff_groups.jsonl
    │   ├── packet_latency.json
    │   ├── sniff.pcap
    │   └── client_error.json
    └── recovery/            # interrupted_cases/、posthoc_backups/
```

`<case-id>` 为 `CPUc_MEMORYg_GPU`，例如 `4c_16g_on`。case 中间产物从创建起即位于
隐藏工作目录；校验完成后原子移动请求样本到 `raw/requests/`，再记录 case 完成。
中断恢复会保存工作文件、已移动但尚未确认完成的请求样本和 idle 诊断，备份保留来源相对路径，
避免不同目录下的同名文件互相覆盖；合并完成后只清理
本次 case 的已知中间文件，未知文件留下供诊断。`.acprof/` 包含恢复依据，不能当作缓存删除。

清单的 `schema_version=1`、`layout_version=2` 与 CSV、输入计划及状态文件的 schema 独立。
`primary` 指向三个主要结果文件；`metadata/raw/plots/logs/debug/internal` 给出目录；
`artifacts` 列出已声明文件及目录的映射，`case_work` 和 `request_samples` 给出含 `{case_id}` 的模板。
所有路径相对于模型结果根目录，清单是固定路径契约，包含尚未生成的可选产物，
不记录文件计数、内容 hash 或成功状态。完成与来源判断仍读状态、计划及实际产物。
清单在准备阶段原子发布，读取与路径路由不在请求窗口内递归扫描目录。
未知版本、损坏清单、被改写的路径映射和越界/符号链接路径明确拒绝，不回退猜测。

没有清单的历史目录按 flat layout 读取，绘图和补采继续写入该目录原有位置，不自动迁移。
旧目录的独立 probe 产物可保留，首次主实验仍可创建 v2；已有正式产物则拒绝重新初始化。
目录布局兼容不放宽现有产物 schema、源码身份或恢复校验；升级前中断的实验仍受源码指纹约束。
运行期间不要移动文件或删除清单。以下路径表使用 v2；旧目录沿用原文件名和位置。

| 文件 | 说明 |
| --- | --- |
| `result_manifest.json` | Artifact Layout v2 的路径契约，不代表文件已生成或测量成功。 |
| `capability_report.json` | 本次采集的能力状态及实际完整性，和静态元数据中的准备阶段快照分开。 |
| `quality_checks.json` | 独立 schema v1 的质量观察；普通 warning 不撤销已验证 Capability 或 `full_profile_complete`。 |
| `runtime_failures.json` | 存在正式请求失败时汇总的 typed failure 列表，保留请求 ID、阶段与环境；不改变测量 CSV 数值协议。 |
| `metadata/interface_validation.json` | schema v1 的源码 import／signature 报告，含模型 SHA、source/runner SHA256、dependency image ID、状态与 `inference=not_run`；失败记录 `failed_stage/error`，清理异常另存 `cleanup_error`。原始输出在 `logs/interface_validation.log`，不含测量指标。 |
| `metadata/runtime_validation.json` | 测量窗口外独立运行验证的结构化报告；原始输出在 `logs/runtime_validation_<device>.log`。 |
| `.acprof/work/cases/<case-id>/result.csv` | 采集期间逐资源配置写入的可恢复中间结果；成功合并后清理。 |
| `raw/requests/<case-id>.jsonl` | 长期保留的紧凑 request-level latency，每窗口一行；含 application 原始样本和按请求 ID 对齐的 packet 样本。详见下方约定，不参与默认统计聚合。 |
| `result_all.csv` | 动态测量结果。每一行对应一个 resource config、一个 input scale、一次 warmup/repeat iteration，并记录归一化指标、PCAP 网络字节、cold-start phases，以及该窗口的 cgroup memory/stat/PID、swap、块 I/O 与压力/事件。 |
| `.acprof/run_state.json` | 主实验状态 schema v1，记录实验 ID、参数、主机与源码/依赖指纹、绑定的镜像和输入计划、case 完成状态与 CSV SHA256、启动/恢复记录、最终完成状态。 |
| `.acprof/recovery/interrupted_cases/` | 恢复时保存中断 case 的原始 CSV、PCAP 与关联 sidecar；备份完成后才开始该 case 的新测量。 |
| `static_meta.json` | 单个 JSON object 的静态元数据。记录模型版本、参数/精度/量化/许可证、输入输出格式、per-scale 静态逻辑 FLOPs、推理后端、镜像、GPU/主机 RAM、主机 swap、Docker 存储和环境信息。 |
| `metadata/model_resolution.json` | 运行准备阶段写入的解析报告，包含候选、字段来源、语义和独立运行验证引用；失败 draft 也可独立导出。与可执行 `acprof_model.json` 分离，静态裁决不是推理／测量成功证据。 |
| `metadata/auto_report.json` | `acprof auto` 的预检与收尾报告，保存请求／实际采集模式及静态决策身份；验证与实际采集结果独立引用，不作为 CSV 的替代证据。 |
| `metadata/collection_history.json` | schema v1 的采集/修复 provenance。分别记录 post-hoc profiler 补采、timeout retry、quality retry 和静态元数据回填历史；最新一次状态由对应 history 的最后一项得到。 |
| `metadata/input_scale_plan.json` | 所有任务族共用的 input scale/payload 计划。schema v2 额外记录 workload provenance、per-scale 输入元数据和模型约束；读取端要求 schema v2，拒绝缺少版本或 v1 计划。主采集和 compute profiler 复用同一份 payload。 |
| `metadata/startup_oom_pruning.json` | 独立 startup probe 证据 schema v2；记录最低 CPU、逐次启动结果、Docker State、错误、时间与连续 confirmed OOM 前缀，不含性能测量。 |
| `metadata/matrix_plan.json` | probe 完成后冻结的正式计划 schema v1；含实际资源与 input scale 执行顺序、算法版本、seed、剪枝来源与内容 hash。resume 原样复用。 |
| `metadata/compute_profile_plan.json` | per-scale FLOP profiling 结果。每个 CPU/GPU scale 可同时记录独立的 `torch_profiler_eager` 与 `ncu` profile；NCU 只存在于 GPU profile。失败信息按工具保存，只读取当前按 profiler 分层的 plan 结构。 |
| `metadata/execution_profile_plan.json` | 显式 execution profiling 的采样与 per-resource-config/per-scale 汇总。Massif 条目对应 `gpu_mode=off`，Nsight Systems 条目对应 `gpu_mode=on`；复用 entry 记录实际 source resource 与 sampling strategy，失败按工具记录且不阻断主实验。 |
| `raw/compute_profiles/` | 默认保留的原始 compute profiler artifacts；`--discard-compute-profiles` 可在汇总后删除。 |
| `raw/posthoc_profiles/` | `acprof profile` 生成的补采 plan、原始报告与可恢复 checkpoint。 |
| `.acprof/recovery/posthoc_backups/<timestamp>/` | 成功补采替换文件前保留的原始 CSV、静态元数据与已有历史记录备份。 |
| `raw/probes/largest_scale_<timestamp>_<pid>/` | `acprof probe` 的独立输入计划与 `largest_scale_probe.json`，不含正式 CSV。 |
| `raw/execution_profiles/` | 默认保留 raw Massif `.out` 与 Nsight Systems `.nsys-rep`；stats 导出的 `.sqlite` 缓存会自动删除。传入 `--discard-execution-profiles` 时 raw artifacts 也会在汇总后删除。 |
| `logs/terminal.log` | 在 tmux pane 内运行 `acprof run` 时自动记录的完整终端显示。实验正常结束或报错退出时落盘，不受 tmux 历史行数上限影响。 |
| `plots/latency_model/latency_model_report.json` | `acprof plot` 生成的 latency 拟合报告。包含分 CPU/GPU 的正值模型、整配置留一与最大尺度外推指标、质量门槛、系数和训练范围。 |
| `plots/latency_model/latency_model_residuals.csv` | `acprof plot` 生成的 case-level residual。每个 `GPU mode × CPU × memory × input scale` 聚合 case 一行，包含重复数/离散度、full-fit、resource-config OOF 和最大尺度 holdout 预测。 |
| `plots/latency_model/latency_model_fit_curves.png` | `acprof plot` 生成的 full-fit 曲线图。横轴为 input scale，CPU-off 与 GPU-on 分面展示，每个 `CPU × memory` 资源配置一条拟合曲线，并叠加实测 case 中位数。 |
| `plots/latency_model/latency_model_residuals.png` | `acprof plot` 在 residual CSV 有有效数据时生成的模型诊断图，包含 OOF 实际值/预测值、相对残差分布及残差随预测延迟和输入尺度的变化。 |
| `debug/idle/<case-id>.jsonl` | 仅 `--idle-debug` 时生成。每行对应一个 workload window 的 idle 诊断记录，包含 GPU NVML idle power trace、`nvidia-smi` GPU/process 快照、CPU idle window 内 RAPL 子窗口功率、host/container CPU delta、top proc CPU delta，以及 after-idle 快照，用于定位 `gpu_idle_power_w` / `cpu_idle_power_w` case 内波动来源。 |
| `plots/<environment_class>/cpu/*.png` | `acprof plot` 生成的该环境 CPU-only 图表；历史身份缺失时归入 `unknown`。 |
| `plots/<environment_class>/gpu/*.png` | `acprof plot` 生成的该环境 GPU-only 图表。 |
| `plots/<environment_class>/gpu+cpu/*.png` | `acprof plot` 生成的同一环境内 GPU/CPU 对比图表。 |

`.acprof/work/cases/` 下本次已完成 case 的中间文件会在 `result_all.csv` 成功 merge、完成状态持久化后清理。
旧布局对应 `result_case_*.csv`、`*.sniff_groups.jsonl`、`lat_case_*.json` 和 `sniff_case_*.pcap`。
若运行被中断，中间文件保留用于恢复。

`raw/requests/*.jsonl`（旧布局为 `*.requests.jsonl`）保留 schema v1、`sniff_group_id`、`input_scale`、`warmup`、`repeat_idx`、
`source=client_http`、`latency_app_s` 数组及请求阶段 `status`。数组下标 `i` 对应请求 ID
`<sniff_group_id>:<i>`，数值单位为秒，保留原始浮点精度；只包含成功返回的请求，失败尝试另记
`failed_request_id` 和 `error`。超时中止时仍写出此前成功的请求，自动预热不进入该文件。
packet merge 在清理 PCAP 前追加同长度的 `latency_packet_s`，未匹配位置为 `null`；
basic 模式不生成该数组。恢复未完成 case 时，旧请求文件随 CSV 一起备份后重采。

序列化和 `fsync` 在全部监测停止后执行，窗口内复用已有 latency 数组；不记录输入或输出 payload。
该设计参考 [MLPerf LoadGen 的延后日志处理](https://github.com/mlcommons/inference/blob/master/loadgen/logging.h)，
沿用本项目窗口结束后的写盘方式，无新增日志线程或依赖。正式分析仍按 CSV 的
`status=ok AND warmup=0` 筛选，以独立窗口为统计单位；请求数组不能替代窗口均值或置信区间。
历史目录缺少请求文件时视为未保存原始样本，不从均值或分位数重建。

### 结果完整性与断点续跑

主实验默认拒绝已有实验产物的目录，避免重写静态元数据或追加上次遗留的测量行。
重新执行原命令并加 `--resume`，或在 TUI 高级参数勾选“恢复未完成实验”，可以恢复由当前
状态协议创建的实验。参数与输出根目录保持一致；`--notify`、`--skip-build` 不影响测量身份。
只有 probe 产物的目录仍可用于首次正式采集。旧实验没有 `run_state.json` 时继续支持读取、
绘图及补采，但不能仅凭残留 CSV 推断恢复状态；新的主实验须使用另一输出目录。

恢复检查主机、Python/依赖、AC-Prof 源码及继承的测量环境参数。已经完成准备的实验直接
使用保存的模型 revision、不可变 image ID、输入计划与 profiler 汇总，不重新查询 Hub、构建
镜像或生成另一组输入。原镜像必须存在。输入计划、静态元数据及 profiler 计划的 hash
不匹配时退出；准备阶段尚未完成、尚无 case 时可重新准备。

源码身份包含执行模块、`extensions/*/manifest.json` 和 `assets/` 输入资源；TUI、绘图及分析展示
模块不参与续跑身份。声明或输入内容变化会拒绝恢复，界面文案变化不会。此源码范围使用新的
摘要域，因此升级前的未完成实验不能跨此变更续跑，应使用新输出目录；原产物仍可读取和分析。

case 只有在抓包回填和原有校验结束，且其测量唯一键完整匹配计划后才记为完成。唯一键为
`CPU × memory × GPU mode × input scale × warmup × repeat_idx`。恢复会复用校验通过的完整
case；中断 case 先备份，再创建新容器，重新执行该 case 的原 warmup/repeat 协议。已记录完整
错误占位的 OOM/timeout case 也属于已完成的尝试，不在恢复时自动改变超时或重测条件。
非整数 input scale 在计划、传参和 CSV 身份中保留可往返的完整数值，例如 `30/224` 不会被
格式化为六位有效数字。唯一键继续精确比较，不通过容差合并相邻尺度；显示用四舍五入不进入恢复身份。
OOM pruning 继续按原有参考 CPU/内存顺序重建证据，复用与推断不会增加正式请求。

合并拒绝缺失/空 case、重复文件、重复测量、截断行和与计划不符的行；保留历史扩展列，
历史缺失的可选指标保持 `nan`。全部校验通过后，在同一目录写临时文件并 flush/fsync，
再用原子替换发布 `result_all.csv`。发布前发生写入错误时保留已有最终文件和 case 产物。
完成状态先持久化，再清理中间文件。`status=complete` 表示计划已执行并完成合并，
`outcome=partial` 表示其中包含错误行，两者不能等同于全部测量成功。

`.acprof/result.lock` 是 v2 采集和补采共享的目录锁；旧布局仍使用 `.acprof-result.lock`。
进程退出自动释放锁，锁文件本身可以保留。
固定位置 `/tmp/acprof-measurement-<uid>.lock` 还会串行化本机同一用户发起的实验，不受
`TMPDIR` 影响。它防止不同输出目录争用固定端口和主机计数器，不协调其它用户或外部负载。
容器名称增加每次启动独有的后缀，case 文件名和测量唯一键保持稳定；清理只接受本次启动
取得的完整 container ID。Docker 创建成功但启动失败时，通过本次 `--cidfile` 回收该容器。
`--resume` 对已完成实验只检查结果结构并报告完成，不重新测量或覆盖后续补采的指标。
所有状态、备份与校验操作均在准备阶段、case 边界或最终发布阶段执行。

monitor 由 `MonitorGroup` 统一持有，按既有顺序启动和停止，随后尝试所有 `close()`。
某个 stop/close 失败仍会清理其余 monitor，并将原始请求错误和清理错误写入请求 JSONL；
清理失败会终止本次 client，避免残余采样器进入下一窗口。取消和请求超时仍保留原退出语义。

### `static_meta.json` 字段

`static_meta.json` 是一个 model/image/run-level JSON object，只保存描述实验对象、环境和最终 profiling 配置/口径的相对稳定信息。补采、重试、回填与备份过程记录放在独立的 `collection_history.json`。数组、布尔值、数字和 `null` 均保留 JSON 原生类型，不再编码成 CSV 字符串。

| 字段 | 含义 |
| --- | --- |
| `schema_version` | `static_meta.json` schema 版本；新增运行环境绑定与验证记录后的当前版本为 `7`。 |
| `model_name` | Hugging Face model ID，例如 `google-bert/bert-base-uncased`。 |
| `model_revision` | 实际解析到的 model revision / commit hash。 |
| `parameter_count` | Hugging Face Hub SafeTensors metadata 的参数总数；Hub 未提供时为 `null`。 |
| `parameter_bytes` | 根据 `parameter_dtype_counts` 的各 dtype 元素数量与字节宽度精确求和得到的逻辑 tensor payload 大小，不含序列化 header；没有 dtype 统计或存在未知 dtype 时为 `null`。 |
| `precision_dtype` | SafeTensors 参数中数量占主导的权重精度，例如 `FP32`、`FP16`、`BF16`、`INT8`；无法确认时为 `null`。 |
| `parameter_dtype_counts` | 按 dtype 统计的参数/张量元素数量，保留混合精度与少量整型 buffer 信息。 |
| `inference_precision_by_device` | 当前 handler 明确请求的 CPU/GPU 推理精度。通常 Transformers NLP/CV/audio 为 CPU FP32、GPU FP16；MOSS adapter 为 CPU FP32、GPU BF16；Encodec/DAC 两者均 FP32。TorchScript 保留导出权重精度、输入 FP32；skops 保留 estimator 内部精度、输入 FP32。Silero 和 skops 仅声明 CPU。 |
| `static_flops` | Torch eager profiler 得到的逻辑 shape FLOPs，按 `input_scale` 保存 `flops_per_request`；未采集成功时为 `null`。 |
| `static_macs` | 静态 MACs。当前不做不可靠的 FLOPs/2 推断，因此未单独采集时为 `null`。 |
| `input_format` | 实际 `/predict` HTTP JSON 输入协议及其 JSON Schema。 |
| `output_format` | 实际 `/predict` HTTP JSON 响应协议及其 JSON Schema。 |
| `quantized` | 是否检测到量化配置、量化 tag 或量化权重 dtype；无法确认时为 `null`。 |
| `quantization_method` | 量化方法，例如 `gptq`、`awq`；不适用或未知时为 `null`。 |
| `quantization_config` | Hub model config 中的完整量化配置；没有时为空 object。 |
| `model_license` | Hugging Face model card 许可证，例如 `apache-2.0`、`mit`；无法确认时为 `null`。 |
| `model_metadata_source` | 参数量、参数 payload、精度、量化和许可证的元数据来源，当前在线 Hub 检测成功时为 `huggingface_hub`。 |
| `model_resolution` | 可选的接口解析 object（内部 schema v1）：任务、backend、library、制品格式、loader、operation、model type、固定 revision、元数据文件和 runtime profile。包含 `candidates/evidence`、`conflicts/missing`、`selection`、`interface_kind`、`pipeline_task`、`code_files/code_revision`、有效 `model_spec`；自动解析追加独立 `contract` provenance 和仅在无缺口时生成的 `generated_spec`。依赖固定、用户审阅、接口检查和真实运行的观察分别记录；`contract.runtime_validation` 从 `not_run` 变为包含 mode、image ID、payload／报告 hash 与设备结果的 object，详见[契约生成](Runtime_Compatibility.md#自动生成模型契约m1m6)。`candidate` 不是执行成功；`ambiguous/needs_configuration` 在镜像准备前拒绝。历史 v7 缺失字段按未知处理，不推算。无数值单位或测量窗口，不增加 CSV 列。 |
| `task_family` | 任务族：`nlp`、`cv`、`audio`、`timeseries`、`diffusion`、`multimodal`、`structured`。 |
| `pipeline_tag` | Hugging Face pipeline tag，例如 `fill-mask`、`image-classification`。 |
| `runtime_backend` | 容器内使用的 runtime backend，例如 `transformers_pipeline`、`chronos`、`diffusers`。 |
| `image_tag` | 本次传给 Docker 的镜像引用；v7 新采集使用不可变 `sha256:` image ID，历史文件可能为可变 tag。 |
| `image_id` | 经 Docker inspect 核验的不可变镜像 ID；补采优先使用该字段。历史文件无法确认时不补造。 |
| `image_name` | 便于查看的构建标签，含模型名和构建请求指纹前缀；执行仍使用 `image_id`。 |
| `runtime_environment` | 镜像内生成的环境清单：profile、adapter、构建指纹、模型及实际 snapshot revision、Python 和已安装包版本、依赖锁与包清单 SHA256、自定义 Python 源码 SHA256，以及 `model_download` 文件清单。新构建追加平台/环境身份、各父镜像 ID、系统锁摘要与实际系统包集合，字段详见下文；历史缺失字段不推算。 |
| `runtime_validation` | 独立容器验证报告。保存实际 image ID、输入尺度／payload SHA256、每个设备的状态、dtype、attention 实现、输出摘要、有效 `model_spec` 和分阶段 `stages`。验证推理接口，不替代 profiler 兼容性检查，也不计入请求或性能测量。失败的完整报告另见 `runtime_validation.json`。 |
| `batch_size` | 本次 profiling 的 batch size。 |
| `input_scale_type` | `result_all.csv/input_scale` 的语义名，例如 `seq_length`。 |
| `workload` | workload 清单的可复现元数据，包括素材 SHA256、来源、变换、推理模式以及模型侧输入约束。 |
| `input_scale_plan_sha256` | 本次实际执行的 `input_scale_plan.json` SHA256。 |
| `run_command` | 启动本次 profiling 的 `acprof run ...` 命令，便于复现实验参数。 |
| `model_download_url` | Hugging Face model page URL。 |
| `gpu` | 存在 GPU case 时为选定物理 GPU 的名称；仅 CPU 实验保留主机设备信息，没有可见 NVIDIA GPU 时为 `unknown`。 |
| `gpu_mem_total_bytes` | 对应上述设备的 total VRAM，单位 bytes；无法读取时为 `null`。 |
| `gpu_device` | schema v7 的新增可选 object：`uuid`、主机 `index`、`pci_bus_id`、`name`、`memory_total_bytes`。GPU case 运行前解析并固定 UUID，Docker、NVML 和独立 profiler 共用；容器内单卡编号为 `0`。CPU 实验为空对象；历史缺字段表示身份未知，重新执行 GPU post-hoc 采集时拒绝猜测设备。 |
| `host_mem_total_bytes` | Linux 测量环境可见 RAM 总量，单位 bytes；Native 沿用主机 RAM，WSL 为 guest 可见内存，不代表 Windows host；无法读取时为 `null`。 |
| `host_swap_total_bytes` | 实验启动时 host 已启用 swap 的总容量，单位 bytes；无法读取时为 `null`，未启用时为 `0`。 |
| `host_swap_used_bytes_at_start` | 静态元数据采集时 host 已使用的 swap 快照，单位 bytes；无法读取时为 `null`。 |
| `host_swap_type` | `/proc/swaps` 中 active swap 的 backing 类型：`none`、`file`、`partition`、`zram`、`mixed` 或无法识别时的 `unknown`。 |
| `host_vm_swappiness` | 实验启动时 `/proc/sys/vm/swappiness` 的整数值；无法读取时为 `null`。 |
| `model_cache_bytes` | mounted 模型为 `model_artifact_bytes` 的兼容别名；历史 baked 模型仍为镜像 `/models/hf` 下按 inode 去重的普通文件逻辑 bytes，不重写历史结果。 |
| `model_storage_mode` | `mounted` 表示主机 Model Store 只读挂载，`baked` 表示历史镜像内权重。历史缺字段时不推断具体缓存/制品大小。 |
| `model_artifact_bytes` | mounted 模型清单中主 snapshot 与离线依赖文件的逻辑 bytes；按清单路径求和，包含配置、tokenizer、代码，不等同于物理共享 blob 占用。历史 baked 结果为未知。 |
| `runtime_image_bytes` | mounted 模式下的 Docker image size，包含 runtime、代码和模型清单，不含挂载权重。历史 baked 镜像无法可靠拆分，保留未知。 |
| `total_deployment_bytes` | mounted 模式为 `model_artifact_bytes + runtime_image_bytes`；baked 模式为完整 `docker_image_bytes`，避免重复加权重。非 registry 压缩流量、非共享层实际占盘量。 |
| `docker_image_bytes` | `docker image inspect <image_tag> --format "{{.Size}}"` 返回的本地 image size，单位 bytes。 |
| `docker_storage_total_bytes` | Docker daemon `DockerRootDir` 所在文件系统的总容量，单位 bytes；无法访问 daemon 路径时为 `null`。 |
| `docker_storage_available_bytes_at_start` | 静态元数据采集时 `DockerRootDir` 所在文件系统对当前用户可用的容量快照，单位 bytes；该值会随磁盘使用变化。 |
| `docker_storage_filesystem` | `DockerRootDir` 所在文件系统类型，例如 `ext4`；无法识别时为 `unknown`。 |
| `docker_storage_device` | 承载 `DockerRootDir` 的 mount source，例如 `/dev/nvme0n1p2`；无法识别时为 `unknown`。 |
| `docker_storage_type` | 根据 `lsblk` transport/rotational 信息得到的 `nvme_ssd`、`ssd`、`hdd` 或内存文件系统 `memory`；证据不足时为 `unknown`。 |
| `environment` | 兼容保留的发行版标签，例如 `ubuntu24.04` 或 `ubuntu24.04+wsl`；不能用于判定采集等级或补全历史身份，环境边界以 `platform` / `comparability_class` 为准。 |
| `cgroup_version` | 本次 preflight 实际检测到的 hierarchy：仅支持 `v2`，其它 hierarchy 在预检退出。 |
| `cgroup_collection_mode` | 当前唯一采集策略为 `strict_v2`。分析正式数据集时应同时要求 `cgroup_version=v2` 和 `cgroup_collection_mode=strict_v2`。 |
| `cpu_power_source` | CPU package 功耗来源。`rapl` 表示使用 Linux RAPL powercap 真实计数器；`unavailable` 表示当前环境没有可用 RAPL。 |
| `rapl_topology` | static schema v7 可选对象。记录全部发现的 powercap 域、父域、真实路径、sysfs aliases、计数器范围、enabled、可读状态、选中来源及 DRAM 覆盖缺口。子域与 MSR/MMIO 重复接口保留元数据，只有明确选中的 package/DRAM 分别计量。历史缺失表示未知。 |
| `vcpu_power_method` | estimated vCPU 功耗计算方法。`rapl_cgroup_cpu_share` 表示对 RAPL package energy 逐采样区间按 container cgroup CPU share 归因；`unavailable` 表示无法估算。 |
| `cpu_governor` | Host CPU frequency governor 汇总值，例如 `performance`、`powersave`、`schedutil`；如果各 CPU policy 不一致，会写成 `mixed:<governor>=<count>,...`；无法读取时为 `unavailable`。 |
| `cpu_boost` | Host CPU boost / turbo 状态。`on` 表示 boost 可用，`off` 表示关闭；无法读取时为 `unavailable`。 |
| `compute_profile_tools` | 本次启用/实际记录的 profiler 列表，例如 `["torch_profiler_eager","ncu"]`。 |
| `torch_profiler_eager_flop_semantics` | Torch eager FLOP 的统计口径说明。 |
| `torch_profiler_eager_attention_implementation` | 独立 Torch probe 强制并验证的 attention 实现，当前为 `eager`。 |
| `torch_profiler_eager_repeat_cpu` | CPU Torch eager probe 的 repeat；未采 CPU profile 时为 `null`。 |
| `torch_profiler_eager_repeat_gpu` | GPU Torch eager probe 的 repeat；未采 GPU profile 时为 `null`。 |
| `ncu_flop_semantics` | NCU GPU 实际执行 FLOP 的计数器、Tensor/Scalar 分类口径说明。 |
| `ncu_repeat` | NCU probe repeat；所有 NCU per-request 指标据此归一化。 |
| `ncu_fma_flop_weight` | NCU FMA 指令的 FLOP 权重，当前为 `2`。 |
| `ncu_metrics` | 本次 NCU 实际请求/解析的 metric 列表。 |
| `torch_version` | Torch probe 使用的 PyTorch 版本。 |
| `transformers_version` | Torch probe 使用的 Transformers 版本。 |
| `ncu_version` | Host NCU 版本；历史回填无法可靠确认时为 `unknown`。 |
| `gpu_compute_capability` | profile 使用 GPU 的 compute capability。 |
| `gpu_sm_count` | profile 使用 GPU 的 SM 数。 |
| `compute_profiles_retained` | raw profiler artifact 是否保留。 |
| `compute_profile_provenance` | profile 来源，例如本次直接采集或历史 `posthoc_backfill`。 |
| `execution_profile_schema_version` | execution profile plan schema 版本。 |
| `execution_profile_tools` | 本次显式启用且适用于所选 GPU modes 的 execution profiler 列表，例如 `["massif","nsys"]`；工具缺失时仍列出，并通过对应 error 字段诊断；默认关闭时为空列表。 |
| `massif_peak_semantics` | Massif peak 的 process-lifetime 口径，明确包含模型加载与预热。 |
| `massif_repeat` | 每个 Massif probe 内的 inference repeat；peak bytes 不按 repeat 归一化。 |
| `massif_version` | 实际执行分析的 container image 中的 Valgrind/Massif 版本；使用共享运行依赖的原模型镜像，未启用或无法确认时为 `unknown`。 |
| `massif_sampling_strategy` | `representative_per_scale` 或 `full_resource_matrix`。 |
| `massif_reference_cpu_cores` / `massif_reference_mem_cap_gb` | 缩减采样实际使用的代表资源；完整矩阵时为 `null`。 |
| `massif_reused_across_resource_cases` | Massif entry 是否从代表资源复用到其他结果行。 |
| `nsys_timeline_semantics` | Nsight Systems timeline 的 NVTX range 与汇总口径。 |
| `nsys_repeat` | 每个 Nsight Systems NVTX range 内的 inference repeat；动态汇总据此归一化到单 request。 |
| `nsys_version` | Host Nsight Systems 版本；未启用或无法确认时为 `unknown`。 |
| `nsys_sampling_strategy` | `representative_per_cpu_scale`、`representative_per_scale` 或 `full_resource_matrix`。 |
| `nsys_reference_cpu_cores` / `nsys_reference_mem_cap_gb` | Nsys 缩减采样使用的代表资源；per-CPU 策略的代表 CPU 为 `null`。 |
| `nsys_reused_across_resource_cases` | Nsys entry 是否从采样资源复用到其他结果行。 |
| `execution_profiles_retained` | raw Massif / Nsight Systems artifacts 是否保留。 |
| `execution_profile_provenance` | execution profile 的来源；默认关闭时为 `disabled`。 |

`static_meta.json` 只接受 schema v7；旧版本或缺失版本直接报错。读取端最多接收 4 MiB 的
UTF-8 JSON object，并拒绝 `NaN`、`Infinity` 及溢出为无穷大的数字。可选读取只在文件不存在时
返回空对象；文件存在但超限、损坏或类型错误时仍严格报错且不改写原件。当前完整 cache artifacts
大小字段为 `model_cache_bytes`，不读取旧 `model_weight_bytes`。采集/修复记录独立写入
`collection_history.json`，不从旧静态元数据补造 swap、cgroup 或运行环境信息。

`runtime_environment` 继续使用 schema v1。新服务镜像构建时增加以下身份字段，保持原有包版本、
模型快照及 `build_fingerprint` 的语义；`static_meta.json` 仍为 v7，CSV 没有新增列。

| 字段 | 含义 |
| --- | --- |
| `platform_id` | 声明的平台键：`cpu`、`cu124` 或 `cu128`。 |
| `platform_definition_id` | 规范化 Python 基础镜像、架构、系统锁和 Torch 闭包的内容摘要。 |
| `environment_id` | 平台声明与完整 Python 依赖制品的内容摘要，独立于 profile、adapter、模型和代码。 |
| `platform_image_id` / `environment_image_id` / `model_image_id` | 本次实际继承的平台、环境与模型快照镜像的不可变 Docker ID。 |
| `platform_build_fingerprint` / `environment_build_fingerprint` | 包含安装配方的构建身份；环境构建还绑定实际平台 image ID。 |
| `request_fingerprint` | 服务构建声明的查找键；原有 `build_fingerprint` 另绑定不可变模型父镜像 ID，继续表示实际构建身份。 |
| `model_spec` | 本地 `--model-spec` 优先于仓库 `acprof_model.json` 的有效内容，无声明时为空 object。规范化内容参与服务构建指纹，加载器消费构建期固化的相同内容；不改变依赖环境或模型文件层身份。本地声明文件的绝对路径和原始字节 SHA256 另参与 run 恢复身份。 |
| `python_base_image` / `architecture` | 固定 OCI digest 的 Python 基础镜像及目标 `linux/amd64`。 |
| `system_lock_sha256` | 规范化系统锁的 SHA256，覆盖 snapshot 来源、索引摘要、系统包集合和 `.deb` 制品。 |
| `system_packages` | 构建时实测的完整已安装系统包集合，键为 `包名:架构`，值为包含 epoch 的版本。 |

新缓存必须同时核对标签、上述身份字段与内部完整包清单。历史 v7 结果仍按记录的原始 image ID
和原构建指纹补采；缺失这些新增字段不会触发推算、回填、升级依赖或重建替代镜像。
`dependency_lock_sha256` 仍是镜像内完整 Python 锁文件字节的摘要；可复用环境中的注释排版
不参与 `environment_id`，因此构建服务时读取实际父环境清单中的该值。

`runtime_environment.model_download` 是可选的独立 schema v1 清单，历史结果可缺失。`requested_policy` 保存 `auto/full`，`effective_policy` 保存实际 `selected/full`，`reason` 说明筛选或回退原因；`weights` 记录组件、格式、variant 和索引／分片文件。`files` 保存路径、实际逻辑大小和准备阶段计算的 SHA256，另保留 Hub 提供的 Git blob／LFS 标识；`excluded_files` 是未下载文件的远端元数据。`verification=sha256` 表示准备阶段已完成完整性检查，`plan_sha256` 校验规范化 JSON（不含自身字段）。`selected_bytes` 按清单路径求和，不对相同内容的多个路径去重，mounted 模式下其含依赖总量成为 `model_artifact_bytes`（及兼容别名 `model_cache_bytes`）；不等同于镜像大小或释放的磁盘空间。新增清单不改变 CSV 字段和历史指标定义。`runtime_environment.model_store` 记录 entry ID、计划 SHA256 和逻辑制品大小；主机结果额外记录 `host_path`，仅用于挂载定位，不参与便携镜像内容身份。

新构建在该清单的 `endpoint` 字符串中记录成功使用的 Hugging Face Hub 基地址，依赖模型在
`dependencies[].download.endpoint` 分别记录。该字段来自执行 metadata/snapshot 请求的地址，
参与 `plan_sha256`，无单位，不属于测量窗口；它不表示重定向后的 CDN/Xet URL，
也不能追溯本地 cache 最初从哪里取得文件。历史清单缺失时视为 unknown，不默认补成官方或镜像。
主地址和显式备用列表参与模型层、服务层指纹；下载失败不发布已验证清单或模型镜像。

`runtime_validation.json` 使用独立 schema v1：`devices.off/on` 分别保存 CPU／GPU 的 `ok`、`error`、`inconclusive` 或明确 cgroup OOM 的 `resource_limit`；总状态为 `ok`、`error`、`inconclusive` 或 `resource_limited`。每个模式只执行一次最小计划输入，资源上限为本次配置的最大 CPU／内存。load、preprocess、predict、postprocess、validate_output 五阶段都须成功；错误、验证预算耗尽和资源限制均阻止正式矩阵，CPU + GPU 必须两者通过。stdout/stderr 保存在 `runtime_validation_off/on.log`，超时也清理验证容器。它们不是 warmup、测量行或 profiler 结果。验证前已有的结果不因此变为本次成功结果。

每个设备的可选 `stages` 依次记录 `execution/load/preprocess/predict/completion/postprocess/validate_output/metadata`，
每项包含阶段名与 `verified/error`；错误保存原异常类型和消息。失败报告的 `failed_stage` 指向
失败步骤，读取输入或进入／退出推理上下文的异常为 `input_or_execution_context`。被终止或超时的
进程可能没有阶段回报，以外层状态和日志为准；未执行阶段和历史缺失字段不补为成功。
这些诊断没有时间单位，不用于比较阶段耗时，也不增加正式测量请求。

这些字段在 profiling 后原子补写，原始 `run_command` 保持不变。`static_flops` 只保存不依赖硬件计数器的 Torch 逻辑 shape FLOPs，并按 input scale 展开；NCU 实际执行 FLOPs、吞吐率以及 execution 数值仍保存在 `result_all.csv`，execution 字段是否来自代表资源由上述 sampling metadata 和 plan entry provenance 说明。

### 质量与失败产物

`quality_checks.json` 为 `{"schema_version": 1, "checks": [...]}`。每个 check 包含
`code`、`severity`、`observed`、`threshold`、`detail`、`evidence`，与 Capability 分开：

| code | 观察值与边界 |
| --- | --- |
| `cpu_idle_baseline_unstable` / `gpu_idle_baseline_unstable` | 该 case 的空闲功率 `(max-min)/mean`，无量纲；保存原始 W 样本与现有阈值，不改变能耗公式或采样窗口 |
| `weights_reinitialized` | Transformers loading info 的 missing/mismatched keys；threshold 为 0 个 key |
| `unused_checkpoint_weights` | loading info 的 unexpected keys；保留 loader 来源，不根据告警文本归类 |
| `missing_cli_exit_code` | process return code 为 null；threshold 表示需要进程退出证据 |

加载质量在独立 probe 的 `devices.<device>.quality_checks` 中保存，case 能耗质量在窗口结束后
写入 case sidecar。资源矩阵退出时（包括中途失败或取消）保存根目录 `quality_checks.json`；失败的 probe 仍可由 audit 直接读取。
quality 数据不进入 capability evidence；普通 warning 不改变 `full_profile_complete=true`。
兼容性报告据此区分 `full_success` 与 `full_success_with_warnings`。
审计和质量消费者只读取不超过 4 MiB 的 UTF-8 JSON 对象，并拒绝 NaN、Infinity 和溢出为无穷值的数字。
损坏的运行元数据记为 `invalid_metadata` error；损坏的 canonical quality evidence 记为
`quality_evidence_invalid` warning，同时保持 `quality_status=unknown` 且禁止自动选择。只读检查不重写原始证据。

typed `failure` 包含 `stage`、`reason_code`、`detail`、`device`、`runtime_profile`、
`retryability`、`evidence`、`exception_type`。`reason_code` 包括 task、dependency、precision、
contract、artifact、processor、access、initialization、inference 与 timeout/resource 原因；
稳定代码定义在 [`failures.py`](../acprof/failures.py)。原始异常链和日志用于诊断，不作为报告分类输入。
CLI 使用 `ACPROF_FAILURE=` JSON 记录，TUI 按该结构展示；HTTP 错误响应携带同一 failure，
client 在请求及 monitor 结束后写入 case sidecar，资源矩阵退出时汇总为根目录 `runtime_failures.json`。
非零 client 退出优先保留结构化原因；恢复运行时，旧 case 失败随该次尝试归档，根侧车只反映当前 case 证据。
两个新增根侧车不更改既有 Artifact Layout v2 manifest 身份或历史 CSV 的列和数值。

timeout evidence 包含 `timeout_seconds`（秒）、`request_phase`、`request_id`、`input_scale`、
`model_loaded`、`service_alive` 与 `timeout_scope`。正式 HTTP 请求沿用 connect/read inactivity
超时语义；兼容性验证沿用整个子进程预算。未观察的布尔字段为 null，不能补成 true 或 false。
`request_timeout`、`compatibility_budget_exhausted` 和只表示已有报告产物损坏的
`recorded_evidence_invalid` 在兼容性报告中为 `inconclusive`；后者不覆盖同目录已有的有效 typed failure。
预算筛查或实测资源限制为 `unverified`，用 `evidence.measured_oom` 区分是否有真实 OOM 证据。
提高预算需显式重试并使用新目录，不覆盖原始预算与失败证据。

历史结果缺少这些字段时保持 unknown；不能从 Capability 推导质量良好，也不回读自然语言日志
伪造结构化告警。`coverage report` 对已有 full 结果但缺少质量记录的条目展示
`full_success_quality_unknown`。2026-09-27 旧审计的 140 个 warning 项没有在本次改动中被重写或重新实测；
新运行会直接产生可消费的结构化原因。

### `collection_history.json` 字段

| 字段 | 含义 |
| --- | --- |
| `schema_version` | `collection_history.json` schema 版本，当前为 `1`。 |
| `posthoc_profile_history` | `acprof profile` 事后补采记录，包括工具、采样策略、完成时间与备份位置。 |
| `timeout_retry_history` | 请求超时后的重采/合并记录。当前仓库没有自动生成该记录的入口，保留独立历史文件中的已有记录。 |
| `quality_retry_history` | 质量检查后的定向重采/合并记录。当前仓库没有自动生成该记录的入口，保留独立历史文件中的已有记录。 |
| `static_meta_backfill_history` | 对历史结果补充静态元数据时的来源、字段、备份位置及无法回溯的字段。 |

不再重复保存 `posthoc_profile_last_run`、`timeout_retry_last_run` 或 `quality_retry_last_run`；需要最新记录时读取对应 `*_history[-1]`。`static_meta.json` 中的 history/last-run 字段会被拒绝，不再执行迁移。

### 最大输入探测结果

`acprof probe` 每次生成独立的 `input_scale_plan.json` 和 `largest_scale_probe.json`。
后者当前为 schema v3，主要字段如下：

| 字段 | 含义 |
| --- | --- |
| `memory_probe.candidate_order_gb` | 去重、升序排列的候选内存上限。 |
| `memory_probe.attempts` | 每档的 `startup_oom`、`runtime_oom`、`cuda_oom`、`timeout`、`error` 或 `ok` 结果，以及错误与分段计时。 |
| `memory_probe.minimum_viable_mem_gb` | 完整返回且有效尺度与计划一致的第一个候选；没有成功档时为空。 |
| `timing.request_s` | 成功档 `/predict` 的 host 端到端耗时。 |
| `cold_start.total_s` | 成功档全新容器启动到 ready 的耗时。 |
| `timing.ready_plus_request_s` | 上述冷启动与请求耗时之和。 |
| `timing.command_s` | 包含预检、检测、可选构建、输入规划、失败候选和清理的整条命令耗时。 |
| `timing.request_timeout_s` | 默认无限等待时为 `null`；显式设置 `--timeout-seconds` 时为对应秒数。旧 schema v2 始终记录有限值。 |

探测不会写入或修改 `result_case_*.csv`、`result_all.csv`、`static_meta.json` 或
`collection_history.json`，结果不包含 idle、能耗或网络测量。用法见[最大输入探测](Getting_Started.md#先探测最大输入)。

## 冷启动

| 字段 | 含义 |
| --- | --- |
| `cold_start_started_at` / `cold_start_ready_at` | host 侧开始执行 `docker run` 与首次成功收到 `/ready` 的本地 ISO-8601 时间戳。 |
| `cold_start_container_launch_s` | host 开始 `docker run` 到 container 内 Python server process 开始执行的时间；使用共享系统 wall clock 对齐。旧镜像未返回启动时间戳时为 `nan`。 |
| `cold_start_server_setup_s` | server process 开始到 model load 开始之间的 setup 时间，扣除单独列出的显式 CUDA 初始化，包括 Python/framework/handler import 与配置。 |
| `cold_start_cuda_init_s` | 现有启动路径中 `torch.cuda.is_available()` 与 `torch.cuda.init()` 的显式计时；CPU case 为 `0`，不发送 probe inference。 |
| `cold_start_model_load_s` | container handler `load()` 的持续时间，使用 `time.perf_counter()`。 |
| `cold_start_ready_wait_s` | model load 完成到 host 首次收到成功 `/ready` 的时间，包括 Flask 开始监听、poll interval 和本地 HTTP 往返。 |
| `cold_start_first_predict_app_s` | workload client 发出的第一个成功 `/predict` 的 application latency；默认 auto-window 模式下通常是已有 auto-warmup 请求，固定窗口模式下是已有首个 warmup/measurement 请求，不会为此新增请求。 |
| `cold_start_s` | 当前 container 从 `docker run` 到 `/ready` 成功的时间，单位秒。 |

## 结果行数和时间成本估算

### CSV 行数与请求数

完整计划的行数为：

```text
资源 case 数 = len(cpus) × len(mems) × len(gpus)
总行数 = 资源 case 数 × 实际 input scale 数 × (warmup + repeat)
正式测量行数 = 资源 case 数 × 实际 input scale 数 × repeat
```

这里使用 `input_scale_plan.json` 中实际确定的档数；自定义音频清单可能不是 6 档。
默认 6 档时，计划行数为 `4 × 4 × 2 × 6 × (2 + 5) = 1,344`，其中 warmup 384 行、
正式测量 960 行。OOM、超时或剪枝可能生成错误占位，异常中断也可能留下部分结果，
因此计划行数不等于成功实测行数。

一行 CSV 是一个请求窗口，不等于一个 batch 或一次 `/predict`：

- 固定 `--repeat-in-window N`：每个成功窗口发送 `N` 次请求，完整成功矩阵的窗口内请求数为 `总行数 × N`。
- 默认 `--repeat-in-window 0`：每行至少发送一次请求，累计请求耗时达到 `--repeat-window-seconds` 后停止；CSV 的 `repeat_in_window` 记录该行实际请求数。
- `--batch-size` 决定一次请求的样本数，不乘入 CSV 行数。样本总数还需按任务支持情况乘以 batch size。

auto 模式在每个资源 case / input scale 开始前默认额外发送 5 次 auto warmup 请求；
它们不写成 CSV 行，也不同于 `--warmup 2` 的两个完整测量窗口。默认完整矩阵另有
`32 × 6 × 5 = 960` 次 auto warmup 请求。固定 `--repeat-in-window N` 时不执行这段预热。

### 每行测量窗口

当前 client 按“冷却 → 无请求对照 → workload”执行；CPU、GPU 与资源监控共享对照窗口：

```text
row_window_s ≈ idle_cooldown_s + matched_control_s + active_workload_s
               + monitor_start_stop_and_csv_overhead_s

默认快速请求场景：5 + 20 + 10 = 35 秒/行
默认完整矩阵：1,344 × 35 / 3,600 ≈ 13.07 小时
```

GPU 开启和关闭都使用同一组 `5 + 20` 秒默认基线开销，不会再为 GPU 单独追加一个对照窗口。
`10` 秒是 auto workload 的目标，结束条件在完整请求返回后判断：若单次请求需要 120 秒，
该行至少约 `5 + 20 + 120 = 145` 秒。它不会在第 10 秒截断请求。
固定请求数时，active workload 约为 `N × 平均请求延迟`；已有成功 CSV 可用
`latency_app_s × repeat_in_window` 估算该行累计请求耗时。

单请求超时默认是 `--request-timeout-seconds 300`；超时会形成错误行，不能当作一次
成功的 300 秒测量。上述 13.07 小时仅适用于完整成功矩阵、请求足够快且默认窗口接近目标的情况，
还未包含 auto warmup、启动、抓包解析、构建、分析器和清理等开销。

### 整条命令的耗时

```text
main_collection_s ≈ sum_over_cases(
  cold_start_s
  + sum_over_scales(auto_warmup_s)
  + sum_over_scales((warmup + repeat) × row_window_s)
  + sniff_parse_and_container_cleanup_s
) + final_merge_s

total_wall_s ≈ preflight_s + model_detection_s + docker_build_and_download_s
               + input_scale_planning_s + runtime_validation_s + compute_profile_s
               + execution_profile_s + main_collection_s
```

auto warmup 约为每个 case/scale 的 `5 × 平均请求延迟`，长耗时模型应单独计入。
镜像复用、失败 case、启动 OOM 剪枝和续跑会改变实际工作量；启用通知和日志收尾也有额外耗时。
内存/PID 峰值、cgroup 增量、冷启动分解和派生能效使用已有采样路径，不增加 `/predict` 数量；
网络字节复用已有 PCAP，但仍需离线解析时间。
