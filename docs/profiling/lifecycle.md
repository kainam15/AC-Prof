# 采集生命周期与冷启动

[← 返回专题目录](protocol.md)

## 采集生命周期

根 CLI 顺序为 `resolve → interface validation → prepare runtime → runtime validation → matrix measurement`。
任务解析和环境预检通过后，先检查无权重的源码接口，再准备正式镜像和输入计划，完成每个选中设备
的最小真实请求，最后运行所选 profiler 与正式矩阵。validation 不属于 measurement。
启用启动剪枝时，先执行独立 startup-OOM probe，再冻结正式矩阵计划并开始采集。
每个正式 case 创建新容器；client 控制已有预热、冷却、无请求对照与 workload 窗口，
监控停止后才计算派生值和写行。case 产物经抓包解析与校验后合并；图表及通知属于测量之外的操作。
模块顺序见[主机编排](../development/orchestration.md#主机编排与测量)，镜像与独立验证见[运行兼容](../models/images.md#构建复用和验证)。

独立接口验证与 startup probe 可能预热宿主机文件缓存。冷启动描述全新容器的进程和模型初始化，
不承诺磁盘冷缓存；`cold_start_first_predict_app_s` 不计入 `/ready` 前的分段和，也不新增推理请求。

请求与采集器边界可通过开销诊断入口的 `--window-boundaries` 单独观察，字段与限制见
[窗口置信区间与开销对照](../results/comparisons.md#窗口置信区间与开销对照)。该诊断记录已有逻辑时间戳，
在采集器全部收尾后保存独立文件；正式窗口默认不启用，不改写既有 CSV、能耗公式或历史产物。

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
[负载报告](../results/comparisons.md#非流式负载报告)。流式 token 指标需另行实现真实服务端事件。

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
