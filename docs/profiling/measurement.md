# 测量协议与输入契约

[← 返回专题目录](protocol.md)

## 协议不变量

- CPU package、估算 vCPU、GPU device 与 container-attributed 数据分别命名和解释，整机测量不能静默替换容器归因。
- 单请求、workload window、进程生命周期、独立 profiler 使用不同窗口与分母；比较前核对请求数、尺度单位及来源资源。
- 数值 `0`、CSV `nan`、JSON `null` 和字段缺失含义不同；未知历史数据不补成零，推导或代表资源复用不冒充独立实测。
- JSON 保留原生类型；schema 版本由协议与兼容策略决定。只接受当前 schema；可选指标不可用时保持 `nan`，不转换旧字段或推测来源。
- 采集器返回的真实运行时错误必须保留于结果行诊断，正式请求成功但可选 monitor 出错时行标记 `warn`，不得标为无诊断的 `ok`；已知不支持的可选 RAPL 能耗能力继续为 `nan`，不冒充采集故障。
- 容器 I/O 计数器下降视为重置/不可信，差值写 `nan` 并记录诊断；有效的零差值才写 `0`。在 `/ready` 失败后，host 按冻结输入规模、warmup/repeat 计划生成完整错误行，避免将未开始的请求误诊为抓包失败。
- 输入计划、`input_scale_plan_sha256`、workload 素材与模型 revision 必须一致；派生字段复用已有采样，不额外发起请求。
- 静态对象描述与采集过程来源分开保存；补采和修复记录进入 `collection_history.json`，保留原始备份和可回溯信息。

## 确定性数值回归基线

`tests/measurement/windows/test_measurement_semantics_baseline.py` 用无硬件合成样本冻结数值语义：CPU 占用率按容器 CPU 秒数／窗口秒数／配额核心数计算，内存用字节均值和峰值；RAPL 微焦耳计数器跨范围回绕后换算为焦耳，matched-control 平均功率只从 effective 能量中扣除；计数器有效零增量保留 `0`，回退、负值或非有限数值归为 `nan`，不能充当真实观测。

独立期望常量来自上述单位与公式，不从被测实现反推；这些仅是**数值计算的确定性回归证据**，不能证明传感器校准、采样对工作负载的扰动、Native Linux / GPU 硬件精度或跨运行方差。后者仍需真实硬件重复实验和 provenance 核对。设计上参考 [MLPerf LoadGen 对样本性能结果与场景上下文的分离](https://github.com/mlcommons/inference/blob/master/loadgen/results.h)，不引入其代码或依赖。

## 周期采样调度

CPU 能耗和资源监控器共用 `monitors.common.sample_periodically`。首个周期采样
安排在 `started_at + interval`，后续 deadline 以原始相位递增；回调收到的是实际
开始采样时的单调时钟时间，不是已经错过的计划时间。

`--sampling-policy burst|skip` 显式选择超时策略，默认仍为 Burst；策略及
`periodic-v1` 协议版本进入窗口原始证据。Burst 的语义是：回调或线程调度延迟后，已过期的 deadline 会连续执行，
直到追上原计划；若每次回调都比周期长，采样线程可能持续执行而无法追上。连续读取
不会恢复过去时刻的瞬时状态，也不代表实际采样间隔仍然均匀。停止信号会终止后续
采样，包括超时回调之后的追赶；回调异常向调用方传播，不自动重试。

`tests/measurement/windows/test_monitor_common.py` 使用随等待和回调耗时前进的假时钟，固定首周期、
绝对相位、单次和持续超时、调度迟到、停止及异常行为。这些测试是调度协议的
回归基线，不能证明采样器的实际开销、传感器精度或对工作负载的扰动。

Skip 保留原计划相位，在回调完成后跳过严格早于完成时刻的计划时点，下一次采样
等待未过期的 deadline；它不会补造丢失的瞬时状态。GPU 保留原先线程启动后立即
首采的协议，CPU/resource 保留等待首个周期的协议；所有采集器的 start/stop 次序保持不变。

每个采集器记录周期回调数量、迟到次数/总量/最大值、回调耗时总量/最大值、超周期次数、
跳过的计划时点数量，以及首尾计划时间/实际时间。统计使用常量空间，不保存另一份逐 tick
数组；边界首尾样本不进入周期回调计数。回调耗时包括既有传感器读取，不包括诊断累计本身，
也不宣称是硬件内部采样时刻。诊断的额外时钟调用和内存操作仍须纳入真实开销对照。
热路径没有新增文件输出或日志，证据序列化和落盘在所有采集器停止后执行。

保留相位的 Skip 不保证任意相邻实际采样都间隔一个完整周期；Burst 与 Skip 的结果
必须保留各自协议身份，不能在缺少对照证据时直接合并或切换默认策略。
术语参考 [Tokio MissedTickBehavior](https://docs.rs/tokio/latest/tokio/time/enum.MissedTickBehavior.html)，
只借鉴策略分类；切换策略仍需独立的 Native Linux 对照证据，不照搬其定时容差。

采样开销对照沿用 `scripts/measure_overhead.py`，从已完成实验复用冻结镜像、输入和资源条件：

```bash
uv run python scripts/measure_overhead.py results/example-run \
  --gpu on --rounds 5 --requests 100 --sample-hz 20,100 \
  --sampling-policies burst,skip --output-dir results/overhead-policy-comparison
```

`--sampling-policies` 默认 `burst`。每轮随机打乱一次 `none` 基线及各 policy × rate 场景，
仍按相同 round 配对延迟差值和置信区间；CPU/GPU 模式分别运行，不跨设备拼接配对。
`overhead.json` schema v2 同时记录每个采集器的周期诊断、策略/频率和主机进程 CPU 秒数。
`host_process_cpu_s` 仅包含请求循环内 Python 进程所有线程的 CPU 时间，排除容器 CPU、
collector start/stop、文件落盘；`host_process_cpu_pct` 以一个 CPU 核心为 100%，不是整机利用率。
`--modes none,basic,full` 的内部采集对照仍要求单频率、单策略。该诊断不产生正式画像，
也不包含原样本序列化成本；窗口后持久化成本需结合整个 case 生命周期另行评估。

## 窗口原始证据与提交记录

新采集的每个已尝试窗口具有 `run_id / case_id / case_attempt_id / window_id` 四层身份。
`case_id` 是资源配置键，`case_name` 继续用于进程、日志和请求 group 名；重跑同一 case 使用
新 attempt，不能覆盖旧窗口证据。控制窗口与正式/预热窗口分别记录，不把控制窗口计入请求总体。

`raw/telemetry/<case_id>/<case_attempt_id>/` 保存独立 JSON schema v1，包含 CPU RAPL 原始微焦耳
计数器与回绕范围、host/container CPU 时间、DRAM domain/拓扑、resource 样本、窗口首尾 cgroup
计数器、NVML 功率样本及累积能耗计数器、单位、逻辑时钟边界、控制基线和实际选用的能耗来源。
GPU 从累积计数器回退到功率积分的原因以及控制基线重选也保留；不重新读取硬件推断历史来源。
GPU 的 `capabilities` 分别记录能量总值、功率样本和功率峰值能力。累计计数器有效但功率
采样不可用时，保留真实的空 samples 并允许重算总能量和平均功率；峰值保持 unknown，
不从平均值补造样本。没有可用样本或计数器证据、或清理失败的 collector 明确标记 unavailable。

`--raw-sample-limit` 默认 100000，限制每个 collector、每个 workload/control 窗口**保存的样本数**。
超过上限时保留首尾和按索引均匀选取的子集，记录 observed/retained/omitted 数量及 `complete=false`。
这限制的是持久化证据和窗口后序列化规模；现有采集器仍保留完整内存样本用于既有聚合公式，
**不是整个采集进程的内存上限**。截断不会改变当次 CSV 聚合值，但不再具备从保存子集精确重算
全部原始指标的能力。将采集 buffer 改为流式聚合，需要按 CPU、GPU、resource 各自算法另做数值与开销验证。

所有 collector 停止后，client 持久化 request journal、telemetry 和 schema v2 receipt，
随后更新 case 工作投影并同步追加根目录模块 CSV，最后发布统一结果 commit。
receipt 保存原始 client 行、请求记录、各自内容 hash、telemetry SHA256 及窗口顺序；
结果 CSV 丢失可从它恢复。无 receipt 的 telemetry 与请求尾部仍不是已完成窗口。
超时或取消后的可用样本保留为未提交诊断，不伪造成成功测量。
历史 schema v1 receipt 保留原先最后发布的语义。PCAP/补采保存显式派生字段，原始观测不改写。

client 在同步持久化完成后才进入下一窗口，没有后台刷盘线程。新增目录的父目录条目也要同步，
避免仅同步文件而在崩溃后丢失整层路径。首次恢复可校验、重放旧记录，正常追加不读取或重算旧窗口。
恢复的新运行使用独立 attempt，必要的预热单独记录在 `raw/recovery_warmup/`，不追加测量行；
它不代表跨重启连续测量。详见[实时提交与后处理](artifacts.md#实时提交与后处理)。

`acprof.window_records.read_window_record` 校验 schema、文件名与窗口身份、记录内容 hash、
请求/行 hash，以及关联 telemetry 的 hash 和身份；整个结果根移动后相对引用仍然有效。
`verify_case_window_records` 可进一步限定 run/case/attempt 身份。历史数据缺 receipt 或 raw 时
保留旧聚合读取能力，并明确 `legacy_or_missing_raw_evidence`，不为旧结果补造样本。

纯分析入口 `acprof.telemetry.recompute_window` 使用 `window-reduction-v1` 重放已有 CPU/DRAM、
GPU 与 resource 数值公式，不初始化 monitor、不查询当前环境、不做文件 I/O。它使用记录的控制
基线和 GPU 来源选择；缺失 collector 保持 unavailable，截断序列明确拒绝精确重算。
`raw_capability` 分别说明 workload 重算能力与 control 样本完整性。CSV 每请求字段仍需按记录的
completed request count 归一化，不能把窗口总能耗直接解释为 J/request。

设计参考 [MLflow 的 run 元数据与 artifact 分离](https://mlflow.org/docs/latest/ml/tracking/)、
[pyperf 的校准与正式样本阶段](https://github.com/psf/pyperf/blob/main/pyperf/_manager.py)，
只借鉴边界设计；未引入外部运行平台或复制上游实现。

## Profiling mode 与能力证据

`profiling_mode=full` 为默认，保留原有 native Linux、本机 Docker、cgroup v2、RAPL、硬件 instructions
和抓包要求。`basic` 允许 Native Linux、WSL2 及受限 Linux VM/容器。
Docker 路径仍需本机 daemon 与 cgroup v2，要求 application latency、throughput、容器 CPU/memory；
Process 路径要求 application latency、throughput，并按权限采集进程树 CPU/RSS 与设备 GPU 遥测。
能耗、PMU、packet latency 在 basic 中不请求、不启动、数值保持 `nan`。
两种模式不能混作同一画像；Runtime、资源控制和指标归属也必须匹配，见 [Linux Runtime](../platforms/linux-runtimes.md)。
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
FULL/PARTIAL 是平台上限，不等于采集成功或 full/basic 模式。完整边界见 [WSL2](../platforms/wsl2.md)。

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
分布不同与窗口请求总数不同分别记录，详见[跨独立实验比较](../results/comparisons.md#跨独立实验比较)。
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
