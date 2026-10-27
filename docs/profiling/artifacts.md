# 实验产物结构与恢复

[← 返回专题目录](protocol.md)

## 输出文件

模型根目录为 `<output-dir>/<model-dir>/`；模型 ID 中的 `/` 替换为 `--`。
新采集按**推理执行配置**和 collection ID 隔离，不能按翻译语种、摘要等语义用途反复执行
同一完整矩阵。下表列出可能生成的文件；probe、补采、调试与绘图产物仅在执行对应操作时出现。

### 按任务保存与恢复

```text
<output-dir>/google-t5--t5-small/
├── metadata/
│   ├── task_runs/<collection-id>.json
│   └── task_workloads/<sha256>.json
└── tasks/
    └── text2text-generation-<profile-hash>/<collection-id>/
```

`google-t5/t5-small` 的三种翻译方向与摘要在相同 Beam ×4 计算配置下自动归为一个
Seq2Seq 推理组，仅生成一套测量结果。若模型还有 Greedy/采样或其他独立推理接口，
才会另建目录。目录仍使用 Artifact Layout v2，包含独立分层 CSV、`run_state.json`、
输入/矩阵计划与原始测量窗口。

集合清单 schema v1 保存固定模型 revision、执行配置签名、所有已声明用途的证据/配置与
指纹、代表输入及各推理组状态，不生成跨组平均延迟。对应 workload、输入计划及
`model_resolution.selected_task` 保留完整 provenance。参数、batch、实际输入/输出 Token、
模板版本/哈希与 seed 仍在输入计划及请求记录中保存；未知观测保持 unavailable。

恢复集合使用原参数与 `--resume --task-run-id <collection-id>`；多个集合时必须指定 ID。
恢复读取冻结组选择、所有用途及 workload，拒绝配置、revision、模板或组成员变化；
已完成组不重测，未开始组按原方案执行 Smoke。历史按语义任务独立保存的集合仍由其
冻结身份恢复，不会自动合并、改写或覆盖既有结果。

发现、选择、下载确认和集合清单写入均在正式窗口之外。独立推理组具有独立冷启动和监控
生命周期；仅共享固定模型文件缓存，不共享运行中的模型实例。

### Artifact Layout v2

新主实验由 `ArtifactLayout` 集中管理路径。准备阶段就创建五个基础 CSV、四个 profiler CSV
的完整表头，以及 `profiling/metadata/raw/plots/logs/debug` 目录；没有测量时只有表头，
不写示例行、零值或占位 `nan` 行。其他原始产物按实际采集创建。

`result_manifest.json` 与其他受限 JSON 产物共用读取器：只接受不超过 4 MiB 的 UTF-8
JSON object，拒绝非有限数值和损坏内容。未知或不一致的 manifest 仍明确报错，不退回 flat layout。
读取器按 64 KiB 分块累积实际内容，避免为小文件预先分配整个上限；最多读取上限加一个溢出
探测字节，不依赖预先查询的文件大小。SHA256 校验与解析使用同一次打开取得的原始字节，
已有 UTF-8 BOM 计入大小和哈希；显式允许增长的结果记录仍保留其原有大小策略。

```text
<model-dir>/
├── result_layers.json         # schema v3：统一提交、状态、偏移、字段、行数与 SHA256
├── summary.csv               # 行身份、输入与状态
├── performance.csv           # 延迟、吞吐、启动
├── resources.csv             # CPU、GPU、内存与 PMU
├── energy.csv                # 功率、能耗与来源
├── network.csv               # 抓包网络统计及 packet_status
├── profiling/                # 未启用的 profiler 仅有表头
│   ├── torch_profiler.csv
│   ├── ncu.csv
│   ├── nsys.csv
│   └── massif.csv
├── result_generations/<generation>/  # 完成/冻结时生成不可变快照；旧代不自动删除
├── static_meta.json
├── capability_report.json
├── result_manifest.json
├── metadata/                # 解析、输入/矩阵/profiler 计划和 collection_history
├── raw/
│   ├── requests/<case-id>.jsonl
│   ├── windows/<case-id>/<attempt-id>/*.json  # CSV 追加前持久化的不可变 receipt
│   ├── telemetry/<case-id>/<attempt-id>/*.json # 原始采样、计数器及调度诊断
│   ├── result_updates/*.json  # 后处理变更字段及其来源
│   ├── case_results/<case-id>/*.csv # 完成 case 的可恢复投影来源
│   ├── recovery_warmup/<case-id>/<attempt-id>/*.json # 未计入测量的恢复预热
│   ├── measurements/<case-id>/<attempt-id>/  # 完整测量的后处理输入快照
│   │   ├── checkpoint.json
│   │   ├── measurement.csv
│   │   ├── sniff.pcap
│   │   ├── sniff_groups.jsonl
│   │   └── requests.jsonl
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
    ├── result_store/         # writer/publication 锁、不可变 commits、pending 与 snapshots
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

新主实验使用带 `result_manifest.json` 与 `result_layers.json` 的 v2 布局；历史 CSV 与分层 schema v1/v2
保留读取路径，不自动迁移。已有正式产物的目录不能直接初始化为新实验。
目录布局兼容不放宽现有产物 schema、源码身份或恢复校验；升级前中断的实验仍受源码指纹约束。
运行期间不要移动文件或删除清单。以下路径表使用 v2；旧目录沿用原文件名和位置。

| 文件 | 说明 |
| --- | --- |
| `result_manifest.json` | Artifact Layout v2 的路径契约，不代表文件已生成或测量成功。 |
| `result_layers.json` | 分层 CSV schema v2 清单，记录 generation、可选导入来源 SHA256、层路径、字段、行数和内容哈希；最后原子切换。schema v1 固定路径结果仍可读取。它不更改固定的 Layout v2 路由清单。 |
| `result_generations/<generation>/{summary,performance,resources,energy,network}.csv` | 同一组测量窗口按来源和职责拆分，保留全部六个测量身份字段，避免每个文件包含全部 200 多列；不同 generation 的文件互不覆盖。 |
| `result_generations/<generation>/profiling/{torch_profiler,ncu,nsys,massif}.csv` | 四种 Profiler 分别输出；没有实际字段证据的工具省略，存在的工具允许仅记录部分测量键。 |
| `capability_report.json` | 本次采集的能力状态及实际完整性，和静态元数据中的准备阶段快照分开。 |
| `quality_checks.json` | 独立 schema v1 的质量观察；普通 warning 不撤销已验证 Capability 或 `full_profile_complete`。 |
| `runtime_failures.json` | 存在正式请求失败时汇总的 typed failure 列表，保留请求 ID、阶段与环境；不改变测量 CSV 数值协议。 |
| `metadata/interface_validation.json` | schema v1 的源码 import／signature 报告，含模型 SHA、source/runner SHA256、dependency image ID、状态与 `inference=not_run`；失败记录 `failed_stage/error`，清理异常另存 `cleanup_error`。原始输出在 `logs/interface_validation.log`，不含测量指标。 |
| `metadata/runtime_validation.json` | 测量窗口外独立运行验证的结构化报告；原始输出在 `logs/runtime_validation_<device>.log`。 |
| `.acprof/work/cases/<case-id>/result.csv` | 采集期间逐资源配置写入的可恢复中间结果；成功合并后清理。 |
| `raw/requests/<case-id>.jsonl` | 长期保留的紧凑 request-level latency，每窗口一行；含 application 原始样本和按请求 ID 对齐的 packet 样本。详见下方约定，不参与默认统计聚合。 |
| `raw/windows/<case-id>/<attempt-id>/*.json` | schema v2 的不可变窗口记录，绑定 run、case、attempt、window 身份、原始行、请求样本、telemetry SHA256 和采集顺序。先持久化 receipt，再追加结果投影；历史 schema v1 保留原先 receipt 最后发布的读取语义。 |
| `raw/telemetry/<case-id>/<attempt-id>/*.json` | schema v1 原始 CPU/GPU/resource 样本、边界计数器、单位、时钟、采集器和 reducer 版本，以及 Burst/Skip 聚合诊断。超过保存上限的样本子集明确标记不完整，不能用于精确全量重算。 |
| `raw/measurements/<case-id>/<attempt-id>/` | 完整 case 测量结束后的不可变输入快照；含原始 CSV、已关闭的 PCAP、请求/分组投影及 `checkpoint.json`，后处理失败时用它恢复，不重新推理。抓包未启用时没有 PCAP。 |
| 分层结果（`result_layers.json` 与各模块 CSV） | 动态测量结果。每一行对应一个 resource config、一个 input scale、一次 warmup/repeat iteration，并记录归一化指标、PCAP 网络字节、cold-start phases，以及该窗口的 cgroup memory/stat/PID、swap、块 I/O 与压力/事件。 |
| `.acprof/run_state.json` | 主实验状态 schema v1，记录实验 ID、参数、主机与源码/依赖指纹、绑定的镜像和输入计划、case 完成状态与 CSV SHA256、启动/恢复记录、最终完成状态。 |
| `.acprof/recovery/interrupted_cases/` | 恢复时保存中断 case 的原始 CSV、PCAP 与关联 sidecar；备份完成后才开始该 case 的新测量。 |
| `static_meta.json` | 单个 JSON object 的静态元数据。记录模型版本、参数/精度/量化/许可证、输入输出格式、per-scale 静态逻辑 FLOPs、推理后端、镜像、GPU/主机 RAM、主机 swap、Docker 存储和环境信息。 |
| `metadata/model_resolution.json` | 运行准备阶段写入的解析报告，包含候选、字段来源、语义和独立运行验证引用；失败 draft 也可独立导出。与可执行 `acprof_model.json` 分离，静态裁决不是推理／测量成功证据。 |
| `metadata/auto_report.json` | `acprof auto` 的预检与收尾报告，保存请求／实际采集模式及静态决策身份；验证与实际采集结果独立引用，不作为 CSV 的替代证据。 |
| `metadata/collection_history.json` | schema v1 的采集/修复 provenance。分别记录 post-hoc profiler 补采、timeout retry、quality retry 和静态元数据回填历史；最新一次状态由对应 history 的最后一项得到。 |
| `metadata/input_scale_plan.json` | 所有任务族共用的 input scale/payload 计划。schema v2 额外记录 workload provenance、per-scale 输入元数据和模型约束；读取端要求 schema v2、UTF-8 的有限数值 JSON object，大小不超过 4 MiB，拒绝缺少版本或 v1 计划。主采集、独立 load、runtime validation 和 profiler 在启动容器或进入测量窗口前复用并校验同一份 payload。补采上下文读取已记录 SHA256 的计划时，对同一次有界读取的原始字节校验并解析，不能先解析再重新打开路径校验另一份内容；已有 UTF-8 BOM 计入字节指纹。 |
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

输入计划在准备阶段写入同目录临时文件；生成新计划前不删除旧计划。序列化拒绝非有限数字，
并按最终 UTF-8 字节检查 4 MiB 上限（包括平台换行），同批字节增量计算 SHA256。
合法计划沿用原有字段顺序、ASCII 转义、缩进和平台换行，不重写历史实验的计划或 hash。
生成、序列化、文件 `fsync` 或原子替换在发布前失败时，保留已有完整计划并向上报错，
不能将旧文件视为本次规划成功。目录 `fsync` 位于替换之后；此时失败仍报错，但完整的新计划可能已经发布。

`.acprof/work/cases/` 下本次已完成 case 的中间文件会在分层 CSV 成功发布、完成状态持久化后清理。
旧布局对应 `result_case_*.csv`、`*.sniff_groups.jsonl`、`lat_case_*.json` 和 `sniff_case_*.pcap`。
若运行被中断，中间文件保留用于恢复。

`sniff_groups.jsonl` 按 CSV 行序保存 `sniff_group_id`。运行中 OOM 或请求超时后，
host 为尚未完成的窗口补写错误占位行时，同步写入空字符串 group，保留已完成窗口的原始对应关系。
写入顺序沿用客户端的 sidecar 先落盘、CSV 后发布；CSV 发布失败后的多余尾行在重试时清理，
已提交 CSV 对应的 sidecar 行缺失或损坏仍明确报错，不猜测 group，也不改写历史实验文件。
旧 flat layout 与 v2 使用同一约定；历史 CSV 没有 sidecar 时不伪造请求对应关系。

`packet_latency.json`（旧布局为 `lat_case_*.json`）按请求和 TCP stream 保存 schema v2 抓包数据，
大小随请求数增长，不适用小型元数据的 4 MiB 限制。读取仍校验 UTF-8、顶层 object、schema、
request record 结构及有限数值；损坏或非有限数据报错，不截断请求，也不覆盖已有合并结果。
当前沿用整份 JSON 解码，内存占用随抓包数据量增长；解析与合并均在测量窗口结束后执行。

`raw/requests/*.jsonl`（旧布局为 `*.requests.jsonl`）保留 schema v1、`sniff_group_id`、`input_scale`、`warmup`、`repeat_idx`、
`source=client_http`、`latency_app_s` 数组及请求阶段 `status`。数组下标 `i` 对应请求 ID
`<sniff_group_id>:<i>`，数值单位为秒，保留原始浮点精度；只包含成功返回的请求，失败尝试另记
`failed_request_id` 和 `error`。超时中止时仍写出此前成功的请求，自动预热不进入该文件。
packet merge 在工作投影中追加同长度的 `latency_packet_s`，未匹配位置为 `null`；
basic 模式不生成该数组。原始请求和 PCAP 保留在不可变测量快照中；恢复未完成测量时，
旧工作请求文件随 CSV 一起备份，然后从 receipt 重建已完成部分，只采集剩余窗口。

### 实时提交与后处理

`result_layers.json` 的 schema v3 与 Artifact Layout v2 独立。它保存 `run_id`、单调提交序号、
schema hash、状态，以及每个模块的 `committed_bytes/row_count/sha256/revision`。
每个窗口使用 `run_id + case_id + case_attempt_id + window_id` 去重；同一计划位置来自不同
attempt 的重复测量也被拒绝。schema 从 `metric_registry.py` 和 `result_layers.py` 派生并冻结。

CPU RAPL、GPU NVML、PMU、资源监控全部停止并完成收尾后，client 持久化窗口证据，
同步追加该窗口的模块行。所有文件 `fsync` 完成后，发布不可变 commit，再原子替换根清单并
同步目录。下一窗口等待上述步骤完成；采样频率、窗口和能耗积分公式不变。
追加只序列化新行、增量更新 SHA256，不读取旧 CSV，也不生成 `live/` 或新的 generation。

普通 CSV 查看器可能短暂看到未提交尾部。正式读取器持有短期共享 publication lock，
只解析同一清单声明的字节前缀，并核对 SHA256、表头、行数、身份及基础层顺序。
读取器逐层校验和解析，同一层的哈希与 CSV 解析使用同一份原始字节；解析完成即释放该层
原始缓冲，避免同时保留全部 CSV 内容。初始化重试也逐层校验后释放原始字节。
结果快照仍保留全部解析行，单层读取和解析仍需要与该层大小相应的内存。
首次打开 writer 或恢复时校验 commit 链和原始来源；可信来源可修复缺失或损坏的 CSV，
多出的未提交尾部会截断。原始 receipt、telemetry、派生记录或 commit 损坏则报错。
恢复过程本身中断可再次重试，不能靠猜测数值修复证据。
完整 commit 文件仍逐条校验 SHA256；历史记录仅保留重放所需的来源列表和预期行数，
避免累积完整 manifest 副本。全链身份和顺序检查完成后，按提交顺序重放并逐次核对行数。
恢复时按 64 KiB 分块比较已有 CSV 的已提交前缀，每次扫描最多额外读取一个尾部探测字节，
避免载入整个未提交尾部。截断前在同一个可写文件句柄上重新核对前缀；不符则从可信来源
原子重建。完整且无需修复的 CSV 仍只读打开。
每层修复后仅保留校验所得的 SHA256 状态供 writer 后续追加使用，释放该层完整 CSV 缓冲区，
避免同时保留全部投影并再次计算哈希。追加交易更新哈希状态的副本，失败不会推进已提交状态。
原始记录重放和已提交行仍有各自的内存需求。

窗口来源的 SHA256 校验与 receipt 解析使用同一次读取取得的原始字节；路径和身份通过检查后，
按这份已加载 receipt 中的引用校验 telemetry，避免重新打开 receipt 时换用了另一份内容。
单次追加只验证一次该窗口，历史 schema v1/v2 receipt 的字段、哈希和大小策略保持不变。
来源文件的独立 SHA256 计算按 1 MiB 分块进行，哈希缓冲内存不随文件总大小增长；
证据仍须完整读取至文件末尾，缺失、内容改变或读取失败都会阻止派生结果提交或恢复。

抓包完成前，`packet_status=pending`；成功匹配为 `ok`，未请求为 `not_requested`，
无对应测量为 `not_applicable`，缺少匹配为 `unavailable`，后处理失败为 `error`。
硬件支持与失败原因继续由既有能力及诊断字段说明，缺值不等于零。
网络解析可能影响 performance、network、resources 和 profiler 派生速率；仅对实际变化的
模块做原子替换。变更字段、原因和 checkpoint/profiler 来源保存在 `raw/result_updates/`，
原始窗口行不改写。跨模块替换中断后，统一按最后提交及原始来源恢复。

实现借鉴 [Delta transaction log](https://github.com/delta-io/delta/blob/master/PROTOCOL.md)
的数据先持久化、再提交可见版本的顺序，以及
[atomicwrites 的目录同步处理](https://github.com/untitaker/python-atomicwrites/pull/15)。
仅使用标准库；未引入 Delta Lake 或已归档的 atomicwrites 依赖。

序列化和 `fsync` 在全部监测停止后执行，窗口内复用已有 latency 数组；不记录输入或输出 payload。
该设计参考 [MLPerf LoadGen 的延后日志处理](https://github.com/mlcommons/inference/blob/master/loadgen/logging.h)，
沿用本项目窗口结束后的写盘方式，无新增日志线程或依赖。正式分析仍按 CSV 的
`status=ok AND warmup=0` 筛选，以独立窗口为统计单位；请求数组不能替代窗口均值或置信区间。
历史目录缺少请求文件时视为未保存原始样本，不从均值或分位数重建。

### 准备阶段诊断

输出目录通过校验并创建 RunState 后，每次启动或恢复都会分配独立的 `attempt_id`。
`.acprof/run_state.json` 的本次 `attempts` 条目用 `preparation_evidence` 引用
`metadata/preparation/<attempt_id>.json`；对应主进程准备日志在
`logs/preparation/<attempt_id>.log`。CLI 和 TUI 使用相同路径，恢复不会覆盖上次尝试。

摘要的 `schema_version=1`、`scope=preparation`，记录最近阶段和阶段状态、各阶段尝试次数、
最终 `complete/failed/cancelled`、经过脱敏和限长的异常、最近一次可恢复错误，以及日志完整性。
`reason_code` 区分 `preparation_failed`、`cancelled_by_user`、
`controller_disconnected`、`interrupted` 和 `preparation_complete`。
同一次准备中的“重新准备环境”沿用同一 attempt；成功后摘要为 complete，最近错误仍保留。
准备成功不代表主实验成功；后续测量失败以 run state 和正式结果证据为准。
恢复是否允许仍由既有身份与产物校验决定，诊断摘要不自行推断可重试性。

日志只覆盖主进程 Python stdout/stderr，不保证收录直接继承文件描述符的子工具输出；
工具日志继续使用各阶段已有路径。每条完整逻辑行先脱敏再保存，去掉 URL 凭据、查询串、
片段和已识别的认证/敏感键值；摘要不保存原始请求、配置或环境字典。
日志正文最多 8 MiB，每条逻辑行最多 64 KiB UTF-8；超限整行省略，
摘要的 `omitted_lines`、`retained_bytes` 和 `log.status` 明示保存状态。
`log.status` 为 `recording/saved/truncated/failed/unavailable`；
字节数是保留文本的 UTF-8 长度，不是进程内存占用。

摘要在阶段切换时原子保存。准备退出时恢复原输出流并关闭日志，然后才进入正式测量；
不会为这份证据向测量窗口增加文件写入。证据保存失败会记录告警，原准备异常继续传播。
强制终止可能只留下最近的 running 阶段，不能把它当作已完成。
早于 RunState 的模型解析或主机预检错误仍由 CLI/TUI 报告，不为诊断提前绕过输出目录和锁校验。

### 结果完整性与断点续跑

主实验默认拒绝已有实验产物的目录，避免重写静态元数据或追加上次遗留的测量行。
重新执行原命令并加 `--resume`，或在 TUI 的“恢复 / 重试”检查页选择“继续实验”（准备阶段为“重试准备”），
可以恢复由当前状态协议创建的实验；TUI 恢复入口见[查找实验与恢复参数](../usage/tui-pages.md#查找实验与恢复参数)。
参数与输出根目录保持一致；`--notify`、`--rebuild-image` 不影响测量身份，但恢复时禁止重建，必须使用原始 image ID。历史 `skip_build` 字段读取时移除，不改变已保存证据。
保存的 options 只记录展开后的实际参数，不把 TUI 的 `extra_options` 容器当作 CLI 参数。
读取已有实验时展开该容器；重复字段值冲突或未知选项会拒绝恢复，不静默丢弃测量条件。
只有 probe 产物的目录仍可用于首次正式采集。旧实验没有 `run_state.json` 时继续支持读取、
绘图及补采，但不能仅凭残留 CSV 推断恢复状态；新的主实验须使用另一输出目录。

恢复检查主机、Python/依赖、AC-Prof 源码及继承的测量环境参数。已经完成准备的实验直接
使用保存的模型 revision、不可变 image ID、输入计划与 profiler 汇总，不重新查询 Hub、构建
镜像或生成另一组输入。原镜像必须存在。输入计划、静态元数据及 profiler 计划的 hash
不匹配时退出；准备阶段尚未完成、尚无 case 时可重新准备。

续跑入口与实际 writer 共用只读校验，按字段报告参数、采集源码、Python 依赖或主机身份的变化，
并核对冻结产物与已完成 case 的文件 hash。界面预检查不代替启动时的镜像和硬件核验；
writer 取得测量锁与目录锁后仍重新检查，避免确认期间发生变化。

同身份重试未完成准备时，在覆盖任何准备产物前复制原状态、元数据、日志和已有 profiler 原始文件，
v2 备份位于 `.acprof/recovery/preparation_attempts/<attempt-id>/`；新 attempt 的
`preparation_backup` 保存其相对路径。备份失败不追加 attempt，也不改写原记录。
已有 case 或正式 CSV 却缺少冻结运行环境时拒绝按准备失败重试，不用空 `cases` 掩盖遗留测量。
源码或环境变化仍不允许原地续跑；TUI 可恢复参数并自动选择新输出目录，保留原实验身份及全部产物。

源码身份包含执行模块、`extensions/*/manifest.json` 和 `assets/` 输入资源；TUI、绘图及分析展示
模块不参与续跑身份。声明或输入内容变化会拒绝恢复，界面文案变化不会。此源码范围使用新的
摘要域，因此升级前的未完成实验不能跨此变更续跑，应使用新输出目录；原产物仍可读取和分析。

case 只有在抓包回填和原有校验结束，且其测量唯一键完整匹配计划后才记为完成。唯一键为
`CPU × memory × GPU mode × input scale × warmup × repeat_idx`。恢复会复用校验通过的完整
case；测量尚未完成的中断 case 先备份，再创建新 attempt，复用可信 receipt 并仅采集计划中剩余窗口。已记录完整
错误占位的 OOM/timeout case 也属于已完成的尝试，不在恢复时自动改变超时或重测条件。
非整数 input scale 在计划、传参和 CSV 身份中保留可往返的完整数值，例如 `30/224` 不会被
格式化为六位有效数字。唯一键继续精确比较，不通过容差合并相邻尺度；显示用四舍五入不进入恢复身份。
OOM pruning 继续按原有参考 CPU/内存顺序重建证据，复用与推断不会增加正式请求。

case 状态分别记录 `measurement_status`、`processing_status` 和 `cleanup_status`。
client 退出、全部 monitor 停止、tcpdump 关闭后，先核对完整计划及已提交 receipt，
再复制原始输入到临时目录；复制体重新校验并 `fsync` 后整体发布到
`raw/measurements/<case-id>/<attempt-id>/`，最后把 checkpoint 引用写入 `run_state.json`。
此后 PCAP 解析、packet merge 或质量检查失败，`--resume` 校验快照、receipt 和 telemetry 的
身份/摘要，重建可变投影，仅重试后处理；不会启动新推理容器，也不会重复 warmup/repeat。
解析命令由当前受信源码重建，状态文件只提供经校验的数据参数。

容器清理失败或进程在清理确认前退出时，恢复先使用既有容器所有权协议清理已确认遗弃的容器，
并验证原不可变 container ID 已不存在；清理未确认时拒绝后处理。旧清理错误归档保留。
新 schema v3 的部分 case 可以逐窗口恢复。每次 attempt 使用独立 UUID，恢复预热按原已完成
warmup 的请求数执行并单独记录，不新增测量行；自动预热仍遵循原请求窗口规划。
这表示多个运行分段，不宣称连续的 cold start、idle 历史或温度状态。各行保留自己的 attempt
和测量证据。PCAP 分段单独保存、解析，通过含 attempt 的请求 ID 合并，禁止直接拼接 PCAP 字节。
请求标识新增 32 位 attempt UUID 和一个分隔符；原始 HTTP 请求头相应增加 33 字节，
网络字节指标仍计入这些协议字节，跨版本比较时应保留这一差异。
合并后的 packet JSON 用请求的 `capture_segment` 指向 `capture_segments`；索引 0 是当前抓包，
后续索引按 checkpoint 的归档顺序排列。`tcp_stream` 只在对应分段的 `streams` 中解释，避免编号碰撞。
历史协议没有该恢复语义时仍使用原来的整 case 重试。旧 receipt、telemetry 和 checkpoint 保留。
成功发布正式结果后，仅删除工作目录中的中间副本；原始 PCAP、测量快照和旧 generation
不会自动删除，因此长时间或高采样率实验需要预留相应磁盘空间。

新增采样策略和原始样本保存上限进入运行身份，包括默认值。升级前未完成实验不能跨源码身份
变化继续采集；已有分层 schema v1/v2 继续读取，新主实验使用 schema v3，不原地迁移历史结果。

合并拒绝缺失/空 case、重复文件、重复测量、截断行和与计划不符的行；不适用的可选指标为 `nan`。
最终校验通过后冻结不可变 generation，将快照引用提交到 schema v3 清单；不会创建 `result_all.csv`。
失败保留可恢复证据，不将未完成结果标记为完整。快照与旧格式读取规则见[不可变结果快照](../results/metrics.md#不可变结果快照)。
完成状态先持久化，再清理中间文件。`status=complete` 表示计划已执行并完成合并，
`outcome=partial` 表示其中包含错误行，两者不能等同于全部测量成功。

`.acprof/result.lock` 是 v2 采集和补采共享的目录锁；旧布局仍使用 `.acprof-result.lock`。
进程退出自动释放锁，锁文件本身可以保留。
主机测量锁以 Linux abstract UNIX socket `acprof.measurement.<uid>` 为内核互斥标识，
并同时取得固定 `/tmp/acprof-measurement-<uid>.lock` 的 `flock`。两者均不依赖工作树、
输出目录、`TMPDIR`、`XDG_RUNTIME_DIR` 或登录方式；SSH、终端和普通系统服务使用同一身份。
单纯迁移到 XDG runtime 目录会受登录生命周期和服务环境影响，因此不采用该方案。
文件锁保留 owner、regular-file、单链接和 no-follow 检查，取得后再次核对路径与 FD 的 inode。
异常文件、竞争或绑定失败均拒绝进入；失败会回收已取得的 FD，不自动删除锁文件。

持锁期间 unlink 或替换文件不会释放内核 socket，另一新版进程仍被阻止；最后一个 socket FD
关闭（包括进程被杀）后自动释放，无 PID 陈旧判定，也不需要正式窗口内轮询。
文件锁用于协调仍在运行的旧版本；旧版本本身无法抵抗其持锁文件被删除。
该互斥覆盖同机同 UID、共享主机 network namespace 的 AC-Prof 进程；服务必须使用主机
network namespace，不能同时用独立 network namespace 和独立临时挂载来运行采集。
独立容器／network namespace、其他 UID、特权管理员和外部负载不属于此 advisory lock 的保护范围。
设计依据 [Linux unix(7) 的 abstract socket 生命周期](https://github.com/mkerrisk/man-pages/blob/master/man7/unix.7)，
仅使用 Python 标准库，不新增依赖或测量期开销。
容器名称增加每次启动独有的后缀，case 文件名和测量唯一键保持稳定；清理只接受本次启动
取得的完整 container ID。Docker 创建成功但启动失败时，通过本次 `--cidfile` 回收该容器。
`--resume` 对已完成实验只检查结果结构并报告完成，不重新测量或覆盖后续补采的指标。
所有状态、备份与校验操作均在准备阶段、case 边界或最终发布阶段执行。

monitor 由 `MonitorGroup` 统一持有，按既有顺序启动和停止，随后尝试所有 `close()`。
某个 stop/close 失败仍会清理其余 monitor，并将原始请求错误和清理错误写入请求 JSONL；
清理失败会终止本次 client，避免残余采样器进入下一窗口。取消和请求超时仍保留原退出语义。
