# 实验产物结构与恢复

[← 返回专题目录](protocol.md)

## 输出文件

输出目录为 `<output-dir>/<model-dir>/`；模型 ID 中的 `/` 替换为 `--`。
下表列出可能生成的文件；probe、补采、调试与绘图产物仅在执行对应操作时出现。

### Artifact Layout v2

新主实验由 `ArtifactLayout` 创建以下布局；根目录的四个文件分别是正式结果、静态描述、
能力报告和 `result_manifest.json`。子目录按需创建，文件缺失不能据此推断实验成功或失败。

`result_manifest.json` 与其他受限 JSON 产物共用读取器：只接受不超过 4 MiB 的 UTF-8
JSON object，拒绝非有限数值和损坏内容。未知或不一致的 manifest 仍明确报错，不退回 flat layout。

```text
<model-dir>/
├── result_layers.json         # 分层结果的 SHA256 与字段/行数清单
├── summary.csv                # 行身份、输入与状态
├── performance.csv            # 延迟、吞吐、启动
├── resources.csv              # CPU、GPU、内存与 PMU
├── energy.csv                 # 功率、能耗与来源
├── network.csv                # 抓包网络统计
├── profiling/                 # 各分析器独立的可选 CSV
│   ├── torch_profiler.csv
│   ├── ncu.csv
│   ├── nsys.csv
│   └── massif.csv
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

当前正式结果只使用带 `result_manifest.json` 与 `result_layers.json` 的 v2 布局，旧实验目录不提供兼容续跑、分析或 Posthoc。已有正式产物的目录不能直接初始化为新实验。
目录布局兼容不放宽现有产物 schema、源码身份或恢复校验；升级前中断的实验仍受源码指纹约束。
运行期间不要移动文件或删除清单。以下路径表使用 v2；旧目录沿用原文件名和位置。

| 文件 | 说明 |
| --- | --- |
| `result_manifest.json` | Artifact Layout v2 的路径契约，不代表文件已生成或测量成功。 |
| `result_layers.json` | 分层 CSV schema v1 清单，记录来源宽表 SHA256、层路径、字段、行数和内容哈希。它是完成时发布的独立产物，不更改固定的 Layout v2 路由清单。 |
| `summary.csv` / `performance.csv` / `resources.csv` / `energy.csv` / `network.csv` | 同一组测量窗口按来源和职责拆分，保留全部六个测量身份字段，避免每个文件包含全部 200 多列。 |
| `profiling/{torch_profiler,ncu,nsys,massif}.csv` | 四种 Profiler 分别输出；没有实际字段证据的工具省略，存在的工具允许仅记录部分测量键。 |
| `capability_report.json` | 本次采集的能力状态及实际完整性，和静态元数据中的准备阶段快照分开。 |
| `quality_checks.json` | 独立 schema v1 的质量观察；普通 warning 不撤销已验证 Capability 或 `full_profile_complete`。 |
| `runtime_failures.json` | 存在正式请求失败时汇总的 typed failure 列表，保留请求 ID、阶段与环境；不改变测量 CSV 数值协议。 |
| `metadata/interface_validation.json` | schema v1 的源码 import／signature 报告，含模型 SHA、source/runner SHA256、dependency image ID、状态与 `inference=not_run`；失败记录 `failed_stage/error`，清理异常另存 `cleanup_error`。原始输出在 `logs/interface_validation.log`，不含测量指标。 |
| `metadata/runtime_validation.json` | 测量窗口外独立运行验证的结构化报告；原始输出在 `logs/runtime_validation_<device>.log`。 |
| `.acprof/work/cases/<case-id>/result.csv` | 采集期间逐资源配置写入的可恢复中间结果；成功合并后清理。 |
| `raw/requests/<case-id>.jsonl` | 长期保留的紧凑 request-level latency，每窗口一行；含 application 原始样本和按请求 ID 对齐的 packet 样本。详见下方约定，不参与默认统计聚合。 |
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
case；中断 case 先备份，再创建新容器，重新执行该 case 的原 warmup/repeat 协议。已记录完整
错误占位的 OOM/timeout case 也属于已完成的尝试，不在恢复时自动改变超时或重测条件。
非整数 input scale 在计划、传参和 CSV 身份中保留可往返的完整数值，例如 `30/224` 不会被
格式化为六位有效数字。唯一键继续精确比较，不通过容差合并相邻尺度；显示用四舍五入不进入恢复身份。
OOM pruning 继续按原有参考 CPU/内存顺序重建证据，复用与推断不会增加正式请求。

合并拒绝缺失/空 case、重复文件、重复测量、截断行和与计划不符的行；不适用的可选指标为 `nan`。校验通过后分别原子写入各模块 CSV，最后发布 `result_layers.json` 完整性清单；不会默认创建 `result_all.csv`。失败保留可恢复的 case 产物，不将未完成结果标记为完整。
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
