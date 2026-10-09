# 运行排障与结果判断

解释预检、抓包、OOM、cgroup、空值或未完成实验时查阅。取证步骤见 [结果审计流程](../../.agents/skills/acprof-result-audit/SKILL.md)，指标定义见 [指标与结果分析](../results/metrics.md)。

[文档导航](../README.md)

## 诊断证据

| 情况 | 判断依据与限制 |
| --- | --- |
| 启动或运行期主机内存不足 | Docker `OOMKilled=true`、明确 cgroup OOM kill 或分配失败日志，并结合发生阶段；仅分配失败不等于已发生 OOM kill |
| CUDA OOM | 明确 CUDA allocation/OOM 日志；主机 RAM 和 GPU VRAM 是不同资源 |
| 请求超时 | 超时记录和当次参数；区分已发请求与 `not_measured_after_timeout` 未执行计划行 |
| Profiler 不适用或失败 | 对应工具计划、错误和支持范围，不能直接推断主实验失败 |
| 断连或退出码 `137` | 单独不足以确定 OOM，需要容器、内核或明确日志证据 |
| 文件或字段缺失 | 先核对任务、设备、工具、schema 与运行阶段，不统一解释为零或失败 |

探测分类见 [`largest_scale_probe.py`](../../acprof/host/largest_scale_probe.py)；探测超时与正式请求超时是独立配置。
容器被清理后无法取得的状态必须标为证据缺口，不能从容器不存在推断退出原因。

## 实时状态检查

仅执行与问题相关的命令。运行中的实验优先看原日志，避免额外高频轮询或新 GPU 负载。
这些命令的结果属于本次检查，不应复制成长期机器配置；`static_meta.json` 的启动快照也只描述对应实验。

```bash
git status --short --branch
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv
df -h .
docker context show
docker info --format '{{.DockerRootDir}} {{.CgroupVersion}}'
docker ps -a --format '{{.Names}}\t{{.Status}}'
docker image ls
```

根据日志中的实际容器名、image ID 或 PID 定向查看 `docker inspect --format '{{json .State}}' <container>`、
`docker image inspect <image-id>` 或 `ps -p <pid> -o pid,etime,comm`。尖括号内容需替换为本次值。
Docker 数据可能在另一挂载点，磁盘问题还需对实际 `DockerRootDir` 执行 `df`，按需查看 `docker system df -v`。

## 常见判断

本节解释结果值、失败边界与采样限制。安装、主机权限、抓包和旧镜像的操作排查见
[README 常见问题](#常见问题)。

| 当前现象 | 优先查阅 |
| --- | --- |
| 空值或统计量不可用 | [预期空值与失败](#先区分预期空值与失败) |
| 模型或任务不支持 | [预检退出](#任务尚未适配的预检退出)、[图像描述兼容范围](../models/supported-tasks.md#图像描述输出与兼容范围) |
| 内存不足或跳过 case | [启动 OOM 与剪枝](#启动-oom-与剪枝占位)、[运行期 OOM](#运行期-oom) |
| 能耗、基线或功率异常 | [GPU 空值](../profiling/energy.md#gpu-energy-字段全是-nan)、[CPU 空值](../profiling/energy.md#cpu--vcpu-energy-字段全是-nan)、[CPU 基线](../profiling/energy.md#cpu-idle-baseline-波动-warning)、[GPU 基线](../profiling/energy.md#gpu-idle-baseline-波动-warning)、[峰值功率](../profiling/energy.md#cpu--vcpu-peak-power-看起来异常) |
| 资源或 PMU 指标异常 | [资源占用率](#资源占用率字段全是-nan)、[MIPS、cache 与 dTLB](#mipscache-miss-与-dtlb-miss) |
| Profiler 字段缺失 | [计算分析器](../profiling/profilers.md#mflops--compute-profiling-字段全是-nan)、[执行分析器](../profiling/profilers.md#massif--nsight-systems-execution-profiling-字段全是-nan) |

### 先区分预期空值与失败

- 常规性能分析筛选 `status=ok` 且 `warmup=0`；`status=error` 行中的部分数值不视为完整测量。
- 关闭的 profiler、不适用的任务/GPU mode 和旧版本缺失字段可以为 `nan`，不能直接解释为采集失败或数值为零。
- 一个窗口只有 1 个有效请求时，标准差、CV、IQR 为 `nan`；增加 `--repeat` 只增加窗口数，不保证每个窗口有多个请求。
- 比较 profiler 数值前检查 plan 的采样来源；代表资源复用值不代表该 CPU/内存配置被独立分析过。

### 任务尚未适配的预检退出

- `acprof run` 与 `acprof probe` 在任务识别及显式覆盖后，检查已知采集缺口、未登记的任务标签和任务族不匹配；失败时显示 `[task-support][ERROR]` 及解决办法，退出码为 `2`。TUI 显示“任务不支持”，详细原因保留在日志中。
- `image-to-text` 已支持 CV 单图请求及图像描述输出，要求 `--batch-size 1`；多图 batch 会在预检退出。`image-text-to-text` 等九类多模态任务及适配边界见 [README](../models/supported-tasks.md#多模态任务)，同样要求单样本请求。预检不是对全部模型架构或依赖版本的兼容保证。
- 预检在模型镜像准备、输入规划和测量前执行，因此不新增测量 CSV、OOM/超时占位行或探测请求记录；已有测量结果保留。不要将此类退出解释为资源不足或一次实际推理失败。
- 支持范围与适配步骤见 [README 的任务支持诊断](#task-supporterror--tui-显示任务不支持)。Hub 连接、鉴权或缺少元数据导致的识别失败继续使用独立诊断。

### 启动 OOM 与剪枝占位

- 容器在模型加载期间触达 `--memory` cgroup 上限并被内核终止；错误会同时记录 memory cap、Docker 状态和 exit code。
- 该 case 的占位行保留为 `status=error`，latency、throughput、energy 和 resource usage 等未执行指标保持 `nan`。`acprof plot` 的性能图与 latency model 只使用 `status=ok` 行；资源可行性热力图会单独读取这些占位行，用来展示失败边界。
- 增大 memory cap，或改用更小/量化模型；不要用推测值回填失败 case 的指标。
- 默认剪枝在正式矩阵前执行独立 startup probe，按最低 CPU 的连续低内存 confirmed Docker OOM 前缀推断；probe 不写性能 CSV。对应全部 CPU（含参考 CPU）的正式 case 写为 `result_origin=inferred_not_measured`，热力图显示 `P-OOM`。正式尝试自身发生启动 OOM 才显示 `OOM-S`，不会回头扩大冻结计划的剪枝范围。需要每个资源格独立实测时使用 `--no-prune-startup-oom`；冻结顺序与证据见[采集协议](../profiling/measurement.md#startup-probe-与冻结矩阵)。

### 运行期 OOM

- 容器已完成启动，但在 workload 期间被 memory cgroup OOM kill。Docker 的 `OOMKilled=true` 优先于客户端的 MIPS、HTTP 断连或其他次生退出码进行分类。
- 已成功写入的测量行原样保留；已有失败行追加 Docker OOM 上下文；其余计划行写为 `status=error`，未采指标保持 `nan`。该 case 返回后矩阵继续执行。
- 运行期 OOM 在资源可行性热力图中显示为 `OOM-R`，但不会用于启动 OOM 剪枝，也不能从一个 CPU 配置外推到其他 CPU 配置。

### 资源占用率字段全是 `nan`

- 正式采集只支持 cgroup v2；cgroup v1 在启动阶段退出，不再提供兼容开关。
  缺失的 cgroup 文件保持未知，不使用 host 全局数据冒充 container 指标。

### MIPS、cache miss 与 dTLB miss

- `cpu_mips_*` 字段来自 Linux `perf` 的 `instructions` 硬件事件，不是 CPU frequency 推导值。`acprof run` 会在 task detection 前检查 `perf` 权限，失败时打印 `[mips][ERROR]`、当前 `perf_event_paranoid` 和恢复步骤。
- TUI 快速检查与正式启动共用 `resolve_perf_command_prefix()`：直接运行 `perf`，先读取有效 `instructions` 计数，再附加主机 PID 1 检查跨用户权限，每个探测最多 5 秒。环境检查不尝试 sudo 或自动 setcap；权限不足时可在 `F2` → “连接与权限 → 采集权限”显式配置，或参照[最小权限安装](../getting_started/installation.md#最小权限安装)。`ACPROF_SUDO_PASSWORD` 已移除，须从进程和本地 env 文件删除。TUI 使用独立环境副本，并将旧配置报告为迁移错误；实际容器 PID 仍由采集时的探测验证。
- `cpu_cache_*` 和 `cpu_dtlb_*` 字段来自 Linux `perf` generic PMU events。可先用 `perf list` 和 `perf stat -e cache-references,cache-misses,dTLB-loads,dTLB-load-misses -- true` 检查当前 CPU / kernel 是否支持；事件不支持不表示 miss 为 0。
- 这些 cache / dTLB 字段用于描述访存行为，不提供 DRAM GB/s。实际 read/write bandwidth 需要 uncore memory-controller、Intel PCM、AMD IBS/DF 或其他硬件专用计数器，不能由 miss 数直接换算。

## 常见问题

### `[infra][ERROR]`

确认主机为原生 Linux，Docker endpoint 是本机 Engine。先读取实际 context 和环境覆盖：

```bash
docker context show
docker context inspect --format '{{.Endpoints.docker.Host}}'
docker info
```

结合 `DOCKER_HOST`、`DOCKER_CONTEXT` 与 CLI 报错确定实际 endpoint；只有需要切换时再调整本次命令环境，
不要仅为诊断就改写用户默认 Docker context。正式采集的主机条件见[运行指南](../getting_started/installation.md#1-检查主机环境)。

### 中断后残留容器或端口占用

`acprof run` 的 SIGTERM 按取消处理，展开当前 case 和实验状态的清理；清理完成后恢复原信号处理器。
SIGKILL 无法执行 Python 清理。服务容器带有 `org.acprof.container.lifecycle=1` 及主机、用户、boot ID、
PID、进程启动时间标签；下一次启动服务前会回收同机同用户且能确认主人进程已退出的容器。
检查与回收在冷启动计时之前完成，不进入正式测量窗口；删除目标始终使用完整容器 ID。

运行中的会话、其他主机/用户、缺少归属信息的旧容器以及无法读取进程身份的容器会保留。
旧版本容器没有这些标签，仍需用 `docker ps -a --no-trunc`、`docker inspect <完整容器ID>` 核对归属及端口，
确认已不用后再单独处理。回收失败会阻止新服务启动并报告错误，不会改用其他端口混入原实验。
恢复采集仍须通过原实验的参数、源码、依赖和产物身份检查；容器回收不会删除实验文件。

`scripts/check_runtime.py` 会记录清理失败并使原本成功的检查失败；不会用清理错误覆盖原有非零返回码，
也不会吞掉 `KeyboardInterrupt` 或未捕获的异常。

### `[sniff][ERROR]` 或 `latency_s` 无法合并

检查 `tcpdump`、`tshark`、capability 和 bridge：

```bash
command -v tcpdump tshark
getcap "$(command -v tcpdump)"
ip link show docker0
```

如果 Docker daemon 修改过默认 bridge，传入 `--sniff-iface <实际网卡>`。

### `[cpu-energy][ERROR]`

AC-Prof 要求 RAPL energy counter 可读。按错误信息检查 `/sys/class/powercap/*/energy_uj` 的存在性和权限；不要用 TDP 或 CPU utilization 伪造缺失功耗。

### `[cgroup][ERROR]`

正式实验要求统一 cgroup v2。检查 `test -f /sys/fs/cgroup/cgroup.controllers` 和
`cat /proc/self/cgroup`；修复主机启动/systemd 配置并重启后再采集。已删除 `--allow-cgroup-v1`。
结果目录留有不同版本或版本未知的 `result_case_*.csv` 时会拒绝续写；使用新输出目录或归档原部分结果。

### `[mips][ERROR]`

先运行：

```bash
perf stat -e instructions -- true
cat /proc/sys/kernel/perf_event_paranoid
```

若权限不足，`acprof run` 会输出适合当前主机的修复步骤。修好权限后用普通用户运行 AC-Prof，不要使用 `sudo acprof run`，以免结果文件归 root 所有。

即使 `perf_event_paranoid=-1`，普通用户也可能无法附加 root 所属的 Docker 服务进程。
若自启动 perf 成功、PID 附加失败，需要给真实 perf ELF 配置 `CAP_PERFMON`；Ubuntu 的
`/usr/bin/perf` 通常只是包装脚本。TUI 的“连接与权限”会解析当前内核对应的真实文件。
多行 `[ERROR]` / `[WARN]` 后续诊断会在测量窗口结束后完整显示，不再作为普通输出隐藏。

### `container_oom_killed during startup`

按[启动 OOM 与剪枝](#启动-oom-与剪枝占位)核对容器阶段、实测前缀与 `startup_oom_pruning.json`，区分实际失败和 `P-OOM` 推断。

### `container_runtime_oom`

按[运行期 OOM](#运行期-oom)核对 Docker 状态与已有成功行；错误占位不进入常规性能分析。

### 运行中还没有 `result_all.csv`

这是正常的：矩阵执行期间先写 `.acprof/work/cases/<case-id>/result.csv`（旧目录为 `result_case_*.csv`），全部 case 完成后才合并为 `result_all.csv`。如果在 tmux 中运行，采集期间查看对应终端；`logs/terminal.log`（旧目录为 `tmux_all.log`）在命令结束或报错退出时落盘。

### `--skip-build` 后接口报错

先区分镜像核验失败与独立推理验证失败。当前实现会核验指纹、模型 revision、环境与下载清单，
不存在匹配镜像时自动构建；`--skip-build` 不会直接使用旧 `:latest`。
指纹或清单不匹配时按报错重建；核验通过但接口失败时，检查 `runtime_validation.json`、设备日志、
任务路由与实际容器依赖，不能把所有错误都归因于镜像陈旧。行为见[运行兼容说明](../models/images.md#构建复用和验证)。

### `[task-support][ERROR]` / TUI 显示“任务不支持”

退出阶段和产物边界见[任务预检退出](#任务尚未适配的预检退出)。根据用户目标处理：

- **想立即采集：** 换用[任务目录](../models/supported-tasks.md#任务支持范围)中已覆盖接口的模型，例如图像分类、目标检测或 ASR 模型。
- **必须采集此类型：** 等待支持该类型的项目版本，或按[适配契约](../models/adaptation.md#新增一个模型适配)补齐输入、推理、输出与指标口径，再验证后采集。
- **确实是识别错误：** 核对模型页的 `pipeline_tag`，通过 `--task`、`--task-family`、`--backend`（TUI 高级配置中的“识别覆盖”）纠正。仅在模型实际支持目标任务时使用；把图像描述模型改填成图像分类不会获得分类能力。

`image-to-text` 已有图像描述输出适配，要求 `--batch-size 1`，且最终 runtime profile 必须同时支持模型架构和 pipeline task。
默认 CV profile 使用 Transformers 4.57.6；需要 5.6.0 架构的 GLM-OCR 会因该版本缺少 `image-to-text` registry
在预检返回 `runtime_task_unsupported`，重建相同环境不能消除这个接口缺口。
镜像实际版本与选定 lock 不符则返回 `runtime_dependency_incompatible`，应按选定环境重建；
只修改主机 `.venv` 不会改变镜像内依赖。历史日志中的 `Unknown task image-to-text` 也应先核对这两类证据。
项目直接调用 4.57.6 的 [Apache-2.0 官方 pipeline](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/pipelines/image_to_text.py)，
未另引入推理框架。该路径不包含多模态对话任务，也不保证所有图像描述架构兼容；详见[Runtime 预检与失败证据](../models/routing.md#runtime-预检与失败证据)。

Hub 已明确给出的未知任务标签会保留并提示，不再被通用架构后缀猜成另一类任务。Hub 无法访问、缺少元数据等识别失败仍保留独立诊断，不统一归为“不支持”。

更多诊断，包括 idle baseline 波动、Profiler `nan`、GPU energy 和 PMU event 问题，见[常见判断](#常见判断)。
