# 指标分组与字段解释

[← 返回专题目录](metrics.md)

## 分层指标字段解释（完整宽表导出）

每行对应一个资源配置、一个 input scale 和一次 warmup/repeat 请求窗口。
常规性能分析只取 `status=ok` 且 `warmup=0`；错误行中的部分数值不作为正式测量。
不可用或不适用的数值为 `nan`，历史 CSV 缺少新字段时不能补成 `0`。

字段按用途分组，列顺序、类型、单位、来源和窗口由 [metric_registry.py](../../acprof/metric_registry.py) 统一登记；
`config.CSV_FIELDS` 引用同一字段列表。完整元数据见[字段速查](metric_reference.md)，绘图数值转换和补采完成条件复用登记表。

显式导出宽表时，列依次为：资源配置与输入输出／网络、
延迟与吞吐、独立 Profiler、GPU／CPU package／DRAM／估算 vCPU 能耗与能效、
CPU 资源与 PMU、容器内存／swap／I/O／PID、GPU 资源、冷启动、请求契约与结果来源。
GPU UUID 紧随 `gpu_mode`，能耗来源紧邻对应 GPU 能耗组，硬件 cycles／IPC 属于 PMU 组。
标准表头最后四列固定为 `workload_contract`、`result_origin`、`status`、`error`；
Profiler 和 DRAM 自身的诊断保留在各自指标组末尾。

合并、packet 回填和 profiler 补采写出时统一此顺序，已有未知扩展列保留在 `status`、`error`
之前。列重排不改变字段名、数值、单位、测量窗口或缺失值含义；分层 CSV 依据字段名和测量键组合。
client 追加到字段集合相同的已有文件时沿用原表头，避免数值错位；字段缺失或重复时在写入前报错，
要求使用新输出文件。已有实验文件不会因升级而自动重写。

`*_per_request` 以及历史 CPU/GPU/vCPU 能量列按窗口内请求数归一化；DRAM 的
`dram_window_energy_j` / `dram_window_effective_energy_j` 保留整段窗口能量，
对应 `*_per_request_j` 才是 J/request。`*_delta` 若未注明归一化，则表示整个窗口的增量。

`result_origin` 区分 `formal_measurement`（正式 client 窗口）、`formal_attempt`（正式尝试失败）
与 `inferred_not_measured`（startup OOM 剪枝占位）。是否成功仍读取 `status`；
独立 startup probe 只有 JSON 证据，没有性能 CSV。历史缺少此列不补造来源。

| 查阅方向 | 字段组 |
| --- | --- |
| 配置、输入与请求 | [资源配置、输入与网络](#资源配置输入与网络)、[延迟与吞吐](#延迟与吞吐)、[两种延迟的区别](#latency_s-和-latency_app_s-的区别) |
| Profiler | [Torch 与 NCU](../profiling/profilers.md#torch-与-ncu-计算指标)、[Massif 与 Nsight Systems](../profiling/profilers.md#massif-与-nsight-systems-执行指标) |
| 能耗与归一化 | [GPU](../profiling/energy.md#gpu-功率与能耗)、[CPU package](../profiling/energy.md#cpu-package-功率与能耗)、[DRAM](../profiling/energy.md#rapl-topology-与-dram)、[估算 vCPU](../profiling/energy.md#估算-vcpu-能耗与派生能效)、[像素口径](#像素归一化口径) |
| 资源与 PMU | [CPU](#cpu-资源频率与-pmu)、[容器内存、swap、I/O 与 PID](#容器内存swapio-与-pid)、[GPU 资源](#gpu-资源与运行状态) |
| 生命周期与失败 | [冷启动](../profiling/lifecycle.md#冷启动)、[运行状态与错误](#运行状态与错误) |

### 资源配置、输入与网络

| 字段 | 含义 |
| --- | --- |
| `cpu_cores` | 当前 Docker container 的 CPU core 限制，来自 `--cpus`。 |
| `mem_cap_gb` | 当前 Docker container 的 memory cap，单位 GB，来自 `--mems`。 |
| `gpu_mode` | `on` 或 `off`。`on` 表示容器仅暴露 `--gpu-device` 选定的物理 GPU。 |
| `gpu_device_uuid` | 主容器和主机 NVML 采集器共用的物理 UUID；CPU 行为 `nan`。设备名称、PCI bus ID 与主机 index 见 `static_meta.json/gpu_device`。 |
| `input_scale` | 本行实际执行的主输入尺度。语义见 `static_meta.json/input_scale_type`。 |
| `input_units_per_request` | `effective input_scale × batch_size`。保留历史尺度语义：NLP/时序通常是 token/context step，Audio 是秒，CV 是缩放倍率，分辨率型 Diffusion/多模态是边长。这些图像尺度都不是像素总数。 |
| `input_num_samples` | 音频 payload 的实际采样点数；其他任务或旧计划无法推导时为 `nan`。它是诊断字段，音频主尺度仍为 `duration_s`。 |
| `input_pixels_per_request` | CV/多模态每请求输入像素总数，单位 pixel/request。由物化计划中的宽高计算；相同尺寸视频为 `batch_size × width × height × frame_count`，多模态中同时存在的图像与视频像素相加。不乘颜色通道数，指 processor 处理前的素材尺寸。 |
| `output_pixels_per_request` | Diffusion 每请求输出像素总数，单位 pixel/request。单张方形图像为 `batch_size × resolution_px²`；视频为 `batch_size × output_pixel_count_per_video`，帧数只计一次。来自输入计划中受 handler 尺寸协议约束的输出几何，不新增采样。不适用或不能确认几何时为 `nan`。 |
| `request_payload_bytes` | `requests` 实际发送的 prepared HTTP JSON body 字节数，在同一 workload window 内取平均。 |
| `packet_request_wire_bytes_per_request` / `packet_response_wire_bytes_per_request` | 从同一 PCAP 中属于该 `/predict` TCP stream 的 client→server / server→client captured `frame.len` 总和，再对本行请求求平均。客户端使用 `Connection: close`，因此每个 stream 对应一个请求；握手、ACK、关闭包和重传都保留。 |
| `packet_total_wire_bytes_per_request` | 上述请求与响应 captured frame bytes 之和。它不含 capture 未保留的 Ethernet FCS，也不包含物理层 preamble / inter-frame gap，不能直接当作插座侧链路能耗输入。 |
| `packet_tcp_payload_bytes_per_request` | 同一 TCP stream 内 `tcp.len` 的总和，再对本行请求求平均；重传 payload 会按实际捕获次数计入。 |
| `packet_protocol_overhead_bytes_per_request` | `packet_total_wire_bytes_per_request - packet_tcp_payload_bytes_per_request`，表示捕获到的 L2/L3/L4 header、ACK/握手/关闭等开销；不拆分 TCP payload 内的 HTTP header 与 JSON body。 |
| `packet_protocol_overhead_ratio` | 本行所有请求的 protocol overhead bytes 总和 / total wire bytes 总和；分母无效时为 `nan`。 |
| `task_param` | 本行 payload 真正发送给 handler 的二级参数，使用稳定排序的 JSON 字符串；通常来自 `params`，时序任务同时记录顶层 `prediction_length`，不再记录未执行的任务族默认值。 |
| `output_length_avg` | 同一 workload window 内响应长度的平均值：文字任务为该请求所有返回文本的 Unicode 字符数之和（不是 UTF-8 字节）；Diffusers 图像生成为图像数，视频生成为总帧数；检索等不适用任务为 `nan`。 |
| `output_token_count_avg` | 同一 workload window 内每次响应文本 tokenizer token 数的平均值；ASR/图像描述按输出文本重新分词，`add_special_tokens=False`。图像描述先逐条分词再求和，不拼接 caption，也不等于实际生成 token ID 数或解码步数。tokenizer 不可用或统计失败时响应为 `null`；窗口内全部不可得时 CSV 为 `nan`，有效空文本为 `0`。 |
| `repeat_idx` | 当前 warmup 或 repeat phase 内的 0-based iteration index。 |
| `warmup` | `1` 表示 warmup 行，`0` 表示正式测量行。`acprof plot` 默认排除 warmup 行。 |
| `repeat_in_window` | 本行内部连续发送的 request 数量。`latency_app_s` 和 `latency_s` 都是该 window 内 request 的平均值。 |

### 延迟与吞吐

| 字段 | 含义 |
| --- | --- |
| `latency_s` | packet-level latency，来自 `tcpdump` PCAP + `tshark` 解析 + `acprof.packet.merge_packet_latency` merge。当前默认要求该字段完整；抓包不可用、PCAP 为空、解析为空或 merge 后仍有缺失时，程序会退出并给出恢复提示。 |
| `latency_s_per_input_unit` | `latency_s / input_units_per_request`，在 packet merge 阶段更新。 |
| `latency_s_per_input_megapixel` / `latency_s_per_output_megapixel` | `latency_s × 1,000,000 / 对应的 pixels_per_request`，单位 s/Mpixel；沿用 packet 请求响应窗口，在 packet merge 后更新。 |
| `latency_request_count` | 本行实际合并到 packet-level latency 分布中的有效 request 数。可与 `repeat_in_window` 对照检查抓包完整性。 |
| `latency_p50_s` / `latency_p90_s` / `latency_p95_s` | packet-level latency 在本行 request window 内的 empirical nearest-rank 分位数，由 merge 阶段从同一组 packet latency 明细计算。 |
| `latency_std_s` / `latency_cv` / `latency_iqr_s` / `latency_max_s` | 同一 packet-level request window 的总体标准差、变异系数 `std / mean`、nearest-rank `P75 - P25` 和最大值。少于 2 个有效 request 时，std、CV 和 IQR 为 `nan`；max 仍保留。 |
| `latency_slow_ratio` | packet-level latency 中超过 `SLOW_LATENCY_THRESHOLD_S` 的 request 比例，默认阈值为 `0.06` 秒，用于观察尾延迟或双峰分布。 |
| `latency_app_s` | host-side application latency。`acprof.host.client` 用 `requests.post()` 外层 `time.perf_counter()` 测得，通常比 `latency_s` 更容易稳定产出。 |
| `latency_app_s_per_input_unit` | `latency_app_s / input_units_per_request`。 |
| `latency_app_s_per_input_megapixel` / `latency_app_s_per_output_megapixel` | `latency_app_s × 1,000,000 / 对应的 pixels_per_request`，单位 s/Mpixel；包含原 application 请求的预处理、推理、后处理和传输，不是仅模型算子的耗时。 |
| `latency_app_request_count` | 本行 application latency 分布中的有效 request 数；正常成功窗口通常等于 `repeat_in_window`。 |
| `latency_app_p50_s` / `latency_app_p90_s` / `latency_app_p95_s` | host-side application latency 在本行 request window 内的 empirical nearest-rank 分位数。 |
| `latency_app_std_s` / `latency_app_cv` / `latency_app_iqr_s` / `latency_app_max_s` | application latency 的总体标准差、变异系数、nearest-rank IQR 和最大值；少于 2 个有效 request 时 std、CV 和 IQR 为 `nan`。 |
| `latency_app_slow_ratio` | host-side application latency 中超过 `SLOW_LATENCY_THRESHOLD_S` 的 request 比例，默认阈值为 `0.06` 秒。 |
| `throughput_samples_per_s` | 吞吐量，约等于 `batch_size / latency`。如果 `latency_s` 成功 merge，会优先按 `latency_s` 更新；否则按 `latency_app_s` 计算。 |
| `throughput_samples_per_s_per_cpu_core` | `throughput_samples_per_s / cpu_cores`；packet latency merge 后会与 throughput 一起重算。它表示按配置 CPU quota 归一化的吞吐，不是实际 CPU utilization 归一化值。 |

### 像素归一化口径

分辨率横轴（`resolution_px` / `resolution_scale`）下的能耗和延迟归一化子图使用
`J/Mpixel` / `s/Mpixel`，分母为每请求像素总数除以一百万。Diffusion 使用输出像素，
CV 和图像多模态使用 processor 处理前的输入像素；视频计入全部帧。横轴仍表示原来的
边长或缩放倍率。`input_units_per_request` 和 `*_per_input_unit` 继续表示任务尺度单位；
文本 token、音频秒数、去噪步数等尺度使用各自的 input unit。

`Mpixel` 表示一百万像素。新字段使用每百万像素单位，避免 CSV 六位小数把很小的
每像素延迟舍入为零。所有分子已经平均到单 request；分母计入该 request 的完整 batch，
不再乘除 `repeat_in_window`。绘图先对每行计算比值，再沿用同一资源配置和尺度内的
mean/median 聚合，默认仅纳入 `status=ok` 且 `warmup=0` 的行。

例如 batch=2、输出为 128×128 时，每请求有 32768 像素；若能耗为 327.68 J，
则为 10000 J/Mpixel。输出改为 256×256、能耗增为 1310.72 J 时，仍为
10000 J/Mpixel。旧字段 `E / (batch_size × resolution_px)` 会从 1.28 升至 2.56，
因此它的上升不能用于判断每像素能效变差。

像素归一化使用 2 个像素计数和 6 个比值字段；partial case 缺少这些
可选列时仍可保留原测量行并补写失败记录，新列填 `nan`。静态 schema 保持 v7，
输入计划要求 v2，多模态视频计划另行记录 `video_frame_width / video_frame_height`。
实时采集在 monitor 停止后从物化计划计算像素数；若服务返回的有效尺度与计划不一致，
像素计数保持 `nan`。packet 回填只更新 packet 像素延迟，不改变 application 延迟或能耗。

绘图只使用 CSV 的显式 `input_pixels_per_request` / `output_pixels_per_request`。
缺少计数、空值或非法计数保持 `nan`；不再读取输入计划为旧行补造像素数，相关面板显示 `No data`。
有效计数对应的比值由原始能耗/延迟重算，避免使用过期派生值；不改写输入 CSV。

输入像素数不等同于模型实际处理的 patch/token 数或 FLOP：processor 可能缩放、切块或
固定尺寸。像素指标描述给定工作负载的成本，比较时仍需核对模型、推理步数、精度和
输出质量；token、音频秒数、context step、去噪步数等尺度不适用像素面积归一化。

### CPU 资源、频率与 PMU

| 字段 | 含义 |
| --- | --- |
| `resource_usage_iters` | resource usage monitor 在本行测量窗口内保留的 sample 数。 |
| `container_cpu_util_avg_pct` | 当前 Docker container 在测量窗口内的平均 CPU 占用率，按 `container_cpu_time_delta / (elapsed_seconds * cpu_cores) * 100` 计算。 |
| `container_cpu_util_peak_pct` | 当前 Docker container 在相邻采样间隔中的峰值 CPU 占用率，单位 `%`。 |
| `container_cpu_nr_periods_delta` / `container_cpu_nr_throttled_delta` | workload 窗口首尾 cgroup `cpu.stat` 的调度周期数和被 quota throttled 周期数之差。支持 cgroup v2，也兼容 v1 `cpu.stat`。 |
| `container_cpu_throttled_period_ratio_pct` | `nr_throttled_delta / nr_periods_delta * 100`；用于判断 `--cpus` quota 是否实际成为瓶颈。没有有效 period 时为 `nan`。 |
| `container_cpu_throttled_time_s_per_request` | cgroup CPU throttled time 的窗口增量换算成秒后除以 `repeat_in_window`。v2 读取 `throttled_usec`，v1 读取 `throttled_time` 纳秒值；它不是单纯的 request wall latency。 |
| `container_cpu_pressure_some_stall_pct` / `container_cpu_pressure_full_stall_pct` | cgroup v2 `cpu.pressure` 的 `some/full total` 在 workload 窗口内的增量除以窗口时长。使用累计 stall time，不使用瞬时 `avg10/60/300`；系统不提供 per-cgroup PSI 时为 `nan`。 |
| `cpu_freq_avg_hz` | 测量窗口内 host online CPU 当前频率的平均值，单位 Hz。每个 sample 先对 online CPU 求平均，最终再对窗口内 sample 求平均；优先读取 Linux cpufreq sysfs，失败时回退到 `/proc/cpuinfo`。 |
| `cpu_freq_peak_hz` | 测量窗口内 host online CPU 当前频率的峰值，单位 Hz。每个 sample 取 online CPU 的最高当前频率，最终再取窗口内最大值。 |
| `cpu_cycles_est_app` | 基于 application latency 的 estimated CPU cycles，公式为 `latency_app_s * cpu_freq_avg_hz * cpu_cores * container_cpu_util_avg_pct / 100`。这是利用率与频率推导值，不是硬件 PMU retired instructions / cycles 计数。 |
| `cpu_cycles_est_packet` | 基于 packet-level `latency_s` 的 estimated CPU cycles，公式同 `cpu_cycles_est_app`，但在 `acprof.packet.merge_packet_latency` 成功回填 `latency_s` 后才会更新；merge 前或 packet latency 缺失时为 `nan`。 |
| `cpu_instructions_per_request` | Linux `perf stat -e instructions` 采集到的 retired instructions，按本行 `repeat_in_window` 平均到单 request。MIPS 采集失败会中止实验而不是写入静默 `nan`。 |
| `cpu_cycles_per_request` / `cpu_ref_cycles_per_request` | 同一 perf PID 窗口的 `cycles` / `ref-cycles` 硬件事件计数除以成功请求数。ref-cycles 是参考频率周期，不作为 IPC 分母。不支持、未计数或混合 PMU 的部分计数缺失时为 `nan`。 |
| `cpu_ipc` | 同一窗口 `instructions / cycles`，只使用相同 PMU/事件修饰范围的完整计数。cycles 非正、计数缺失或范围不一致时为 `nan`，不会用 estimated cycles 补分母。 |
| `cpu_perf_running_pct` | instructions、cycles、ref-cycles 有效读数中最小的 counter running percentage，用于识别 multiplexing；没有有效运行比例时为 `nan`。perf 默认已缩放计数，不再次乘除该比例。 |
| `cpu_cache_references_per_request` | Linux `perf` generic event `cache-references` 的窗口计数，按本行 `repeat_in_window` 平均到单 request。其对应的 cache level 由 CPU 架构和 kernel PMU 映射决定。 |
| `cpu_cache_misses_per_request` | Linux `perf` generic event `cache-misses` 的窗口计数，按本行 `repeat_in_window` 平均到单 request；不能跨架构固定解释为某一级 cache miss。 |
| `cpu_cache_miss_rate_pct` | `cache-misses / cache-references * 100`。用于观察 cache access locality，不表示实际内存带宽；分母无效或为 0 时为 `nan`。 |
| `cpu_dtlb_loads_per_request` | Linux `perf` event `dTLB-loads` 的窗口计数，按本行 `repeat_in_window` 平均到单 request。 |
| `cpu_dtlb_load_misses_per_request` | Linux `perf` event `dTLB-load-misses` 的窗口计数，按本行 `repeat_in_window` 平均到单 request，用于观察数据地址转换未命中。 |
| `cpu_dtlb_load_miss_rate_pct` | `dTLB-load-misses / dTLB-loads * 100`。用于观察数据地址转换开销；分母无效或为 0 时为 `nan`。 |
| `cpu_mips_app` | 基于 `latency_app_s` 的真实 retired-instruction MIPS，公式为 `cpu_instructions_per_request / latency_app_s / 1e6`。 |
| `cpu_mips_packet` | 基于 packet-level `latency_s` 的真实 retired-instruction MIPS，在 `acprof.packet.merge_packet_latency` 成功回填 `latency_s` 后更新；merge 前或 packet latency 缺失时为 `nan`。 |
| `cpu_perf_elapsed_s` | perf 统计窗口报告的 elapsed time，单位秒，用于诊断 perf 窗口是否覆盖本行 workload。 |

cycles / ref-cycles 为可选 PMU 事件，不改变 instructions 的必需性。IPC 基于硬件计数，
仍受 perf attach 边界、进程线程范围及 multiplexing 的影响；窗口不等于模型单一算子。
分析 CPU 执行效率优先使用 `cpu_cycles_per_request` 与 `cpu_ipc`；`cpu_cycles_est_*`
继续保留其估算含义，供旧文件读取和诊断。解析格式参考
[Linux perf stat 源码文档](https://github.com/torvalds/linux/blob/master/tools/perf/Documentation/perf-stat.txt)，
使用现有 perf 进程，不增加独立采样轮次。

### 容器内存、swap、I/O 与 PID

| 字段 | 含义 |
| --- | --- |
| `container_mem_usage_avg_bytes` | 当前 Docker container 在测量窗口内的平均 memory usage，单位 bytes，来自 cgroup memory 文件。 |
| `container_mem_usage_peak_bytes` | 当前 Docker container 在测量窗口内的峰值 memory usage，单位 bytes。 |
| `container_mem_util_avg_pct` | 当前 Docker container 平均 memory usage / `mem_cap_gb` 的百分比。 |
| `container_mem_util_peak_pct` | 当前 Docker container 峰值 memory usage / `mem_cap_gb` 的百分比。 |
| `container_mem_peak_cgroup_bytes` | workload 窗口结束时读取 cgroup v2 `memory.peak`。它是该新建 container cgroup 自创建以来的内存峰值，因此能捕获 `sample_hz` 之间的瞬时峰值，也可能包含模型加载期；不是单个 workload window 可重置的峰值。文件不存在时为 `nan`。 |
| `container_mem_anon_bytes_end` / `container_mem_file_bytes_end` / `container_mem_slab_bytes_end` | workload 窗口结束时 cgroup v2 `memory.stat` 的匿名内存、文件页和 slab 当前字节数。`slab` 缺失时使用 `slab_reclaimable + slab_unreclaimable`；无法读取时为 `nan`。 |
| `container_mem_pgfault_delta` / `container_mem_pgmajfault_delta` | cgroup v2 `memory.stat` 的 page fault / major page fault 计数器在 workload 窗口首尾的增量。它们是整个 cgroup 的事件数，不按 request 归一化。 |
| `container_mem_workingset_refault_delta` | cgroup v2 `memory.stat` 的 workingset refault 窗口增量；内核只提供 anon/file 分项时取两者之和，用于观察页被回收后再次访问。 |
| `container_mem_high_events_delta` / `container_mem_max_events_delta` | cgroup v2 `memory.events` 的 `high` 和 `max` 计数器在 workload 窗口内的增量，分别表示 memory high 边界触发和 memory max 边界命中次数。文件不可用时为 `nan`。 |
| `container_mem_oom_events_delta` / `container_mem_oom_kill_events_delta` | cgroup v2 `memory.events` 的 `oom` 与 `oom_kill` 窗口增量；前者表示 cgroup 内分配进入 OOM，后者表示实际发生进程 OOM kill。 |
| `container_mem_pressure_some_stall_pct` / `container_mem_pressure_full_stall_pct` | cgroup v2 `memory.pressure` 的 `some/full total` 窗口增量占窗口时长的比例。`full` 表示窗口内所有相关任务同时因内存压力停顿。 |
| `container_swap_limit_bytes` | 当前 container cgroup 的独立 swap hard limit。cgroup v2 来自 `memory.swap.max`。`-1` 表示 cgroup 未设上限，无法读取时为 `nan`。 |
| `container_swap_usage_avg_bytes` | 本行 workload 测量窗口内 container swap 使用量的平均值。cgroup v2 读取 `memory.swap.current`，按现有 `sample_hz` 采样。 |
| `container_swap_usage_peak_bytes` | 同一测量窗口内采样到的 container swap 使用量峰值，单位 bytes；它不是 host 全局 swap 使用量。 |
| `container_io_read_bytes_per_request` | 本行测量窗口首尾 container cgroup block-I/O read bytes 计数器之差，再除以实际 `repeat_in_window`；聚合 cgroup 报告的全部设备，无法读取或本行未完成请求时为 `nan`。 |
| `container_io_write_bytes_per_request` | 本行测量窗口首尾 container cgroup block-I/O write bytes 计数器之差，再除以实际 `repeat_in_window`；page-cache hit 不产生块设备读取，因此该值不等于应用读取的文件字节数。 |
| `container_io_read_ops_per_request` / `container_io_write_ops_per_request` | cgroup v2 `io.stat` 中全部设备 `rios/wios` 的窗口增量除以实际 `repeat_in_window`。它们表示 block-I/O operation 数，不是 POSIX read/write syscall 数；cgroup v1 兼容模式为 `nan`。 |
| `container_io_pressure_some_stall_pct` / `container_io_pressure_full_stall_pct` | cgroup v2 `io.pressure` 的 `some/full total` 窗口增量占窗口时长的比例，用于区分真实 I/O stall 与仅有块 I/O 字节增量的情况。 |
| `container_pids_current_end` | workload 窗口结束时 cgroup v2 `pids.current`，统计该 cgroup 当前 tasks 数。 |
| `container_pids_peak_cgroup` | workload 窗口结束时读取 cgroup v2 `pids.peak`；与 `memory.peak` 一样是新建 container cgroup 生命周期峰值。旧内核未提供该文件时为 `nan`。 |
| `container_pids_max_events_delta` | cgroup v2 `pids.events/max` 在 workload 窗口内的增量，表示因 PID hard limit 拒绝创建 task 的次数。 |

### GPU 资源与运行状态

| 字段 | 含义 |
| --- | --- |
| `gpu_sm_clock_mhz` | NVML 选定物理 GPU 在 workload 测量窗口内的 SM clock 成功采样值算术平均，单位 MHz。仅 `gpu_mode=on` 且 NVML 支持该查询时有值；不采集 graphics clock。 |
| `gpu_memory_clock_mhz` | NVML 选定物理 GPU 在 workload 测量窗口内的 memory clock 成功采样值算术平均，单位 MHz。 |
| `gpu_pstate` | NVML 选定物理 GPU 在 workload 测量窗口内出现次数最多的 performance state（`P0`–`P15`）；次数并列时取性能等级更高的较小编号。它是运行状态解释变量，不代表锁频。 |
| `gpu_temp_c` | NVML 选定物理 GPU 在 workload 测量窗口内的 GPU temperature 成功采样值算术平均，单位 °C。 |
| `gpu_util_avg_pct` | NVML 选定物理 GPU 在测量窗口内的平均 GPU utilization，单位 `%`。这是 device-level 口径，不做 container process attribution。 |
| `gpu_util_peak_pct` | NVML 选定物理 GPU 在测量窗口内的峰值 GPU utilization，单位 `%`。 |
| `gpu_mem_used_avg_bytes` | NVML 选定物理 GPU 在测量窗口内的平均 used VRAM，单位 bytes。 |
| `gpu_mem_used_peak_bytes` | NVML 选定物理 GPU 在测量窗口内的峰值 used VRAM，单位 bytes。 |
| `gpu_mem_util_avg_pct` | NVML 选定物理 GPU 平均 used VRAM / total VRAM 的百分比。 |
| `gpu_mem_util_peak_pct` | NVML 选定物理 GPU 峰值 used VRAM / total VRAM 的百分比。 |

### 运行状态与错误

| 字段 | 含义 |
| --- | --- |
| `status` | `ok`、`warn` 或 `error`。`warn` 标记存在异常的行；常规性能图与延迟模型只使用 `ok` 行，资源可行性图保留失败状态。 |
| `error` | 错误或 warning 文本。正常行为空；`status=error` 时强制非空。请求超时会区分实际发出但未完成的请求（`client_request_timeout`）与未发请求、因前序超时跳过的计划行（`not_measured_after_timeout`），并记录触发尺度、timeout 下界、请求阶段和 request ID。 |

### `latency_s` 和 `latency_app_s` 的区别

- `latency_app_s` 是 client 侧应用层计时，只要 `/predict` 请求成功，一般就能写出。
- `latency_s` 是 packet-level 计时，需要完整完成 `tcpdump` capture、`acprof.packet.sniff_parse_pcap` parse、`acprof.packet.merge_packet_latency` merge。
- 当前默认行为是严格模式：如果无法保证 `latency_s` 有值，`acprof run` 会退出，不继续 merge 最终结果。

`workload_contract` 是新增的 JSON 文本列，不参与数值聚合。每行保存已完成请求的实际工作量摘要，
以 `request_count` 和 `variants[{count, contract}]` 保留请求数量与不同工作量的分布；不保存请求顺序。
其 generation 上限与实际 token 数、图像和音频模态的区别见[Workload Contract](../profiling/measurement.md#workload-contract)。
`basic` 使用 `latency_app_s` 和相应 application 分布指标；`latency_s` 仍专指 packet latency，不能互相替填。
分析能耗／PMU 时先核对模式和[能力证据](../profiling/measurement.md#profiling-mode-与能力证据)，缺失数值继续为 `nan`。
