# 指标与结果分析

查字段、分析过滤、CSV 格式要求或图表时查阅。能耗详见 [能耗测量](../profiling/energy.md)，独立工具详见 [Profiler](../profiling/profilers.md)，产物结构见 [采集协议](../profiling/protocol.md)。

[文档导航](../README.md)

新 CSV 包含 `environment_class`；旧数据缺环境证据时为 `unknown`。分析按环境分组，图表写入
`plots/<environment_class>/...`，混合身份 CSV 明确拒绝。`audit --compare` 同时检查
`comparability_class`，Native/WSL 或未知身份不能生成跨环境性能结论；缺失能耗显示
`not comparable`，不补零、不加入 Native baseline。范围与字段见 [WSL2](../platforms/wsl2.md)
和[环境协议](../profiling/measurement.md#环境身份与能力支持)。

## 采集能力概览

以下为项目可采集的指标范围；实际列值取决于 [profiling mode](../profiling/measurement.md#profiling-mode-与能力证据)、设备和显式启用的工具。

- 性能：application / packet-level latency、P50/P90/P95、标准差/CV/IQR/最大值、吞吐量、每任务尺度单位及每百万像素延迟、每 CPU core 吞吐，以及容器启动、server setup、CUDA 初始化、模型加载、ready wait 和首次推理的冷启动分解。
- 能耗：CPU package、估算 vCPU 和 GPU 的 idle、平均/峰值功率与能量，以及不增加采集轮次的 container-attributed 能效派生值（包括 J/input unit，以及图像任务的 J/Mpixel）。
- 资源：容器 CPU / 内存、cgroup CPU throttling、memory events、CPU/内存/I/O PSI、`memory.peak`、anon/file/slab、page fault/refault、块 I/O 字节与操作数、PID 当前值/峰值/上限事件、CPU 频率与估算 cycles，以及 GPU utilization、VRAM、SM/显存时钟、P-state 和温度。
- 网络：从同一份 PCAP 派生每请求的请求/响应 frame bytes、TCP payload 和 L2–L4 协议开销，不增加抓包轮次。
- PMU：retired-instruction MIPS、cycles / ref-cycles、IPC、计数运行比例、cache miss 和 dTLB miss。
- 计算：PyTorch eager 逻辑 FLOP，以及 NVIDIA Nsight Compute 实际 GPU FLOP。
- 可选 execution profile：Valgrind Massif 内存峰值、Nsight Systems CUDA timeline。

## 从结果目录开始

结果目录为 `<output-dir>/<model-dir>/`，模型 ID 中的 `/` 替换为 `--`。
例如 `google-bert/bert-base-uncased` 对应 `google-bert--bert-base-uncased/`。

| 阅读目的 | 入口 |
| --- | --- |
| 查看测量值 | `result_layers.json` 指向的 `summary.csv` 和各指标模块 CSV；正式分析筛选 `status=ok` 且 `warmup=0`。 |
| 查找产物 | `result_manifest.json` 的布局路径与 `result_layers.json` 的校验记录。 |
| 复现实验对象和输入 | `static_meta.json` 与 `metadata/input_scale_plan.json`。 |
| 追踪补采或修复 | `metadata/collection_history.json` 与对应 profiler plan。 |
| 查看图表和拟合 | `plots/cpu/`、`plots/gpu/`、`plots/gpu+cpu/` 与 `plots/latency_model/`。 |

`latency_app_s` 是客户端应用层计时，`latency_s` 是抓包解析得到的 packet-level 计时。
关闭 GPU 或未启用某个 profiler 时，对应字段为 `nan` 属于预期结果。
新实验运行中先写 `.acprof/work/cases/<case-id>/result.csv`，矩阵完成后汇总为独立分层 CSV；不自动创建完整宽表。详见[产物结构](../profiling/artifacts.md#artifact-layout-v2)。

### 分层结果 CSV（唯一正式结果）

新实验完成后生成 `result_layers.json`，指向 `result_generations/<generation>/` 中的
`summary.csv`、`performance.csv`、`resources.csv`、`energy.csv`、`network.csv`。
如果 profiler 有记录，则在该 generation 的 `profiling/` 下生成对应的
`torch_profiler.csv`、`ncu.csv`、`nsys.csv`、`massif.csv`；未运行且没有有效数据的分析器不写空表。

- 每个分层文件保留 `cpu_cores`、`mem_cap_gb`、`gpu_mode`、`input_scale`、`warmup`、`repeat_idx` 六个测量身份字段；基础层与 summary 一一对应，profiler 可按实际采集记录稀疏输出。
- 除身份字段外，每个指标只有一个所属层，分类从 `metric_registry.py` 的来源及 tool 声明派生；未知历史扩展列归入 summary，不会丢弃。
- `result_layers.json` 的 schema v2 记录 generation、模块路径、字段、行数和文件哈希；`acprof results verify <目录>` 检查文件损坏、归属和关联键。缺失字段或无效关联不以 `0` 填补。已有 schema v1 的固定路径分层结果继续只读兼容；新发布一律使用 v2，不改写原有文件。
- `acprof results export <分层目录> <新文件.csv>` 在显式请求时重建完整宽表，拒绝覆盖既有文件；日常采集和分析不生成它。
- 分层 CSV 是唯一正式存储。Posthoc 补采生成完整的新快照，保持上一代的文件不变；不再兼容旧实验的宽表目录。

#### 不可变结果快照

generation 是各层字段、行数与内容 SHA256 的确定性摘要。同一结果重复发布复用同一 generation，
不重写现有文件或相同的清单。发布先在临时目录写完并同步所有层，再原子安装 generation 目录，
最后原子替换根目录 `result_layers.json`。写层、安装目录或提交清单之前失败，读者仍可读取上一份完整结果；
如果替换清单之后目录同步失败，操作仍报错，但完整的新快照可能已生效。现有损坏的 generation 会被明确拒绝，
不会静默用新数据覆盖修复。

读者按一次读取的清单访问不可变文件；文件的 SHA256 与 CSV 解析使用同一次字节读取。
并发发布各自准备完整快照，以最后一次清单原子替换为准，不会混合两代的 CSV；需要读取后修改的
Posthoc 仍必须持有实验目录锁，不能把这一规则用作并发合并指标的机制。
旧 generation 与发布中断留下的未引用 generation 不自动删除，以免影响进行中的读取或恢复证据。
这会增加多次补采后的磁盘占用；自动垃圾回收不属于当前协议。

这一原子快照的范围是清单及其引用的结果层集合。`static_meta.json` 与采集历史仍分别发布，
Posthoc 失败时通过备份恢复；它们不属于同一次清单切换。Posthoc 获得目录锁后会重读结果、
元数据、输入计划和历史，与锁前准备的数据比较；发生变化就停止，避免旧输入覆盖另一轮补采的结果。
普通只读分析并未共同持有该写入锁，因此在补采发布期间仍可能短暂读到新元数据与旧 generation。
需要同时依据结果、元数据与历史作结论时，应在补采完成后读取；当前协议不承诺整个实验目录的原子事务。

`audit.valid` 表示已有聚合结果、计划及元数据检查通过，不代表逐一验证了全部原始样本的哈希。
窗口 receipt 与关联 telemetry 由 `read_window_record` / `verify_case_window_records` 校验，
后处理恢复还会校验 measurement checkpoint 的文件集合与内容。原始证据的缺失、损坏或截断
应通过这些入口判断，不能从聚合结果可读取推断原始证据完整或可精确重算。

显式从宽 CSV 导入时，schema v2 的 `source_name` / `source_sha256` 记录导入时的来源，
导入过程会核对源文件未改变；发布完成后，来源 CSV 可以移动、删除或修改，不影响快照本身的校验。
schema v1 保留原来的源文件变更校验。路径必须符合声明的 generation 与层名，拒绝越界、符号链接及
对 generation 目录的宽表导出写入。测量字段、单位、六字段关联键及统计口径没有随存储协议变更。

跨实验比较将 `sampling_policy` 与 `raw_sample_limit` 纳入已记录的测量协议条件，
适用于同硬件、跨硬件和资源伸缩比较。不同 Burst/Skip 策略或保存上限不自动视为同条件；
历史结果缺少这些记录时保持 `unknown`，不会补写当前默认值来证明可比。

可只读检查结果完整性，并按独立测量窗口估计均值区间：

```bash
acprof audit results/<model-dir>/ --require-complete --require-ok
acprof stats results/<model-dir>/ --metric latency_app_s
```

实验尚未完成时可省略 `--require-complete` 查看审计说明。区间的样本单位、
连续窗口相关性与开销对照方法见[统计说明](comparisons.md#窗口置信区间与开销对照)。

完整说明集中在[输出文件](../profiling/artifacts.md#输出文件)、[CSV 字段字典](fields.md#分层指标字段解释完整宽表导出)
和[常见判断](../usage/troubleshooting.md#常见判断)。

## 专题索引

- [指标分组与字段解释](fields.md)：资源、性能、延迟、网络和状态字段。
- [报告与只读审计](reports.md)：统一分析模型、交互报告和结果审计。
- [统计分析与跨实验比较](comparisons.md)：置信区间、开销、比较范围和非流式报告。
- [图表与绘图入口](plotting.md)：绘图命令、目录和生成规则。
