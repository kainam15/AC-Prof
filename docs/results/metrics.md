# 指标与结果分析

查字段、分析过滤、CSV 格式要求或图表时查阅。能耗详见 [能耗测量](../profiling/energy.md)，独立工具详见 [Profiler](../profiling/profilers.md)，产物结构见 [采集协议](../profiling/protocol.md)。

[文档导航](../README.md)

新 CSV 包含 `environment_class`；旧数据缺环境证据时为 `unknown`。分析按环境分组，图表写入
`plots/<environment_class>/...`，混合身份 CSV 明确拒绝。`audit --compare` 同时检查
`comparability_class`，Native/WSL 或未知身份不能生成跨环境性能结论；缺失能耗显示
`not comparable`，不补零、不加入 Native baseline。范围与字段见 [WSL2](../platforms/wsl2.md)
和[环境协议](../profiling/measurement.md#环境身份与能力支持)。

## 采集能力概览

Process basic 使用独立的 `process_cpu_*` / `process_rss_*` 字段；`auto` 资源上限保留为空。
Runtime 身份、实际限制来源、GPU 整卡范围及缺失状态见 [Runtime 指标口径](../platforms/linux-runtimes.md#资源与指标口径)。
这些字段不能替换 Docker 的 cgroup 指标，也不改变其历史定义。

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

新实验准备阶段即生成 `result_layers.json` 与根目录的 `summary.csv`、`performance.csv`、
`resources.csv`、`energy.csv`、`network.csv`，以及 `profiling/` 下的四个 profiler CSV。
CSV 从完整表头开始，每个窗口结束后追加对应模块。未运行的 profiler 只有表头，没有数据行。

- 每个分层文件共享资源/计划键 `cpu_cores`、`mem_cap_gb`、`gpu_mode`、`input_scale`、`warmup`、`repeat_idx`，并共享 `runtime_kind` 与 `run_id/case_id/case_attempt_id/window_id`。基础层与 summary 一一对应，profiler 只输出实际采集记录。
- 除身份字段外，每个指标只有一个所属层，分类从 `metric_registry.py` 的来源及 tool 声明派生；未知历史扩展列归入 summary，不会丢弃。
- schema v3 清单记录统一提交序号、模块路径、字段、行数、已提交字节偏移及 SHA256。`acprof results verify <目录>` 检查已提交前缀、字段归属与关联键；不会把未提交尾部当成测量。历史 schema v1/v2 仍可读取，不自动转换已有结果。
- `acprof results export <分层目录> <新文件.csv>` 在显式请求时重建完整宽表，拒绝覆盖既有文件；日常采集和分析不生成它。
- 根 CSV 是可恢复的实时投影，原始窗口 receipt 与显式派生记录提供可追溯来源。Posthoc 仅替换受影响模块，完成后重新冻结快照；旧快照保持不变。

运行中可用 `read_result_csv(ArtifactLayout.discover(root).result_csv)` 获取完整逻辑宽表，
初始化后合法返回完整字段和空行列表。`read_result_csv_snapshot` 将内容与同一次锁定的清单
SHA256 绑定。读取会短暂等待提交；普通 CSV 查看器不参与锁，可能看到部分模块先更新。
schema v3 从同一个文件句柄流式解码并哈希已提交前缀，普通读取不预读未提交尾部；
初始化的严格校验只额外检查一个尾部字节。短读会继续读取，提前结束、would-block 或 I/O 失败不会被当成完整结果。
原始字节缓冲保持有界，返回逻辑宽表仍需保存解析结果；不会宣称整个读取过程使用固定内存。
提交和恢复协议见[实时提交与后处理](../profiling/artifacts.md#实时提交与后处理)。

#### 不可变结果快照

完成实验或调用 `ResultWriter.freeze()` 时，按字段、行数及内容 SHA256 生成确定性 generation，
在 `result_generations/<generation>/` 保存不可变 CSV；相同内容复用同一 generation。
先持久化全部快照文件与快照清单，再通过统一 commit 发布 `snapshot` 引用和 `state=complete`。
根 CSV 与快照使用不同 inode，后续追加或模块替换不会改写已发布快照。
替换清单后目录同步失败仍报错，但新提交可能已生效，恢复必须按实际 HEAD 判断。
损坏的不可变 generation 明确拒绝，不用新数据静默覆盖。

schema v3 采集使用独占 writer lease，读者与提交使用独立的短期 publication lock。
历史 schema v2 的读者仍按一次清单访问不可变文件。Posthoc 继续持有实验目录锁并核对输入身份，
不能同时修改同一实验。所有 CSV 哈希与解析均使用同一次读取的字节。
旧 generation 与发布中断留下的未引用 generation 不自动删除，以免影响进行中的读取或恢复证据。
这会增加多次补采后的磁盘占用；自动垃圾回收不属于当前协议。

这一原子快照的范围是清单及其引用的结果层集合。`static_meta.json` 与采集历史仍分别发布，
Posthoc 通过持久化恢复记录协调元数据与结果，失败后按最后提交恢复；它们不属于同一次清单切换。Posthoc 获得目录锁后会重读结果、
元数据、输入计划和历史，与锁前准备的数据比较；发生变化就停止，避免旧输入覆盖另一轮补采的结果。
普通只读分析并未共同持有该目录写入锁，因此在补采发布期间仍可能短暂读到新元数据与旧提交。
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
