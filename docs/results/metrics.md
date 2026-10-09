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
| 查看测量值 | `summary.csv` 和各指标模块 CSV，由 `result_layers.json` 统一关联；正式分析筛选 `status=ok` 且 `warmup=0`。 |
| 查找产物 | `result_manifest.json` 的布局路径与 `result_layers.json` 的校验记录。 |
| 复现实验对象和输入 | `static_meta.json` 与 `metadata/input_scale_plan.json`。 |
| 追踪补采或修复 | `metadata/collection_history.json` 与对应 profiler plan。 |
| 查看图表和拟合 | `plots/cpu/`、`plots/gpu/`、`plots/gpu+cpu/` 与 `plots/latency_model/`。 |

`latency_app_s` 是客户端应用层计时，`latency_s` 是抓包解析得到的 packet-level 计时。
关闭 GPU 或未启用某个 profiler 时，对应字段为 `nan` 属于预期结果。
新实验运行中先写 `.acprof/work/cases/<case-id>/result.csv`，矩阵完成后汇总为独立分层 CSV；不自动创建完整宽表。详见[产物结构](../profiling/artifacts.md#artifact-layout-v2)。

### 分层结果 CSV（唯一正式结果）

新实验完成后生成 `result_layers.json`、`summary.csv`、`performance.csv`、`resources.csv`、`energy.csv`、`network.csv`。如果 profiler 有记录，则生成 `profiling/torch_profiler.csv`、`profiling/ncu.csv`、`profiling/nsys.csv`、`profiling/massif.csv` 中对应文件；未运行且没有有效数据的分析器不写空表。

- 每个分层文件保留 `cpu_cores`、`mem_cap_gb`、`gpu_mode`、`input_scale`、`warmup`、`repeat_idx` 六个测量身份字段；基础层与 summary 一一对应，profiler 可按实际采集记录稀疏输出。
- 除身份字段外，每个指标只有一个所属层，分类从 `metric_registry.py` 的来源及 tool 声明派生；未知历史扩展列归入 summary，不会丢弃。
- `result_layers.json` 的 schema v1 记录模块路径、字段、行数和文件哈希；`acprof results verify <目录>` 检查文件损坏、归属和关联键。缺失字段或无效关联不以 `0` 填补。
- `acprof results export <分层目录> <新文件.csv>` 在显式请求时重建完整宽表，拒绝覆盖既有文件；日常采集和分析不生成它。
- 分层 CSV 是唯一正式存储。Posthoc 补采按事务刷新模块，未变化的模块文件无需重写；不再兼容旧实验的宽表目录。

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
