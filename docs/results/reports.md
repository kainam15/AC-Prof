# 报告与只读审计

[← 返回专题目录](metrics.md)

## 图表与延迟拟合产物

### PNG 窗口统计与区间

PNG 的适用数值曲线、指标概览和 Process 输入尺度图复用 `analysis.uncertainty.summarize_windows`，
按资源、输入尺度、环境和运行时身份筛选 `status=ok,warmup=0` 窗口，等权计算均值与标准差。
95% within-run percentile bootstrap 使用 5000 次重采样、seed=0、block-size=1，与 TUI 默认窗口统计一致。
曲线保留均值点，置信带及竖线表示区间；有效窗口不足三个或无有效数据时不伪造区间，缺失尺度不连成完整观测。
仅在一次绘图调用内复用包含实际观测和身份的统计缓存；重新生成图表会重新读取数据。

这不是跨独立实验的区间；独立重复使用 `acprof compare`。冷启动按启动身份去重，Profiler、
派生归一化指标及显式 median 等原有聚合不强行转换为请求窗口区间。缺少窗口身份的库调用也不推断独立样本。
图表、TUI 与可选统计 CSV 使用相同基础单位；GUI 可显示 ms，CSV 保留原始单位字段。

### 统一分析模型与精简汇总

`acprof.analysis.model.load_analysis()` 将实验目录或 CSV 转为只读 `AnalysisModel`。
采集 CSV 和字段顺序保持原样；`raw_rows` 保留包括未知扩展列在内的原始行，`records` 提供长表：

```text
run_id, model, runtime, device, cpu, memory, concurrency, metric, value, unit
```

长表同时保留 `config_id`、task、input case、experiment batch、GPU 身份、环境、
`source`、CSV 行号、`eligible` 和 `evidence`。`source_row` 从 2 开始（第 1 行是表头），
来源快照记录 CSV SHA256。未记录 run ID 的旧 CSV 使用内容摘要生成 `legacy-*` 标识，
它只用于定位文件，不证明实验独立性。缺少 model/runtime/concurrency 或环境时保持 unknown/null；
不能用当前主机身份补全历史实验。相邻 `static_meta.json` 按已有明确字段读取，不要求历史 schema
升级；未知列完整保留，但已废弃且归因不明的能耗/计算字段不自动映射为当前指标。
每个来源的数值列按表头准备一次，长表指标仍按登记顺序输出。已知缺失标记直接归为 null；
其他值经过有限数值检查，派生能耗复用同一行已解析的 CPU、GPU 能耗和请求数。

`config_id` 使用记录的 run ID 和配置条件摘要；目录名和 `experiment_batch` 展示标签不参与身份。
移动或重命名原始目录不会改变配置 ID，来源路径和目录展示名仍保留。
同时导入原件与副本时，同一记录的 `run_id + measurement_key` 明确拒绝为重复；数据或配置条件
不同则报告内容冲突，并给出两个 CSV 的行号。旧 CSV 的内容标识仅能识别完全相同的已知内容，
不能推断缺失的真实 run ID。CSV 与 run state 各自记录的 run ID 不一致时也拒绝导入。
同一次加载复用来源解析快照，完成后复核内容；读取期间 CSV 或 manifest 引用的数据层改变时
拒绝生成报告。历史 CSV 能显示原始观测，不代表它已通过严格的实验身份和可比性校验。

TUI 的「选择实验」、结果查看和离线统计／绘图／HTML／对比入口均按实际产物选择来源：
优先使用已存在的 `result_layers.json`，仅在旧版实验目录缺少该清单时读取
`result_all.csv`。新版布局的结果清单缺失或损坏不会自动降级到旧 CSV，
历史 CSV 可用于分析也不代表实验已完成或通过可比性审计。
TUI 的单实验入口会展开仅含一个任务的集合及其上级输出目录；补采命令使用展开后的
实验目录，统计和绘图再选择该目录中的结果文件。

`summary` 默认提供 19 列：配置 ID、model/runtime/device、CPU quota、memory cap、GPU 身份，
应用 P50/P95、samples/s、QPS、CPU/memory/GPU/VRAM peak、cold start、观测总能量、
观测 energy/request 和 status。完整长表及原始字段仍可由同一模型访问。
summary 是每个配置的派生数据，不替代主实验 CSV，也不写回原结果。

配置按 run、model、runtime、device、CPU、memory、GPU、concurrency、环境、task、
input case 和 batch 区分；未显式命名的 input case 由原记录的 input scale、类型、task 参数、
batch size 和 input units 构成。不同条件不混合求平均。仅 `status=ok`、`warmup=0` 且
非 `inferred_not_measured` 的行参与汇总；warn/error、warmup、推断和缺失值保持可追溯。
有成功及失败正式窗口的配置标为 `partial`，没有正式窗口标为 `no_formal_windows`。
`ok` 仅表示已写入的正式窗口均成功，不证明整个实验计划已完成；完成度仍由原始 run state 和 audit 验收。
CSV 重复列、损坏行、重复测量键和元数据损坏会明确报错；只读取明确给出的输入，不递归扫描备份目录。

聚合约定：

- 延迟、吞吐、IPC 等对成功窗口等权求均值；P50/P95 是各窗口分位数的均值，不能解释成
  合并全部请求后的分位数。峰值取成功窗口最大值。每项保留有效/缺失窗口数与 aggregation。
- Cold start 按[冷启动汇总规则](../profiling/lifecycle.md#冷启动汇总)去重，`n` 表示启动观测数。
  同一启动值冲突或缺失身份与已知身份混用时，该指标保留为 `lifecycle_conflict` / null。
- `observed_energy_j` 是成功窗口的 CPU package 与所选 GPU 总能量之和：
  每行 `(cpu_energy_total_j + gpu_energy_total_j) × repeat_in_window`；CPU-only 行只计 CPU package。
  GPU 行任一分量缺失、成功窗口能量或实际请求数缺失时，总量 unavailable，避免把部分和呈现为总量。
  `observed_energy_per_request_j` 对相同窗口按实际请求数加权。未包含 DRAM、启动和窗口外 idle，
  也不是墙插整机能量；总能量默认 `neutral`，因为窗口数量和持续时间会影响它。
- QPS 只接受 CSV 显式的 `qps`（request/s）。`throughput_samples_per_s` 保持 sample/s；
  不通过 batch size 或 latency 猜测并发 QPS。estimated cycles 不补入 PMU cycles/IPC。
- 长表 evidence 区分 `measured`、`derived`、`inferred_not_measured`，派生能耗保留其范围；
  数值缺失使用 null，不填 0。本视图不生成 CI，也不将窗口数称为请求样本数。

Metric Registry 在原有采集声明上增加 `label`、`group`、`direction`、`scale`、`summary`
和 `aggregation`。单位沿用原协议，Dashboard 的标题、单位、颜色方向、默认刻度与预设从同一登记表读取。
首批覆盖 Performance、Resource、Energy、Startup、CPU PMU、GPU；其余未审定字段保留 neutral。
CPU/GPU utilization 不默认判定越高或越低越好。

### 单负载点流式容量报告

OpenAI-compatible `load.json` 可生成独立 HTML，不进入正式实验的 `AnalysisModel`：

```bash
acprof report results/qwen-capacity/load.json
acprof report results/qwen-capacity/ --output results/qwen-capacity/capacity.html
```

报告展示 offered／成功／失败／丢弃计数、客户端并发、成功请求吞吐、SLO Goodput、Token/s 及
coverage／scope、TTFT／TPOT／E2E 分布和服务身份。v2 会从原始 requests、`duration_s`、声明 SLO 和
客户端并发重新计算 capacity 与完整 streaming 分布，并核对 `identity_sha256`、两份持久化摘要及
服务声明／probe；不一致即拒绝生成，实际渲染重算后的分布。早期 v2 未在 `identity.service` 重复
model／revision 时，使用哈希身份中的对应声明校验，不会跳过一致性检查。v1 可读取，新增能力显示
unavailable，不以零替代。实验与结果的 `schema_version` 必须为相同的整数 1 或 2；v2 的
`identity.service` 与 `service_identity` 必须为整数版本 1。布尔值、浮点数、缺失及未知版本均拒绝，
CLI 明确报错且不发布 HTML。CLI 只接受一个流式 load 输入，拒绝与 baseline、比较用途或
其他结果组合；默认生成 `load_report.html`，已有文件、目录或符号链接均不覆盖，并遵守离线分析锁。
该报告只有一个负载点，不生成容量曲线或最大容量结论。字段公式见[流式测量](../profiling/streaming.md)。

### 交互式配置比较报告

HTML 是统一的离线实验分析报告，在配置图表之外展示窗口统计、数据质量与实验完整性。

#### 实验结束后自动生成

`acprof run` 在实验结果提交、必需指标检查通过并保存窗口统计 JSON 后，自动生成 HTML。
TUI 启动的实验使用同一完成流程。新版实验保存到 `plots/analysis/report-<UTC时间>-<随机标识>.html`，
旧布局保存到 `analysis/`；终端打印实际路径，不自动打开浏览器。报告优先复用刚保存到根目录
`analysis/` 的有效统计 JSON，生成期间继续持有测量锁，避免与下一次正式采集重叠。

已完成实验的 `--resume` 会补生成一份新 HTML，不重跑测量、不覆盖已有报告；统计 JSON 仍按内容复用。
报告生成失败时明确报错，保留已提交结果和统计 JSON，可修复后用原参数加 `--resume` 重试，
或执行错误信息中的 `acprof report ... --output ...` 命令。无结果、必需指标缺失或采集中断时，
不会自动发布 HTML；中断实验在 Resume 完成后生成。历史及部分结果仍可使用下面的 CLI 手动分析。

完成阶段生成的设计参考 [pytest-html 的 session finish hook](https://github.com/pytest-dev/pytest-html/blob/master/src/pytest_html/basereport.py)
（MPL-2.0）；复用 AC-Prof 的分析和原子发布实现，不复制第三方代码、不新增依赖。

#### 手动生成与比较

```bash
acprof report results/<model>/
acprof report results/run-a/ results/run-b/ --output comparison.html
acprof report results/<model>/ --baseline '<config_id>' --output baseline.html
```

手动命令默认在第一个实验目录生成新的 `report.html`，已有输出拒绝覆盖。HTML 内嵌数据与 Plotly.js，
不依赖 CDN、服务器或 GUI；可复制到 Windows 用浏览器直接打开。生成过程在 Linux 遵守主机测量锁，
不能与同用户正式采集并行。打开浏览器分析也应避开正式测量窗口。
发布时仍会原子检查目标是否存在：若其他写入方抢先创建文件、目录或符号链接，生成失败并保留目标，
不发出成功回执。输出文件系统不支持这种拒绝覆盖的发布操作时明确报错，不退回覆盖写入。

报告提供三个图表视图及「窗口统计」，共用 model/task/runtime、CPU/GPU、CPU count、memory、concurrency、
input case、experiment batch 和环境筛选；点击表格行或图中点后，其他视图保留同一配置的选中状态。

- **Comparison Matrix**：默认六项 latency/throughput/memory/energy/IPC/startup；显示实际值、
  单位、状态和窗口数。列标题按数值排序，空值始终置后；可选 Summary、六个指标组或自定义指标。
  同条件内逐列着色，lower/higher 从 Registry 读取，neutral 与 unavailable 保持灰色；颜色不是跨指标总分。
  每次筛选按“比较组 × 指标”预计算一次颜色范围，单元格仅做常数时间归一化；该部分为 O(MN)，
  不因单元格逐一重扫整组。排序不改变颜色范围，筛选后按剩余符合条件的观测重新计算。
- **Baseline**：从任意配置选择，显示绝对值、`current - baseline` 和相对百分比。
  baseline 为 0 时保留绝对差、百分比 unavailable。按所选用途共用严格比较的条件规则；
  不兼容或证据不足时显示原因，保留绝对值，不计算改善比例。
  `--baseline` 接受 config ID，只有单配置 run 才能直接用 run ID。
- **Pareto / Trade-off**：四种 latency/throughput 与 energy/memory/VRAM 预设，X/Y 可选，
  点大小可映射第三指标，runtime/device 由颜色及形状区分。frontier 只比较当前用途下条件已核验
  且质量允许优选的成功配置；选择 baseline 后还需与之可比。缺失坐标和 partial/failed 不参与支配判断，重复最优点全部保留。
  neutral 轴只画 scatter，不自动指定优化方向。筛选变化后重新计算 frontier。
  二维 frontier 在各比较组内排序扫描，最坏时间复杂度为 O(N log N)，避免逐对比较。
  使用原始数值的精确大小关系，保留配置输入顺序及所有并列点，不引入近似容差。
- **Scaling**：CPU cores、memory、concurrency 可作资源轴，其他指标作 Y 轴；按 model/runtime、
  环境和输入条件分面。每条线固定其他资源、device、GPU 和 run，空值处断线；不跨配置条件连线。
  历史 CSV 没有 concurrency 时明确显示缺少该资源轴，不推断为 1。
- **窗口统计**：按配置与指标展示均值、样本标准差（ddof=1）、95% CI、有效／缺失窗口数，
  分开列出指标缺失原因和预热、warn/error、推断行等排除数量；支持指标筛选和跨视图配置选择。
  来源区展示 Bootstrap 参数、抽样单位、统计假设、JSON 复用／重新计算原因，以及当前质量、
  运行状态、实验完整性、计划覆盖和可展开的原始审计证据。

#### HTML 窗口统计与 JSON 复用

每个实验分别搜索 `plots/analysis/`（新版布局）、`analysis/` 和根目录中的 `window-statistics-*.json`，
兼容自动统计、手动统计以及时间戳和历史 UUID 文件名。
按修改时间从新到旧检查；修改时间只决定候选顺序，不证明有效性。只复用 schema v1、95% 置信水平、
既有循环块 percentile Bootstrap 口径且来源 SHA256、当前质量与完整性状态一致的报告。
还会以同一分析快照校验配置分组、均值／标准差／窗口数和区间结构；命中时不再运行 Bootstrap，
保留该 JSON 的指标选择、resamples、seed 和 block_size。不同 run、concurrency、输入或环境不混合，
旧 JSON 的分组无法唯一对应 HTML 配置时重新计算。

缺失、过期、损坏或不兼容的 JSON 在报告中保留原因，并继续寻找可用候选；没有命中时，
从本次已读取的原始测量窗口调用既有 `summarize_windows`，默认计算 `latency_app_s`、`latency_s`、
`container_attributed_energy_eff_j`，confidence=0.95、resamples=5000、seed=0、block_size=1。
候选 JSON 有 32 MiB 读取上限，超限标为不可复用；权限与存储故障明确报错。
HTML 生成器不改写原始数据或统计 JSON；实验完成流程会先保存 JSON，手动统计仍使用 `acprof stats`。
分层结果只消费 manifest 声明的 committed 数据，不读取未提交尾部；统计结束及发布 HTML 前
复核结果与已解析元数据的内容，发生变化时拒绝发布。

均值和标准差描述窗口观测，CI 描述**同一实验内**窗口均值的不确定性；有效窗口不是请求样本或独立实验。
标准差至少需要两个有效窗口；CI 至少需要 `max(3, 3 × block_size)` 个有效窗口，连续块还要求
无指标缺失、序号连续。不可用时显示空值与原因，不用零或单点区间补齐。
指标缺失数只统计正式成功窗口中的缺失；被过滤的行及计划中尚未写入的窗口分别展示。
没有可核验计划时完整性保持 unknown；`run_status=complete`、观测窗口成功或 CI 可用均不单独证明实验完整。
只有失败或预热记录的配置也保留不可用统计。历史记录缺少有效窗口身份时说明限制，不推断样本身份。

设计参考 [SciPy 的 Bootstrap 参数与结果契约](https://github.com/scipy/scipy/blob/main/scipy/stats/_resampling.py)
和 [Joblib 的缓存有效性检查](https://github.com/joblib/joblib/blob/main/joblib/memory.py)（均为 BSD-3-Clause）。
借鉴参数显式记录和复用前校验的设计，直接复用 AC-Prof 的统计、缺失原因与审计实现，
不复制第三方代码、不新增依赖；所有检查与计算位于正式测量窗口之外。

HTML 的 `comparison_profiles` 从相同 Python 规则生成，按当前配置选择物化输入与实际 workload；
浏览器只比较归一化证据和精确有理数分布，不自行放宽 task 或硬件策略。
同一次加载复用各来源的固定条件哈希、输入规模索引和 workload 归一化证据，原始输入顺序及内容
仍参与比较。资源坐标、硬件策略和 runtime threads 继续按配置与用途核对。
复用限定在本次加载，各用途的返回结果独立；后续加载会重新读取并准备来源证据。
`--comparison-purpose` 指定初始用途，报告内也可切换。没有 baseline 时按完整证据分组；选择后立即显示成对原因。
每次刷新在独立的比较上下文中核验各配置一次，摘要、所选配置详情、各指标差值和 Pareto
共享该结果；显示更多指标不会重复核验 workload。下一次刷新重新建立上下文，baseline、
用途或筛选变化后不沿用上一轮的判断，也不在配置对象上保存缓存。
这些比较用于探索已记录数据，不证明不同模型质量等价、持续硬件隔离或统计显著性。
独立实验的条件审计和 CI 继续使用[跨独立实验比较](comparisons.md#跨独立实验比较)；已有 `acprof compare` 语义保持不变。
窗口统计不改变独立 JSON / CSV 导出入口。

实现直接复用 [Plotly](https://github.com/plotly/plotly.py) 的图形与离线 bundle，参考
[HTML renderer](https://github.com/plotly/plotly.py/blob/main/plotly/io/_html.py) 和
[renderer 设计讨论 #1459](https://github.com/plotly/plotly.py/issues/1459)；Pareto 借鉴
[Optuna 的方向归一化、二维排序扫描与非支配判断](https://github.com/optuna/optuna/blob/master/optuna/study/_multi_objective.py)
及[可行性可视化讨论 #2397](https://github.com/optuna/optuna/issues/2397)，没有复制优化器或引入 Optuna。
两者使用 MIT；Plotly/Optuna 的公开仓库仍在维护。主机锁固定 Plotly 7.1.0（支持本项目 Python 3.12+），
增加 Plotly 和 narwhals，复用已有 packaging；无 Web 服务、前端构建链或优化框架依赖。
报告保留 bundle 许可注释并内嵌 [Plotly.js v4.1.1 MIT](https://github.com/plotly/plotly.js/blob/v4.1.1/LICENSE)。
Plotly 只在写报告时导入，所有处理均在采集窗口外进行，不改变 collector 的测量成本。

### 只读审计

```bash
acprof audit results/<model>/
acprof audit results/<model>/ --json
acprof audit results/<model>/ --require-complete --require-ok
acprof audit --metrics
```

审计检查 CSV 结构、唯一测量键、状态枚举、非法数值、输入计划 hash 和新实验的计划覆盖，
另核对可由同一行确定的能耗分量之和及 packet 字节关系。它不重建底层 RAPL 归因或证明全部测量准确。
`valid` 表示已执行的检查通过；`completion` 单独报告主实验是否正式完成。旧实验没有 `run_state.json`
时为 `unknown`，不会根据存在 `result_all.csv` 就宣布完成。

报告分别统计所有行、warmup、正式 `ok`、`warn` 和 `error`。正式 `ok` 行的数值缺失按证据分为
`not_recorded`（旧文件未记录）、`not_applicable`、`tool_not_enabled`、`profiler_reported_error`
或 `unavailable_unspecified`；后者表示现有产物无法确定原因。数值 `0` 保留为有效观测，
不会被算作缺失，合法的负 effective 能耗也不会自动被判错。审计不会修改结果、计划或元数据。

同一次审计复用从该来源元数据校验得到的平台身份，并按需解析一次 profiler 启用配置。
Runtime、设备、状态和 profiler 错误只在各行内复用，缺失原因仍按原有优先顺序判断。
每次调用重新读取来源元数据，历史 unknown 和不一致的身份仍保持 unknown。
公式核对仅转换参与公式的字段，逐项非法数值检查仍覆盖全部数值指标。比较中的指标可用性
要求正式 `ok` 行中至少存在一个有限数值；已知缺失标记直接判为不可用，其余值仍进行数值解析。

`--require-complete` 要求正式完成状态及完整计划；`--require-ok` 要求所有非 warmup 行为 `ok`
且至少有一行。默认允许审计历史或失败实验，但结构/一致性错误仍以退出码 1 表示。

跨后端比较沿用同一只读入口：

```bash
acprof audit results/<left-model>/ --compare results/<right-model>/ --json
# 条件不完整或不一致时要求非零退出：
acprof audit results/<left-model>/ --compare results/<right-model>/ --require-comparable
```

比较分别返回 `compatible`、`incompatible`、`unknown`，检查任务/场景、物化输入内容与顺序、
资源设置、已记录的有效线程数、测量口径、质量约束和 actual workload。它允许预期中的 backend、
环境、镜像、模型与制品身份差异，不要求整份输入计划或配置 hash 相同。`run_id`、续跑身份和
恢复锁仍按原规则严格匹配，运行后 actual 不进入执行前身份。
比较报告 schema v2 使用 `--comparison-purpose same-hardware`（默认）检查同机 CPU/GPU、
实际 affinity、线程数、驱动与功耗策略；`cross-hardware` 将已知硬件差异记录为
`expected_difference`，继续检查输入、线程与测量协议。示例：

```bash
acprof audit results/left --compare results/right \
  --comparison-purpose cross-hardware --require-comparable --json
```

`resource-scaling` 明确允许 CPU 和内存配额不同，报告写入
`allowed_resource_dimensions=["cpu","memory"]`；GPU 模式/身份、batch、实际输入输出分布、
有效线程、affinity、电源策略、环境与测量协议仍严格检查。它是描述性资源对比，CPU 与内存同时
变化不能归因到单独一个轴。不同资源坐标只在该用途下投影，不能绕过实际 workload 检查；
一个完整矩阵内不同资源 case 的 workload 不一致时，整体资格保留 unknown，HTML 每配置检查仍能定位差异。

硬件证据来自每个正式 case 开始前的 `hardware_conditions.json`，字段、范围与未知值见
[硬件条件证据](../profiling/lifecycle.md#硬件条件证据)。缺失值始终为 `unknown`；
跨硬件模式不会把未记录条件视为预期差异。实际正式服务的线程数优先于独立 probe 近似值；
只有旧 probe 证据时仍需显式正整数线程请求，并保留硬件证据未知的限制。
`quality_constraints` 必须由输入计划显式记录。新计划记录当前采集要求的
`sanity_only_not_accuracy_benchmark` 策略及 `protocol` / `task` 均须为 `verified` 的门槛；
独立 runtime validation 的成功回执保存同一份约束。相同策略允许在其他条件也一致时比较性能，
不表示执行了准确率评测或证明不同模型质量等价。历史计划缺少该字段时仍为 `unknown`，
不会按当前默认策略补齐；恢复继续复用原计划。条件一致也不保证整个运行期间独占硬件
或热状态不变。比较不修改 run identity，不改变恢复校验。
actual workload 的 `variants` 计数必须覆盖 `request_count`，已有 `repeat_in_window` 时还需一致；
显式失败案例的终态与成功状态分别显示，不将 OOM/timeout 归为未开始或成功。
