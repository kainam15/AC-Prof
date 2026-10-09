# 统计分析与跨实验比较

[← 返回专题目录](metrics.md)

## 窗口置信区间与开销对照
```bash
acprof stats results/<model>/ --metric latency_app_s --metric latency_s \
  --confidence 0.95 --resamples 5000 --seed 0 --output internal-testing/window-statistics.json
```

每个资源配置和尺度分别汇总正式 `ok` 窗口，窗口均值等权；`repeat_in_window=1000` 的一行
仍只贡献一个统计单位。报告包含有效/缺失窗口数、均值、样本标准差和 percentile bootstrap 区间。
少于 3 个有效窗口时保留均值，区间为 `null`；少量窗口的区间本身也不稳定。
默认假设窗口之间可视为独立。存在连续时间相关性时可设置 `--block-size` 使用循环移动块，
至少需要 3 倍块长的窗口，缺失或不连续的 repeat 不组成块。它不能排除温度、主机负载等系统性偏差，
也不是跨机器或跨实验的一般置信保证。冷启动、cgroup 生命周期峰值及独立 profiler 的复用值会被拒绝。

可选 `--precision-target 0.05` 评估预先选择的相对区间宽度目标。参数为有限正比例，
`0.05` 表示 `(ci_high - ci_low) / (2 * mean) ≤ 5%`；非对称区间的半宽并不等于
端点到均值的最大距离。它沿用当前 confidence、resamples、seed 和 block-size，按资源、
输入尺度、环境和指标分别评估；窗口内增加请求数不会增加独立统计单位。

启用后 schema v1 报告顶层新增 `precision`，保存目标、公式、比例单位和
`scope=within_run_observed_windows`；每组新增 `precision_status`（`met`、
`not_met`、`not_assessable`）、`relative_ci_half_width`、`precision_reason` 和
`precision_excluded_windows`。窗口不足、指标缺失、已有正式 warn/error 行、窗口序号间断、
负观测、非正均值或区间无效时不可评估，比率为 `null`。重采样少于两次，或非恒定观测
得到零宽区间时也不可评估；保留原始均值和区间。排除的正式行单独计数，不改变既有
`n_windows`、`missing_windows` 的含义。不传参数时不新增精度字段。

`met` 只表示已观察窗口在既有 bootstrap 假设下满足用户的区间宽度目标，不证明实验完整、
独立 run 数量足够或论文证据充分。CSV 无法推断所有尚未写入的计划首尾缺口，完整性仍由
`audit` 验收；此诊断不提供自动补采、自动停止或按结果反复加样直至通过的流程。

采集开销验证使用两个独立入口：

- `scripts/measure_overhead.py <完成的实验目录> --gpu off --output-dir <新目录>`：复用原 image ID、
  输入计划和资源，随机化每轮未启用 monitors 与 5/20/100 Hz 监测线程的顺序，按同轮请求均值计算相对变化。
  默认取源实验最大的 CPU/内存配额与输入计划第一项；`--cpu`、`--mem`（GiB）和
  `--input-scale` 可选择源实验已记录的其他坐标，输入尺度必须唯一匹配。使用原始 payload，
  不重新生成输入；不存在的坐标在启动容器前拒绝，所选坐标和 payload SHA256 写入报告。
  它记录 RAPL/NVML/cgroup monitor 的采样成功状态；不运行 PCAP、perf 或 TUI，也不生成正式能耗 CSV。
  添加 `--modes none,basic,full --sample-hz 20` 可比较完整采集器组合：none 无采集器，basic 仅容器
  CPU/memory，full 使用现有 PCAP、perf、RAPL、NVML 和容器资源采集器，并复用主 client 的无请求对照。
  三组使用同一常驻服务、物化 payload、请求数、源实验线程／设备设置和串行 `/predict` 完成边界；
  共享 `ExecutionConditions` 固定源 GPU UUID、CPU 绑定及请求超时，清除调用者的设备／线程覆盖，
  退出时恢复环境。源条件不能满足时拒绝诊断；硬件观测与恢复的条件一并保存在输出目录。
  HTTP 返回前服务端已完成 runtime completion hook。未完成或错误响应使诊断失败，不记录成功时延。
  这些是内部诊断场景，未新增正式 profiling mode，不输出正式画像 CSV；结果只比较请求窗口，
  不包含容器启动、离线合并、profiler 或 TUI 成本。full 对照要求已有抓包权限，不修改系统权限。
  full 在请求计时结束后沿用生产采集的抓包等待落盘步骤，保存 PCAP 和解析结果；请求覆盖不足时
  诊断失败，该轮不能进入开销汇总。
- `scripts/compare_ui.py <check_hardware 的 command.json> --output-dir <新目录>`：单 case 下随机化
  CLI/TUI 的配对顺序，每次运行完整正式协议并通过审计，镜像/revision/输入 hash 必须一致。
  默认 `--ui headless` 只测试 TUI 调度与日志路径；`--ui terminal` 用于实际终端绘制对照。

两者默认 5 轮监测器对照 / 3 对 UI 实验，输出所有原始轮次和配对均值变化的 95% 区间。
采集器对照还报告绝对时延差、各组轮次均值的样本标准差和配对差的标准差。每轮内多个请求
只形成一个窗口均值，不能把请求数当成独立重复次数。
负值表示该次对照中更快，不能直接解释成监测器提升了推理性能；区间跨零时没有检测到稳定方向。
所有报告均在窗口结束后写出，新目录保护失败和中断证据。正式采集的 monitor 生命周期保持原协议。

开销入口可显式添加 `--window-boundaries`。每个比较请求窗口单独保存
`overhead-<round>-<scenario>.boundaries.json`，不包含预热或无请求对照。
该 schema v1 诊断记录相对于首个诊断事件的秒数、请求起止和完成数、各采集器
start/stop/close 调用耗时，以及它们已有的逻辑采样边界；`window_id` 与报告轮次对应。
全部 stop 尝试后、close 前复制内存时间戳，全部 close 尝试后才写文件；失败和取消窗口也保留证据。
诊断文件写出失败会使本来成功的诊断失败；已有请求或收尾异常时保留该异常并另报写出错误。

缺少已有边界时记录 `unknown`，不推算精确 counter read 时刻。CPU 的结束时间戳先于 join
和最后一次 counter read，单凭边界偏移不能断言能量偏差显著。诊断自身有额外计时和记录开销，
其 `successful` 只反映请求、生命周期和快照状态，不代表硬件准确度通过。
默认关闭时不增加计时、快照或文件，正式 CSV、采集器启停顺序与能耗公式保持原义。

TUI 的“统计报告”页调用同一个 `acprof stats`，默认分析应用延迟、抓包延迟和容器归因有效能耗，
使用 95% 区间、5000 次重采样、seed=0、block-size=1。点击计算后显示表格：完整报告内容相同则复用已有 JSON 并提示，否则按日期时间保存新文件；
需要其他指标或连续块参数时，可先用 CLI 生成报告，再在 TUI 打开。
表格将秒转换为 ms，能耗保留报告中的单位；真实零、无有效窗口、窗口不足和连续块中断分别显示。
统计只描述已写入的正式成功窗口，不能代替 `acprof audit --require-complete --require-ok` 对实验完成度的验收。

该页还可直接读取上述两种开销工具的成功报告，显示配对轮数、延迟变化和区间；headless 与 terminal
的测量范围分别注明。未完成、失败、损坏或未知版本的报告显示错误，不保留上一份结果冒充新报告。
读取只展示报告记录的实验，不重新测量或核验源 CSV 的当前版本；比较时须使用同一实验口径。
操作步骤见 [TUI 使用说明](../usage/tui-pages.md#统计报告)。

## 跨独立实验比较
```bash
acprof compare --left results/a1/model --left results/a2/model --left results/a3/model \
  --right results/b1/model --right results/b2/model --right results/b3/model \
  --metric latency_app_s --metric container_attributed_energy_eff_j \
  --output internal-testing/independent-comparison.json
```

两侧分别代表同一实验定义的多次独立重启。每份源文件需要唯一 `run_id`；复制目录不能增加
重复次数，同组中的模型／镜像／源码身份变化会阻止比较。使用现有条件审计匹配资源矩阵、
物化输入、实际 Workload Contract、线程和硬件条件；`--purpose cross-hardware` 允许已知的硬件差异。
不兼容或条件未知时仍输出审计、失败／缺失统计，但差值、比值与区间为 `null`。

实际 workload 按资源和输入尺度累计每个 variant 的请求计数，以精确约分后的比例比较分布。
`99:1` 与 `1:99` 不等价；`99:1` 与 `990:10` 分布相同，仅 `request_counts_changed=true`。
单一 workload 的自动窗口执行次数不同仍可比较；请求总数不充当单请求工作量。
`conditions.actual_workload` 保留两侧全部计数、比例、`reason` 和 `changed_dimensions` 的输入／输出维度分布。
固定输出定义的 tabular／time-series 任务发生输出分布变化时不兼容；生成、检测或未声明固定输出的任务
若只有输出观测变化，则为 `unknown` / `output_distribution_equivalence_unverified`，不假设随机差异无害，
也不引入任意统计容差。输入、请求上限、任务或场景变化仍不兼容。
部分输出维度未知时仍保留全部 variant 及计数，并报告可观察的输入差异；相同的未知值不构成等价证据。

每个资源与尺度先对单个实验的正式 `status=ok,warmup=0` 窗口均值等权平均，再对各实验均值
等权平均。`repeat_in_window` 不作为权重。差值定义为右组减左组，单位与原指标一致；
比值为右组除左组，无单位，左组均值不大于零时不可用。95% percentile bootstrap 在左右两组
分别重采样独立实验均值，默认 5000 次、seed=0。任一组少于三个有效独立实验时不生成区间；
它不声称请求独立、相同热状态、连续独占硬件或模型质量合格。

报告 `kind=independent_experiment_comparison`、schema v1，逐项包含有效／请求的实验数、缺失实验、
失败与缺失窗口、各次均值、标准差、差值／比值及区间；保留完整条件检查和源文件 SHA256。
统计期间源产物变化会拒绝结果。只对成功窗口的估计可能存在选择偏差，失败计数必须一起报告。
即使组间使用 `cross-hardware` 或 `resource-scaling`，每组内部也必须通过 `same-hardware`
复验；`condition_checks` 用 `group`、`within_group` 标识组内问题。
独立 `resource-scaling` 按 device/input 配对，每次实验在每个 device/input 下只能有一个资源配置，
否则明确拒绝，避免把多个配置当成独立样本。`resource_coordinates.left/right` 保留双方 CPU/内存，
原 group 顶层坐标仍为左侧 baseline；每个 run 保留自己的坐标。
质量证据单独贯通 `audit`、`compare`、`stats`、配置摘要、TUI 和 HTML：
`run_status` 是原运行状态，`measurement_status` 根据计划覆盖和正式成功窗口分为 complete/incomplete/unknown；
这里的 complete 仅表示这些窗口完整，不代表所有硬件指标可用，缺失指标及原因继续保留。
比较报告的 `status` / 每组 `comparability` 仍只表示条件可比性，不由质量告警覆盖。
`quality_status` 为 passed（明确记录的检查为空）、warning（已解释的未使用 checkpoint 权重）、
blocked（weights_reinitialized 或 error）、unknown（历史缺证据、损坏或尚未解释的 warning）。
passed 不是模型准确率合格证明；warning 保留 detail、observed、threshold 和原始 source/artifact。
`quality_checks` 保存全部原证据；`quality_reasons` 解释 `auto_selection_eligible` 的决定。
blocked/unknown 暂停默认自动优选，但数值观测和描述统计仍保留；独立比较保留每个实验、每侧和每个 run 的证据，
顶层 `quality.left/right` 提供聚合质量状态。读到部分历史缺证据时整组继续 unknown，不能由其他成功 run 覆盖。
`quality_checks.json` 与回退来源 `runtime_validation.json` 纳入比较的源 SHA256，一旦变化便拒绝报告。

该入口读取正式实验；独立的 `load.json` 不可作为 `result_all.csv` 输入混入统计。

## 非流式负载报告
`acprof load` 输出 `load.json`（`nonstream_load_experiment`，schema v1），内嵌
`nonstream_load_result`。协议与命令见[独立负载协议](../profiling/lifecycle.md#独立非流式负载)。
请求时间戳均为客户端 `perf_counter` 相对实验起点的秒数：`planned_s` 是计划提交时间，
`sent_s` 是调用 HTTP 发送前的时间，`completed_s` 是完整读取响应后的时间，均不是网卡时间戳。
`latency_s=completed_s-sent_s`；`scheduled_latency_s=completed_s-planned_s` 包含调度与排队等待；
`queue_delay_s=sent_s-planned_s`。各自输出均值及 P50/P95/P99，使用线性插值。
前两种分布仅包含成功响应，排队分布包含全部已发送请求。丢弃／未发送请求时间为 `null`。

计数分别记录 offered、sent、succeeded、failed、dropped、cancelled；成功率以 offered 为分母。
发送、完成响应及成功吞吐分别除以从计划起点到全部任务收尾的时长，单位 requests/s；
目标请求率单独保留，不能用成功吞吐替代目标到达率。HTTP 错误、JSON 错误和超时不计入成功时延。
连接模式、并发数、调度、随机种子、超时、队列上限、请求数、预热数、抓包与源设备条件都进入身份。
服务返回的 Workload Contract 描述单请求工作量；真实客户端调度以报告的 `protocol` 为准。
本报告不采集能耗，不输出 token 首响应或间隔，也不与正式串行关闭连接的 CSV 合并。

HTML 默认显示全部观测；“质量范围”可筛选满足质量要求的候选，“质量状态”可单独筛选。
配置行分开展示运行、完整性、可比性和质量状态；选中行后可展开完整原始质量证据。
矩阵优劣配色与 Pareto frontier 排除 blocked/unknown 候选，不隐藏其原始数值。
HTML 条件资格与严格比较一致，数值变化仍是描述性汇总；跨独立实验区间使用 `acprof compare`。
