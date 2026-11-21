# Docker / Process 统一画像契约

[← 指标与分析](metrics.md) · [逐字段覆盖矩阵](runtime-coverage.md)

## 审计基线与数据流

`results/metrics.py` 是字段、单位、窗口、聚合及层归属的唯一登记依据。
`results/coverage.py` 补充来源的实现路径、范围、条件和成本，不另存字段表。
`scripts/render_runtime_coverage.py --check` 检查矩阵漂移。229 字段审计基线为
`ecefdcaf`（2026-11-17）；后续矩阵反映当前代码，登记数量不能等同于硬件能力。

两个 Runtime 都经 `host/client.py` → `measurement/execution.py` → `measurement/metrics.py`
生成窗口行；`measurement/publication.py` 写请求 journal、telemetry 与 receipt，
`results/` 写分层 CSV。PCAP 在窗口之外解析回填；独立 Profiler 经其 plan 和补采流程写出。
原始证据和 receipt 为恢复依据，CSV 保持投影语义。静态元数据在准备阶段采集。

本轮基线 `ea719e76` 登记 250 个字段，两个 Runtime 共有实现路径的字段为 82 个。本轮后的 253 个登记字段中，两端共有实现路径的字段为 153 个；其中包含诊断与协议字段。Docker 与 Process 均从同批请求获得应用层 P50/P90/P95、
吞吐和正文大小。窗口的 `throughput_app_samples_per_s` 始终以应用延迟为分母；
历史 `throughput_samples_per_s` 仍可在 PCAP 回填时采用包延迟。冻结结果按原字段集恢复。

## 共同概念与统计范围

P1 已接入共同进程树采集。Docker 将观察嵌入既有 resource 线程，复用已解析的容器 PID；
Process 使用原有线程。CPU user/system 与 RSS 同时读取，I/O、缺页、Swap、线程/FD 数只在首尾读取。
累计边界指标要求全窗口观察到的成员身份稳定；变化或回退时保留缺失及原因。PID 完全替换且
没有同身份计数对的新窗口不再写 CPU 零值。`process_reduction_version=2` 标识新证据，旧 raw 的 v1 计算保持。
`process_leader_*_ctx_switches` 为各进程主线程之和；不是所有线程总和。`process_swap_bytes_end`
读取 Linux `VmSwap`，不包含 swapped shmem。`process_rss_growth_bytes` 允许负数，不截零。

Process 独立、已验证 cgroup 的 CPU 时间、内存平均/采样峰值、块 I/O 由 `service_cgroup_*` 输出；
复用 Docker 的 cgroup readers，不写任何限制或委派文件。路径 inode 和根 PID 成员身份在边界复核。
继承父级限制时状态为 unsupported。Docker `container_*` 原义与字段顺序保持。

NVML 查询与聚合已经复用。Process 增加显存比例、SM/显存频率、P-state、温度；
利用率/显存采用请求的 sample rate。自动计划只读取预检可用的 NVML 查询，两种 Runtime 的慢变状态最多每秒读取一次。历史 Docker Full 使用原采样频率。
缺查询支持不抹掉其它有效读数；有缺口的已知值标 partial，不能声称完整验证。
不同状态采样频率不能直接当作等口径比较。

`resource_metric_quality` 使用 `{"schema_version":1,"metrics":{...},"collectors":{...}}` 保存逐指标
`source/scope/window/quality/derivation/reason` 及采样口径；原始 telemetry 保留独立证据。
它和新增数值字段进入 resources 层，读取端兼容早期无 wrapper 的映射。缺字段的历史行保持 unknown。
逐指标记录缺失时，审计、比较和离线 CSV 导入读取对应采集器的窗口状态。例如，
`client_http` 记录 `partial` 时，应用延迟保留原值并排除在汇总和差值之外。
该采集器的状态条目若为 `null`、数组或字符串，其指标报告 `error`，其他采集器的有效指标仍可使用。
嵌套 process/cgroup 的 raw 保留数量参与精确重算检查，截断不宣称可精确重算。
Process 样本数在自己的 raw retention 中，新的 Process 行不再借用来源为 cgroup 的 `resource_usage_iters`。

准备阶段新增 `static_meta.json.cpu_topology`，记录可见逻辑 CPU、物理核心、online CPU、
去重共享 cache 和 NUMA CPU 列表，字段缺失为 null 并保留错误；它不是服务独占资源证明。
CPU 型号、GPU 型号/驱动/PCI bus ID、内存和每 case 的限制复用已有 metadata / hardware_conditions，
不在采样线程重复探测。USS/PSS 仍未默认读取。

| 概念 | 共同值 / 语义 | 必须独立保留 |
| --- | --- | --- |
| 延迟、吞吐 | `latency_app_*`、成功请求与既有输入规模分母 | `latency_s` 始终为 PCAP；窗口分位数均值不是总体分位数 |
| CPU | 观察进程树的 CPU 秒数；一个逻辑核为 100% | cgroup 占用率按配额核心归一化；host frequency 不是进程实际 cycles |
| 内存 | 进程树 RSS 之和、时间加权平均、采样峰值 | cgroup memory 包含缓存/内核；RSS 可能重复计共享页；生命周期峰值不是窗口峰值 |
| I/O | 进程读写字节与 syscall 次数首尾差分 | cgroup 块设备 I/O 与 syscall 次数不同；`read_chars` 不替代磁盘字节 |
| GPU | 选定 UUID 的整卡 NVML 指标 | 不能标记为模型独占；MIG/驱动不支持逐项缺失 |
| 网络 | PreparedRequest 请求正文和解码后的 Response 正文，各按窗口取均值 | TCP payload、抓包 frame bytes、物理线上字节各自保留 |
| 能耗 | 包/DRAM/整卡来源一致时同名展示 | 总能量、matched-control 残差、CPU 时间归因估算不得混用 |
| Profiler | 独立工具、来源、版本与窗口 | 独立 forward / 生命周期数据不能写成正式请求窗口实测 |

## 缺失、质量与回退

每项证据携带 `source`、`scope`、`window`、`quality`、`derivation`，保留原始采样引用。
静态契约说明默认口径，运行证据说明实际读数。零只来自有效零计数或合法差值；
CSV 未知为 `nan`，JSON 未知为 `null`，不填充历史未采字段。

`quality` 区分 `complete`、`partial`、`unsupported`、`permission_denied`、
`not_applicable`、`tool_missing`、`not_requested`、`error`、`unavailable`；`complete` 只表示声明范围的窗口证据完整，
不宣称硬件准确度验收。`derivation` 区分 `observed`、`recorded`、`derived`、`estimated`。
进程身份使用 PID + start time；成员退出、计数器下降、权限缺失、瞬时子进程遗漏
须保留限制。无法证明完整的累计计数不能写零；独立成立的 RSS 或 GPU 数据继续保留。

回退只允许相同语义，例如同一 GPU 累计能量与功率积分，并保留选择与原因；
RSS ↔ cgroup memory、HTTP bytes ↔ PCAP、设备能量 ↔ 模型归因均不构成回退。
不为得到数值改权限、重跑模型或开启额外 Profiler。必需指标失败继续阻断，
可选运行错误保留诊断，不能标为无诊断成功。历史缺少新证据时保持 unknown。

## 开销与高级接入边界

无额外采样：请求结果派生、准备阶段静态信息。低开销候选：首尾计数器、轻量
procfs/psutil 读取；周期 CPU/GPU 遥测按中等开销管理；PCAP 与深度 Profiler 按高开销管理。
这是设计分级，**不是实测结论**。新实验按项启用，历史 Basic / Full 使用原协议恢复。详见[采集器契约](../profiling/collector-capabilities.md)。
USS/PSS 遍历映射且权限更严格，不默认高频读取；FD、Swap、缺页和 I/O 优先窗口边界。
温度、频率、P-state 不要求跟随高频利用率查询。

Process cgroup 只能使用经验证的独立服务 cgroup，共享父级不能归因到服务。
读取权限与限制权限分别判断，不把可读目录误认为可写委派。Process 能力驱动的能耗、
PID/thread perf、PCAP 及 Torch/NCU/Nsys/Massif 必须处理准入、身份、清理及补采来源，
不能只移除 basic 限制。PMU 保留线程继承、退出子进程、multiplex running ratio 和实际窗口；
CPU 时间能耗归因必须标记为估算。
当前 Docker perf 使用容器根 PID 与任务继承，不是 cgroup PMU；覆盖矩阵按实际命令标记。

开销验收复用 `scripts/measure_overhead.py`：同硬件、model revision、输入和资源条件，
按 round 配对 none / enabled，记录延迟、采集进程 CPU/RSS、采样丢失/迟到、窗口偏移及存储成本。
未经明确授权不执行真实配对实验；Mock 只验证计算和边界。跨 Runtime 比较须逐指标检查
来源、scope、window、derivation、质量及已有 workload/hardware/输入身份。跨 Runtime
描述性差值还要求 Docker 容器 cgroup 的实读配额与可见父级上限吻合，并且 Process 已核实服务自身配额。
历史数据缺少任一侧配额证据时保留各自的原始数值。
报告的 CPU、Memory、I/O、GPU 分组展示共同字段，表头/数值提示与“所选配置的质量证据”
展示范围、窗口、采样和逐项质量。已知不完整读数保留原始值，不进入汇总、统计区间或差值；
历史没有质量证据时沿用原统计筛选，同时明确 unknown。独立实验比较增加指标口径校验，
Runtime、模型、输入和硬件的既有条件检查仍然生效；同硬件、固定模型和推理配置、匹配的工作负载及资源控制均有证据时，共同字段可单独显示描述性差值。独立实验推断和自动排序仍核对 Runtime。

## 参考与复用决定

- [psutil 7.2.2](https://github.com/giampaolo/psutil/blob/release-7.2.2/docs/index.rst)：沿用已锁定依赖、`oneshot()` 和已有查询 API，不升级 master 或引入服务。
- [process-exporter](https://github.com/ncabatoff/process-exporter/blob/e52ab0a1c03d6baf8ed60154c254ecc0008549f3/proc/read.go)：借鉴 PID/starttime、逐项失败、累计计数分离，不复制 exporter 架构。
- [cAdvisor](https://github.com/google/cadvisor/blob/98929c4f2a2e7ba06f72fd6838bf77731dd2c288/docs/storage/prometheus.md)：保留 cgroup、进程与设备范围。
- [OpenTelemetry process metrics](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/system/process-metrics.md)：参考 CPU、I/O、线程与上下文切换分类；开发中的规范不作为重命名历史字段的理由。
- [psutil Issue #1210](https://github.com/giampaolo/psutil/issues/1210)：计数器可能回退，须记录异常，不能把负增量截成合法零值。

以上只借鉴思想并复用项目已有依赖。硬件支持、真实开销、能耗精度及 Docker Full 验收须另附同版本证据。
