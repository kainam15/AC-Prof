# CLI 辅助与分析命令

[← 返回专题目录](cli.md)

## `acprof probe`
`--revision` 接受 branch、tag 或完整 commit SHA；TUI 的模型候选、采集前自动解析和最大输入探测共用当前填写的 revision。

复用 `--model`、`--task`、`--task-family`、`--backend`、`--model-spec`、`--batch-size`、`--workload-spec`、
`--output-dir` 和 `--skip-build` 的参数及默认值。
资源列表与超时的用途如下：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--cpus` | `1,2,4,8` | 只选择列表中的最小 CPU。 |
| `--mems` | `2,4,8,16` | 从低到高实测的候选内存上限。 |
| `--gpus` | `off,on` | 包含 `off` 时优先 CPU-only，否则使用 `on`。 |
| `--input-scales` | 自动规划 | 只探测已确定尺度中的最大值。 |
| `--timeout-seconds` | 不设超时 | 单次探测请求的等待上限；显式值必须有限且大于 0。 |

`acprof probe` 不接收 `acprof run` 的 `--request-timeout-seconds`、warmup/repeat、能耗采样或 profiler 参数。
详细用法见[先探测最大输入](experiments.md#先探测最大输入)。

## `acprof profile`
位置参数 `result_dir` 是已完成的模型结果目录。操作和恢复规则见
[补采说明](../profiling/profilers.md#补采已有结果)。

TUI 使用四项复选框选择补采工具（初始勾选 `torch`、`ncu`），将勾选结果传给
`--tools`；未勾选任何工具时不启动补采。工具适用范围、采样策略和指标口径与 CLI 相同。

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--tools` | `torch,ncu,nsys,massif` | 选择需要补齐的工具，逗号分隔。 |
| `--dry-run` | 关闭 | 验证已有文件并展示计划，不启动分析器或改写结果。 |
| `--force-reprofile` | 关闭 | 强制重采并替换所选工具已经成功的值。 |
| `--massif-sampling` | `per-scale` | 可选 `per-scale`、`full`。 |
| `--nsys-sampling` | `per-cpu-scale` | 可选 `per-cpu-scale`、`per-scale`、`full`。 |
| `--massif-reference-cpu` / `--massif-reference-mem` | 结果矩阵最大值 | 在已有 CPU-only 资源配置中选择代表值。 |
| `--nsys-reference-cpu` / `--nsys-reference-mem` | 结果矩阵最大值 | 在已有 GPU 资源配置中选择代表值；`per-cpu-scale` 只用代表内存。 |
| `--torch-profiler-repeat` / `--torch-repeat` | `1` | 同一参数的两个名称；控制 Torch probe 内推理次数。 |
| `--ncu-repeat` / `--nsys-repeat` / `--massif-repeat` | `1` | 对应工具的 probe 内推理次数，归一化口径同 `acprof run`。 |
| `--ncu-root` / `--nsys-root` | 自动检测 | Host 工具安装目录或可执行文件。 |
| `--compute-profile-cpus` / `--compute-profile-mem` | host 逻辑 CPU / 75% host memory | 临时 compute profiler 的 CPU/内存上限，内存单位 GB。 |

## 其他入口
`acprof results verify <分层目录>` 校验正式结果的 SHA256、指标字段归属、行数和测量身份；
`acprof results export <分层目录> <新文件.csv>` 按需重建完整宽表，不允许覆盖已有文件。
新实验不会默认写出 `result_all.csv`；Posthoc 成功后仅更新发生变化的模块。旧宽表实验
不再提供默认兼容或续跑。详见[分层结果 CSV](../results/metrics.md#分层结果-csv唯一正式结果)。

`acprof report <实验目录或 CSV> [更多输入 ...]` 生成 Comparison Matrix、Pareto 与 Scaling 的离线 HTML。
`--output <新文件.html>` 指定输出，默认首个实验目录下 `report.html`，拒绝覆盖已有文件；
`--baseline <config_id>` 预选配置，单配置 run 也可直接使用 run ID。无需 GUI 或 Web 服务，
不递归扫描输入目录中的备份，源 CSV 保持不变。详细语义见[交互式配置比较报告](../results/reports.md#交互式配置比较报告)。

`acprof plot` 接收结果 CSV 路径，`acprof tui` 可用 `--model` 预填模型、用 `--preset` 选择预设。
`acprof audit <目录或 CSV>` 只读校验结果；`--json` 输出报告，`--require-complete --require-ok`
用于验收新实验。`acprof stats <目录或 CSV>` 按测量窗口计算置信区间，支持重复 `--metric`、
`--confidence`、`--resamples`、`--seed`、`--block-size`。可选 `--precision-target 0.05`
按既有区间评估 5% 的相对半宽目标，输出 met/not_met/not_assessable；默认不评估，
不改变原有统计字段或采集行为。定义与不可评估条件见[结果分析](../results/comparisons.md#窗口置信区间与开销对照)。
省略输出选项时向 stdout 输出报告 JSON。`--output FILE` 保存到指定新文件，禁止覆盖；
`--output-dir DIR` 在指定目录中比较完整 JSON 内容，相同则复用已有文件，否则以本地日期时间
`window-statistics-YYYYMMDD-HHMMSS-ffffff.json` 保存。两个输出选项互斥。
目录模式向 stdout 输出一行 `ACPROF_STATS {"report_path": "绝对路径", "reused": false}`；复用时 `reused` 为 `true`。
TUI“统计报告”页的“计算统计”使用目录模式和默认统计参数，报告位于 v2 结果目录的 `plots/analysis/`（旧目录为 `analysis/`），复用时提示已有报告并显示其内容。
`/stats [csv/dir]` 与按钮等价；`/report [json]` 或“查看报告”读取已有窗口统计、监测开销或 CLI/TUI 对照报告。
这些操作需要 TUI 空闲；开销实验仍通过独立脚本显式运行。报告展示与路径带入方式见 [TUI 说明](tui-pages.md#统计报告)。
`/images` 打开“镜像管理”页并自动读取数据，空闲时每轮读取完成后 5 秒更新；离开页面或运行任务时暂停。
默认视图为镜像树，可切换到镜像列表或层共享。
树中 `←/→` 折叠/展开、空格勾选；列表表头点击排序，再次点击反向。筛选支持模型、逻辑环境名、标签、ID 和平台。
镜像列表优先显示逻辑名称和完整/新增大小，右侧保留 `Repository` 和 `Tag`；无标签镜像显示短 image ID，`Tag` 为 `—`。
“选择同模型”勾选模型镜像；“删除所选”确认全部标签及按集合去重的释放估算。层共享视图只浏览层，仍可清除或删除先前勾选的镜像。
视图、筛选、排序、折叠、勾选与手动调整的详情高度只在本次会话保留，不写入设置文件；自动刷新保留仍有效的勾选和浏览位置。
详情高度通过列表下方的分隔条调整；窗口缩小时限制显示高度，放大后恢复手动值，`Home` 恢复默认高度。
镜像消失、标签变化或新增容器引用时取消对应勾选，Docker 环境变化时清空勾选。
关系证据和空间口径见[镜像管理与清理](../models/images.md#镜像管理与清理)。
各入口的完整帮助可直接运行：

```bash
acprof run --help
acprof probe --help
acprof profile --help
acprof plot --help
acprof audit --help
acprof stats --help
acprof tui --help
```

## 独立比较与负载
`acprof compare --left <实验> --right <实验>` 支持重复指定两侧独立实验，输出差值、比值和跨实验区间；
参数与统计假设见[跨独立实验比较](../results/comparisons.md#跨独立实验比较)。
`--purpose` 支持 `same-hardware`、`cross-hardware`、`resource-scaling`；组内重复始终按相同硬件核验。
`audit --compare` 和 `report` 使用同名取值的 `--comparison-purpose`；HTML 可在生成后切换用途。
资源扩容只放开 CPU/内存配额，双方资源坐标和仍需匹配的条件见[比较规则](../results/comparisons.md#跨独立实验比较)。
TUI“统计报告 → 独立实验比较”复用同一入口，可选择左右组、基线和比较用途，或直接读取 CLI 输出的 JSON。
`acprof load <源实验> --gpu off --scenario concurrent --concurrency 4 --output-dir <新目录>`
执行独立 HTTP 负载；到达率使用 `--scenario arrival-rate --rate 10 --arrival poisson`。
协议、连接复用前提和失败口径见[独立非流式负载](../profiling/lifecycle.md#独立非流式负载)。
