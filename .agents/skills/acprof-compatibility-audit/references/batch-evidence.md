# 批量兼容性证据

## 复用现有入口

先核对当前 [CLI 契约](../../../../docs/CLI_Reference.md#acprof-coverage)和 `--help`。以下示例从仓库根目录使用已有 `.venv` 执行，无需先安装 console script；输出目录换成本次独立目录。已安装的 `acprof` 与 `python -m acprof` 使用相同命令分发。

```bash
.venv/bin/python -m acprof inspect MODEL --revision FULL_SHA --explain --output-dir internal-testing/compatibility/inspect
.venv/bin/python -m acprof coverage run examples/coverage/regression.json --output-dir internal-testing/compatibility/static
```

`MODEL` 与 `FULL_SHA` 来自冻结样本。已有 `coverage snapshot/run` 能处理指定 task/library 的下载量样本和静态/Probe 证据，优先复用；全 task 排名应先保存任务全集及查询条件，不把几个 stratum 的样本描述为全 Hub。

完整 profiling 使用 `acprof auto` 或项目正式入口；显式指定资源、输入尺度、repeat、模式与单次预算。Probe 的 full 是一次实际推理，不能用它代替 full profiling。`coverage run` 生成报告退出 0，也不表示所有模型通过。

## 批次身份与资源

记录样本 revision、采样时间、权重/分母、任务选择依据和项目来源。若因为歧义显式提供 `--task`，保存自动判断及覆盖后的两份证据，解释选择来源；榜单标签不能直接成为语义正确率的标准答案。

记录每次尝试的 CPU、内存、GPU/dtype、磁盘预算与超时范围。构建、下载、Probe、正式采集各自计时，Probe 超时参数未必覆盖准备阶段。缺少资源时标为未实测，不偷换较小模型或把 full 降成 basic 后报告 full 成功。

正式采集串行遵循项目主机锁；元数据扫描、构建、绘图和额外诊断不进入测量窗口。批次恢复依据冻结清单和已有产物，不能刷新榜单后续写同一份报告。设备切换产生新尝试，保留原始失败。

## 证据层级

| 层级 | 可以证明 | 仍不能证明 |
| --- | --- | --- |
| 静态解析 | 所读配置、路由和依赖证据 | 权重可下载、真实推理、任务质量 |
| Interface Probe | 当前定义下的导入/签名检查 | 完整推理和正式采集 |
| Runtime Validation | 所测设备与最小输入的推理及输出检查 | full profiling 或所有输入尺度 |
| basic profiling | 所选 basic 指标与计划的采集情况 | full 所需能耗、perf、packet 均完整 |
| full profiling | 所选计划及必需指标的实际验收结果 | 未选择的 profiler、其它设备和模型质量 |

full 的通过条件以当版[模式与能力证据](../../../../docs/Profiling_Protocol.md#profiling-mode-与能力证据)为准，结合退出码、完整计划、正式成功行、能力报告和只读审计；任何必需证据缺失保留为缺口。警告单列，不能把未知当成零或通过。

报告同时给出样本总数、已尝试、未尝试及分类计数。权重存在时沿用冻结权重；没有实测或独立语义审阅的比率使用未知/null，不能虚构 0% 或 100%。
