# Code Health 实施记录（2026-10-02）

本轮在 AC-Prof 的 `wt-codex2` 工作树实施，起点为干净的
`2465c10e522395888f5a3c29cb0419dfe9738632`。用户附件中的 `484e814` 是较早快照；
以下结果以本轮工作区核验为准。实现与验证完成后，按用户要求分模块提交 Git；未推送。

现行设计见 [Architecture](../development/architecture.md)，检查与执行命令见
[Testing](../development/static-checks.md#跨模块-private-api-检查)。本文件是实施证据快照，不替代这些专题。

## 清单处理结果

| 项目 | 结果 |
| --- | --- |
| private API baseline 与 CI | 标准库 AST 检查三类 import、alias、相对导入与局部作用域；生产 tuple 精确登记，新增和失效条目都失败；测试单独统计 |
| `docker_runtime` 能力归属 | 生命周期留在原模块；状态证据、镜像准备/身份、模型 token、GPU mode 分别归入 `container_state`、`runtime_images`、`runtime_identity`、`gpu_device`；移除 `_run` 包装 |
| profiler 共享边界 | 删除 `profiler_common`，建立 `profiler_support`；复用 `artifacts.atomic_write`，保留 profiler JSON 的 NaN 表达与字段格式 |
| logging / 用户输出 | 按下表分类；内部诊断用 DEBUG，用户警告用 stderr，进度与 machine events 保留；未增加 verbose/quiet 选项 |
| coverage baseline | 固定 coverage.py 7.15.2；Python 3.12 四个 host shard 上传原始数据，汇总 XML/JSON/HTML；不设置百分比 gate |
| CSV ignore | 保留 `*.csv`，放行 `tests/fixtures/**/*.csv` 和 `examples/**/*.csv`；临时 Git 仓库回归覆盖根层、嵌套与结果目录 |
| `execution_profile` 拆分 | Massif 与 Nsys 的执行、错误、恢复和清理各归入独立模块；原文件只保留 plan/orchestration |
| TUI / CLI 后续边界 | TUI action 仍依赖 `query_one()`、busy 状态、screen 与 worker 回调，保留 widget ownership；CLI 的 tmux 日志单独提取到 `terminal_log` |
| mypy 第二批 | 白名单由 8 个扩大到 18 个模块，覆盖 matrix/run state、profiler plan/support/parser 和纯数据分析；继续只维护 mypy |
| 已完成或按策略保留项 | subprocess 与 MonitorGroup preparation 不重做；compute/resource/client 不继续按行数拆；Ruff 不扩大规则集；国际化维持按需求推进 |

## private dependency 与模块边界

同一 AST 口径下，生产 private 引用从 **177 次 / 164 个唯一 dependency** 降至
**109 次 / 102 个唯一 dependency**；测试为 199 次 / 127 个，仅作诊断。
减少的条目已从 baseline 删除，`--prune` 只能删除，不能批准新增依赖。
错误扫描根目录、重复 baseline、新增替换旧条目与删除后重新引入均有回归。

`client_metrics`、`resource_metrics` 的 companion-module 协作及 `latency_model`
的 analysis/plotting 调用保留具体历史条目，不对整个子系统开无限制豁免。
检查器不追踪动态 import/getattr、运行时 re-export 或任意赋值数据流。

| 模块 | 起点行数 | 完成本轮拆分后 |
| --- | ---: | ---: |
| `host/docker_runtime.py` | 568 | 316 |
| `host/execution_profile.py` | 1265 | 543 |
| `cli/run.py` | 1265 | 1122 |

行数用于描述结果，不作为质量 gate。Massif/Nsys 迁出的 17 个函数经过 API 名称归一化后的
AST 对比，与起点函数结构一致；原有 checkpoint、resume、错误条目与清理回归改在实际所属模块 mock。
新 support 模块可独立导入，TaskInfo 只用于类型检查；没有保留旧路径 shim 或重复采集入口。

## 输出审计

| 模块 | 分类与处理 |
| --- | --- |
| `host/orchestrator` | case/matrix/sniff 进度保留 stdout；idle 稳定性警告移到 stderr |
| `host/posthoc/service` | dry-run、更新清单、产物路径属于用户结果展示，保留 |
| `host/preflight` | 面向用户的前置条件失败继续写 stderr |
| `host/client` | 启动 pipeline tag 改为 DEBUG；ready、窗口完成与用户错误保留原职责 |
| `host/execution_profile` | 工具进度与计划路径保留；Nsys importer 成功诊断改为 DEBUG |
| `host/input_plan` | 输入计划与规模输出保留；内部 scale reason 改为 DEBUG |
| `host/largest_scale_probe` | machine result marker 保持 stdout；逐内存档位及最终错误改为 stderr |
| `host/profilers/ncu` | collecting/resume/工具进度属于用户可见状态，保留 |
| `cli/terminal_log` | tmux pipe 进度保留 stdout，失败及已有 pipe 的提示写 stderr |

回归涵盖 library import 静默且不配置 root logger、client DEBUG、用户 stderr、
machine event 解析以及真实短子进程的 TUI stdout/stderr 合并。
采样开始至停止间启用 DEBUG 仍无日志，成功、超时和取消均验证；请求和结果发布仍在 monitor finish 后。
这些是离线或 headless 证据，不是用户终端实测。

## 验证范围与产物

本轮代码及配置输入清单保存于 `internal-testing/code-health/source_manifest.json`，
480 个文件的内容清单 SHA-256 为
`5ea2ecdd8ddab072784deec3d27b13dc462c6b3ced2e607b154247b00b00e259`。
该清单包含源码、测试、脚本、requirements、CI 和检查配置，不包含本记录等文档。
原始日志和 JSON evidence 均在已忽略的 `internal-testing/code-health/`，不提交生成产物。

| 完整 host 回归 | 执行 | 通过 | 跳过 | 失败 / error |
| --- | ---: | ---: | ---: | ---: |
| Python 3.10.21 | 1771 | 1698 | 73 | 0 / 0 |
| Python 3.12.3，4 shards 合并 | 1771 | 1699 | 72 | 0 / 0 |

两版发现的 test ID 清单 SHA-256 均为
`47ca8101eb8ce5cc25c6e1c503bdd560d060878b576a1bc4f2a5cd151f2a172b`。
3.12 的 `aggregate_test_reports.py` 验收通过，四片齐全、没有重复或遗漏；报告为
`host-summary-3.12.json`，单片原始证据在 `shards-3.12/`。3.10 的报告为 `host310.json`。
72 个共同 skip 需要具体容器或锁定的推理依赖；3.10 另有一个 host metadata 检查要求 Python 3.11+。
没有将这些 skip 记为运行环境已通过。

定向回归：profiler 拆分 73/73、输出及 runtime 70/70、CLI 边界 67/67、窗口日志断言 14/14，均无 skip。
AST guard、mypy 18 个模块、pre-commit（含新增文件）、pip check、host/container lock 检查、
指标文档同步检查、六个根 CLI 的 `--help`、compileall 和 `git diff --check` 通过。
修改的四篇 Markdown 核对了 fence 配对及 85 个本地链接和锚点。
PyCharm 对 checker、runtime image、execution facade、Massif/Nsys 和 terminal log 的单文件检查无 error；
不将空的批量 lint 返回视为全文件分析证明。

### Coverage 快照与优先缺口

四份 `.coverage.*` 已用 `combine --keep` 合并，并生成
`coverage/coverage.xml`、`coverage/coverage.json`、`coverage/html/index.html`。
全仓语句覆盖 **24244/28882（83.94%）**，分支覆盖 **7771/10664（72.87%）**；
Coverage.py 的语句与分支合计指标为 **80.96%**。没有百分比门槛，也没有为了数字新增 omit/exclude。

| 优先模块 | 语句已覆盖 / 总数 | 分支已覆盖 / 总数 | 未覆盖重点 |
| --- | ---: | ---: | --- |
| `extensions/schema` | 167 / 178 | 68 / 78 | 非法声明与字段验证的拒绝分支 |
| `packet/merge_packet_latency` | 187 / 220 | 55 / 76 | 异常输入、缺失匹配和部分 CLI 路径 |
| `host/matrix_plan` | 56 / 67 | 23 / 34 | 非法资源列表、schema/seed/覆盖集合与 pruning 证据不匹配 |
| `host/run_state` | 262 / 280 | 70 / 84 | 损坏状态、layout 不符、冻结计划被改、非 immutable image ID 和异常产物类型 |
| `monitors/energy_cpu` | 302 / 376 | 67 / 112 | 负计数器、空样本/无效时间窗，以及部分真实 reader 和线程路径 |
| `monitors/energy_nvml` | 177 / 200 | 33 / 54 | 设备不可用、采样边界和部分线程/关闭路径 |
| `host/measurement_window` | 80 / 86 | 34 / 38 | 重复 monitor、重复 prepare、非法 idle duration 与 control-window 中断 |
| `host/container_state` | 42 / 64 | 8 / 16 | inspect 失败、非 JSON/空状态与异常退出字段 |
| `profilers/massif` | 66 / 110 | 14 / 24 | 真实工具执行、checkpoint 无效及报告读取失败的部分分支 |
| `profilers/nsys` | 83 / 102 | 14 / 20 | stats/import 失败及报告/临时产物缺失的部分分支 |

这是后续补充回归的依据，不是本轮继续追求覆盖率的目标。该数据只覆盖 host runner 进程，
子进程没有自动注入 coverage；mock 触达的分支也不能证明真实容器、硬件或 profiler 可用。

**Native validation: required。** 本轮调整了容器生命周期所在接口、GPU mode 的代码归属与 profiler 组织，
未运行真实 Docker/GPU/RAPL/perf 采集或真实 tmux 终端验证。相关回归使用 mock 与 headless 测试；
不据此宣称硬件路径已验证。远程 GitHub Actions 尚未运行，本地通过不等于远程 CI 已通过。

## 参考与复用取舍

- private dependency guard 参考 [Import Linter protected contract](https://github.com/seddonym/import-linter/blob/main/src/importlinter/contracts/protected.py)
  的依赖规则和失效忽略项检查；核对 Pylint/Pyright 后，选择适配项目 symbol tuple 的标准库 AST，不叠加另一套 linter。
- 容器职责参考 [Docker SDK container API](https://github.com/docker/docker-py/blob/main/docker/api/container.py)，
  继续复用现有 Docker CLI 与 command runner；没有新增 Docker SDK 或采样期进程。
- coverage 采用 [Coverage.py 官方合并流程](https://github.com/coveragepy/coveragepy/blob/main/coverage/data.py)，
  只增加开发依赖，保留 unittest 与现有 evidence runner。
- 日志与 tmux 分别核对 CPython logging 和 tmux pipe-pane 的源码；许可证、维护情况及完整来源见
  [Architecture](../development/orchestration.md#host-command-与-diagnostics) 与 [Testing](../development/ci.md#host-coverage-baseline)。
