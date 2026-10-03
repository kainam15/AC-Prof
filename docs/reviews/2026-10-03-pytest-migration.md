# pytest 迁移验收记录（2026-10-03）

本记录针对 `refactor/pytest-migration`，基准提交为
`6a9aa029dfeedea8478010ec741a8aef49867680`。在独立 worktree 中实施，未修改原工作区的并行改动。
现行执行方式和 evidence 契约见[测试指南](../Testing.md)，本文只保存本次审计的输入、结果和限制。

## 基线与测试集合

迁移前使用原 runner、Python 3.12.3 和 branch coverage 执行完整测试：

| 项目 | 基线 |
| --- | ---: |
| 收集并执行 | 1997 |
| passed | 1913 |
| skipped | 84 |
| failed / error / expected failure / unexpected success | 均为 0 |
| suite SHA256 | `e4f86a4d59081523604da5962018ec8931e889a3961ad81985359b3d34dbf4cc` |

原始 evidence、日志、测试源码副本、ID 清单和 coverage 保存在本机被忽略的
`internal-testing/pytest-migration/before/`；完整基线运行耗时 3509.73 秒。
9 份原 SVG 另行执行，全部通过，迁移未更新 SVG 内容。

正式转换前同时用 pytest 收集旧测试，1997 个旧 ID 无遗漏；还发现旧 runner
未执行的两个原生 pytest 平台用例。迁移后，这两个用例、9 个 visual 用例和新增的
测试基础设施回归均进入统一集合。`subTest` 展开为可独立重跑的参数化 cases，
因此新旧总数不能直接作为行为覆盖增减的依据。

迁移后共收集 **4234** 项；逐个映射旧类名、方法名及参数化父用例，1997 个旧用例均存在。
映射仅保存在本次审计产物中，不向正式测试保留旧类名 alias 或另一套 runner。
Python 3.10.21 与 3.12.3 的 ID 集合及 suite hash 相同：
`be40ece55206f8115124d17170f67b84222a675f94cf860899ee714acf70c573`。

参数审计还逐组核对索引参数的数据来源，修复初次转换少算的 15 项：设置校验中展开的
9 个 payload、动态能力列表追加的 4 项，以及架构检查追加的 2 个文件。三个数据集与基线
逐值比较一致。动态文件、翻译、主题和任务集合直接生成 pytest 参数，不再冻结集合长度；
其余 102 组索引参数经静态审计，无长度不符、未解析来源或追加数据遗漏。

## evidence、分片与隔离

保留 schema v1 的字段与状态分类；ID 改为 pytest node ID。setup、call、teardown
归并成每个 case 的一条记录；assertion failure 与非断言异常仍分别记录为 failed 和 error。
无 reason 的 xfail / XPASS、严格 skip、collection error、空 pattern、空 shard 及只收集未执行
均有独立 CLI 回归，不能生成虚假的成功证据。wrapper 同时验证同步函数和原生 async 函数。

每个 Python 版本重复两轮四分片收集，分配均为 **1059 / 1059 / 1058 / 1058**，
交集为空，并集恰为 4234 项。真实执行的小型四分片报告经原
`scripts/aggregate_test_reports.py` 消费成功；未修改聚合器实现。

session autouse fixture 隔离 `MEASUREMENT_LOCK_ROOT`，直接 pytest 和 wrapper 共用，
覆盖 fixture 初始化、测试及清理。TUI 验证子进程显式继承同一临时目录，并通过
Python audit hook 拒绝打开生产锁文件；父进程 patch 不被误当成子进程隔离证据。

## 本地执行与 coverage

Python 3.10 按 `unit` 与 `not unit` 两个互斥 marker 集合执行，分别为 **4092 passed / 1 skipped**
与 **10 passed / 116 skipped**。Python 3.12 完整 pytest 回归为 **4103 passed / 116 skipped**，
耗时 2901.99 秒。以上完整回归对应参数审计前的 4219 项；修复后重新执行全部 8 个受影响文件，
并以最新 collection 核对执行记录，补回的 15 项均须具备实际执行证据。

两版 Python 的 8 个受影响文件均 **1213 passed**；evidence、wrapper、子进程锁隔离、
runtime 调度和框架门禁也在两版 Python 上重新验证。将实际执行记录按最新 ID 核对：
Python 3.10 为 **4117 passed / 117 skipped**，Python 3.12 为 **4118 passed / 116 skipped**，
4234 个 ID 无缺失、无失败或 error。这是完整回归与修正后复验的覆盖审计，
不是虚构一次 4234 项的完整执行；原始分次报告及来源映射分别保留在 `after/` 和
`after/final-execution-audit.json`。

在迁移前后共有的生产文件上，branch coverage 对照完全一致：

| 指标 | 迁移前 | 迁移后 |
| --- | ---: | ---: |
| covered lines / statements | 27304 / 32172 | 27304 / 32172 |
| covered branches / branches | 8765 / 11820 | 8765 / 11820 |
| combined coverage | 81.99% | 81.99% |

直接 `pytest --cov` 的初始结果因插件在 pytest-cov 启动前导入模块，漏计 57 行导入期代码；
使用 CI 相同的外层 `coverage run --branch -m pytest` 定向复核导入及受影响文件，
再合并本次完整回归数据。其余 1 行与 1 条分支下降来自上述设置参数遗漏，补齐后消失。
对照固定原有文件集合，4 个新增 `acprof/testing/` 模块单列，不稀释原有代码覆盖率。
保留原始 pytest-cov 数据、合并输入及最终 XML、JSON、HTML，未新增全局百分比门槛。

84 个旧 skip 父用例的原因逐项对应，没有意外新增原因；参数化增加了 31 条 skip 记录，另有
旧 runner 未收集到的 WSL2 平台测试，在本机按既有原因跳过。Python 3.10 比 3.12 多出的
1 条 skip 是原有 host lock 元数据检查要求 Python 3.11+，并非迁移后放宽。

单独的 `tests/visual` 命令也执行通过，
9 份 SVG 的 SHA256 与基线逐文件一致。所有 TUI 结果均为 Textual headless / Pilot 证据。

真实 Chrome 报告交互测试 10 项通过；这是本机 Chrome 验证，不代表 CI 固定 Chromium 的重跑结果。
Ruff、mypy、依赖一致性、runtime/host lock 检查、private API 门禁及 pre-commit 均独立执行。
构建 wheel / sdist 后，在只安装 `host.lock` 的新环境中通过 26 项发行 smoke 检查，
确认 pytest 未被安装。PyCharm 的原项目读取和符号工具可用；本 worktree 未在 IDE 打开，
因此 IDE diagnostics 未完成，当前代码验证来自上述 CLI 工具。

## 真实容器接口

先核对已有镜像的 environment ID、Python packages 和系统依赖 manifest，再以不可变 image ID
运行。测试工具安装在 `/tmp/acprof-tests` 临时 venv，wheelhouse 带 hash，容器使用
`--network none`、2 CPU、4 GiB；执行时启用 `--require-no-skips`。

| 已有环境 | passed | skipped |
| --- | ---: | ---: |
| onnxruntime-cpu | 35 | 0 |
| nlp-cpu | 6 | 0 |
| cv-cpu | 7 | 0 |
| audio-cpu | 6 | 0 |
| multimodal-transformers4576-cpu | 11 | 0 |
| diffusion-cpu | 19 | 0 |
| structured-cpu | 5 | 0 |
| timeseries-cpu | 2 | 0 |
| nlp-transformers560-cpu | 2 | 0 |
| multimodal-transformers560-cpu | 3 | 0 |
| custom-multimodal-cpu | 9 | 0 |

共 **105 项真实 CPU 接口测试通过**。证据保存在
`internal-testing/pytest-migration/cached-runtime/`，包含实际 image ID、manifest、日志和 schema v1 报告。
这些已有镜像的依赖内容匹配当前锁，但其构建 recipe 指纹早于当前 checkout，不能冒充本次新构建。

本次还实际调用了 `check_runtime.py --profile onnxruntime-cpu --build-only`：
预构建拉取回退后，BuildKit 请求 `docker/dockerfile:1` 超时，结果为失败。
因此 **当前 recipe 的完整构建入口尚未验收通过**；未运行 ONNX basic 真实采集，也未重跑 GPU/hardware workflow。
未更改推理依赖锁、Dockerfile、生产测量代码或硬件 workflow。
**Native validation: not applicable**：本轮改动限于测试及开发验证工具，不改变硬件指标或正式采集语义。

## 远程 CI 与后续边界

迁移开始时查询的最近一次
[CI run 37107319047](https://github.com/kainam15/AC-Prof/actions/runs/37107319047)
对应 `c06b345972aa123e89e81f6d642695c0b2e228e6`，不是上述本地基准提交：
Python 3.10 四个 host shards 均成功；Python 3.12 两个成功、两个取消；
CV runtime 与 host-summary 失败，snapshot、ONNX 和其余 runtime jobs 成功。
原始 API 响应保存在基线目录，记录查询时状态，不把旧 CI 结果当作迁移后的验证。

本分支未 push，故没有当前提交的 GitHub Actions 通过证据。CI 保留 Python 3.10 / 3.12、
每版本四分片、schema v1 artifacts、Python 3.12 branch coverage 合并和独立 snapshot job。
WSL 平台判断有离线回归，本次没有真实 WSL 主机执行；headless 结果也不等于真实终端验证。

pytest-xdist 尚未进入正式依赖或默认命令。当前 `unit` 分类仍包含 TUI、进程模拟及主机状态 patch，
不能直接将整个 `-m unit` 集合视为可并行安全；正式 evidence 插件明确拒绝多 worker，保留四个确定性 CI shards。

另用独立临时工具目录评估 pytest-xdist 3.8.0 / execnet 2.1.2，仅选择经过读取审计的
`test_uncertainty.py` 与 `test_load_protocol.py`：前者使用局部数据和固定 seed，后者使用
OS 分配的临时 TCP port，线程均回收；没有 Docker、TUI、硬件、生产锁或全局环境修改。
一次串行和两次 `-n auto`（限制为 2 workers）均 **14 passed**，JUnit case 集合相同。
观察到的进程耗时为 3.99 / 2.82 / 2.82 秒，只作为当前负载下的小样本，不据此修改 CI。
此探测使用仅允许上述两个模块的临时 marker 插件，未使用正式 evidence 插件，结果不冒充 CI 分片报告。
worker 数量控制参考 [pytest-xdist 官方实现与说明](https://github.com/pytest-dev/pytest-xdist/blob/master/docs/distribution.rst)。
