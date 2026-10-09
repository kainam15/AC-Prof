# 测试依据与文档检查

[← 返回专题目录](testing.md)

## 参考实现与复用取舍

测试和硬件证据的身份约束参考 [MLPerf 审计指南的软硬件一致性检查](https://github.com/mlcommons/inference_policies/blob/master/MLPerf_Audit_Guidelines.adoc)。
测试调度核对了 [pytest-xdist 的 work-stealing 实现](https://github.com/pytest-dev/pytest-xdist/blob/master/src/xdist/scheduler/worksteal.py)
与 [PR #858](https://github.com/pytest-dev/pytest-xdist/pull/858)（MIT）。本项目保持串行分片和现有
pytest evidence 协议，先消除重复 CI 检查，再依据当前节点耗时评估分片；不直接引入 worker 调度依赖。
身份计算复用现有标准库 fingerprint，相关逻辑只在测试或测量窗口外执行。

表单边框复用 [Textual 8.2.8 的 `solid` 字符集](https://github.com/Textualize/textual/blob/v8.2.8/src/textual/_border.py)，
并根据 [Select 上游说明](https://github.com/Textualize/textual/discussions/4061)覆盖 `SelectCurrent`
及[源码中的展开菜单 `SelectOverlay`](https://github.com/Textualize/textual/blob/v8.2.8/src/textual/widgets/_select.py)。
Textual 为 MIT 许可且由上游维护；这里只覆盖现有 TCSS，
保留控件尺寸和事件处理，不增加依赖、终端自动探测或测量期间的后台处理。

开发检查复用 [Ruff 官方 hook](https://github.com/astral-sh/ruff-pre-commit)
和 [pre-commit 官方基础 hooks](https://github.com/pre-commit/pre-commit-hooks)（MIT，持续维护，
所选版本支持 Python 3.10）。参考 [HTTPX 的工具配置](https://github.com/encode/httpx/blob/master/pyproject.toml)
把 Ruff 规则放在 `pyproject.toml`，通过 pytest 插件保存本项目的 evidence。
hooks 固定完整 commit SHA，CI 直接执行同一份配置，避免维护第二份检查清单；只新增开发依赖，
没有采集期间的后台进程或测量开销。

提交信息配置依据 [VS Code 官方 instructions 设置](https://code.visualstudio.com/docs/agent-customization/custom-instructions#specify-instructions-for-generated-content)
及 [Copilot Chat 的设置定义](https://github.com/microsoft/vscode-copilot-chat/blob/main/package.json)。
语言冲突依据本机 Copilot 0.67.0 和上游
[提交生成提示](https://github.com/microsoft/vscode-copilot-chat/blob/main/src/extension/prompts/node/git/gitCommitMessagePrompt.tsx)、
[ResponseTranslationRules](https://github.com/microsoft/vscode-copilot-chat/blob/main/src/extension/prompts/node/base/responseTranslationRules.tsx)
确认；生成提示还会引用近期提交风格，项目输出格式由仓库中的提交消息模板维护。
格式检查复用 [commitlint](https://github.com/conventional-changelog/commitlint) 和
[现成的 pre-commit adapter](https://github.com/alessandrojcm/commitlint-pre-commit-hook)（均为 MIT，
由上游维护），保留现有 pre-commit 管理方式；adapter 固定完整 commit SHA，CLI 和规则包固定
直接版本，不引入 Husky 或项目级 npm package。仅在提交和手动校验时运行。

结果原子发布采用 [CPython 的 tempfile](https://github.com/python/cpython/blob/main/Lib/tempfile.py)
和标准库文件同步、替换机制；实验身份参考 [ASV 的结果管理](https://github.com/airspeed-velocity/asv/blob/main/asv/results.py)。
两者的通用做法与现有 CSV/目录协议兼容，恢复仍按 AC-Prof 的 case 与测量窗口实现。
依赖解析复用持续维护的 [uv](https://github.com/astral-sh/uv)（MIT / Apache-2.0），只在更新锁时使用。
指标元数据参考 [Prometheus Python client](https://github.com/prometheus/client_python)（Apache-2.0）的类型与单位声明，
窗口区间参考 [SciPy bootstrap](https://github.com/scipy/scipy/blob/main/scipy/stats/_resampling.py)（BSD-3-Clause）的重采样方法。
本项目只需离线登记表和均值区间，使用标准库实现，无需在采集服务加入 exporter 或 SciPy 依赖。
profiler 调研了 [NVIDIA nsight-python](https://github.com/NVIDIA/nsight-python)（Apache-2.0）；其 kernel profiling 接口
不替代现有完整请求和旁路 probe 契约，因此保留 CLI/CSV 集成，提取纯解析与环境发现模块。
这些选择不增加正式测量窗口内的服务或网络调用，工具和环境验证均在采集前后进行。

生成文档的编码契约参考 [CPython 3.10 pathlib](https://github.com/python/cpython/blob/3.10/Lib/pathlib.py)
（PSF 许可）和 [PEP 597](https://github.com/python/peps/blob/main/peps/pep-0597.rst)，以及
[PyPA 对 README 显式使用 UTF-8 的修复](https://github.com/pypa/packaging.python.org/pull/682)。
采用标准库的显式编码、LF 写入与 `EncodingWarning` 回归保护；兼容 Python 3.10+，
不复制上游代码、不增加依赖，仅影响开发文档生成和检查。

主机分片参考 [pytest plugin hooks](https://github.com/pytest-dev/pytest/blob/8.4.x/src/_pytest/hookspec.py)
（MIT）和 [Textual 测试配置](https://github.com/Textualize/textual/blob/main/pyproject.toml)（MIT），
复用官方维护的 collection/report/session hooks 与 async fixture 生命周期；不复制源码。
阶段完成语义核对 [pytest 8.4.2 runner](https://github.com/pytest-dev/pytest/blob/8.4.2/src/_pytest/runner.py)
与 [setup-plan](https://github.com/pytest-dev/pytest/blob/8.4.2/src/_pytest/setupplan.py)：
fixture 诊断跳过 call，setup-plan 复用 setuponly；报告按实际 phase 判断完成状态。
[pytest #11706](https://github.com/pytest-dev/pytest/issues/11706)说明退出时的 teardown 证据也必须保留。
这些检查沿用已锁定的 MIT 依赖，不增加 runtime 依赖或正式测量开销。
[pytest-asyncio 清理讨论](https://github.com/pytest-dev/pytest-asyncio/issues/222)提示需核对
pending task、async generator 与 executor 的退出；锁定版本并保持 function loop scope。
项目独立实现 schema v1 聚合与排序分片，避免新增通用报告依赖。所有开销只在测试进程中，
不会进入正式测量窗口。涉及 `sys.modules` 隔离的测试，先用
`importlib.import_module` 获取真实目标再 `patch.object`，避免 Python 3.10 的字符串 patch
沿父包属性找到已脱离导入缓存的模块；请求完成、异常和超时断言保持原语义。

统计报告页复用 [Textual 官方 DataTable](https://github.com/Textualize/textual/blob/main/docs/widgets/data_table.md)
及 [Worker API](https://github.com/Textualize/textual/blob/main/docs/guide/workers.md)（MIT，官方持续维护），
已在项目使用的 Textual 8.2.8 中验证；不增加表格库或统计依赖。
窗口统计调用既有 CLI，JSON 读取在后台执行；只在用户操作和任务完成时更新表格，采集期间禁止启动，
避免给正式窗口增加轮询或统计计算。回归覆盖三种终端尺寸、中英文切换、失败恢复、原 CSV 不变及测量互斥。
报告保存参考 [pytest-benchmark 的文件存储](https://github.com/ionelmc/pytest-benchmark/blob/master/src/pytest_benchmark/storage/file.py)
（BSD-2-Clause）和 [Joblib 的稳定内容表示](https://github.com/joblib/joblib/blob/main/joblib/hashing.py)
（BSD-3-Clause）；仅借鉴思路，使用标准库比较完整 JSON、Linux 目录锁与既有原子写入，不增加依赖或测量开销。
`test_stats.py` 覆盖时间戳命名、旧 UUID 报告复用、格式无关的内容比较、参数/CSV/统计值变化、损坏文件、重名和并发发布；
TUI 回归核对重复计算时打开已有文件、显示中英文提示及控制恢复。

表头拖动复用 [Textual DataTable](https://github.com/Textualize/textual/blob/v8.2.8/src/textual/widgets/_data_table.py)
的列元数据、渲染与鼠标捕获。沿用官方维护的 MIT 依赖，无额外包、后台轮询或测量窗口内的诊断。
当前 8.2.8 没有公开的列宽 setter；兼容逻辑集中在 `tui/table.py`，调整列宽时清理渲染缓存并更新虚拟尺寸。
升级 Textual 时需运行拖动回归，覆盖中英文 cell 宽度、固定列与横向滚动、表头点击排序、释放后的 Click、
拖出表格、禁用/隐藏/清空时的鼠标释放，以及三种终端尺寸下的会话列宽保留和采集锁定。
末列右沿在完整显示、横向滚动及单列表格中均不显示手柄或捕获拖动，普通表头点击仍有效。
报告页覆盖数据更新、报告重建和语言切换；镜像树另检查父子行的大小、未知值和容器数量与表头左对齐、横向滚动同步和折叠状态保留。

## 文档与 Skill 检查

修改文档后，可在仓库根目录执行 `.venv/bin/python scripts/check_docs.py`。
该检查扫描 Git 跟踪的 `docs/`、根 README、`AGENTS.md` 和 Skill 文档/YAML，
核对本地相对链接、Markdown 标题锚点、图片路径、UTF-8、行尾和 YAML/frontmatter；
不访问外部链接，也不声称远程链接依然可用。GitHub CI 的 `docs-check` 始终运行此检查。


检查新增文件也包括被 Git 忽略的文件；`git diff --check` 只覆盖已跟踪差异，不能替代完整文件清单。
迁移章节时核对原有锚点、相对链接、代码中的文档引用及字段表是否有遗漏；代码示例中的路径以注明的执行目录为准。
技能格式可用已安装 `skill-creator` 的 `scripts/quick_validate.py <skill-dir>` 检查；该工具是开发辅助，不是项目运行依赖。

交付说明实际执行的命令、结果、跳过原因及未验证范围。只改文档时，不宣称完成真实 Docker/GPU 或用户终端验证。

开销与负载契约测试包括 `test_overhead_contract.py`、`test_overhead_entrypoint.py`、
`test_execution_conditions.py`、`test_load_protocol.py`；外部 Docker／硬件边界模拟，内部编排真实执行。
负载测试使用本机临时 HTTP 服务证明并发与真实连接复用；它不等于目标模型服务支持 keep-alive。
跨实验统计使用 `test_independent_comparison.py`；CI 完整性使用 `test_report_aggregation.py`。
所有本地测试／真实 PCAP 复现均应避开另一个正式采集窗口，不绕过主机测量锁。

本轮配置与统计设计参考 [pyperf](https://github.com/psf/pyperf)（MIT），生命周期参考
[CPython ExitStack](https://github.com/python/cpython/blob/3.12/Lib/contextlib.py)（PSF），
负载计划参考 [MLCommons LoadGen](https://github.com/mlcommons/inference/blob/master/loadgen/test_settings.h)
（Apache-2.0）。仅借鉴方法，使用标准库和已有采集器，不复制框架或增加采集依赖。
CI 复用官方 [download-artifact v4](https://github.com/actions/download-artifact/tree/v4)（MIT），
与现有 upload-artifact v4 配对并固定 SHA；汇总离线运行，不增加采集开销。
