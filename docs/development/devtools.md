# 开发辅助工具与 MCP

[← 返回专题目录](testing.md)

## 辅助开发工具
`requirements/test.in` / `test.lock` 统一固定 pytest、pytest-asyncio、pytest-cov 和
pytest-textual-snapshot；`dev.in` 包含它，主机和推理 runtime 的依赖声明不包含测试工具。
Textual 与主机锁一致；pytest 8.4.2 / syrupy 4.8.0 保留已有 SVG 序列化格式。
`textual-dev` 与 Hypothesis 仍是按需工具，添加普通测试依赖前须同步开发锁和 CI。

| 工具 | 适用场景 | 运行入口 |
| --- | --- | --- |
| `textual-dev` | Textual 开发日志、事件和调试界面 | `.venv/bin/textual` |
| Hypothesis | 同步输入与状态转换的性质测试 | pytest 中的 `@given` / `strategies` |
| `pytest-textual-snapshot` | 固定场景 SVG 视觉回归 | `.venv/bin/python -m pytest tests/visual` |

```bash
.venv/bin/python -m pip install --require-hashes -r requirements/host.lock -r requirements/dev.lock
.venv/bin/python -m pytest --version
.venv/bin/python -m pip check
```

交互调试时，在两个终端中分别从仓库根目录运行：

```bash
# 终端 A：开发控制台
.venv/bin/textual console
# 终端 B：连接控制台运行 TUI
TEXTUAL=devtools,debug acprof tui
```

Hypothesis 可用于同步 pytest 测试，并使用 `python -m pytest <文件>`
执行选定用例。每个生成样例应隔离可变状态；发现失败后保留最小输入，加入稳定回归。
不要直接用 `@given` 包装异步 `Pilot` 交互；优先生成同步输入处理或状态转换的样例。
新增依赖这些工具的常规测试时，先补齐相应开发依赖与 CI 环境，不能仅依赖本机安装。

快照用例使用 `snap_compare` fixture，显式指定 `terminal_size`，固定语言、主题和输入，
仓库的 `tests/visual/test_snapshots.py` 固定十一个场景：中文窄终端、英文常规尺寸、宽终端、
弹窗覆盖、实际拖动表格之后、测量中、清理未完成、中英文 WSL2 模型检测等待，以及中英文模型确认对话框；
覆盖 `80×24`、`120×30`、`150×45`。
通用场景固定 Native Linux 身份，WSL2 场景固定 PARTIAL，避免基线随运行测试的主机变化。
WSL2 检测等待场景通过 F5 启动模拟受控任务，在 Monitor 内显示接口阶段和原始日志，
不打开被动 Preparation Modal；截图前验证测量未开始、日志可见且停止按钮可用。
模型确认场景从上述等待状态接收明确的 review 请求后打开 Preparation Modal，核对模型字段、
中英文确认标签、按钮无遮挡与 Tab 可达；确认请求绑定当前模拟进程和 request ID。
基线在 `tests/visual/__snapshots__/`。测试隔离设置、固定主题和显示路径，不启动采集或外部服务。
普通功能测试与 SVG 回归共用开发环境和 pytest 配置：

```bash
TZ=UTC PYTHONHASHSEED=0 .venv/bin/python -m pytest tests/visual -q \
  --snapshot-report internal-testing/tui-snapshot-report.html
```

更新测试工具和开发锁（保持 `host.lock` 约束，Python 3.10+）：

```bash
.venv/bin/uv pip compile requirements/runtime-test.in --python-version 3.10 --universal \
  --generate-hashes --no-annotate --no-header -o requirements/runtime-test.lock
.venv/bin/uv pip compile requirements/test.in --python-version 3.10 --universal \
  --generate-hashes --no-annotate --no-header -o requirements/test.lock
.venv/bin/uv pip compile requirements/dev.in --python-version 3.10 --universal \
  --generate-hashes --no-annotate --no-header -o requirements/dev.lock
```

快照用例固定主题、语言、路径与颜色模式，规范化 SVG 行末空白以兼容仓库格式检查；
截取前检查目标页面已经激活，避免把错误场景保存成基线。
正式测量场景须显式设置接口解析与运行验证为 `passed`、测量为 `running`；
不能只设置 `measurement_active=True` 而让阶段状态保留 `not_started`。
测量中和清理未完成的场景还须模拟由应用管理、`poll()` 返回 `None` 的进程，
并隔离停止操作；截取前断言停止按钮可用，不启动真实采集或向真实进程发送信号。

先查看失败报告的 HTML / SVG 差异，再在预期变更或首次建立基线时对选定用例追加
`--snapshot-update`；普通验证不更新基线。快照与功能测试使用同一 pytest runner，
不能代替行为断言、evidence JSON 或[真实终端证据](terminal-validation.md#tui-与终端证据)。
上述调试、样例生成和截图均在正式测量窗口之外运行；按任务选择流程见
[TUI 回归 Skill](../../.agents/skills/acprof-textual-regression/SKILL.md#按需选择辅助工具)。

用法参考上游维护的 [textual-dev](https://github.com/Textualize/textual-dev)
与 [pytest-textual-snapshot](https://github.com/Textualize/pytest-textual-snapshot)（MIT），
以及 [Hypothesis](https://github.com/HypothesisWorks/hypothesis)（MPL-2.0）。
复用现有 CLI、性质测试和快照机制；通过统一测试锁及按需运行控制依赖与维护成本，
不向推理环境或正式采集进程加入开发工具。

## PyCharm MCP 的验证边界
使用项目的 `.venv` 解释器；符号搜索按需限定 `paths=["acprof/**", "tests/**"]`，
避免将临时虚拟环境中的第三方代码视为项目实现。调用分析无法解析已找到的 Python 符号时，
结合符号文档和源码核对调用者；空结果不能证明没有依赖。

若 `analyze_calls` 返回两条相同的候选标识，使用 `search_symbol(include_external=true)`
核对是否同时找到源码和 `.venv/.../acprof/_bundle` 中的安装副本。`.venv` 的项目排除规则
不排除 Python SDK 库索引。editable 安装不应复制这份 bundle；修复 build hook 后，执行
`uv pip install --python .venv/bin/python --no-deps --reinstall-package acprof -e .` 更新安装，
再验证入向和出向调用。资源打包约定见[安装包说明](distribution.md#工作目录与资源)。

PyCharm 2026.2.3（build `262.10968.92`）已复现一种 MCP 兼容问题：
`analyze_calls` 的 `isCallableSymbol` 依赖显示文本中的 `name(...)`，
而 Python 函数的 Usage View 文本只有名称，因此真实 `PyFunction` 也会被过滤。
这类错误应修复 IDE 工具的函数识别，不需要改动业务函数或重建 Python 环境。
修复验收应包含真实函数的入向、出向调用和不存在符号的错误路径；
本机兼容补丁还需验证版本匹配、撤销与启动加载，并区分独立 JVM 检查和完整 IDE 重启。
实现依据见 [JetBrains Call Hierarchy](https://github.com/JetBrains/intellij-community/blob/master/plugins/mcp-server/mcpserver.toolsets/src/general/CallHierarchyAnalysisSupport.kt)
与 [Python Usage View](https://github.com/JetBrains/intellij-community/blob/master/python/src/com/jetbrains/python/findUsages/PyElementDescriptionProvider.java)。

PyCharm 在 **Settings → Tools → Python Integrated Tools → Testing** 中选择 **pytest**，
使用项目 `.venv`；已有 Run Configuration 的 target 改为 pytest 的文件或 node ID。
VS Code 的 Python Test Explorer 使用 `python.testing.pytestEnabled: true`、
`python.testing.pytestArgs: ["tests"]`，解释器选择同一 `.venv`。
CLI、IDE、host CI 与 SVG job 使用同一配置。需要 CI evidence 时追加 `--report <文件.json>`；
普通控制台输出不能代替 JSON，证据必须包含真实输出和退出码。
`build_project` 若提示无法收集构建诊断，不能替代 Python 编译和相关测试；
依赖查询返回空列表也不能证明 Python 环境没有安装依赖。
临时重命名和工具测试文件放在任务独立的 `internal-testing/` 子目录中。
