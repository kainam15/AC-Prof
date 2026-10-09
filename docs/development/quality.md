# 开发质量检查

[← 返回专题目录](testing.md)

产物布局改动的定向入口包括 `test_artifact_layout.py`、`test_run_recovery.py`、
`test_auto.py`、`test_posthoc.py`、`test_result_audit.py`、`test_result_comparison.py`、
`test_latency_model_report.py` 和 `test_tui_reports.py`。覆盖新目录写入、flat 目录读取及恢复、
中断文件保留、补采备份与分析输出路由；这些离线用例不代替真实 Docker/GPU 采集。

矩阵计划或 RAPL/DRAM 协议改动的定向入口：

```bash
.venv/bin/python scripts/run_tests.py \
  --pattern 'test_matrix_plan.py' --pattern 'test_startup_probe.py' \
  --pattern 'test_run_recovery.py' --pattern 'test_energy_cpu.py' \
  --pattern 'test_capabilities.py' --pattern 'test_client_energy_warnings.py' \
  --pattern 'test_metric_registry.py' \
  --report internal-testing/matrix-dram-tests.json
.venv/bin/python scripts/render_metric_reference.py --check
```

这些用例验证同 seed 的实际计划、resume 不重排、probe 不写性能结果、异常不扩大 OOM
前缀，以及模拟 sysfs 的 alias、不同 package、回绕、部分覆盖和读取失败。
mock 测试不证明物理服务器的 DRAM 域可读或实际 Docker OOM 行为；真实验收须另查
拓扑、启动失败的 Docker State，以及可读 RAPL/DRAM 设备上的 full 测量产物。

RAPL 的模拟 sysfs 必须包含用于识别域类型的 `name`（如 `package-0`、`core`、`dram`），
不能只创建 `energy_uj`。`tests/test_tui.py` 的循环链接用例也调用生产拓扑读取器，
修改发现逻辑时应一并验证该用例。

指标登记表新增字段时，在 `test_metric_registry.py` 中显式列出新增字段，保留历史字段
顺序的基准哈希；运行 `scripts/render_metric_reference.py` 更新速查文档后再执行 `--check`。
生成器显式读写 UTF-8，写入无 BOM 的 LF 文本，不依赖 Windows 或 Linux 的默认 locale。
新字段插入对应用途组，整体 `status`、`error` 保持在最后两列。列顺序调整还需验证
旧表头的追加、case 合并、packet 回填和 profiler 补采，确保按列名保留数值及未知扩展列。

Ruff、pre-commit 和锁生成工具 uv 由 [`requirements/dev.in`](../../requirements/dev.in) 声明，
完整版本与制品哈希保存在 [`requirements/dev.lock`](../../requirements/dev.lock)。开发锁以主机锁
为约束，避免在同一个 `.venv` 安装时引入冲突；不加入主机运行依赖或容器环境身份。

```bash
.venv/bin/python -m pip install --require-hashes -r requirements/dev.lock
.venv/bin/python -m pip check
.venv/bin/python -m pre_commit install
.venv/bin/python -m pre_commit run --all-files --show-diff-on-failure
```

`install` 给当前 clone 安装 `pre-commit` 和 `commit-msg` 两个 Git hooks；新 clone 需执行一次。
手动运行和 CI 读取同一份
[`.pre-commit-config.yaml`](../../.pre-commit-config.yaml)，检查尾随空白、文件末尾换行、YAML、JSON、
TOML、冲突标记、文件大小和 Python 代码。大文件检查对所有文件执行，限额为 1 MiB，覆盖现有
约 938 KiB 的固定音频输入；Markdown 的两个行尾空格保留为换行。临时证据目录和禁止修改的
`docs/Original_Project_Definition.md` 不参与 hooks。

### AI 提交信息与格式校验

VS Code 的 Source Control 面板中，点击 ✨ **Generate Commit Message** 生成提交信息。
在工作区 `.vscode/settings.json` 中合并以下设置，保留已有配置：

```json
{
  "github.copilot.chat.localeOverride": "en",
  "github.copilot.chat.commitMessageGeneration.instructions": [
    { "file": ".github/commit-message.instructions.md" }
  ]
}
```

生成模板只在 [commit-message.instructions.md](../../.github/commit-message.instructions.md)
维护：`type(scope): concise summary` 标题、空行，以及核心改动、测试或兼容性说明的列表。
scope 可选，标题保持简洁但不设字符数上限。工作区设置只引用该文件。
先暂存本次提交的改动再生成，并在提交前核对内容。

工作区设置若被个人 `.git/info/exclude` 或全局规则忽略，换电脑或新 clone 时需重新合并
上述设置。规则文件本身可随仓库保存。模型沿用 VS Code 当前的 utility model 配置；
需要更换时，在 Settings 中搜索 `Chat: Utility Small Model` 并选择账号可用的模型。

Copilot 0.67.0 的提交生成提示包含 `ResponseTranslationRules`；默认 `localeOverride: auto`
会按 VS Code 界面语言加入系统级语言要求。上述工作区配置使用 `en` 去掉自动追加的中文
要求，使英文提交规则不再与它冲突；该设置也影响当前项目 Copilot Chat 的默认输出语言，
不会改变 VS Code 界面语言。修改设置后清空旧提交消息，再点击 ✨ 重新生成；旧消息不会
自动重写，必要时执行 **Developer: Reload Window**。

[`commitlint`](https://github.com/conventional-changelog/commitlint) 在 `commit-msg` 阶段读取
[.commitlintrc.json](../../.commitlintrc.json)，强制检查 Conventional Commits 结构、允许的类型、
scope 大小写及正文/页脚前的空行；标题、正文和页脚均不限制行长度。
不合规的消息会阻止提交；修改消息后重试。提交内容由生成模板指导并由提交者复核，
格式校验不能证明内容属实。
保留 commitlint 对 Git 自动生成的 merge、revert、fixup!/squash! 等消息的默认豁免。

commitlint 需要 Node.js 22.12+，依赖安装在 pre-commit 的隔离缓存中，不进入 Python 开发锁
或推理环境。安装 hooks 后，首次提交可能需要联网初始化依赖；可提前执行：

```bash
.venv/bin/python -m pre_commit install --install-hooks
```

手动校验一个已存在的消息文件（不创建 Git commit）：

```bash
.venv/bin/python -m pre_commit run commitlint --hook-stage commit-msg \
  --commit-msg-filename /path/to/commit-message.txt
```

`pre_commit run --all-files` 执行文件检查，不校验提交消息或历史提交；当前 CI 的 `lint` job
也仅检查文件。AI 生成功能还需要当前 VS Code 中的 Copilot 或可用的 utility model；
本地格式校验通过不代表已经验证界面中的模型生成结果。

VS Code 提交失败弹窗可能只显示 hook 输出的第一行；点击“显示命令输出”查找真正的
`Failed` 和具体规则名。`body-leading-blank` 表示标题后直接出现正文，缺少空行；应改为
一个标题，或在标题与正文之间加入空行。不要通过关闭该规则来接受拼接的多个标题。

### Ruff 与代码检查

Ruff 版本由 [`pyproject.toml`](../../pyproject.toml) 的 `required-version` 强制核验，Python 目标为
3.10，显式启用 `E4`、`E7`、`E9`、`F`，以及 `B006`（可变默认值）、`B012`（finally 跳转）、
`B904`（异常链）、`I`（import 排序），以及 `RUF010`（f-string 显式转换）、
`RUF013`（显式 Optional）、`SIM101`（合并同一对象的 isinstance）。不启用全量 RUF/SIM、`E501` 或 formatter；
`line-length = 100` 本身不检查行长。Ruff hook 只检查，不自动修复；空白和末尾换行 hooks
会修正文件并返回失败，检查 `git diff` 后重新运行。不得用扩大 `ignore` 或排除目录掩盖新问题。
公共导出用显式重导出或 `__all__` 表达；必须先设置路径、环境或验证缺失依赖的 import，
只在对应行标注具体规则及原因，不统一忽略 `__init__.py`。

本地 Python 修改先运行 Ruff，再按受影响行为选择 pytest 文件、node ID 或 marker。
需要 evidence 时追加 `--report`，或使用 CI wrapper；后者的 `--pattern` 可重复。
局部修改默认验证受影响范围。Git hook 不运行业务测试或硬件采集。

```bash
.venv/bin/ruff check acprof/host/env_utils.py tests/test_env_utils.py
.venv/bin/python -m pytest tests/test_env_utils.py \
  --report internal-testing/env-tests.json
git diff --check
```

更新开发工具时，修改输入并使用开发锁中的 uv 版本重新生成；Ruff 需同步修改版本约束及 hook 的
完整 commit SHA，pre-commit 需同步最低版本。随后核对锁和 diff，重新安装开发锁并运行完整 hooks。

```bash
.venv/bin/uv pip compile requirements/dev.in --python-version 3.10 --universal \
  --generate-hashes --no-annotate --no-header --output-file requirements/dev.lock
```

`scripts/compile_locks.py --check` 仍只验证既有容器锁与 profile 映射，不代替开发锁的重新解析。
主机依赖变更还需用 Python 3.11+ 执行 `scripts/compile_locks.py --host-only --check`，核对发行声明、
已验证 pin 与主机 lock；重新生成方式见[运行兼容](../models/environment.md#当前配置)。CI 的 Python 3.12 job
运行该检查；Python 3.10 job 保留容器锁检查。
wheel CI 同时验证两层依赖契约：锁定环境先按 `requirements/host.lock` 安装并以 `--no-deps` 核对制品；另起空环境直接安装构建出的 wheel，让解析器按 `pyproject.toml` 的公开版本范围选择当前可用依赖，再执行 `pip check`、隔离 import 与 CLI help。这个解析环境只额外固定 pytest/pytest-asyncio 测试工具版本，并从源码树外执行 `test_hf_auto_download.py`，因此 Hugging Face SDK 等运行依赖保持按公开范围解析，可在发布前暴露 API 漂移；锁定版本仍提供可复现基线。
这些开发工具只在编辑、提交和 CI 验证时运行，不进入正式测量窗口。

离线可视化的定向入口为 `test_metric_registry.py`、`test_analysis_model.py`、`test_report.py`；检查旧 CSV、
分组隔离、能量范围、缺失值、冷启动去重、转义及公共 report 入口。浏览器交互由 CI 的独立
`report-browser` job 执行，Python 3.12、Playwright 和绘图依赖锁在 `requirements/browser.lock`，
不安装到共享 `.venv` 或 runtime。锁使用现有 uv 0.12.13 生成：

```bash
.venv/bin/uv pip compile requirements/browser.in --python-version 3.12 \
  --generate-hashes --no-annotate --no-header -o requirements/browser.lock
.venv/bin/uv venv internal-testing/browser-venv --python .venv/bin/python
.venv/bin/uv pip install --python internal-testing/browser-venv/bin/python \
  --require-hashes -r requirements/browser.lock
internal-testing/browser-venv/bin/python -m playwright install chromium
ACPROF_BROWSER_TESTS=1 ACPROF_BROWSER_ARTIFACT_DIR=internal-testing/browser-evidence \
  internal-testing/browser-venv/bin/python scripts/run_tests.py --pattern test_report_browser.py \
  --report internal-testing/browser-evidence/tests.json --require-no-skips
```

CI 在准备阶段用 `playwright install --with-deps chromium` 安装锁定 Playwright 对应的 Chromium。
默认使用这份浏览器；`ACPROF_BROWSER_EXECUTABLE` 仅供显式验证其他浏览器版本时覆盖，不自动选择系统 Chrome。
测试使用 `offline=True`，页面不得产生 HTTP(S) 请求或 JavaScript 异常；覆盖原有五项 baseline、筛选、
排序、跨图选择、Pareto、窄屏交互，以及质量筛选和原始证据展示。未设置 `ACPROF_BROWSER_TESTS=1` 时
普通主机测试明确 skip，浏览器 job 使用 `--require-no-skips` 要求实际执行。
报告固定保存在 artifact 目录，测试失败另保存页面截图、DOM、console/page error、原报告和 Playwright trace。
测试报告和日志无论成功失败均由 CI 上传。浏览器安装需要网络，页面交互本身离线。
条件回归检查输入顺序不匹配时保留散点但暂停改善比例及 baseline frontier，并对照 Python/JavaScript
的精确 workload 分布判断。颜色范围回归检查 200 个配置、2 个指标只读取 400 次数值，验证线性/log、
neutral、零值和分组。合成页面依次渲染 100、400、1000 个配置的六指标矩阵，检查排序颜色及筛选数值，
将首次就绪、排序与筛选耗时写入 `matrix-scaling.json`；这些时间包含浏览器与共享开发机开销，
不是模型性能数据，也不设脆弱的跨机器绝对时限。当前范围保留完整表格；更大规模是否分页或虚拟化需另行实测。
Headless Chromium 证据不等于实际 Windows 浏览器验收，也不证明 Docker/GPU 采集正确。
所有验证均避开正式测量窗口。
