# 自动化测试入口

[← 返回专题目录](testing.md)

## 自动化验证入口

安装分发修改除普通回归外，还需构建 sdist/wheel，在隔离环境从空目录执行
`scripts/check_distribution.py`；standalone 用 `--binary <path>` 执行相同验收。
该脚本核对全部公共帮助入口、参数错误、无 Docker 时的 doctor JSON、内置资源和 packet worker 的实际输出。
日常 CI 的独立 `wheel` job 分别从当前 checkout 和 sdist 构建 wheel，
在两个新的 venv 中按 `host.lock` 安装依赖及对应 wheel，并检查归档内容与 Docker context 暂存。
验证脚本复制到源码树外，从空目录执行，清除 `PYTHONPATH` 并使用 Python `-I`；同时核对
`acprof.__file__` 位于安装环境内，拒绝意外使用源码或 editable 安装。公共入口从安装包 dispatcher
读取，包含 `compare`、`load`、`model-store`；实际生成离线 HTML 并运行 packet worker，验证
静态资源和子进程随包完整分发。报告和日志作为 `wheel` artifact 保存。
standalone 和真实 `uv tool install` 保留在发布或按需流程；构建步骤见[发行包说明](distribution.md#linux-standalone)。
这些检查不代替 Docker/GPU 推理和完整 profiling。

初始化入口修改运行 `test_setup.py` 和 `test_tui_onboarding.py`，覆盖缺失 uv、安装/诊断失败、
重复执行、非交互终端、首次配置与已有设置保留；使用隔离 uv 目录实际执行
`./setup.sh --no-tui --no-modify-path`。真实推理另用安装后的命令运行最小 basic CPU 实验并审计结果。

测试使用 pytest 原生函数或有明确共同主题的 `Test*` class；文件名与函数名为 `test_*`。
保留 `unittest.mock`；临时目录、环境、输出、日志与异常优先使用 `tmp_path`、`monkeypatch`、
`capsys`、`caplog` 和 `pytest.raises`。初始化与清理使用 function scope 的 yield fixture；
共同的数据构造器放在领域 fixture 模块，避免实例化另一测试 class。
模拟网络、Docker、硬件与通知边界；避免测试触发真实采集或改写用户设置。

```bash
# 按受影响的行为选择测试文件
.venv/bin/python -m pytest tests/test_env_utils.py -v
# 符合“验证范围”中的全套回归条件时
.venv/bin/python -m pytest -v
acprof run --help
.venv/bin/python -m compileall -q acprof scripts packaging
git diff --check
```

单测试、标记与异步测试：

```bash
python -m pytest tests/test_env_utils.py::test_local_settings_override_env_file_but_not_explicit_process_values
python -m pytest --collect-only
python -m pytest -m unit
python -m pytest -m "runtime and not hardware"
python -m pytest -m hardware
python -m pytest tests/test_tui_input.py
```

单个 node ID 以 `--collect-only -q` 的实际输出为准。`asyncio_mode="auto"` 收集
`async def test_*`，测试与 async fixture 的 event loop 均为 function scope，
Textual `run_test()` 通过 async context manager 完成退出；不要遗留后台 task。
`--strict-markers` 拒绝未注册标记，`--import-mode=importlib` 允许领域目录中同名测试模块。
`unit` 自动赋予无 integration/runtime/visual/platform/hardware 标记的测试；
平台 skip 只作用于显式 `wsl`、`native_linux`，`hardware` 本身不跳过失败。
原有缺依赖或显式 opt-in skip 原因保留；CI runtime 使用 `--require-no-skips` 拒绝缺失验证。

`scripts/check_test_framework.py` 在 pre-commit 与 CI 中禁止旧执行框架的导入，允许 mock；
不通过排除文件或忽略报错维持门禁。正式测试环境暂不安装或启用 pytest-xdist：全套测试包含
全局模块 patch、进程锁、TUI 和资源敏感场景。未来仅在完成 port、临时目录、全局 cache、
Docker 名称与 host 状态审计后，对独立 unit 子集比较串行/并行结果；CI 沿用
[确定性分片与完整性验收](ci.md#ci-与环境测试)，数量以 workflow 的 `HOST_TEST_SHARDS` 为准。

`test_architecture.py` 禁止根目录出现任何 `.py` 文件，检查标准库 `profile/cProfile` 可直接导入。
`test_distribution.py`、`test_tui_interaction.py` 与 terminal-log 回归保护公开帮助、
TUI 预览及日志中的 `acprof <command>` 展示，并保留含空格或 shell 特殊字符的参数。

| 实现范围 | 测试入口示例 |
| --- | --- |
| 模块依赖与导入副作用 | `tests/test_architecture.py` |
| 环境、依赖与镜像 | `tests/test_runtime_profiles.py`、`tests/test_environment_identity.py`、`tests/test_lock_compiler.py`、`tests/test_image_layers.py`、`tests/test_runtime_image_build.py`、`tests/test_image_reuse.py`、`tests/test_runtime_validation.py` |
| 模型文件与离线加载 | `tests/test_model_files.py`、`tests/test_model_download.py`、`tests/test_offline_model_loading.py` |
| 字段、能耗、资源与补采 | `tests/test_energy_cpu.py`、`tests/test_resource_usage.py`、`tests/test_posthoc.py` |
| 页面、设置与焦点 | `tests/test_tui_layout_settings.py`、`tests/test_tui_interaction.py`、`tests/test_tui_input.py` |
| 启动环境检查 | `tests/test_tui_startup_preflight.py`；首屏可编辑、静默成功、异常入口、重试、全部采集入口阻断、配置失效、测量锁与迟到回调 |
| 固定页头、底栏与操作颜色 | `tests/test_tui_page_chrome.py`；中英文、三种尺寸、滚动与缩窗、命令框显隐、八种主题和按钮状态 |
| 日志、语言与命令 | `tests/test_tui_log_view.py`、`tests/test_tui_i18n.py`、`tests/test_tui.py` |
| 终端色深与 RGB 输出 | `tests/test_tui_colors.py`；检查缺失/空 `COLORTERM`、真彩色、256 色与自动检测 |
| 中文浮层缺字 | `tests/test_tui_cjk_rendering.py`；通知覆盖按钮、真实遮挡、宽字符两半的局部刷新、中英文和奇偶列宽缩放 |
| 统计报告、异步读取与计算 | `tests/test_tui_reports.py`、`tests/test_report_views.py`、`tests/test_uncertainty.py`、`tests/test_stats.py` |
| Docker 镜像树、层空间、标签删除与采集互斥 | `tests/test_image_management.py`、`tests/test_tui_images.py` |
| 表头拖动、固定列、滚动范围与鼠标释放 | `tests/test_tui_table_resize.py`、`tests/test_tui_all_tables.py`、`tests/test_tui_images.py` |

以上是定位入口，不是每次必须运行的清单。先用 `rg --files tests` 查实际受影响的测试；新改动、失败或未解决问题才需要扩大或重复验证。

`test_runtime_image_build.py` 在 Docker 边界模拟环境中验证四层构建、跨 profile 的环境共享、
模型 commit、Torch 来源、主机下载令牌不进入 Docker 构建、标签错配、额外包、输入变化及构建失败停止。
环境身份测试覆盖当前声明的 profile / environment、无 Torch 环境、跨任务族精确共享及 cu128 的版本差异；CV 新增 timm 后按新包集计算身份。注释、锁文件名
和条目顺序不影响身份，版本、制品、来源、平台和系统锁影响身份。配方变化改变构建缓存，业务
代码变化只重建服务层。它们不能替代实际容器构建与推理验证。

镜像分层测试的临时目录需包含指纹依赖的全部文件，包括 `acprof/model_spec.py`；
修改模型文件筛选或声明解析代码时，断言模型层和服务层指纹变化、依赖层身份不变。
模型接口新增解析字段时，同步更新 CLI 委派测试中独立声明的完整 `model_resolution`
期望值，保留完整对象比较，不从被测函数的返回值生成期望。

自动解析与编排回归使用 `test_resolution_decisions.py`、`test_auto.py`、`test_model_coverage.py`
和 `test_model_inspection.py`，覆盖同源证据、显式冲突处理、验证不提升静态裁决、固定 SHA、
主机失败与旧 CLI 选项拒绝、原生模型验证、冻结覆盖率分母及人工语义参考。已有模型／契约／镜像／
恢复测试继续保护协议。滚动模型检查使用 [coverage 命令](../usage/cli-discovery.md#acprof-coverage)，
其静态、接口检查、容器运行验证与正式测量证据分别验收；4 GiB 或超时限制不等同于模型语义错误。

采集准备与重试使用 `test_collection_workflow.py`、`test_run_native_docker.py`、
`test_run_notifications.py` 和 `test_tui_collection_workflow.py`。当前顺序为采集平台策略检查、
开始通知、模型解析、主机预检、运行环境准备和验证、正式测量；预检失败必须阻止后续准备。
策略拒绝时不发送开始通知、不访问 Hub 或 Docker；策略通过后，开始通知早于模型解析和 Docker 预检。
预检和通知测试在
`acprof.host.detect.detect_task` 边界提供静态 fixture，避免假模型 ID 访问真实 Hub。
监控布局测试同时检查状态网格、准备状态行和日志的衔接，以及各尺寸下日志和按钮的可用空间。
