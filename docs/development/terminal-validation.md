# 终端与采集环境验证

[← 返回专题目录](testing.md)

## TUI 与终端证据

终端兼容验收面向 Windows Terminal、PyCharm / JetBrains Terminal、VS Code Terminal、
Linux 原生终端、SSH 会话及浏览器 Web Terminal。SSH 只传输终端数据，记录结果时仍需注明
客户端终端、版本、字体和窗口列数 / 行数；每种环境分别标记通过、失败或未验证。
至少检查非浮层控件不重叠、边框可辨认、中文无乱码、`Tab` / `Input` / `Select` 可操作，
以及缩放后的布局和焦点可达性。字形能力有限的终端应检查基础 Unicode 线框和键盘路径，
不能依赖 `tall` 块状边框无缝拼接；纯 ASCII 终端不属于当前中文 TUI 的支持范围。

使用 pytest 原生 async 测试和 `pytest-asyncio`、Textual `run_test()` / `Pilot` 和临时 `settings_path`。
普通界面测试使用 `tests/tui_fixtures.py` 的就绪环境替身，保留启动状态机但不触发真实硬件探测；
启动 preflight 专项使用真实 `AcprofTui` 和 thread worker，只模拟 host diagnostics 边界。
覆盖整个 fixture 生命周期的环境隔离使用 `tests/environment_fixtures.py` 的
`isolated_environment()`，保留 `PYTEST_` 与 `TEXTUAL_SNAPSHOT_` 测试运行状态，
不继承应用设置、凭据或代理。`KeyboardInterrupt` 后，session 退出钩子可能先于 fixture
清理运行；不能依赖稍后的环境恢复来保护快照临时目录。`test_environment_fixtures.py`
通过真实子进程核对六组 fixture 的正常退出和中断退出码、失败证据及快照插件收尾，
不关闭插件、不把中断或存储失败改报成功。
快照与验证 runner 的测试同样隔离 host 探测；runner 实际启动子进程前等待启动检查，检查失败则保存失败画面并退出。
尺寸覆盖用户报告的场景，并按布局变更检查 `80×24`、`120×30`、`150×45` 及运行中 resize。
七个页面的标题或状态摘要和底部操作栏应保持可见；次要／导航动作在左下角，主要操作在右下角，
内容滚动、切换语言和隐藏快捷命令框后仍可点击。检查按钮标签完整、左右边缘未被容器裁切、
`Tab` / `Shift+Tab` 按左右分组顺序切换；实验页切入高级参数及监控页放大日志后仍满足该布局。
按钮颜色检查主题切换、悬停、聚焦、禁用与恢复；删除和终止的红色、补采的黄色不能在交互中丢失，服务就绪和保存成功使用绿色。
镜像树路径高亮检查最终屏幕的连接线颜色，覆盖点击、方向键、折叠、搜索、失焦和中英文/深浅主题切换，防止祖先线未重绘或其它分支误亮。
镜像详情检查摘要与诊断分离、默认折叠、长包清单的末项可达、点击/Enter 展开，以及换行选择、空筛选、语言切换和缩放时的折叠状态；这些操作不得触发额外 Docker 查询。
详情分隔条检查上下拖动的实际高度变化、拖出边界后的最小可见区域、按键调整与默认值恢复、缩窗后再放大的手动高度和阅读位置；切换页面、采集开始、失去捕获、缩放或按 `Esc` 后均须释放鼠标，三个镜像视图与中英文均应可用。
分隔条滚动回归使用足够溢出视口的清单，按住鼠标逐行上拖并返回，检查顶部及非零滚动位置保持不变。
Textual 8.2.8 的 `Pilot.hover()` 固定传入 `delta_y=0`，此场景需向 App 投递带实际移动增量、`button=1` 的 `MouseMove`，才能覆盖真实拖动触发的选区自动滚动。
分隔条沿用 [Textual 控件的 `ALLOW_SELECT=False`](https://github.com/Textualize/textual/blob/v8.2.8/src/textual/widgets/_button.py)，避免 [选区自动滚动](https://github.com/Textualize/textual/pull/6440) 抢占拖动语义；只使用锁定版本已有的 MIT 许可 API，不增加依赖或测量期开销。
镜像自动刷新检查首次打开、定时更新、失败重试与有效勾选/浏览位置保留；验证后台页面、确认框、并发操作和测量窗口不启动扫描，以及任务结束后恢复。
退出阶段还需验证已排队的 timer 和 worker 回调：Textual 停止应用后、控件部分卸载而 `on_unmount` 尚未执行时，不再扫描或访问页面控件。
`test_tui_interaction.py` 在退出期间主动投递计时回调，检查耗时控件卸载后无异常且定时器释放；
同时保留运行时刷新、正式测量期间暂停和测量结束后恢复的断言，无需延长固定等待。
此边界依据锁定的 [Textual 8.2.8 退出生命周期](https://github.com/Textualize/textual/blob/v8.2.8/src/textual/app.py)
和 [Timer.stop](https://github.com/Textualize/textual/blob/v8.2.8/src/textual/timer.py)（MIT），
复用现有 API，不增加依赖或测量期刷新。
页面切换、挂载和布局更新后等待框架处理事件，再判断点击和焦点，不用堆叠固定 `sleep` 掩盖竞态。
设置持久化测试先进入实际设置页，按锁定版本的
[Textual 测试流程](https://github.com/Textualize/textual/blob/v8.2.8/docs/guide/testing.md)
等待初始化和控件变更事件完成，确认主题等偏好已应用后再保存并检查文件。
`scripts/run_tui_validation.py` 在子进程结束后，通过 `call_after_refresh` 等待监控页显示且日志控件
进入实际屏幕布局，再保存 `tui-finished.svg` 并返回退出码。排队的焦点事件若切回配置页，辅助入口
会重新选中监控页并等待刷新；10 秒内未就绪则明确报错。`test_tui_validation_runner.py` 覆盖真实子进程、
首次刷新前立即完成、排队的页面切换及启动失败；这些检查仅验证辅助入口的
headless 画面，不代表真实采集或用户终端已通过。
Textual 8.2.8 的 `Pilot.pause()` 可能在返回前的布局刷新中才排入 `Hide`；隐藏后的鼠标捕获断言需继续等待目标控件的 `wait_for_refresh()`，并设置超时，确保已排队的事件完成处理。
拖动中断的各个场景使用独立 `run_test()` 应用实例，避免某次断言失败遗留的页面或 busy 状态引发连带失败。
光标阶段与点击、编辑后恢复的测试只将光标闪烁 Timer 以 `pause=True` 创建，由测试显式推进阶段，
并等待输入框刷新完成后断言；重复点击同一位置也应恢复可见阶段。真实 0.5 秒闪烁间隔及无额外重绘
由独立的 idle-cursor 测试验证，避免较慢的 `Pilot` 操作跨过闪烁周期而误报失败。

Headless 能检查布局、键盘路径和输出状态；SVG、tmux 与真实 VS Code/SSH 终端是不同证据。
通知回归需显式使用 `run_test(notifications=True)`；默认测试模式不显示通知，无法发现浮层缺字。
隐藏快捷命令栏后让通知覆盖底部按钮，核对最终 compositor 输出和终端更新字节中的汉字完整性；
只断言通知原文或 Toast 自身的内容不能证明叠加后的显示正确。
颜色回归还需检查实际渲染器输出的 SGR 颜色序列：Textual 的 SVG 导出固定使用真彩色，
单看 SVG 无法发现终端输出被降级为 256 色的问题。终端颜色变更覆盖输入选中、下拉菜单、
确认按钮与深浅主题；PTY 输出仍不能代替用户客户端实际显示的验收。
原生光标、剪贴板、闪烁和宿主快捷键路由只能在相应终端确认。交付注明实际验证环境；
流程见 [TUI 回归 Skill](../../.agents/skills/acprof-textual-regression/SKILL.md)，模块分工见[架构](consumers.md#tui-与兼容维护)。

## 真实采集与实验隔离

手动 `hardware.yml` 默认选择 `examples/hardware-validation.json` 的固定小模型矩阵，
也可选择 `custom` 单模型。矩阵中的 BERT fill-mask 与 ViT image-classification 使用固定
checkpoint SHA；它们是随机权重接口样例，验证采集链路，不代表生产模型质量或完整模型兼容性。
专用主机须预先准备对应 revision 和当前源码的 CPU/GPU 镜像，以及 host/test 锁环境。
脚本沿用 `skip_build=True` 优先复用已验证的镜像；缺少匹配镜像时仍会按正常流程构建。
执行前应核对所需缓存和下载预算，不能将此选项视为禁止下载。

```bash
.venv/bin/python scripts/check_hardware.py \
  --matrix examples/hardware-validation.json --gpus off,on \
  --output-dir internal-testing/hardware-matrix
```

矩阵 schema v1 要求 1–16 个 case，每项必须包含唯一安全目录名 `name`、`model`、40 位
`revision`、`task` 和 `input_scales`；CPU、内存、GPU 和采样参数沿用命令行公共选项。
全部配置先验证，再串行执行，各 case 使用独立结果目录。失败、清理错误或取消会停止后续 case，
`hardware_matrix.json` 保存源码/主机身份以及 `passed`、`failed`、`cancelled`、`not_run`，
中断时保留已完成证据；强制终止可能留下 `running`，不能解释为通过。
workflow 先将取消、恢复、监测器清理和采样失败的离线回归保存到 `offline-faults/`，再把真实
采集保存到 `measurements/`。离线故障回归使用 mock，不代表真实硬件故障已实测。

`.github/workflows/hardware.yml` 只支持手动触发，在带 `acprof` 标签的专用 Linux x86_64 runner
上使用预先准备的 `.venv`、本机 Docker/cgroup v2、RAPL、perf 和抓包权限。不会由 PR 自动触发。
本地同一入口为：

```bash
.venv/bin/python scripts/check_hardware.py --model hf-internal-testing/tiny-random-bert \
  --task fill-mask --gpus off,on --output-dir internal-testing/hardware-smoke
```

默认每个设备 1 个 case、warmup=1、repeat=3、2 秒请求窗口及 2 秒 idle，属于短 smoke。
`command.json`、`run.log`、原始结果及 `audit.json` 一同保留；验收要求正式行为 `ok`、计划完整，
且两种延迟、CPU 能量、PMU instructions 和 GPU 模式下的 GPU 能量均为有效数值。
未通过不自动更改原参数或覆盖产物。`--compute-profile-tool` / `--execution-profile-tool`
可另测指定工具，工具字段仍需根据其计划和错误列验收，不能用主采集成功代替工具成功。

采样线程及 CLI/TUI 开销的独立对照入口和统计假设见[指标分析](../results/comparisons.md#窗口置信区间与开销对照)。
`compare_ui.py --ui terminal` 继承当前终端，要求 stdout 为 TTY；自动化可用 `script` 分配 PTY
并保存会话。PTY、headless 和用户的 VS Code/SSH 终端须分别标明，不能互相替代。

镜像依赖变化后，主机 `.venv` 测试不能证明容器已更新；构建与复用契约见[运行兼容](../models/images.md#构建复用和验证)。
最小采集示例见[运行指南](../usage/experiments.md#3-跑一个最小-smoke-test)。用独立输出目录运行验证，保留模型 revision、输入计划与日志。
`examples/` 下脚本是手动接口示例，不会自动运行，也不产生与正式 `acprof run` 等价的测量证据。

`internal-testing/` 用于本地临时验证和截图；原始实验结果留在对应结果目录。
普通推理成功不能证明 Torch/NCU/Massif/Nsys 都支持；每种设备、dtype 和工具分别报告实际覆盖范围。
