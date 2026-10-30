# PNG Golden Snapshot 视觉回归（VHS）

保留 [SVG/Textual 回归](devtools.md#辅助开发工具)作为稳定的控件、行为与结构基线；
另用 [charmbracelet/vhs](https://github.com/charmbracelet/vhs) 驱动 ttyd + xterm.js + 无头 Chrome，
通过 VHS 的 `Output frames/` PNG 帧序列取稳定的最终文本帧，检查中文字符、连续边框、截断、弹窗与实际终端模拟器的布局。这样不会依赖 VHS v0.12.1 偶发静默漏文件的单独 `Screenshot` 命令。
这是**开发测试工具**，不是运行时依赖，不在正式实验的测量窗口调用。

## 前置环境

需要在执行截图的 Linux 开发机或临时 CI runner 安装：
- VHS（固定 v0.12.1，支持 `Set Columns` / `Set Rows` 与 `Screenshot`；v0.12.0 存在 [退出码为零却不生成文件的上游缺陷](https://github.com/charmbracelet/vhs/issues/787)）
- `ttyd`、`ffmpeg`、Chrome/Chromium（支持 headless）、`fontconfig`
- 字体 `Noto Sans Mono CJK SC`（Ubuntu: `fonts-noto-cjk`）
- Golden 基线采用 Ubuntu 24.04 / VHS v0.12.1 / Chrome 147.0.7727.116；CI 从 Chrome for Testing 下载固定版本，并验证字体文件 SHA-256。跨系统运行可能产生噪声，请在与 CI 一致的环境制作基线。

Ubuntu 示例（VHS 本体可从 [官方 Releases](https://github.com/charmbracelet/vhs/releases) 下载、核验校验和后安装）：

```bash
sudo apt-get update
sudo apt-get install -y ttyd ffmpeg fonts-noto-cjk
fc-match 'Noto Sans Mono CJK SC'
uv sync --locked --no-default-groups --group test
```

无需 X11、Wayland、远程桌面、Windows、dor-jump、Docker、GPU 或真实模型；
Chrome 仅由 VHS 无头启动。系统安装的工具不会加入 `pyproject.toml` 正式依赖或任何推理镜像。

## 运行与结果

```bash
uv run python scripts/capture_tui_png.py --scenario form
uv run python scripts/capture_tui_png.py --scenario error-modal
uv run python scripts/capture_tui_png.py --scenario near-match
uv run python scripts/capture_tui_png.py --all --output internal-testing/tui-png

# 与 Git 中已审阅的 PNG 基线比较，失败时生成 diff 与 report.json
uv run python scripts/compare_tui_png.py

# 显式启用真实 VHS 截图 + Golden PNG 比较；普通 pytest 不要求 VHS
ACPROF_PNG_VISUAL=1 uv run pytest tests/visual_png -v
```

默认结果为 `internal-testing/tui-png/` 下的 6 张 PNG；其中 `form-small` 使用 **80×24**，其他场景使用 **120×36**，目录已由 Git 忽略。
所有场景使用固定字体、16 px 字体大小、中文语言、
Graphite 主题、UTC、固定测试数据与隔离的临时 HOME/缓存。
`--all` 逐场景启动，避免一个场景遗留 UI 状态或本地设置影响另一个。

测试复用真实 `AcprofTui` 的页面与弹窗组件，但由
`tests/visual_png/fixture_app.py` 注入测试设置、模拟配置和错误；
禁用真实预检、采集子进程、历史结果读取与镜像刷新，设置离线模式；near-match 场景仅注入一份不访问宿主机的模拟环境检查成功状态。
不会解析下载模型、联系外部服务、采样真实硬件或写入正式实验结果。
VHS 只操作该测试应用，没有向 AC-Prof CLI 添加截图选项。

## 长期维护的 Golden PNG

仓库将人工审阅的基线保存在 `tests/visual_png/__snapshots__/*.png`：

- `form`：中文普通表单（120×36）
- `form-small`：小终端、截断与滚动（80×24）
- `error-modal`：中文长错误弹窗
- `near-match`：BAAI/bge-m3 自定义配置
- `measuring`：模拟测量阶段与日志布局
- `task-selection`：多任务选择弹窗

比较器使用 Pillow 对每个 RGB 像素取最大通道差，差异小于 `32/255` 的像素视为渲染噪声，每张图**最多允许 120 个明显不同的像素**；尺寸不一致、缺失、无效图片或超过阈值均判失败。比较器只读基线，绝不会自动创建/修改它。失败时在 `internal-testing/tui-png/diff/` 生成红色差异覆盖图和 `diff/expected/` 中的原基线副本，同时生成机器可读的 `report.json`。

有意调整界面时，用相同的 Ubuntu/Chrome/VHS/字体环境生成候选截图：

```bash
uv run python scripts/capture_tui_png.py --all --output internal-testing/tui-png/candidates
# 仔细人工审查全部 6 张 PNG，不应包含中文重叠/边框缺口
cp internal-testing/tui-png/candidates/*.png tests/visual_png/__snapshots__/
ACPROF_PNG_VISUAL=1 uv run pytest tests/visual_png -v
```

必须在 PR 审阅中明确确认 Golden PNG 变化；**CI 绝不在测试失败时更新基线**。基线是经过审阅的变化检测依据，不代表程序能够从零判断中文排版是否美观。Textual 几何断言与 SVG 快照继续发挥互补作用。

## CI 与验证边界

独立的 [TUI PNG Visual](../../.github/workflows/tui-png.yml) 工作流在 **push main / Pull Request 修改 TUI、任务呈现、视觉夹具、锁文件或测试配置时自动运行**，也可通过 `workflow_dispatch` 手动触发。它安装截图工具、运行 PNG Golden 比对，并在失败时上传 actual / expected / diff / report.json artifacts；仍不影响原有 CI 的 SVG 快照。正式运行与完整测试环境无需安装 VHS 或桌面。

VHS 的 xterm.js + Chrome 反映的是一个确定的无头终端模拟环境，
不是 VS Code Windows 终端、SSH/tmux 或用户本地字体渲染的完全复刻；
涉及终端原生渲染差异时仍需在实际客户端截图验收。
参考上游 [VHS 官方命令文档](https://github.com/charmbracelet/vhs#vhs-command-reference)
与 [Textual 测试指导](https://textual.textualize.io/guide/testing/)。
