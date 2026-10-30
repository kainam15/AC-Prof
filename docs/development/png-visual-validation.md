# 可选 PNG 终端视觉回归（VHS）

保留 [SVG/Textual 回归](devtools.md#辅助开发工具)作为稳定的控件、行为与结构基线；
另用 [charmbracelet/vhs](https://github.com/charmbracelet/vhs) 驱动 ttyd + xterm.js + 无头 Chrome，
以独立 PNG 检查中文字符、连续边框、截断、弹窗与实际终端模拟器的布局。
这是**开发测试工具**，不是运行时依赖，不在正式实验的测量窗口调用。

## 前置环境

需要在执行截图的 Linux 开发机或临时 CI runner 安装：
- VHS（固定 v0.12.1，支持 `Set Columns` / `Set Rows` 与 `Screenshot`；v0.12.0 存在 [退出码为零却不生成文件的上游缺陷](https://github.com/charmbracelet/vhs/issues/787)）
- `ttyd`、`ffmpeg`、Chrome/Chromium（支持 headless）、`fontconfig`
- 字体 `Noto Sans Mono CJK SC`（Ubuntu: `fonts-noto-cjk`）

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

# 显式启用基于真实 VHS 的 pytest 包装；普通 pytest 仅测 tape 生成，不启动 VHS
ACPROF_PNG_VISUAL=1 uv run pytest tests/visual_png/test_png_capture.py -v
```

默认结果为 `internal-testing/tui-png/{form,error-modal,near-match}.png`，目录已由 Git 忽略。
每张 PNG 使用 **120 列 × 36 行**、固定字体、16 px 字体大小、中文语言、
Graphite 主题、UTC、固定测试数据与隔离的临时 HOME/缓存。
`--all` 逐场景启动，避免一个场景遗留 UI 状态或本地设置影响另一个。

测试复用真实 `AcprofTui` 的页面与弹窗组件，但由
`tests/visual_png/fixture_app.py` 注入测试设置、模拟配置和错误；
禁用真实预检、采集子进程、历史结果读取与镜像刷新，设置离线模式；near-match 场景仅注入一份不访问宿主机的模拟环境检查成功状态。
不会解析下载模型、联系外部服务、采样真实硬件或写入正式实验结果。
VHS 只操作该测试应用，没有向 AC-Prof CLI 添加截图选项。

pytest 包装验证 PNG 格式、文件大小、合理的输出像素范围、所有场景尺寸相同，
以及不同场景实际输出的像素数据不同；**不设脆弱的逐像素基线**。
截图仍应由人检查中文字形、边框、截断和遮挡。若需严格对比，建议
先审阅固定环境下的 PNG 基线，再考虑 SSIM/感知差异；不能将字体渲染小差异视为必然回归。

## CI 与验证边界

独立的 [TUI PNG Visual](../../.github/workflows/tui-png.yml) 工作流通过
`workflow_dispatch` 按需运行，安装截图工具、执行 pytest 并上传 PNG artifacts。
**不会默认随 push/PR 运行**，也不影响现有 CI 中的 SVG 快照。

VHS 的 xterm.js + Chrome 反映的是一个确定的无头终端模拟环境，
不是 VS Code Windows 终端、SSH/tmux 或用户本地字体渲染的完全复刻；
涉及终端原生渲染差异时仍需在实际客户端截图验收。
参考上游 [VHS 官方命令文档](https://github.com/charmbracelet/vhs#vhs-command-reference)
与 [Textual 测试指导](https://textual.textualize.io/guide/testing/)。
